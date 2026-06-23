"""
Pipeline — Top-level runner
=====================================
Wires Stage 1 (VLM) → Stage 1b (PromptEncoder) → Stage 2 (SAM) → Stage 3 (KG)
into a single callable.

Each stage is independently configurable and its output is preserved so
callers can inspect intermediate results without re-running earlier stages.

Typical usage
-------------
    import numpy as np
    from pipeline import Pipeline, PipelineConfig

    cfg = PipelineConfig()   # all stubs by default
    pipeline = Pipeline(cfg)

    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    result = pipeline.run(frame, task_prompt="find traversal zones and anomalies")

    print(result.vlm_output.scene_summary)
    print(f"Nodes: {len(result.seg_output.masks)}")
    print(f"Edges: {len(result.kg_result.output.edges)}")

    # NetworkX graph available for path planning / export.
    G = result.kg_result.graph
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from vlm_reasoning import VLMReasoningPass, VLMConfig
from prompt_encoder import PromptEncoder, EncoderConfig, EncodedPrompts
from sam_stage import SAMStage, SAMConfig
from kg import KGStage, KGConfig, KGRunResult
from structure import VLMSceneOutput, SegmentationOutput

from visualization.render_frame import overlay_masks
from visualization.render_graph import render_graph
from visualization.video_writer import VideoWriter, stitch

from heatmap import build_heatmap

log = logging.getLogger(__name__)

import queue
import threading
import time

from pathlib import Path
import cv2
import networkx as nx

out_w, out_h = 1280, 720

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    vlm:     VLMConfig     = field(default_factory=VLMConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    sam:     SAMConfig     = field(default_factory=SAMConfig)
    kg:      KGConfig      = field(default_factory=KGConfig)


# ---------------------------------------------------------------------------
# Per-frame result
# ---------------------------------------------------------------------------

@dataclass
class PipelineResult:
    """All intermediate and final outputs for one frame."""
    frame_id:       int
    task_prompt:    str
    vlm_output:     VLMSceneOutput
    encoded_prompts: EncodedPrompts
    seg_output:     SegmentationOutput
    kg_result:      KGRunResult


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    End-to-end aerial perception pipeline.

    Stages
    ------
    1.  VLMReasoningPass  — scene understanding + priority region extraction
    1b. PromptEncoder     — converts VLM regions to SAM point/box prompts
    2.  SAMStage          — instance segmentation → masks
    3.  KGStage           — knowledge graph construction + edge inference

    Parameters
    ----------
    config : PipelineConfig
        Nested config covering all four components.

    Notes
    -----
    The pipeline keeps the previous frame's SegmentationOutput in memory
    so that Tier 2 temporal edges can be computed on each frame without
    the caller needing to manage state.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.vlm     = VLMReasoningPass(self.config.vlm)
        self.encoder = PromptEncoder(self.config.encoder)
        self.sam     = SAMStage(self.config.sam)
        self.kg      = KGStage(self.config.kg)
        self._prev_seg: Optional[SegmentationOutput] = None
        self._frame_counter = 0
        self.threads = []

        self.frame_q = queue.Queue(maxsize=2)
        self.sam_q = queue.Queue(maxsize=2)
        self.kg_q = queue.Queue(maxsize=2)

        self.writer = cv2.VideoWriter(
            "output.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,
            (out_w, out_h)
        )

    # ------------------------------------------------------------------

    def frame_reader(self, frame_iterable, stop_event):
        for frame in frame_iterable:
            if stop_event.is_set():
                break
            frame_q.put(frame)

    def sam_worker(self):
        while True:
            frame = self.frame_q.get()

            seg_output = self.sam.run(
                frame,
                task_prompt="find traversable areas and cars"
            )

            self.sam_q.put((frame, seg_output))

    def kg_worker(self):
        while True:
            frame, seg_output = sam_q.get()

            heatmap = build_heatmap(seg_output.masks)

            kg_result = self.kg.run(
                seg_output,
                prev_seg_output=self._prev_seg
            )

            self._prev_seg = seg_output

            self.kg_q.put((frame, seg_output, kg_result))

    def vlm_worker(self, interval=5):
        i = 0
        self.latest_output = None

        while True:
            frame, seg_output, kg_result = kg_q.get()

            vlm_output = None
            if i % interval == 0:
                vlm_output = self.vlm.run(
                    frame,
                    task_prompt="describe scene and vehicles"
                )

            i += 1

            self.latest_output = (frame, seg_output, kg_result, vlm_output)

    def start_workers(self):

        self.threads = [
            threading.Thread(target=self.sam_worker, daemon=True),
            threading.Thread(target=self.kg_worker, daemon=True),
            threading.Thread(target=self.vlm_worker, daemon=True),
            threading.Thread(target=self.render_loop, daemon=True),
        ]

        for t in self.threads:
            t.start()

    def render_loop(self):
        import time

        while True:
            if not hasattr(self, "latest_output") or self.latest_output is None:
                time.sleep(0.01)
                continue

            frame, seg, kg, vlm = self.latest_output

            frame_viz = overlay_masks(frame, seg.masks)
            graph_img = render_graph(kg.graph)

            combined = stitch(frame_viz, graph_img)

            cv2.imshow("pipeline", combined)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def run(self, frame: np.ndarray, task_prompt: str, timestamp: float = 0.0,) -> PipelineResult:
        """
        Run the full pipeline on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            H×W×3 uint8 RGB image.
        task_prompt : str
            Natural-language task description (can change per frame).
        timestamp : float
            Optional sensor/wall-clock timestamp.

        Returns
        -------
        PipelineResult
        """
        fid = self._frame_counter
        self._frame_counter += 1
        log.info("=== Frame %d ===", fid)

        # Stage 1 — VLM scene reasoning
        vlm_output = self.vlm.run(
            frame=frame,
            task_prompt=task_prompt,
            frame_id=fid,
            timestamp=timestamp,
        )
        log.info("Stage 1 done: %d priority regions", len(vlm_output.priority_regions))

        # Stage 1b — Prompt encoding
        encoded = self.encoder.encode(vlm_output)

        log.info("Prompt encoding done: %d SAM prompts (%d skipped)",
                 len(encoded.prompts), len(encoded.skipped_regions))
        
        # -----------------------------
        # VALIDATE ENCODER OUTPUT
        # -----------------------------
        assert hasattr(encoded, "prompts"), "EncodedPrompts missing .prompts"
        assert hasattr(encoded, "image_hw"), "EncodedPrompts missing .image_hw"
        assert encoded.image_hw is not None, "Encoder must set image_hw (H, W)"

        from prompt_encoder import SAMPrompt

        assert len(encoded.prompts) > 0, "No SAM prompts generated"
        assert all(
            isinstance(p, SAMPrompt) for p in encoded.prompts
        ), "Encoder must output list[SAMPrompt]"

        # Stage 2 — SAM segmentation
        seg_output = self.sam.run(frame=frame, encoded_prompts=encoded)

        if len(seg_output.masks) == 0:
            log.warning("No masks returned from SAM — skipping KG stage for this frame")
            return PipelineResult(
                frame_id=fid,
                task_prompt=task_prompt,
                vlm_output=vlm_output,
                encoded_prompts=encoded,
                seg_output=seg_output,
                kg_result = KGRunResult(
                    graph=self.kg.empty_graph(),
                    output=self.kg.empty_output(),
                )
            )

        log.info("Stage 2 done: %d masks", len(seg_output.masks))

        heatmap = build_heatmap(
            masks=[m.mask_array for m in seg_output.masks if m.mask_array is not None],
            priorities=[m.priority for m in seg_output.masks if m.mask_array is not None],
            image_hw=frame.shape[:2],
        )

        encoded.prompts = self.encoder.bias_prompts_with_heatmap(encoded.prompts, heatmap)

        # Stage 3 — Knowledge graph construction
        kg_result = self.kg.run(
            seg_output=seg_output,
            prev_seg_output=self._prev_seg,
        )

        self.kg.update_from_heatmap(kg_result.graph, heatmap)

        log.info(
            "Stage 3 done: nodes=%d  edges=%d  traversability_edges=%d",
            kg_result.graph.number_of_nodes(),
            kg_result.graph.number_of_edges(),
            kg_result.traversability.number_of_edges(),
        )

        self._prev_seg = seg_output

        return PipelineResult(
            frame_id=fid,
            task_prompt=task_prompt,
            vlm_output=vlm_output,
            encoded_prompts=encoded,
            seg_output=seg_output,
            kg_result=kg_result,
        )

    def reset(self) -> None:
        """Clear temporal state (call when starting a new sequence)."""
        self._prev_seg = None
        self._frame_counter = 0
        log.info("Pipeline state reset.")

IMAGE_DIR = Path("/home/arianna/vlm-kg-tester/tests/neighborhood_subset")
frames = sorted(IMAGE_DIR.glob("*.jpg"))

cfg = PipelineConfig(
    vlm=VLMConfig(
        backend="gemma",
        model_name="google/gemma-4-E2B-it",
        grpc_endpoint="http://localhost:8000/v1",
        device="cuda",
        temperature=0.2
    ),
    sam=SAMConfig(
        backend="sam3",
        device="cuda"
    )
)

G = nx.DiGraph()
writer = VideoWriter("output.mp4")
pipeline = Pipeline(cfg)

for i, frame_path in enumerate(frames):
    frame = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)

    result = pipeline.run(
        frame,
        task_prompt="find traversable areas and identify locations where you would find cars"
    )

    frame_viz = overlay_masks(frame, result.seg_output.masks)

    G = nx.compose(G, result.kg_result.graph)
    graph_img = render_graph(G)

    combined = stitch(frame_viz, graph_img)

    combined = cv2.resize(combined, (out_w, out_h))
    #cv2.imshow("pipeline", combined)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

    writer.write(combined)

    print(f"\nFrame {i}")
    print(result.vlm_output.scene_summary)
    print(len(result.seg_output.masks))

cv2.destroyAllWindows()
writer.close()
