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
from visualization.video_writer import stitch

from heatmap import build_heatmap

log = logging.getLogger(__name__)

import queue
import threading
import time

from pathlib import Path
import cv2
import networkx as nx

import os
import json
from networkx.readwrite import json_graph

OUTPUT_DIR = Path("./results")
JSON_DIR = OUTPUT_DIR / "json"
GRAPH_DIR = OUTPUT_DIR / "graph"
HEATMAP_DIR = OUTPUT_DIR / "heatmaps"

JSON_DIR.mkdir(parents=True, exist_ok=True)
GRAPH_DIR.mkdir(parents=True, exist_ok=True)
HEATMAP_DIR.mkdir(parents=True, exist_ok=True)

TARGET_H, TARGET_W = 720, 1280

writer = cv2.VideoWriter(
            "output.avi",
            cv2.VideoWriter_fourcc(*"XVID"),
            2,
            (TARGET_W, TARGET_H)
        )

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    force=True,
)

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

@dataclass
class KGItem:
    fid: int
    frame: np.ndarray
    vlm_output: object
    seg_output: object
    encoded_prompts: object
    heatmap: np.ndarray | None
    prev_seg: object | None
    prev_heatmap: np.ndarray | None

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
        log.info("Building VLM...")
        self.encoder = PromptEncoder(self.config.encoder)
        log.info("Building encoder...")
        self.sam     = SAMStage(self.config.sam)
        log.info("Building SAM stage...")
        self.kg      = KGStage(self.config.kg)
        log.info("Building KG stage...")
        self._prev_seg: Optional[SegmentationOutput] = None
        self._frame_counter = 0
        self.threads = []
        log.info("Pipeline init complete.")

        self.kg_q = queue.Queue(maxsize=128)
        self.vlm_q = queue.Queue(maxsize=128)
        self.sam_q = queue.Queue(maxsize=128)
        self.output_q = queue.Queue(maxsize=64)
        self.sam_refine_q = queue.Queue(maxsize=32)

        self.latest_output = None
        self._kg_graph = nx.DiGraph()

        self._prev_heatmap: Optional[np.ndarray] = None

        self._output_lock = threading.Lock()
        self._graph_lock = threading.Lock()

        self.total_frames = len(frames)
        self.processed_frames = 0

        self._last_vlm_output = None
        self.vlm_skip_interval = 2  # run VLM every n frame

    # ------------------------------------------------------------------

    def frame_reader(self, frame_iterable, stop_event):
        for frame in frame_iterable:
            if stop_event.is_set():
                break
            self.vlm_q.put(frame)

    def vlm_worker(self):
        try:
            while True:
                try:
                    item = self.vlm_q.get(timeout=1)
                except queue.Empty:
                    continue
                if item is None:
                    self.sam_q.put(None)
                    break

                fid, frame = item

                log.info("vlm_worker received frame %s", fid)

                if fid % self.vlm_skip_interval == 0 or self._last_vlm_output is None:
                    vlm_output = self.vlm.run(
                        frame=frame,
                        task_prompt="describe the scene",
                        frame_id=fid,
                        timestamp=0.0,
                    )
                    self._last_vlm_output = vlm_output
                    log.info("vlm_worker ran VLM for frame %s", fid)
                else:
                    vlm_output = self._last_vlm_output
                    log.info("vlm_worker skipped VLM for frame %s, reusing frame %s output",
                            fid, fid - (fid % self.vlm_skip_interval))

                encoded = self.encoder.encode(vlm_output)
                self.sam_q.put((fid, frame, vlm_output, encoded))


        except queue.Empty:
            log.warning("VLM worker stalled waiting for input")

        except Exception as e:
            log.exception("vlm_worker crashed: %s", e)
            self.sam_q.put(None)

    def sam_worker(self):
        try:
            while True:
                # --- get next item, falling back to refine queue ---
                try:
                    item = self.sam_q.get(timeout=1)
                except queue.Empty:
                    try:
                        item = self.sam_refine_q.get_nowait()
                    except queue.Empty:
                        continue

                # --- shutdown signal ---
                if item is None:
                    self.kg_q.put(None)
                    return

                # --- unpack based on item type ---
                if isinstance(item, dict) and item.get("type") == "kg_refine":
                    if item["frame"] is None or item.get("fid") is None:
                        continue
                    fid = item["fid"]
                    frame = item["frame"]
                    vlm_output = None
                    encoded = self.encoder.encode_from_text(item["prompts"])
                else:
                    fid, frame, vlm_output, encoded = item

                seg_output = self.sam.run(frame=frame, encoded_prompts=encoded)

                log.info("sam_worker received frame %s", fid)

                # --- heatmap ---
                masks = [m for m in seg_output.masks if m.mask_array is not None]
                heatmap = None
                log.info("sam_worker building heatmap for frame %s (%d masks)", fid, len(masks))
                if masks:
                    heatmap = build_heatmap(
                        masks=[m.mask_array for m in masks],
                        priorities=[m.priority for m in masks],
                        image_hw=frame.shape[:2],
                    )

                log.info("sam_worker finished heatmap for frame %s", fid)
                self.kg_q.put(KGItem(
                    fid=fid,
                    frame=frame,
                    vlm_output=vlm_output,
                    seg_output=seg_output,
                    encoded_prompts=encoded,
                    heatmap=heatmap,
                    prev_seg=self._prev_seg,
                    prev_heatmap=None
                ))
                log.info("sam_worker put frame %s onto kg_q", fid)

                if heatmap is not None:
                    vis = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
                    vis = vis.astype(np.uint8)
                    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
                    cv2.imwrite(str(HEATMAP_DIR / f"frame_{fid:05d}.png"), vis)

        except Exception as e:
            log.exception("sam_worker crashed: %s", e)
            self.kg_q.put(None)  # ensure kg_worker doesn't hang if sam dies

    def kg_worker(self):
        try:
            while True:
                try:
                    item = self.kg_q.get(timeout=1)
                except queue.Empty:
                    continue

                if item is None:
                    with self._output_lock:
                        self.latest_output = "DONE"
                    return

                # -------------------------
                # unpack structured input
                # -------------------------
                fid = item.fid
                frame = item.frame
                vlm_output = item.vlm_output
                seg_output = item.seg_output
                encoded_prompts = item.encoded_prompts
                prev_seg = item.prev_seg

                log.info("kg_worker received frame %s", fid)

                heatmap = item.heatmap

                # -------------------------
                # KG reasoning step
                # -------------------------
                kg_result = self.kg.run(
                    seg_output,
                    prev_seg_output=prev_seg,
                    heatmap=heatmap,
                    vlm_output=vlm_output
                )
                with self._graph_lock:
                    self._kg_graph = nx.compose(self._kg_graph, kg_result.graph)
                    log.info("kg_worker finished frame %s, setting latest_output", fid)

                # -------------------------
                # FEEDBACK LOOP (this is what you were missing)
                # -------------------------
                refined_prompts = getattr(kg_result, "refined_prompts", None)

                if refined_prompts:
                    self.sam_refine_q.put({
                        "type": "kg_refine",
                        "fid": fid,
                        "frame": frame,
                        "prompts": refined_prompts
                    })

                    log.info("KG refined prompts for frame %s: %s", fid, refined_prompts)

                # -------------------------
                # expose latest output
                # -------------------------
                with self._output_lock:
                    self.latest_output = (
                        fid,
                        frame,
                        seg_output,
                        kg_result,
                        vlm_output,
                        heatmap
                    )
                    self.processed_frames += 1

        except Exception as e:
            log.exception("kg_worker crashed: %s", e)
            with self._output_lock:
                self.latest_output = "DONE"   # or signal failure
            return
        
    def output_worker(self):
        while True:
            try:
                item = self.output_q.get(timeout=2)
            except queue.Empty:
                continue
            if item is None:
                break

            fid, frame, seg_output, kg_result, vlm_output, heatmap = item
            with self._graph_lock:
                n_nodes = self._kg_graph.number_of_nodes()
                n_edges = self._kg_graph.number_of_edges()

            frame_data = {
                "frame_id": fid,
                "task_prompt": "Identify vehicles and areas where you would most likely find a truck",

                "scene_summary": getattr(vlm_output, "scene_summary", None),

                "num_objects": len(seg_output.masks),

                "objects": [
                    {
                        "id": getattr(m, "node_id", None),
                        "priority": getattr(m, "priority", None)
                    }
                    for m in seg_output.masks
                ],
                "global_graph_nodes": n_nodes,
                "global_graph_edges": n_edges,
                "graph": json_graph.node_link_data(kg_result.graph)
            }
            
            log.info("WRITING JSON for frame %s", fid)
            with open(JSON_DIR / f"frame_{fid:05d}.json", "w") as f:
                json.dump(frame_data, f, indent=2)

            frame_viz = overlay_masks(frame, seg_output.masks)

            if heatmap is not None:
                heatmap_vis = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_vis, cv2.COLORMAP_JET)
                heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
                frame_viz = cv2.addWeighted(frame_viz, 0.6, heatmap_color, 0.4, 0)

            graph_img = render_graph(kg_result.graph)

            cv2.imwrite(str(GRAPH_DIR / f"frame_{fid:05d}.png"), cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR))
            frame_viz_bgr = cv2.cvtColor(frame_viz, cv2.COLOR_RGB2BGR)
            graph_bgr = cv2.cvtColor(graph_img, cv2.COLOR_RGB2BGR)
            combined = stitch(frame_viz_bgr, graph_bgr)
            combined = cv2.resize(combined, (TARGET_W, TARGET_H))

            writer.write(combined)
            log.info("output_worker wrote frame %s", fid)

    def start_workers(self):

        self.threads = [
            threading.Thread(target=self.sam_worker, daemon=True),
            threading.Thread(target=self.kg_worker, daemon=True),
            threading.Thread(target=self.vlm_worker, daemon=True),
            threading.Thread(target=self.output_worker, daemon=True)
        ]

        for t in self.threads:
            t.start()

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

pipeline = Pipeline(cfg)
pipeline._cached_vlm = None
pipeline._prev_heatmap = None
pipeline.start_workers()

def feed_frames():
    log.info("feed_frames starting, %d frames found", len(frames))
    for fid, frame_path in enumerate(frames):
        frame = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (TARGET_W, TARGET_H))
        log.info("queuing frame %s (%s)", fid, frame_path.name)
        pipeline.vlm_q.put((fid, frame))
    pipeline.vlm_q.put(None)
    log.info("feed_frames done")

feed_thread = threading.Thread(target=feed_frames, daemon=True)
feed_thread.start()

while True:
    if not hasattr(pipeline, "latest_output") or pipeline.latest_output is None:
        time.sleep(0.01)
        continue

    with pipeline._output_lock:
        result = pipeline.latest_output
        pipeline.latest_output = None

    if pipeline.processed_frames >= pipeline.total_frames:
        break

    if result == "DONE" or pipeline.processed_frames >= pipeline.total_frames:
        pipeline.output_q.put(None)
        print("All frames processed.")
        break

    pipeline.output_q.put(result)

cv2.destroyAllWindows()
writer.release()
print(f"Done.")
