"""
Stage 2 — SAM Segmentation
============================
Receives EncodedPrompts from the prompt encoder and runs SAM3
(or a lightweight stub) to produce instance masks.

Each mask becomes one InstanceMask, which is the direct input for
Stage 3's node constructor.

Supported backends
------------------
  sam3      — SAM3 (drop-in compatible interface assumed)
  nanosam   — NVIDIA NanoSAM for Jetson on-device use
  stub      — Deterministic fake masks, no GPU needed

Typical usage
-------------
    from sam import SAMStage, SAMConfig
    from prompt_encoder import PromptEncoder, EncoderConfig

    encoder = PromptEncoder(EncoderConfig(image_hw=(720, 1280)))
    encoded = encoder.encode(vlm_output)

    seg = SAMStage(SAMConfig(backend="sam3", checkpoint="checkpoints/sam3_hiera_large.pt"))
    result = seg.run(frame=img_array, encoded_prompts=encoded)
"""

from __future__ import annotations

import os
import sys
import time
import logging
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from PIL import Image
from prompt_encoder import SAMPrompt
import torch

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model.box_ops import box_xywh_to_cxcywh
from sam3.visualization_utils import normalize_bbox

from prompt_encoder import EncodedPrompts
from structure import InstanceMask, SegmentationOutput, BBox

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SAMConfig:
    backend: Literal["sam3", "nanosam", "stub"] = "stub"
    checkpoint: str = ""            # path to model weights
    config_file: str = ""           # SAM3 model config yaml
    device: str = "cuda"
    # Confidence threshold: masks below this are discarded.
    mask_confidence_threshold: float = 0.5
    # If True, run tracking (between frames.
    use_tracking: bool = False
    # IoU threshold for merging overlapping masks from different prompts.
    merge_iou_threshold: float = 0.8


# ---------------------------------------------------------------------------
# Stub backend
# ---------------------------------------------------------------------------

def _stub_segment(
    frame: np.ndarray, prompts: list[SAMPrompt], cfg: SAMConfig
) -> list[dict]:
    """
    Returns fake masks shaped correctly for testing.
    Each mask is a small filled rectangle around the prompt bbox.
    """
    H, W = frame.shape[:2]
    results = []
    for i, prompt in enumerate(prompts):
        if prompt.box is not None:
            x1, y1, x2, y2 = prompt.box.astype(int)
        else:
            cx, cy = int(prompt.point_coords[0, 0]), int(prompt.point_coords[0, 1])
            pad = 40
            x1, y1 = max(0, cx - pad), max(0, cy - pad)
            x2, y2 = min(W, cx + pad), min(H, cy + pad)

        mask = np.zeros((H, W), dtype=bool)
        mask[y1:y2, x1:x2] = True
        area = mask.sum() / (H * W)
        cx_n = ((x1 + x2) / 2) / W
        cy_n = ((y1 + y2) / 2) / H

        results.append({
            "mask": mask,
            "confidence": round(0.75 + 0.2 * (prompt.priority), 3),
            "centroid": (cx_n, cy_n),
            "mask_area": round(area, 4),
            "prompt": prompt,
        })
    return results

def _sam3_segment(processor, frame: np.ndarray, prompts: list[SAMPrompt], cfg: SAMConfig,) -> list[dict]:

    import torch
    import numpy as np
    from PIL import Image

    from sam3.model.box_ops import box_xywh_to_cxcywh
    from sam3.model.sam3_image_processor import normalize_bbox

    # -------------------------------------------------------
    # Convert image ONCE
    # -------------------------------------------------------
    image = Image.fromarray(frame)
    H, W = frame.shape[:2]

    # -------------------------------------------------------
    # Create ONE inference state
    # -------------------------------------------------------
    inference_state = processor.set_image(image)
    processor.reset_all_prompts(inference_state)

    # -------------------------------------------------------
    # BATCH PROMPT INJECTION (KEY OPTIMIZATION)
    # -------------------------------------------------------
    box_prompts = 0
    point_prompts = 0

    for prompt in prompts:

        # ---------------------------
        # BOX PROMPTS
        # ---------------------------
        if prompt.box is not None:

            box = torch.tensor(prompt.box).view(-1, 4)

            box_cxcywh = box_xywh_to_cxcywh(box)

            norm_box = normalize_bbox(
                box_cxcywh, W, H
            ).flatten().tolist()

            inference_state = processor.add_geometric_prompt(
                state=inference_state,
                box=norm_box,
                label=True
            )

            box_prompts += 1

        # ---------------------------
        # POINT PROMPTS (optional)
        # ---------------------------
        elif len(getattr(prompt, "point_coords", [])) > 0:

            x, y = prompt.point_coords[0]

            norm_box = [
                x / W,
                y / H,
                0.01,
                0.01
            ]

            inference_state = processor.add_geometric_prompt(
                state=inference_state,
                box=norm_box,
                label=True
            )

            point_prompts += 1

    # -------------------------------------------------------
    # SINGLE INFERENCE RESOLUTION
    # -------------------------------------------------------
    masks = None
    scores = None

    if isinstance(inference_state, dict):
        masks = inference_state.get("masks", None)
        scores = inference_state.get("scores", None)

    if masks is None and hasattr(processor, "model"):
        with torch.no_grad():
            outputs = processor.model(inference_state)

        masks = outputs.get("masks", None)
        scores = outputs.get("scores", None)

    if masks is None:
        raise RuntimeError(
            "SAM3 batch inference failed. "
            f"inference_state keys: {list(inference_state.keys())}"
        )

    # -------------------------------------------------------
    # FLATTEN RESULTS
    # -------------------------------------------------------
    results = []

    # ensure iterable
    if isinstance(masks, np.ndarray) and masks.ndim == 3:
        masks = list(masks)

    num_masks = len(masks)

    for i in range(num_masks):

        mask = masks[i].astype(bool)
        score = float(scores[i]) if scores is not None else 1.0

        if score < cfg.mask_confidence_threshold:
            continue

        ys, xs = np.where(mask)

        if xs.size == 0:
            continue

        cx_n = float(xs.mean()) / W
        cy_n = float(ys.mean()) / H

        area = mask.sum() / (H * W)

        # IMPORTANT:
        # We cannot assume 1:1 mapping between prompts and masks
        # SAM3 may merge prompts internally

        results.append({
            "mask": mask,
            "confidence": score,
            "centroid": (cx_n, cy_n),
            "mask_area": float(area),
        })

    return results

# ---------------------------------------------------------------------------
# NanoSAM backend (Jetson on-device)
# ---------------------------------------------------------------------------

def _nanosam_segment(
    frame: np.ndarray, prompts: list[SAMPrompt], cfg: SAMConfig
) -> list[dict]:
    """
    Uses NVIDIA NanoSAM (distilled encoder for Jetson).
    Install: pip install nanosam
    Checkpoint: download from NVIDIA NGC.
    """
    try:
        from nanosam.utils.predictor import Predictor
    except ImportError:
        raise RuntimeError("nanosam not installed. pip install nanosam")

    predictor = Predictor(
        image_encoder=cfg.config_file or "resnet18_image_encoder.engine",
        mask_decoder=cfg.checkpoint or "mobile_sam_mask_decoder.engine",
    )
    predictor.set_image(frame)

    results = []
    H, W = frame.shape[:2]
    for prompt in prompts:
        # NanoSAM takes a single center point.
        cx, cy = prompt.point_coords[0]
        mask, iou = predictor.predict(np.array([[cx, cy]]), np.array([1]))
        mask = mask[0, 0].astype(bool)  # (H, W)
        score = float(iou[0, 0])
        if score < cfg.mask_confidence_threshold:
            continue

        ys, xs = np.where(mask)
        cx_n = float(xs.mean()) / W if xs.size else 0.5
        cy_n = float(ys.mean()) / H if ys.size else 0.5
        area = mask.sum() / (H * W)

        results.append({
            "mask": mask,
            "confidence": round(score, 4),
            "centroid": (cx_n, cy_n),
            "mask_area": round(area, 4),
            "prompt": prompt,
        })
    return results


_BACKENDS = { 
    "nanosam": _nanosam_segment,
    "stub":    _stub_segment,
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SAMStage:

    def __init__(self, config: SAMConfig):
        self.config = config
        self._track_id_counter = 0

        if config.backend == "sam3":

            sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
            bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

            model = build_sam3_image_model(
                bpe_path=bpe_path
            ).cuda().eval()

            self.processor = Sam3Processor(
                model,
                confidence_threshold=0.5
            )

        else:
            self.processor = None

    def run(self, frame: np.ndarray, encoded_prompts: EncodedPrompts,) -> SegmentationOutput:

        if self.config.backend == "sam3":
            raw_results = _sam3_segment(
                self.processor,
                frame,
                encoded_prompts.prompts,
                self.config
            )
        else:
            raw_results = _BACKENDS[self.config.backend](
                frame,
                encoded_prompts.prompts,
                self.config
            )

        # --------------------------------------------------
        # YOU WERE MISSING THIS ENTIRE PIPELINE STEP
        # --------------------------------------------------
        raw_results = self._merge_overlapping(raw_results)

        masks = [
            self._to_instance_mask(r, i, encoded_prompts.image_hw)
            for i, r in enumerate(raw_results)
        ]

        return SegmentationOutput(
            masks=masks,
            frame_id=encoded_prompts.frame_id,
            sam_model=self.config.backend,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _to_instance_mask(
        self,
        raw: dict,
        prompt: SAMPrompt,
        index: int,
        image_hw: tuple[int, int],
    ):
        H, W = image_hw
        mask_arr: Optional[np.ndarray] = raw.get("mask")

        # Derive pixel-space bbox from mask if available.
        if mask_arr is not None:
            ys, xs = np.where(mask_arr)
            if xs.size:
                bx = float(xs.min()) / W
                by = float(ys.min()) / H
                bw = float(xs.max() - xs.min()) / W
                bh = float(ys.max() - ys.min()) / H
            else:
                bx, by, bw, bh = 0.0, 0.0, 0.0, 0.0
        else:
            b = prompt.box
            bx, by = float(b[0]) / W, float(b[1]) / H
            bw, bh = float(b[2] - b[0]) / W, float(b[3] - b[1]) / H

        track_id = self._next_track_id() if self.config.use_tracking else -1

        return InstanceMask(
            node_id=f"node_{index:03d}",
            label=prompt.label,
            semantic_class=prompt.semantic_class,
            bbox=BBox(x=bx, y=by, w=bw, h=bh),
            mask_array=mask_arr,
            centroid=raw["centroid"],
            mask_area=raw["mask_area"],
            confidence=raw["confidence"],
            track_id=track_id,
            source_region_id=prompt.region_id,
            priority=prompt.priority,
        )

    def _merge_overlapping(self, results: list[dict]) -> list[dict]:
        """Remove duplicate masks with IoU above threshold."""
        if len(results) < 2:
            return results

        keep = list(range(len(results)))
        for i in range(len(results)):
            if i not in keep:
                continue
            for j in range(i + 1, len(results)):
                if j not in keep:
                    continue
                m_i = results[i].get("mask")
                m_j = results[j].get("mask")
                if m_i is None or m_j is None:
                    continue
                intersection = (m_i & m_j).sum()
                union = (m_i | m_j).sum()
                iou = intersection / union if union > 0 else 0.0
                if iou > self.config.merge_iou_threshold:
                    # Keep the one with higher confidence.
                    if results[i]["confidence"] >= results[j]["confidence"]:
                        keep.remove(j)
                    else:
                        keep.remove(i)
                        break

        return [results[k] for k in sorted(set(keep))]

    def _next_track_id(self) -> int:
        tid = self._track_id_counter
        self._track_id_counter += 1
        return tid