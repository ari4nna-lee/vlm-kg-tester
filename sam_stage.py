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
import cv2

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

def extract_rois(frame, prompts, padding=40):
    rois = []
    H, W = frame.shape[:2]

    for p in prompts:
        x1, y1, x2, y2 = p.box.astype(int)

        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(W, x2 + padding)
        y2 = min(H, y2 + padding)

        crop = frame[y1:y2, x1:x2]

        rois.append({
            "crop": crop,
            "offset": (x1, y1),
            "prompt": p
        })

    return rois

def project_mask(mask, offset, full_shape):
    H, W = full_shape
    ox, oy = offset

    full = np.zeros((H, W), dtype=bool)

    h, w = mask.shape
    full[oy:oy+h, ox:ox+w] = mask

    return full

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

class PromptFilter:
    def __init__(self, max_prompts=5):
        self.max_prompts = max_prompts

    def filter(self, prompts):
        # sort by priority (highest first)
        prompts = sorted(prompts, key=lambda p: p.priority, reverse=True)

        # keep top-k only
        return prompts[:self.max_prompts]

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

def _sam3_segment(self, processor, frame, prompts, cfg):

    # 1. filter prompts
    prompts = self.prompt_filter.filter(prompts)

    # 2. build ROIs
    rois = extract_rois(frame, prompts)

    results = []
    H, W = frame.shape[:2]

    # 3. per-ROI SAM inference
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for roi in rois:
            crop = roi["crop"]
            prompt = roi["prompt"]
            ox, oy = roi["offset"]

            image = Image.fromarray(crop)

            inference_state = processor.set_image(image)
            processor.reset_all_prompts(inference_state)

            # adjust box to crop coordinates
            x1, y1, x2, y2 = prompt.box.astype(int)
            box = np.array([
                x1 - ox,
                y1 - oy,
                x2 - ox,
                y2 - oy
            ])

            inference_state = processor.add_geometric_prompt(
                state=inference_state,
                box=box,
                label=True
            )

            masks = inference_state.get("masks")
            scores = inference_state.get("scores")

            if masks is None:
                continue

            mask = masks[0].detach().cpu().numpy()

            # collapse extra dims safely
            mask = np.squeeze(mask)

            # ensure binary
            mask = mask.astype(bool)

            if mask.ndim != 2:
                raise ValueError(f"Unexpected mask shape after squeeze: {mask.shape}")
            
            score = float(scores[0].detach().cpu()) if scores is not None else 1.0

            crop_h, crop_w = crop.shape[:2]
            if mask.shape != (crop_h, crop_w):
                mask_resized = cv2.resize(
                    mask.astype(np.uint8),
                    (crop_w, crop_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                mask_resized = mask

            # 4. project back to full frame
            print(f"crop shape: {crop.shape[:2]}, mask shape: {mask.shape}, offset: {(ox, oy)}, full: {(H, W)}")
            full_mask = project_mask(mask_resized, (ox, oy), (H, W))

            ys, xs = np.where(full_mask)
            cx_n = float(xs.mean()) / W if xs.size else 0.5
            cy_n = float(ys.mean()) / H if ys.size else 0.5

            results.append({
                "mask": full_mask,
                "confidence": score,
                "priority": round(0.6 * prompt.priority + 0.4 * score, 4),
                "centroid": (cx_n, cy_n),
                "mask_area": full_mask.sum() / (H * W),
                "prompt": prompt,
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
        self.prompt_filter = PromptFilter(max_prompts=5)

        if config.backend == "sam3":
            # 1. Enable TF32 for Ampere+ GPUs as seen in the working notebook
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

            sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
            bpe_path = f"{sam3_root}/sam3/assets/bpe_simple_vocab_16e6.txt.gz"

            with torch.device("cuda"):
                model = build_sam3_image_model(bpe_path=bpe_path)
            model = model.to(device="cuda").eval()

            # 2. Open the persistent global autocast context manager
            # torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

            self.processor = Sam3Processor(
                model,
                confidence_threshold=0.5,
            )
        else:
            self.processor = None

    def run(self, frame: np.ndarray, encoded_prompts: EncodedPrompts,) -> SegmentationOutput:

        if self.config.backend == "sam3":
            raw_results = _sam3_segment(
                self,
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

        raw_results = self._merge_overlapping(raw_results)

        masks = [
            self._to_instance_mask(
                r,
                encoded_prompts.prompts[i],
                i,
                encoded_prompts.image_hw
            )
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
            priority=raw.get("priority", prompt.priority),
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