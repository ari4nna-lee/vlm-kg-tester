"""
Stage 2 — SAM Segmentation
============================
Receives EncodedPrompts from the prompt encoder and runs SAM2 / SAM3
(or a lightweight stub) to produce instance masks.

Each mask becomes one InstanceMask, which is the direct input for
Stage 3's node constructor.

Supported backends
------------------
  sam2      — Meta SAM2 via sam2 Python package
  sam3      — SAM3 (drop-in compatible interface assumed)
  nanosam   — NVIDIA NanoSAM for Jetson on-device use
  edgetam   — EdgeTAM lightweight tracker (seeds from SAM, propagates cheaply)
  stub      — Deterministic fake masks, no GPU needed

Typical usage
-------------
    from sam import SAMStage, SAMConfig
    from prompt_encoder import PromptEncoder, EncoderConfig

    encoder = PromptEncoder(EncoderConfig(image_hw=(720, 1280)))
    encoded = encoder.encode(vlm_output)

    seg = SAMStage(SAMConfig(backend="sam2", checkpoint="checkpoints/sam2_hiera_large.pt"))
    result = seg.run(frame=img_array, encoded_prompts=encoded)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

from prompt_encoder import EncodedPrompts, SAMPrompt
from structure import (
    BBox, InstanceMask, SemanticClass, SegmentationOutput,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class SAMConfig:
    backend: Literal["sam2", "sam3", "nanosam", "edgetam", "stub"] = "stub"
    checkpoint: str = ""            # path to model weights
    config_file: str = ""           # SAM2 model config yaml
    device: str = "cuda"
    # Confidence threshold: masks below this are discarded.
    mask_confidence_threshold: float = 0.5
    # If True, run tracking (EdgeTAM / SAM2 memory) between frames.
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


# ---------------------------------------------------------------------------
# SAM2 backend
# ---------------------------------------------------------------------------

def _sam2_segment(
    frame: np.ndarray, prompts: list[SAMPrompt], cfg: SAMConfig
) -> list[dict]:
    """
    Uses Meta's SAM2 Python package.
    Install: pip install sam2   (or from source: facebookresearch/sam2)
    """
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ImportError:
        raise RuntimeError(
            "SAM2 not installed. pip install sam2 or clone facebookresearch/sam2"
        )

    predictor = SAM2ImagePredictor(
        build_sam2(cfg.config_file, cfg.checkpoint, device=cfg.device)
    )
    predictor.set_image(frame)

    results = []
    for prompt in prompts:
        masks, scores, _ = predictor.predict(
            point_coords=prompt.point_coords if len(prompt.point_coords) > 0 else None,
            point_labels=prompt.point_labels if len(prompt.point_labels) > 0 else None,
            box=prompt.box,
            multimask_output=False,
        )
        # masks: (1, H, W); scores: (1,)
        mask = masks[0].astype(bool)
        score = float(scores[0])
        if score < cfg.mask_confidence_threshold:
            continue

        H, W = mask.shape
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


# ---------------------------------------------------------------------------
# EdgeTAM backend (tracking mode — amortizes VLM calls)
# ---------------------------------------------------------------------------

def _edgetam_segment(
    frame: np.ndarray, prompts: list[SAMPrompt], cfg: SAMConfig
) -> list[dict]:
    """
    EdgeTAM: seeds masks on first frame from SAM-style prompts, then
    propagates cheaply via on-device tracker.

    This stub seeds with SAM2 on the first call; subsequent calls
    propagate without re-running the full predictor.
    """
    try:
        import torch
        from edgetam import EdgeTAMPredictor  # hypothetical import
    except ImportError:
        log.warning("EdgeTAM not available, falling back to stub.")
        return _stub_segment(frame, prompts, cfg)

    # EdgeTAM maintains internal state between frames — real implementation
    # should keep the predictor as a class-level attribute and call
    # propagate_in_video() on subsequent frames.
    raise NotImplementedError(
        "EdgeTAM integration requires stateful predictor management. "
        "Instantiate EdgeTAMPredictor at the SAMStage level and call "
        "predictor.propagate() for frame N>0."
    )


_BACKENDS = {
    "sam2":    _sam2_segment,
    "sam3":    _sam2_segment,    # SAM3 is API-compatible; swap checkpoint only
    "nanosam": _nanosam_segment,
    "edgetam": _edgetam_segment,
    "stub":    _stub_segment,
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class SAMStage:
    """
    Stage 2 of the pipeline.

    Parameters
    ----------
    config : SAMConfig

    Example
    -------
    >>> stage = SAMStage(SAMConfig(backend="stub"))
    >>> out = stage.run(frame, encoded_prompts)
    >>> for m in out.masks:
    ...     print(m.node_id, m.semantic_class, m.confidence)
    """

    def __init__(self, config: SAMConfig):
        self.config = config
        self._track_id_counter = 0

    def run(
        self,
        frame: np.ndarray,
        encoded_prompts: EncodedPrompts,
    ) -> SegmentationOutput:
        """
        Parameters
        ----------
        frame : np.ndarray
            H×W×3 uint8 RGB image (same frame passed to Stage 1).
        encoded_prompts : EncodedPrompts
            Output of PromptEncoder.encode().

        Returns
        -------
        SegmentationOutput
            List of InstanceMask objects ready for Stage 3.
        """
        assert frame.ndim == 3 and frame.shape[2] == 3

        t0 = time.perf_counter()
        raw_results = _BACKENDS[self.config.backend](
            frame, encoded_prompts.prompts, self.config
        )
        # Deduplicate highly overlapping masks from different prompts.
        raw_results = self._merge_overlapping(raw_results)

        masks = [self._to_instance_mask(r, i, encoded_prompts.image_hw)
                 for i, r in enumerate(raw_results)]

        elapsed = time.perf_counter() - t0
        log.info("SAM pass done in %.2fs  backend=%s  masks=%d",
                 elapsed, self.config.backend, len(masks))

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
        index: int,
        image_hw: tuple[int, int],
    ) -> InstanceMask:
        prompt: SAMPrompt = raw["prompt"]
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