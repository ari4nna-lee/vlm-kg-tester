"""
Stage 1b — Prompt Encoder
==========================
Converts VLMSceneOutput into SAM-compatible prompt dicts.
SAM2/SAM3 accept three prompt types: points, boxes, and masks.
This module produces point + box prompts derived from the VLM's
priority regions, prioritized by score.

Typical usage
-------------
    from prompt_encoder import PromptEncoder, EncoderConfig

    encoder = PromptEncoder(EncoderConfig(image_hw=(720, 1280)))
    sam_prompts = encoder.encode(vlm_output)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import numpy as np
import logging
log = logging.getLogger(__name__)

from structure import (
    BBox, PriorityRegion, VLMSceneOutput, SemanticClass,
)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EncoderConfig:
    image_hw: tuple[int, int] = (720, 1280)  # (H, W) of the input frame
    # Minimum VLM priority to bother prompting SAM at all.
    priority_threshold: float = 0.3
    # Number of positive prompt points per region (center + optional corners).
    points_per_region: int = 1
    # Whether to also pass the bbox as a box prompt (more accurate than points alone).
    use_box_prompts: bool = True
    # Negative point strategy: place one point in background for each region.
    add_background_points: bool = False


# ---------------------------------------------------------------------------
# Output structures
# ---------------------------------------------------------------------------

@dataclass
class SAMPrompt:
    """
    Prompt for a single SAM instance request.
    All coordinates are in pixel space (not normalized).
    """
    region_id: str
    label: str
    semantic_class: SemanticClass
    priority: float
    # point_coords: (N, 2) array of [x, y]
    point_coords: np.ndarray
    # point_labels: (N,) array — 1=positive, 0=negative
    point_labels: np.ndarray
    # box: [x1, y1, x2, y2] in pixel space, or None
    box: Optional[np.ndarray]


@dataclass
class EncodedPrompts:
    """Output of the prompt encoder, fed directly into Stage 2."""
    prompts: list[SAMPrompt]
    image_hw: tuple[int, int]
    frame_id: int = 0
    # Regions that were filtered out due to low priority.
    skipped_regions: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------

class PromptEncoder:
    """
    Converts VLMSceneOutput → EncodedPrompts for SAM.

    Coordinate convention
    ---------------------
    VLM bboxes are normalized [0,1].
    SAM expects absolute pixel coordinates.
    This class handles the conversion so neither Stage 1 nor Stage 2
    needs to know the image dimensions.
    """

    def __init__(self, config: EncoderConfig):
        self.config = config

    def encode(self, vlm_output: VLMSceneOutput) -> EncodedPrompts:
        """
        Parameters
        ----------
        vlm_output : VLMSceneOutput
            Output of VLMReasoningPass.run().

        Returns
        -------
        EncodedPrompts
            SAM-ready prompts, sorted by priority descending.
        """
        H, W = self.config.image_hw
        prompts: list[SAMPrompt] = []
        skipped: list[str] = []

        for region in vlm_output.priority_regions:
            if region.priority < self.config.priority_threshold:
                skipped.append(region.region_id)
                continue
            prompt = self._region_to_prompt(region, H, W)
            prompts.append(prompt)

        # Keep sorted by priority so SAM processes the most important first.
        prompts.sort(key=lambda p: p.priority, reverse=True)

        return EncodedPrompts(
            prompts=prompts,
            image_hw=(H, W),
            frame_id=vlm_output.frame_id,
            skipped_regions=skipped,
        )
    
    def encode_from_regions(
        self,
        regions: list[dict],
        frame_id: int = 0,
    ) -> EncodedPrompts:
        """
        Encode SAM prompts directly from KG-derived region dicts, used by kg_refine feedback path
        """
        priority_regions = []
        for i, r in enumerate(regions):
            try:
                b = r["bbox"]
                bbox = BBox(
                    x=float(b["x"]),
                    y=float(b["y"]),
                    w=float(b["w"]),
                    h=float(b["h"]),
                )
                region = PriorityRegion(
                    label=r["label"],
                    bbox=bbox,
                    priority=float(r["priority"]),
                    semantic_class=SemanticClass(r.get("semantic_class", "unknown")),
                    reason=r.get("reason", "kg_refine"),
                    region_id=f"refine_{frame_id}_{i:03d}_{r['label']}",
                )
                priority_regions.append(region)
            except (KeyError, ValueError) as e:
                log.warning("encode_from_regions: skipping malformed region %d: %s", i, e)

        synthetic_vlm = VLMSceneOutput(
            task_prompt="kg_refine",
            scene_summary="",
            priority_regions=priority_regions,
            frame_id=frame_id,
        )
        return self.encode(synthetic_vlm)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _region_to_prompt(
        self, region: PriorityRegion, H: int, W: int
    ) -> SAMPrompt:
        bbox = region.bbox
        # Pixel-space corners.
        x1 = int(bbox.x * W)
        y1 = int(bbox.y * H)
        x2 = int((bbox.x + bbox.w) * W)
        y2 = int((bbox.y + bbox.h) * H)
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        pos_points = self._positive_points(x1, y1, x2, y2, cx, cy)
        coords = pos_points.copy()
        labels = [1] * len(pos_points)

        if self.config.add_background_points:
            bg_pts = self._background_points(x1, y1, x2, y2, H, W)
            coords = np.vstack([coords, bg_pts])
            labels += [0] * len(bg_pts)

        box = np.array([x1, y1, x2, y2], dtype=np.float32) \
              if self.config.use_box_prompts else None

        return SAMPrompt(
            region_id=region.region_id,
            label=region.label,
            semantic_class=region.semantic_class,
            priority=region.priority,
            point_coords=np.array(coords, dtype=np.float32),
            point_labels=np.array(labels, dtype=np.int32),
            box=box,
        )
    
    def bias_prompts_with_heatmap(self, prompts, heatmap):
        H, W = heatmap.shape

        biased = []

        for p in prompts:
            if p.box is not None:
                cx = int((p.box[0] + p.box[2]) / 2)
                cy = int((p.box[1] + p.box[3]) / 2)

                score = heatmap[cy, cx]

                p.priority = p.priority + 0.5 * score

            biased.append(p)

        return biased

    def _positive_points(
        self, x1: int, y1: int, x2: int, y2: int, cx: int, cy: int
    ) -> np.ndarray:
        if self.config.points_per_region == 1:
            return np.array([[cx, cy]], dtype=np.float32)
        # 5-point grid: center + four quadrant centers.
        qx1, qy1 = (x1 + cx) // 2, (y1 + cy) // 2
        qx2, qy2 = (cx + x2) // 2, (cy + y2) // 2
        pts = [[cx, cy], [qx1, qy1], [qx2, qy1], [qx1, qy2], [qx2, qy2]]
        return np.array(pts[: self.config.points_per_region], dtype=np.float32)

    @staticmethod
    def _background_points(
        x1: int, y1: int, x2: int, y2: int, H: int, W: int
    ) -> np.ndarray:
        """Place one background point just outside the bbox."""
        bx = max(0, x1 - int((x2 - x1) * 0.15))
        by = max(0, y1 - int((y2 - y1) * 0.15))
        return np.array([[bx, by]], dtype=np.float32)