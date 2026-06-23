"""
Stage 1 — VLM Scene Reasoning Pass
====================================
Accepts a raw frame + task prompt, calls the configured VLM,
and returns a VLMSceneOutput containing:
  - a natural-language scene summary
  - a list of PriorityRegion objects (label, bbox, priority, class, reason)

The VLM is expected to respond in structured JSON so the output can be
deterministically parsed into typed dataclasses.  A stub backend is
included for testing without a live model.

Typical usage
-------------
    from vlm_reasoning import VLMReasoningPass, VLMConfig

    cfg = VLMConfig(backend="gemma", model_name="google/gemma-4-e4b-it")
    stage = VLMReasoningPass(cfg)
    result = stage.run(frame=img_array, task_prompt="find traversal zones")
"""

from __future__ import annotations

import json
import time
import uuid
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np

from structure import (
    BBox, PriorityRegion, SemanticClass, VLMSceneOutput,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VLMConfig:
    backend: Literal["gemma", "stub"] = "stub"
    model_name: str = "stub"
    device: str = "cuda"
    max_new_tokens: int = 1024
    temperature: float = 0.2
    # If running the VLM on a ground station over the wire, set this.
    grpc_endpoint: Optional[str] = None
    # How many regions to request at most.
    max_regions: int = 8


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are the scene reasoning module of an aerial drone perception system.
You receive aerial imagery and a task prompt and output structured JSON.

Return ONLY a single JSON object with two keys:
  "scene_summary": <one-sentence description of the scene>
  "priority_regions": [
    {{
      "label": <short descriptive name>,
      "bbox": {{"x": <float 0-1>, "y": <float 0-1>, "w": <float 0-1>, "h": <float 0-1>}},
      "priority": <float 0-1>,
      "semantic_class": <one of: road, vehicle, building, vegetation, person, anomaly>,
      "reason": <one sentence explaining why this region is high priority>
    }},
    ...
  ]

Order regions by descending priority.
Do not include markdown fences or any text outside the JSON object.
"""


# ---------------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------------

def _call_stub(frame: np.ndarray, task_prompt: str, cfg: VLMConfig) -> dict:
    """Returns deterministic fake output for unit testing."""
    return {
        "scene_summary": "Urban intersection with mixed vehicle and pedestrian traffic observed from nadir view.",
        "priority_regions": [
            {"label": "main_road_corridor", "bbox": {"x": 0.1, "y": 0.3, "w": 0.8, "h": 0.2},
             "priority": 0.95, "semantic_class": "road",
             "reason": "Primary traversal corridor spanning scene width."},
            {"label": "vehicle_cluster_nw", "bbox": {"x": 0.05, "y": 0.1, "w": 0.25, "h": 0.2},
             "priority": 0.85, "semantic_class": "vehicle",
             "reason": "Dense vehicle cluster may impede access to northwest zone."},
            {"label": "building_footprint_a", "bbox": {"x": 0.6, "y": 0.05, "w": 0.35, "h": 0.25},
             "priority": 0.4, "semantic_class": "building",
             "reason": "Static structure; low priority unless blocking LOS."},
            {"label": "anomaly_debris", "bbox": {"x": 0.45, "y": 0.55, "w": 0.1, "h": 0.08},
             "priority": 0.78, "semantic_class": "anomaly",
             "reason": "Unclassified debris on road surface, potential obstruction."},
            {"label": "pedestrian_crossing", "bbox": {"x": 0.35, "y": 0.28, "w": 0.12, "h": 0.1},
             "priority": 0.7, "semantic_class": "person",
             "reason": "Pedestrian activity at road crossing requires safety monitoring."},
        ],
    }


def _call_gemma(frame: np.ndarray, task_prompt: str, cfg: VLMConfig) -> dict:
    """
    Calls Gemma 4 (or any model served via vLLM/HuggingFace generate).
    Expects cfg.grpc_endpoint if off-device, otherwise uses local pipeline.
    """
    try:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        import torch
    except ImportError:
        raise RuntimeError("transformers not installed. pip install transformers torch")

    # Lazy-load to avoid import cost when using other backends.
    processor = AutoProcessor.from_pretrained(cfg.model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.model_name, torch_dtype=torch.bfloat16, device_map=cfg.device
    )

    from PIL import Image
    pil_img = Image.fromarray(frame)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "image", "image": pil_img},
            {"type": "text",  "text": f"Task prompt: {task_prompt}"},
        ]},
    ]
    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_tensors="pt", return_dict=True,
    ).to(cfg.device)

    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=cfg.max_new_tokens,
                                temperature=cfg.temperature, do_sample=True)
    text = processor.decode(output[0], skip_special_tokens=True)
    # Strip anything before the first '{' in case the model prefixes text.
    text = text[text.index("{"):]
    return json.loads(text)


_BACKENDS = {
    "gemma":    _call_gemma,
    "stub":     _call_stub,
}


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class VLMReasoningPass:
    """
    Stage 1 of pipeline.

    Parameters
    ----------
    config : VLMConfig
        Backend and model settings.

    Example
    -------
    >>> stage = VLMReasoningPass(VLMConfig(backend="stub"))
    >>> out = stage.run(frame=np.zeros((720,1280,3), dtype=np.uint8),
    ...                 task_prompt="find high-priority traversal zones",
    ...                 frame_id=0)
    >>> for r in out.priority_regions:
    ...     print(r.label, r.priority)
    """

    def __init__(self, config: VLMConfig):
        self.config = config
        if config.backend not in _BACKENDS:
            raise ValueError(f"Unknown backend '{config.backend}'. "
                             f"Choose from: {list(_BACKENDS)}")

    def run(
        self,
        frame: np.ndarray,
        task_prompt: str,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> VLMSceneOutput:
        """
        Run the VLM reasoning pass on a single frame.

        Parameters
        ----------
        frame : np.ndarray
            H×W×3 uint8 RGB image.
        task_prompt : str
            Natural-language instruction (e.g. "find traversal zones").
        frame_id : int
            Frame index for temporal tracking.
        timestamp : float
            Wall-clock or sensor timestamp.

        Returns
        -------
        VLMSceneOutput
            Structured output ready to feed into Stage 2.
        """
        assert frame.ndim == 3 and frame.shape[2] == 3, \
            "frame must be H×W×3 uint8 RGB"

        t0 = time.perf_counter()
        raw = _BACKENDS[self.config.backend](frame, task_prompt, self.config)
        elapsed = time.perf_counter() - t0
        log.info("VLM pass done in %.2fs  backend=%s  regions=%d",
                 elapsed, self.config.backend, len(raw.get("priority_regions", [])))

        regions = self._parse_regions(raw.get("priority_regions", []))
        # Respect max_regions cap, already sorted by priority descending.
        regions = regions[: self.config.max_regions]

        return VLMSceneOutput(
            task_prompt=task_prompt,
            scene_summary=raw.get("scene_summary", ""),
            priority_regions=regions,
            frame_id=frame_id,
            timestamp=timestamp,
            model_name=self.config.model_name,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_regions(raw_list: list[dict]) -> list[PriorityRegion]:
        regions = []
        for i, item in enumerate(raw_list):
            try:
                b = item["bbox"]
                bbox = BBox(x=float(b["x"]), y=float(b["y"]),
                            w=float(b["w"]), h=float(b["h"]))
                sem = SemanticClass(item.get("semantic_class", "unknown"))
                region = PriorityRegion(
                    label=item["label"],
                    bbox=bbox,
                    priority=float(item["priority"]),
                    semantic_class=sem,
                    reason=item.get("reason", ""),
                    region_id=f"region_{i:03d}_{item['label']}",
                )
                regions.append(region)
            except (KeyError, ValueError) as e:
                log.warning("Skipping malformed region %d: %s", i, e)
        regions.sort(key=lambda r: r.priority, reverse=True)
        return regions