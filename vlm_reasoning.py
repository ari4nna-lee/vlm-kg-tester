"""
Stage 1 — VLM Scene Reasoning Pass
====================================
Accepts a raw frame + task prompt, calls the configured VLM,
and returns a VLMSceneOutput containing:
  - a natural-language scene summary
  - a list of PriorityRegion objects (label, bbox, priority, class, reason)

The VLM is expected to respond in structured JSON so the output can be
deterministically parsed into typed dataclasses.

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
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np
import base64
import re

from io import BytesIO
from PIL import Image
import requests

from structure import (
    BBox, PriorityRegion, SemanticClass, VLMSceneOutput,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VLMConfig:
    backend: Literal["gemma", "stub"] = "gemma"
    model_name: str = "gemma"
    device: str = "cuda"
    max_new_tokens: int = 512
    temperature: float = 0.2
    # If running the VLM on a ground station over the wire, set this.
    grpc_endpoint: Optional[str] = None
    # How many regions to request at most.
    max_regions: int = 5


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

Return ONLY valid JSON.
No markdown.
No explanation.
No trailing commas.
"""


# ---------------------------------------------------------------------------
# Backend adapters
# ---------------------------------------------------------------------------

def _empty_vlm_output() -> dict:
    return {
        "scene_summary": "",
        "priority_regions": [],
    }

def _call_gemma(frame: np.ndarray,
                task_prompt: str,
                cfg: VLMConfig) -> dict:
    """
    Calls Gemma running on a local vLLM server via raw HTTP.
    """

    if cfg.grpc_endpoint is None:
        raise ValueError(
            "Set cfg.grpc_endpoint to your vLLM URL "
            "(e.g. http://localhost:8000/v1)"
        )

    # Convert image -> base64
    pil_img = Image.fromarray(frame)
    pil_img = pil_img.resize((640, 360), Image.LANCZOS)  # half res

    buffer = BytesIO()
    pil_img.save(buffer, format="JPEG")

    image_b64 = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    payload = {
        "model": cfg.model_name,
        "messages": [
            {
                "role": "system",
                "content": _SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Task prompt: {task_prompt}"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_b64}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": cfg.max_new_tokens,
        "temperature": cfg.temperature,
    }

    response = requests.post(
        f"{cfg.grpc_endpoint}/chat/completions",
        json=payload,
        timeout=300,
    )

    response.raise_for_status()

    result = response.json()

    text = result["choices"][0]["message"]["content"].strip()

    # Extract JSON from model output
    start = text.find("{")
    end = text.rfind("}") + 1

    if start == -1:
        raise ValueError(
            f"Gemma returned invalid JSON:\n{text}"
        )

    text = text[start:end]

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # attempt to extract the first valid JSON object from the response
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
        log.warning("Gemma returned unparseable JSON, returning empty output. Raw: %s", text[:200])
        return _empty_vlm_output()  # whatever your fallback/default return looks like

_BACKENDS = {
    "gemma":    _call_gemma,
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
    >>> stage = VLMReasoningPass(VLMConfig(backend="gemma"))
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