"""
Shared dataclasses for the pipeline.
All inter-stage data contracts are defined here so each module
imports from one place and the types stay consistent.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import numpy as np


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SemanticClass(str, Enum):
    ROAD        = "road"
    VEHICLE     = "vehicle"
    BUILDING    = "building"
    VEGETATION  = "vegetation"
    PERSON      = "person"
    ANOMALY     = "anomaly"
    UNKNOWN     = "unknown"


class EdgeTier(int, Enum):
    SPATIAL   = 1   # geometric, every frame
    TEMPORAL  = 2   # tracker-driven, on dynamic object detection
    SEMANTIC  = 3   # VLM-inferred, low frequency


class EdgePredicate(str, Enum):
    # Tier 1 — spatial
    SPATIALLY_ADJACENT   = "spatially_adjacent"
    CONTAINS             = "contains"
    CONNECTED_TO         = "connected_to"
    OCCLUDES             = "occludes"
    OCCLUDED_BY          = "occluded_by"
    # Tier 2 — temporal
    MOVING_TOWARD        = "moving_toward"
    CO_MOVING_WITH       = "co_moving_with"
    ON_TRAJECTORY_TOWARD = "on_trajectory_toward"
    # Tier 3 — semantic
    BLOCKS_ACCESS_TO     = "blocks_access_to"
    LANDMARK_OF          = "landmark_of"
    ANOMALOUS_IN         = "anomalous_in"
    PRIORITY_RELATIVE_TO = "priority_relative_to"


TIER_PREDICATES: dict[EdgeTier, list[EdgePredicate]] = {
    EdgeTier.SPATIAL: [
        EdgePredicate.SPATIALLY_ADJACENT,
        EdgePredicate.CONTAINS,
        EdgePredicate.CONNECTED_TO,
        EdgePredicate.OCCLUDES,
        EdgePredicate.OCCLUDED_BY,
    ],
    EdgeTier.TEMPORAL: [
        EdgePredicate.MOVING_TOWARD,
        EdgePredicate.CO_MOVING_WITH,
        EdgePredicate.ON_TRAJECTORY_TOWARD,
    ],
    EdgeTier.SEMANTIC: [
        EdgePredicate.BLOCKS_ACCESS_TO,
        EdgePredicate.LANDMARK_OF,
        EdgePredicate.ANOMALOUS_IN,
        EdgePredicate.PRIORITY_RELATIVE_TO,
    ],
}


# ---------------------------------------------------------------------------
# Stage 1 output: VLM priority regions
# ---------------------------------------------------------------------------

@dataclass
class BBox:
    """Normalized [0,1] bounding box."""
    x: float
    y: float
    w: float
    h: float

    def to_xyxy(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.x + self.w, self.y + self.h)

    def to_points(self) -> list[tuple[float, float]]:
        """Center point — used as SAM positive prompt."""
        return [(self.x + self.w / 2, self.y + self.h / 2)]

    def area(self) -> float:
        return self.w * self.h


@dataclass
class PriorityRegion:
    """
    Output unit of the VLM reasoning pass.
    One per semantically significant area the VLM identified.
    """
    label: str
    bbox: BBox
    priority: float                    # 0.0 (low) – 1.0 (critical)
    semantic_class: SemanticClass
    reason: str                        # VLM natural-language justification
    region_id: str = ""                # assigned by VLMReasoningPass

    def __post_init__(self):
        assert 0.0 <= self.priority <= 1.0, "priority must be in [0, 1]"


@dataclass
class VLMSceneOutput:
    """
    Full output of Stage 1.
    Passed as-is into Stage 2's prompt encoder.
    """
    task_prompt: str
    scene_summary: str
    priority_regions: list[PriorityRegion]
    frame_id: int = 0
    timestamp: float = 0.0
    model_name: str = ""


# ---------------------------------------------------------------------------
# Stage 2 output: segmentation masks → nodes
# ---------------------------------------------------------------------------

@dataclass
class InstanceMask:
    """
    Raw mask output from SAM2/SAM3 for a single instance.
    mask_array is H×W boolean numpy array (or None for stub/bbox mode).
    """
    node_id: str
    label: str
    semantic_class: SemanticClass
    bbox: BBox
    mask_array: Optional[np.ndarray]   # H×W bool; None when running bbox-only
    centroid: tuple[float, float]      # normalized (cx, cy)
    mask_area: float                   # fraction of image area [0,1]
    confidence: float
    track_id: int                      # -1 if tracking not active
    source_region_id: str              # which PriorityRegion prompted this mask
    priority: float                    # inherited from VLM
    attributes: dict = field(default_factory=dict)


@dataclass
class SegmentationOutput:
    """
    Full output of Stage 2.
    Passed as-is into Stage 3.
    """
    masks: list[InstanceMask]
    frame_id: int = 0
    timestamp: float = 0.0
    sam_model: str = ""


# ---------------------------------------------------------------------------
# Stage 3 output: knowledge graph
# ---------------------------------------------------------------------------

@dataclass
class KGNode:
    """
    A node in the scene knowledge graph.
    Constructed 1-to-1 from each InstanceMask.
    """
    node_id: str
    label: str
    semantic_class: SemanticClass
    centroid: tuple[float, float]
    bbox: BBox
    mask_area: float
    priority: float
    track_id: int
    frame_id: int
    attributes: dict = field(default_factory=dict)


@dataclass
class KGEdge:
    """
    A directed edge in the scene knowledge graph.
    """
    subject_id: str
    predicate: EdgePredicate
    object_id: str
    tier: EdgeTier
    confidence: float
    attributes: dict = field(default_factory=dict)

    def as_triple(self) -> tuple[str, str, str]:
        return (self.subject_id, self.predicate.value, self.object_id)


@dataclass
class KGOutput:
    """
    Full output of Stage 3.
    Contains the NetworkX graph plus the flat node/edge lists for inspection.
    """
    nodes: list[KGNode]
    edges: list[KGEdge]
    frame_id: int = 0
    timestamp: float = 0.0
    # traversability_subgraph is extracted separately by the path planner