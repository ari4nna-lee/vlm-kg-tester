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
    PAVED_ROAD        = "paved_road"
    DIRT_ROAD         = "dirt_road"
    GRASSLAND         = "grassland"
    DENSE_VEGETATION  = "dense_vegetation"
    SPARSE_VEGETATION = "sparse_vegetation"
    STEEP_SLOPE       = "steep_slope"
    WATER             = "water"
    BARE_EARTH        = "bare_earth"
    BUILDING          = "building"
    BUILDING_ROOF     = "building_roof"
    PARKING_LOT       = "parking_lot"
    LOADING_DOCK      = "loading_dock"
    VEHICLE           = "vehicle"
    LARGE_VEHICLE     = "large_vehicle"
    TRUCK             = "truck"
    CAR               = "car"
    TRAILER           = "trailer"
    PERSON            = "person"
    ANOMALY           = "anomaly"
    UNKNOWN           = "unknown"

TRUCK_PRIOR: dict[str, float] = {
    "paved_road":        0.85,
    "dirt_road":         0.65,
    "grassland":         0.30,
    "parking_lot":       0.80,
    "loading_dock":      0.90,
    "bare_earth":        0.35,
    "sparse_vegetation": 0.15,
    "dense_vegetation":  0.02,
    "building_roof":     0.01,
    "building":          0.05,
    "steep_slope":       0.01,
    "water":             0.00,
    "vehicle":           0.60,
    "large_vehicle":     0.85,
    "truck":             1.00,
    "car":               0.20,
    "trailer":           0.75,
    "person":            0.25,
    "anomaly":           0.40,
    "unknown":           0.15,
}

TRAVERSABLE_CLASSES: set[str] = {
    SemanticClass.PAVED_ROAD.value,
    SemanticClass.DIRT_ROAD.value,
    SemanticClass.GRASSLAND.value,
    SemanticClass.BARE_EARTH.value,
    SemanticClass.PARKING_LOT.value,
}

DYNAMIC_CLASSES: set[str] = {
    SemanticClass.VEHICLE.value,
    SemanticClass.LARGE_VEHICLE.value,
    SemanticClass.TRUCK.value,
    SemanticClass.CAR.value,
    SemanticClass.TRAILER.value,
    SemanticClass.PERSON.value,
}

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

SEMANTIC_CLASS_ALIASES: dict[str, str] = {
    "road":             "paved_road",
    "vegetation":       "dense_vegetation",
    "building":         "building",
    "vehicle":          "vehicle",
    "person":           "person",
    "anomaly":          "anomaly",
    "unknown":          "unknown",
    "grass":            "grassland",
    "dirt":             "bare_earth",
    "dirt road":        "dirt_road",
    "gravel":           "dirt_road",
    "pavement":         "paved_road",
    "street":           "paved_road",
    "highway":          "paved_road",
    "forest":           "dense_vegetation",
    "trees":            "dense_vegetation",
    "shrubs":           "sparse_vegetation",
    "water":            "water",
    "roof":             "building_roof",
    "parking":          "parking_lot",
    "parking lot":      "parking_lot",
    "car":              "car",
    "truck":            "truck",
    "large vehicle":    "large_vehicle",
    "trailer":          "trailer",
}

def resolve_semantic_class(raw: str) -> SemanticClass:
    """
    Resolve a raw VLM string to a SemanticClass, with alias fallback.
    Returns SemanticClass.UNKNOWN rather than raising on unrecognized input.
    """
    s = raw.strip().lower()
    # try direct match first
    try:
        return SemanticClass(s)
    except ValueError:
        pass
    # try alias table
    if s in SEMANTIC_CLASS_ALIASES:
        try:
            return SemanticClass(SEMANTIC_CLASS_ALIASES[s])
        except ValueError:
            pass
    # give up gracefully
    import logging
    logging.getLogger(__name__).warning(
        "resolve_semantic_class: unrecognized class %r, defaulting to UNKNOWN", raw
    )
    return SemanticClass.UNKNOWN

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
class NodeObservation:
    frame_id: int
    priority: float
    centroid: tuple[float, float]
    confidence: float
    global_centroid: Optional[tuple[float, float]] = None
    timestamp: float = 0.0

@dataclass
class KGNode:
    node_id: str
    label: str
    semantic_class: SemanticClass
    centroid: tuple[float, float]
    bbox: BBox
    mask_area: float
    priority: float
    confidence: float           
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