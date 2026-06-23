"""
Stage 3 — Knowledge Graph Construction
========================================
Converts SegmentationOutput into a NetworkX scene graph.

Three tiers of edges are computed here (or skipped based on config):

  Tier 1 — Spatial (every frame, geometric, cheap)
      spatially_adjacent, contains, connected_to, occludes, occluded_by

  Tier 2 — Temporal/Dynamic (triggered by dynamic object detection)
      moving_toward, co_moving_with, on_trajectory_toward

  Tier 3 — Semantic (VLM-inferred, low-frequency, expensive)
      blocks_access_to, landmark_of, anomalous_in, priority_relative_to

The traversability subgraph (all `connected_to` edges among road/ground
nodes) is extracted and stored separately for direct consumption by path
planners.

Typical usage
-------------
    from kg import KGStage, KGConfig

    kg = KGStage(KGConfig(enable_tier2=True, enable_tier3=False))
    result = kg.run(seg_output, prev_seg_output=None)
    G = result.graph           # full NetworkX DiGraph
    T = result.traversability  # road-only subgraph
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    nx = None  # type: ignore

from structure import (
    BBox, EdgePredicate, EdgeTier,
    InstanceMask, KGEdge, KGNode, KGOutput,
    SemanticClass, SegmentationOutput,
)

log = logging.getLogger(__name__)

# Semantic classes considered traversable surface.
_TRAVERSABLE = {SemanticClass.ROAD}
# Classes considered dynamic (trigger Tier 2).
_DYNAMIC = {SemanticClass.VEHICLE, SemanticClass.PERSON}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class KGConfig:
    enable_tier1: bool = True
    enable_tier2: bool = True
    enable_tier3: bool = False   # requires a VLM call; off by default

    # Tier 1 thresholds (normalized coordinates).
    adjacency_distance_threshold: float = 0.05  # centroid distance < this → adjacent
    containment_area_ratio: float = 0.75         # inner/outer area ratio for "contains"

    # Tier 2: velocity threshold — below this, object is considered static.
    velocity_threshold: float = 0.01  # normalized units/frame

    # Tier 3: VLM config for semantic edge inference.
    vlm_backend: str = "stub"
    vlm_model: str = "stub"

    # Minimum confidence to keep an edge.
    min_edge_confidence: float = 0.4


# ---------------------------------------------------------------------------
# Tier 1 — Spatial edge inference
# ---------------------------------------------------------------------------

class SpatialEdgeInferrer:
    """Geometric computation on normalized centroids and mask areas."""

    def __init__(self, cfg: KGConfig):
        self.cfg = cfg

    def infer(self, nodes: list[KGNode]) -> list[KGEdge]:
        edges: list[KGEdge] = []
        n = len(nodes)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = nodes[i], nodes[j]
                edges.extend(self._compare(a, b))
        return edges

    def _compare(self, a: KGNode, b: KGNode) -> list[KGEdge]:
        edges: list[KGEdge] = []
        dist = self._centroid_dist(a, b)

        # --- spatially_adjacent ---
        if dist < self.cfg.adjacency_distance_threshold:
            conf = round(1.0 - dist / self.cfg.adjacency_distance_threshold, 3)
            edges.append(self._edge(a, EdgePredicate.SPATIALLY_ADJACENT, b,
                                    EdgeTier.SPATIAL, conf))
            edges.append(self._edge(b, EdgePredicate.SPATIALLY_ADJACENT, a,
                                    EdgeTier.SPATIAL, conf))

        # --- contains / within ---
        containment = self._containment_ratio(a.bbox, b.bbox)
        if containment > self.cfg.containment_area_ratio:
            # a contains b
            edges.append(self._edge(a, EdgePredicate.CONTAINS, b,
                                    EdgeTier.SPATIAL, round(containment, 3)))
        containment_rev = self._containment_ratio(b.bbox, a.bbox)
        if containment_rev > self.cfg.containment_area_ratio:
            edges.append(self._edge(b, EdgePredicate.CONTAINS, a,
                                    EdgeTier.SPATIAL, round(containment_rev, 3)))

        # --- connected_to (traversable surfaces only) ---
        if (a.semantic_class in _TRAVERSABLE and b.semantic_class in _TRAVERSABLE):
            if self._bboxes_touch(a.bbox, b.bbox):
                conf = round(min(a.confidence if hasattr(a, 'confidence') else 0.9,
                                 b.confidence if hasattr(b, 'confidence') else 0.9), 3)
                edges.append(self._edge(a, EdgePredicate.CONNECTED_TO, b,
                                        EdgeTier.SPATIAL, conf))
                edges.append(self._edge(b, EdgePredicate.CONNECTED_TO, a,
                                        EdgeTier.SPATIAL, conf))

        # --- occludes / occluded_by (approximate via bbox overlap + priority) ---
        overlap = self._bbox_iou(a.bbox, b.bbox)
        if overlap > 0.1:
            # Higher-priority node is assumed to be "in front".
            if a.priority >= b.priority:
                edges.append(self._edge(a, EdgePredicate.OCCLUDES, b,
                                        EdgeTier.SPATIAL, round(overlap, 3)))
                edges.append(self._edge(b, EdgePredicate.OCCLUDED_BY, a,
                                        EdgeTier.SPATIAL, round(overlap, 3)))
            else:
                edges.append(self._edge(b, EdgePredicate.OCCLUDES, a,
                                        EdgeTier.SPATIAL, round(overlap, 3)))
                edges.append(self._edge(a, EdgePredicate.OCCLUDED_BY, b,
                                        EdgeTier.SPATIAL, round(overlap, 3)))

        return edges

    # ------------------------------------------------------------------

    @staticmethod
    def _centroid_dist(a: KGNode, b: KGNode) -> float:
        return float(np.hypot(a.centroid[0] - b.centroid[0],
                              a.centroid[1] - b.centroid[1]))

    @staticmethod
    def _containment_ratio(outer: BBox, inner: BBox) -> float:
        """Fraction of inner bbox area that lies within outer bbox."""
        ox1, oy1, ox2, oy2 = outer.to_xyxy()
        ix1, iy1, ix2, iy2 = inner.to_xyxy()
        inter_x = max(0, min(ox2, ix2) - max(ox1, ix1))
        inter_y = max(0, min(oy2, iy2) - max(oy1, iy1))
        inter_area = inter_x * inter_y
        inner_area = inner.area()
        return inter_area / inner_area if inner_area > 0 else 0.0

    @staticmethod
    def _bboxes_touch(a: BBox, b: BBox, tol: float = 0.02) -> bool:
        ax1, ay1, ax2, ay2 = a.to_xyxy()
        bx1, by1, bx2, by2 = b.to_xyxy()
        return (ax1 - tol <= bx2 and ax2 + tol >= bx1 and
                ay1 - tol <= by2 and ay2 + tol >= by1)

    @staticmethod
    def _bbox_iou(a: BBox, b: BBox) -> float:
        ax1, ay1, ax2, ay2 = a.to_xyxy()
        bx1, by1, bx2, by2 = b.to_xyxy()
        ix = max(0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        union = a.area() + b.area() - inter
        return inter / union if union > 0 else 0.0

    @staticmethod
    def _edge(
        subj: KGNode, pred: EdgePredicate, obj: KGNode,
        tier: EdgeTier, confidence: float,
    ) -> KGEdge:
        return KGEdge(
            subject_id=subj.node_id,
            predicate=pred,
            object_id=obj.node_id,
            tier=tier,
            confidence=confidence,
        )


# ---------------------------------------------------------------------------
# Tier 2 — Temporal edge inference
# ---------------------------------------------------------------------------

class TemporalEdgeInferrer:
    """
    Computes motion-based edges by comparing centroids between frames.
    Requires the previous SegmentationOutput for velocity estimation.
    """

    def __init__(self, cfg: KGConfig):
        self.cfg = cfg

    def infer(
        self,
        curr_nodes: list[KGNode],
        prev_masks: Optional[list[InstanceMask]],
    ) -> list[KGEdge]:
        if prev_masks is None:
            return []

        # Build track_id → prev centroid lookup.
        prev_by_track: dict[int, tuple[float, float]] = {
            m.track_id: m.centroid for m in prev_masks if m.track_id >= 0
        }

        # Compute velocity vectors for dynamic nodes.
        velocity: dict[str, np.ndarray] = {}
        dynamic_nodes = [n for n in curr_nodes if n.semantic_class in _DYNAMIC]
        for node in dynamic_nodes:
            if node.track_id in prev_by_track:
                prev_c = np.array(prev_by_track[node.track_id])
                curr_c = np.array(node.centroid)
                v = curr_c - prev_c
                if np.linalg.norm(v) > self.cfg.velocity_threshold:
                    velocity[node.node_id] = v

        if not velocity:
            return []

        edges: list[KGEdge] = []
        node_by_id = {n.node_id: n for n in curr_nodes}

        for node_id, v in velocity.items():
            node = node_by_id[node_id]
            curr_pos = np.array(node.centroid)

            for other in curr_nodes:
                if other.node_id == node_id:
                    continue
                other_pos = np.array(other.centroid)
                to_other = other_pos - curr_pos
                dist = np.linalg.norm(to_other)
                if dist < 1e-6:
                    continue

                # --- moving_toward ---
                cosine = float(np.dot(v, to_other) / (np.linalg.norm(v) * dist))
                if cosine > 0.7:
                    conf = round(cosine * (1.0 - min(dist, 1.0)), 3)
                    if conf >= self.cfg.min_edge_confidence:
                        edges.append(KGEdge(
                            subject_id=node_id,
                            predicate=EdgePredicate.MOVING_TOWARD,
                            object_id=other.node_id,
                            tier=EdgeTier.TEMPORAL,
                            confidence=conf,
                        ))

                # --- on_trajectory_toward ---
                # Project current velocity 5 steps and check proximity.
                projected = curr_pos + v * 5
                proj_dist = float(np.linalg.norm(projected - other_pos))
                if proj_dist < 0.1:
                    conf = round(1.0 - proj_dist / 0.1, 3)
                    if conf >= self.cfg.min_edge_confidence:
                        edges.append(KGEdge(
                            subject_id=node_id,
                            predicate=EdgePredicate.ON_TRAJECTORY_TOWARD,
                            object_id=other.node_id,
                            tier=EdgeTier.TEMPORAL,
                            confidence=conf,
                        ))

        # --- co_moving_with (correlated velocity clusters) ---
        v_items = list(velocity.items())
        for i in range(len(v_items)):
            for j in range(i + 1, len(v_items)):
                id_a, v_a = v_items[i]
                id_b, v_b = v_items[j]
                mag_a, mag_b = np.linalg.norm(v_a), np.linalg.norm(v_b)
                if mag_a < 1e-6 or mag_b < 1e-6:
                    continue
                cosine = float(np.dot(v_a, v_b) / (mag_a * mag_b))
                speed_ratio = min(mag_a, mag_b) / max(mag_a, mag_b)
                conf = round(cosine * speed_ratio, 3)
                if cosine > 0.85 and conf >= self.cfg.min_edge_confidence:
                    edges.append(KGEdge(
                        subject_id=id_a,
                        predicate=EdgePredicate.CO_MOVING_WITH,
                        object_id=id_b,
                        tier=EdgeTier.TEMPORAL,
                        confidence=conf,
                    ))

        return edges


# ---------------------------------------------------------------------------
# Tier 3 — Semantic edge inference (VLM pass)
# ---------------------------------------------------------------------------

class SemanticEdgeInferrer:
    """
    Calls the VLM with the current node list and asks for semantic edge proposals.
    This is low-frequency — call every N frames or on explicit triggers.
    """

    def __init__(self, cfg: KGConfig):
        self.cfg = cfg

    def infer(self, nodes: list[KGNode]) -> list[KGEdge]:
        raw_edges = self._call_vlm(nodes)
        return self._parse_edges(raw_edges, nodes)

    def _call_vlm(self, nodes: list[KGNode]) -> list[dict]:
        if self.cfg.vlm_backend == "stub":
            return self._stub_edges(nodes)

        # Real VLM call: build a text summary of the node list, prompt
        # the model for (subject, predicate, object) triples.
        node_summary = "\n".join(
            f"  {n.node_id}: {n.label} [{n.semantic_class.value}] "
            f"centroid=({n.centroid[0]:.2f},{n.centroid[1]:.2f}) "
            f"priority={n.priority:.2f}"
            for n in nodes
        )
        system = (
            "You are a semantic edge inference module for an aerial drone perception system. "
            "Given a list of scene nodes, return a JSON array of semantic edges. "
            "Each edge: {subject_id, predicate, object_id, confidence}. "
            "Valid predicates: blocks_access_to, landmark_of, anomalous_in, priority_relative_to. "
            "Return ONLY valid JSON, no markdown."
        )
        user = f"Nodes:\n{node_summary}\nInfer semantic edges."

        # Delegate to whichever backend is configured.
        from vlm_reasoning import _BACKENDS as VLM_BACKENDS
        # We reuse the VLM text-only path — frame=None signals text-only call.
        # In practice you would call the VLM API directly here with a text prompt.
        import json
        try:
            import requests
            response = requests.post(
                self.cfg.vlm_backend,
                json={"system": system, "user": user},
                timeout=10,
            )
            return response.json()
        except Exception as e:
            log.warning("Semantic VLM call failed: %s — falling back to stub", e)
            return self._stub_edges(nodes)

    @staticmethod
    def _stub_edges(nodes: list[KGNode]) -> list[dict]:
        """Synthetic semantic edges for testing."""
        edges = []
        node_ids = [n.node_id for n in nodes]
        node_classes = {n.node_id: n.semantic_class for n in nodes}

        vehicles = [nid for nid, cls in node_classes.items() if cls == SemanticClass.VEHICLE]
        roads    = [nid for nid, cls in node_classes.items() if cls == SemanticClass.ROAD]
        anomalies= [nid for nid, cls in node_classes.items() if cls == SemanticClass.ANOMALY]
        buildings= [nid for nid, cls in node_classes.items() if cls == SemanticClass.BUILDING]

        for v in vehicles[:1]:
            for r in roads[:1]:
                edges.append({"subject_id": v, "predicate": "blocks_access_to",
                               "object_id": r, "confidence": 0.82})
        for b in buildings[:1]:
            if node_ids:
                edges.append({"subject_id": b, "predicate": "landmark_of",
                               "object_id": node_ids[0], "confidence": 0.74})
        for a in anomalies[:1]:
            for r in roads[:1]:
                edges.append({"subject_id": a, "predicate": "anomalous_in",
                               "object_id": r, "confidence": 0.91})
        return edges

    @staticmethod
    def _parse_edges(raw: list[dict], nodes: list[KGNode]) -> list[KGEdge]:
        valid_ids = {n.node_id for n in nodes}
        pred_map = {p.value: p for p in EdgePredicate}
        edges = []
        for item in raw:
            try:
                subj = item["subject_id"]
                obj  = item["object_id"]
                pred_str = item["predicate"]
                conf = float(item["confidence"])
                if subj not in valid_ids or obj not in valid_ids:
                    continue
                if pred_str not in pred_map:
                    continue
                edges.append(KGEdge(
                    subject_id=subj,
                    predicate=pred_map[pred_str],
                    object_id=obj,
                    tier=EdgeTier.SEMANTIC,
                    confidence=conf,
                ))
            except (KeyError, ValueError) as e:
                log.warning("Skipping malformed semantic edge: %s", e)
        return edges


# ---------------------------------------------------------------------------
# Node constructor
# ---------------------------------------------------------------------------

def _mask_to_node(mask: InstanceMask, frame_id: int) -> KGNode:
    return KGNode(
        node_id=mask.node_id,
        label=mask.label,
        semantic_class=mask.semantic_class,
        centroid=mask.centroid,
        bbox=mask.bbox,
        mask_area=mask.mask_area,
        priority=mask.priority,
        track_id=mask.track_id,
        frame_id=frame_id,
        attributes={
            "confidence": mask.confidence,
            "source_region_id": mask.source_region_id,
        },
    )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class KGStage:
    """
    Stage 3 of the pipeline.

    Parameters
    ----------
    config : KGConfig

    Example
    -------
    >>> kg = KGStage(KGConfig(enable_tier2=True, enable_tier3=False))
    >>> result = kg.run(seg_output, prev_seg_output=None)
    >>> print(nx.info(result.graph))
    """

    def __init__(self, config: KGConfig):
        if not _NX_AVAILABLE:
            raise ImportError("networkx is required: pip install networkx")
        self.config = config
        self._spatial   = SpatialEdgeInferrer(config)
        self._temporal  = TemporalEdgeInferrer(config)
        self._semantic  = SemanticEdgeInferrer(config)

    def run(
        self,
        seg_output: SegmentationOutput,
        prev_seg_output: Optional[SegmentationOutput] = None,
    ) -> "KGRunResult":
        """
        Parameters
        ----------
        seg_output : SegmentationOutput
            Output of SAMStage.run() for the current frame.
        prev_seg_output : SegmentationOutput, optional
            Previous frame's segmentation, used for Tier 2 temporal edges.

        Returns
        -------
        KGRunResult
            Dataclass containing: .output (KGOutput), .graph (nx.DiGraph),
            .traversability (nx.DiGraph — road-only subgraph).
        """
        t0 = time.perf_counter()

        nodes = [_mask_to_node(m, seg_output.frame_id) for m in seg_output.masks]
        edges: list[KGEdge] = []

        if self.config.enable_tier1:
            t1_edges = self._spatial.infer(nodes)
            edges.extend(t1_edges)
            log.debug("Tier 1: %d spatial edges", len(t1_edges))

        if self.config.enable_tier2:
            prev_masks = prev_seg_output.masks if prev_seg_output else None
            t2_edges = self._temporal.infer(nodes, prev_masks)
            edges.extend(t2_edges)
            log.debug("Tier 2: %d temporal edges", len(t2_edges))

        if self.config.enable_tier3:
            t3_edges = self._semantic.infer(nodes)
            edges.extend(t3_edges)
            log.debug("Tier 3: %d semantic edges", len(t3_edges))

        # Filter low-confidence edges.
        edges = [e for e in edges if e.confidence >= self.config.min_edge_confidence]

        G = self._build_graph(nodes, edges)
        T = self._traversability_subgraph(G)

        elapsed = time.perf_counter() - t0
        log.info(
            "KG built in %.3fs  nodes=%d  edges=%d  traversability_edges=%d",
            elapsed, G.number_of_nodes(), G.number_of_edges(), T.number_of_edges(),
        )

        kg_output = KGOutput(
            nodes=nodes,
            edges=edges,
            frame_id=seg_output.frame_id,
        )
        return KGRunResult(output=kg_output, graph=G, traversability=T)

    # ------------------------------------------------------------------

    @staticmethod
    def _build_graph(nodes: list[KGNode], edges: list[KGEdge]) -> "nx.DiGraph":
        G = nx.DiGraph()
        for node in nodes:
            G.add_node(node.node_id, **{
                "label":          node.label,
                "semantic_class": node.semantic_class.value,
                "centroid":       node.centroid,
                "priority":       node.priority,
                "track_id":       node.track_id,
                "mask_area":      node.mask_area,
                "frame_id":       node.frame_id,
                **node.attributes,
            })
        for edge in edges:
            G.add_edge(
                edge.subject_id, edge.object_id,
                predicate=edge.predicate.value,
                tier=edge.tier.value,
                confidence=edge.confidence,
                **edge.attributes,
            )
        return G

    @staticmethod
    def _traversability_subgraph(G: "nx.DiGraph") -> "nx.DiGraph":
        """
        Extract only traversable nodes and connected_to edges.
        This subgraph is what path planners consume directly.
        """
        traversable_nodes = [
            n for n, d in G.nodes(data=True)
            if d.get("semantic_class") in {c.value for c in _TRAVERSABLE}
        ]
        T = G.subgraph(traversable_nodes).copy()
        # Remove non-traversability edges.
        non_trav = [(u, v) for u, v, d in T.edges(data=True)
                    if d.get("predicate") != EdgePredicate.CONNECTED_TO.value]
        T.remove_edges_from(non_trav)
        return T


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class KGRunResult:
    output: KGOutput
    graph: "nx.DiGraph"           # full scene graph
    traversability: "nx.DiGraph"  # road-only connected_to subgraph