"""
search_planner.py
=================
Extracts actionable search waypoints from the global KG.

Given the accumulated, priority-propagated global graph, clusters nodes
by their global_pose centroid and returns ranked waypoints the drone can
actually fly to.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
from scipy.spatial import cKDTree

import numpy as np

if TYPE_CHECKING:
    import networkx as nx

log = logging.getLogger(__name__)

# Object types that are directly search-relevant for a truck-finding task.
# Nodes of these classes anchor clusters — a cluster without any of these
# is lower value regardless of aggregate priority.
_ANCHOR_CLASSES = {"vehicle", "anomaly", "person"}


@dataclass
class SearchWaypoint:
    """
    A ranked search location in global mosaic coordinates.

    global_x, global_y : float
        Centroid of the cluster in global canvas pixel space.
        Pass these through MosaicTracker's inverse transform to get
        frame pixel coords, or use directly to overlay on global_mosaic.png.
    priority : float
        Aggregate priority of the cluster [0, 1].
    node_count : int
        Number of KG nodes in this cluster.
    anchor_count : int
        Number of anchor-class nodes (vehicle/anomaly/person).
        Higher = stronger evidence this is worth investigating.
    semantic_classes : list[str]
        Unique semantic classes present in the cluster.
        Diversity here is a signal — vehicle + road + person together
        is more compelling than three vehicle nodes in one spot.
    node_ids : list[str]
        IDs of contributing KG nodes, for traceability back to the graph.
    """
    global_x: float
    global_y: float
    priority: float
    node_count: int
    anchor_count: int
    semantic_classes: list[str]
    node_ids: list[str] = field(default_factory=list)


def extract_search_waypoints(
    G: "nx.DiGraph",
    cluster_radius: float = 80.0,
    min_priority: float = 0.4,
    top_k: int = 5,
) -> list[SearchWaypoint]:
    # --- collect nodes that have global_pose and meet priority threshold ---
    candidates = []
    for node_id, data in G.nodes(data=True):
        gp = data.get("global_pose")
        if gp is None:
            continue
        priority = data.get("priority", 0.0)
        if priority < min_priority:
            continue
        candidates.append({
            "node_id": node_id,
            "gx": gp["centroid"][0],
            "gy": gp["centroid"][1],
            "priority": priority,
            "semantic_class": data.get("semantic_class", "unknown"),
        })

    if not candidates:
        log.info("search_planner: no candidates above priority threshold %.2f", min_priority)
        return []
    
    log.info("search_planner: %d candidates entering clustering", len(candidates))
    candidates.sort(key=lambda c: c["priority"], reverse=True)
    points = np.array([[c["gx"], c["gy"]] for c in candidates])
    tree = cKDTree(points)

    assigned = np.zeros(len(candidates), dtype=bool)
    clusters:list[list[dict]] = []

    for i in range(len(candidates)):
        if assigned[i]:
            continue
        neighbor_idx = tree.query_ball_point(points[i], r=cluster_radius)
        unassigned_neighbors = [idx for idx in neighbor_idx if not assigned[idx]]
        for j in unassigned_neighbors:
            assigned[j] = True
        clusters.append([candidates[j] for j in unassigned_neighbors]) 

    # --- score each cluster and build waypoints ---
    waypoints = []
    for cluster in clusters:
        gxs = [c["gx"] for c in cluster]
        gys = [c["gy"] for c in cluster]
        priorities = [c["priority"] for c in cluster]
        classes = list({c["semantic_class"] for c in cluster})
        anchor_count = sum(1 for c in cluster if c["semantic_class"] in _ANCHOR_CLASSES)

        # aggregate priority: mean of top-3 nodes in cluster, boosted by
        # anchor presence and class diversity. Using top-3 mean rather than
        # max avoids a single outlier node dominating, and avoids mean being
        # dragged down by lower-priority nodes absorbed at the cluster edge.
        top3_mean = float(np.mean(sorted(priorities, reverse=True)[:3]))
        anchor_boost = min(0.15, anchor_count * 0.05)
        diversity_boost = min(0.1, (len(classes) - 1) * 0.03)
        agg_priority = min(1.0, top3_mean + anchor_boost + diversity_boost)

        # priority-weighted centroid — center of mass of the cluster,
        # pulled toward higher-priority nodes rather than geometric center
        total_w = sum(priorities)
        cx = sum(gx * p for gx, p in zip(gxs, priorities)) / total_w
        cy = sum(gy * p for gy, p in zip(gys, priorities)) / total_w

        waypoints.append(SearchWaypoint(
            global_x=cx,
            global_y=cy,
            priority=agg_priority,
            node_count=len(cluster),
            anchor_count=anchor_count,
            semantic_classes=classes,
            node_ids=[c["node_id"] for c in cluster],
        ))

    waypoints.sort(key=lambda w: w.priority, reverse=True)
    top = waypoints[:top_k]

    for i, w in enumerate(top):
        log.info(
            "waypoint %d: (%.1f, %.1f) priority=%.3f nodes=%d anchors=%d classes=%s",
            i + 1, w.global_x, w.global_y, w.priority,
            w.node_count, w.anchor_count, w.semantic_classes,
        )

    return top