"""
postprocess.py
===============
Regenerate the global mosaic, priority map, and search waypoints from a
saved pipeline_state.pkl — no VLM/SAM/KG frame processing required.
"""

import pickle
import logging
from pathlib import Path

import cv2
import numpy as np

from search_planner import extract_search_waypoints

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
log = logging.getLogger(__name__)

OUTPUT_DIR = Path("./results")

with open(OUTPUT_DIR / "pipeline_state.pkl", "rb") as f:
    state = pickle.load(f)

kg_graph = state["kg_graph"]
mosaic = state["mosaic"]

log.info("Loaded graph: %d nodes, %d edges", kg_graph.number_of_nodes(), kg_graph.number_of_edges())

# --- global mosaic ---
global_canvas = mosaic.get_canvas()
if global_canvas is not None:
    cv2.imwrite(str(OUTPUT_DIR / "global_mosaic.png"), cv2.cvtColor(global_canvas, cv2.COLOR_RGB2BGR))
    log.info("global_mosaic.png written")

# --- priority map ---
log.info("Generating priority map...")
priority_map = mosaic.get_priority_map(kg_graph)
if priority_map is not None:
    vis = cv2.normalize(priority_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    cv2.imwrite(str(OUTPUT_DIR / "global_priority_map.png"), vis)
    log.info("global_priority_map.png written")

# --- waypoints ---
log.info("Extracting waypoints...")
waypoints = extract_search_waypoints(kg_graph, top_k=5)

if global_canvas is not None and waypoints:
    viz = cv2.cvtColor(global_canvas, cv2.COLOR_RGB2BGR)
    for i, w in enumerate(waypoints):
        cx = int(w.global_x - mosaic._canvas_origin[0])
        cy = int(w.global_y - mosaic._canvas_origin[1])
        cv2.circle(viz, (cx, cy), 20, (0, 0, 255), 3)
        cv2.putText(viz, f"W{i+1} p={w.priority:.2f}",
                    (cx + 5, cy - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1)
    cv2.imwrite(str(OUTPUT_DIR / "global_mosaic_waypoints.png"), viz)
    log.info("global_mosaic_waypoints.png written")

log.info("Postprocessing complete.")