"""
centroid_map.py
===============
Builds a top-down map of all detected object centroids across frames,
compensating for camera pan using homography estimation.

Run after pipeline.py:
    python centroid_map.py
"""

import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from collections import defaultdict

JSON_DIR = Path("./results/json")
IMAGE_DIR = Path("/home/arianna/vlm-kg-tester/tests/neighborhood_subset")
OUTPUT_DIR = Path("./results")

# ---------------------------------------------------------------------------
# 1. Load frames and centroids
# ---------------------------------------------------------------------------

def load_data(json_dir, image_dir):
    frame_data = []
    for path in sorted(json_dir.glob("frame_*.json")):
        with open(path) as f:
            data = json.load(f)
        fid = data["frame_id"]

        # find matching image
        img_paths = sorted(image_dir.glob("*.jpg"))
        if fid < len(img_paths):
            img = cv2.cvtColor(cv2.imread(str(img_paths[fid])), cv2.COLOR_BGR2RGB)
        else:
            img = None

        nodes = data["graph"]["nodes"]
        frame_data.append({"fid": fid, "image": img, "nodes": nodes})

    return frame_data


# ---------------------------------------------------------------------------
# 2. Estimate cumulative homographies between frames
# ---------------------------------------------------------------------------

def estimate_homographies(frame_data):
    """
    Returns a list of 3x3 homography matrices H[i] that map frame i
    into the coordinate space of frame 0.
    """
    H_cumulative = [np.eye(3)]  # frame 0 is the reference

    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    for i in range(1, len(frame_data)):
        img_prev = frame_data[i - 1]["image"]
        img_curr = frame_data[i]["image"]

        if img_prev is None or img_curr is None:
            H_cumulative.append(H_cumulative[-1])  # fallback: no motion
            continue

        gray_prev = cv2.cvtColor(img_prev, cv2.COLOR_RGB2GRAY)
        gray_curr = cv2.cvtColor(img_curr, cv2.COLOR_RGB2GRAY)

        kp1, des1 = orb.detectAndCompute(gray_prev, None)
        kp2, des2 = orb.detectAndCompute(gray_curr, None)

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            H_cumulative.append(H_cumulative[-1])
            continue

        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:200]

        if len(matches) < 4:
            H_cumulative.append(H_cumulative[-1])
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        H, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC, 5.0)

        if H is None:
            H_cumulative.append(H_cumulative[-1])
            continue

        # compose with previous to get cumulative transform
        H_cumulative.append(H_cumulative[-1] @ H)

    return H_cumulative


# ---------------------------------------------------------------------------
# 3. Project centroids into global space
# ---------------------------------------------------------------------------

def project_centroids(frame_data, homographies, img_hw=(720, 1280)):
    H_img, W_img = img_hw
    all_points = []  # (global_x, global_y, label, semantic_class, priority, fid)

    for frame, H in zip(frame_data, homographies):
        fid = frame["fid"]
        for node in frame["nodes"]:
            centroid = node.get("centroid", None)

            # centroid may be stored as scalar (the bug) or as [cx, cy]
            if isinstance(centroid, (list, tuple)) and len(centroid) == 2:
                cx_n, cy_n = centroid
            else:
                # fall back to bbox center
                bbox = node.get("bbox", None)
                if bbox and isinstance(bbox, dict):
                    cx_n = bbox.get("x", 0.5) + bbox.get("w", 0) / 2
                    cy_n = bbox.get("y", 0.5) + bbox.get("h", 0) / 2
                else:
                    continue

            # convert normalized to pixel
            px = cx_n * W_img
            py = cy_n * H_img

            # apply homography
            pt = np.array([[[px, py]]], dtype=np.float32)
            pt_global = cv2.perspectiveTransform(pt, H)[0][0]

            all_points.append({
                "x": float(pt_global[0]),
                "y": float(pt_global[1]),
                "label": node.get("label", "?"),
                "semantic_class": node.get("semantic_class", "?"),
                "priority": node.get("priority", 0.0),
                "fid": fid,
            })

    return all_points


# ---------------------------------------------------------------------------
# 4. Render the map
# ---------------------------------------------------------------------------

def render_map(all_points, output_path):
    if not all_points:
        print("No points to render")
        return

    xs = [p["x"] for p in all_points]
    ys = [p["y"] for p in all_points]
    priorities = [p["priority"] for p in all_points]
    labels = [f"{p['label']}\nf{p['fid']}" for p in all_points]

    cmap = cm.get_cmap("jet")
    p_min, p_max = min(priorities), max(priorities)
    p_range = p_max - p_min if p_max > p_min else 1.0
    colors = [cmap((p - p_min) / p_range) for p in priorities]

    fig, ax = plt.subplots(figsize=(20, 14))

    # scatter
    sc = ax.scatter(xs, ys, c=priorities, cmap="jet",
                    vmin=p_min, vmax=p_max,
                    s=200, zorder=3, edgecolors="black", linewidths=0.5)

    # labels
    for x, y, label in zip(xs, ys, labels):
        ax.annotate(label, (x, y),
                    textcoords="offset points", xytext=(6, 6),
                    fontsize=6,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6))

    # connect same-class nodes across frames with lines
    from collections import defaultdict
    buckets = defaultdict(list)
    for p in all_points:
        key = (p["semantic_class"], p["label"])
        buckets[key].append(p)

    for key, pts in buckets.items():
        pts_sorted = sorted(pts, key=lambda p: p["fid"])
        for i in range(len(pts_sorted) - 1):
            a, b = pts_sorted[i], pts_sorted[i + 1]
            if b["fid"] - a["fid"] <= 1:
                ax.plot([a["x"], b["x"]], [a["y"], b["y"]],
                        "steelblue", linewidth=0.8,
                        linestyle="--", alpha=0.5, zorder=2)

    plt.colorbar(sc, ax=ax, label="priority")
    ax.set_title("Scene Centroid Map (camera-motion compensated)")
    ax.set_xlabel("global x (pixels)")
    ax.set_ylabel("global y (pixels)")
    ax.invert_yaxis()  # image Y is top-down
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading data...")
    frame_data = load_data(JSON_DIR, IMAGE_DIR)

    print("Estimating camera motion...")
    homographies = estimate_homographies(frame_data)

    print("Projecting centroids...")
    all_points = project_centroids(frame_data, homographies)

    print(f"Total centroids: {len(all_points)}")
    render_map(all_points, OUTPUT_DIR / "centroid_map.png")