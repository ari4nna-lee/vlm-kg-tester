"""
mosaic.py
=========
Incremental, streaming version of centroid_map.py's homography logic.

Owned once by Pipeline. Call `update(fid, frame)` exactly once per unique
frame_id, in frame_id order, from kg_worker. Returns the 3x3 homography
that maps THIS frame's pixel coords -> the running global canvas space
(global space == frame 0's pixel space, same convention as
centroid_map.estimate_homographies, just computed online instead of
batch).

Safe against re-delivery: if `update()` is called again with an fid it
has already seen (e.g. a kg_refine re-pass), it returns the cached H
instead of re-running ORB / advancing the chain.

Also exposes `project_point` / `project_bbox` helpers and an optional
running canvas image accumulator for visualization.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class MosaicConfig:
    orb_features: int = 2000
    match_count: int = 200
    ransac_thresh: float = 5.0
    min_matches: int = 4
    # canvas accumulation (optional, only needed if you want the actual
    # stitched image / heatmap raster, not just node global_pose)
    build_canvas: bool = True
    canvas_pad: int = 2000           # extra pixels of slack before a resize
    canvas_blend_alpha: float = 0.5  # weight given to NEW frame on overlap
    keyframe_interval = 5

class MosaicTracker:
    def __init__(self, frame_hw: tuple[int, int], cfg: Optional[MosaicConfig] = None):
        self.cfg = cfg or MosaicConfig()
        self.frame_h, self.frame_w = frame_hw

        self._orb = cv2.ORB_create(self.cfg.orb_features)
        self._bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self._lock = threading.Lock()

        self._prev_gray: Optional[np.ndarray] = None
        self._prev_kp = None
        self._prev_des = None

        self._H_cumulative = np.eye(3, dtype=np.float64)  # maps newest frame -> global(frame 0)
        self._fid_to_H: dict[int, np.ndarray] = {}
        self._last_fid_seen: Optional[int] = None

        # --- canvas state ---
        # canvas_origin: global-space coordinate that maps to canvas pixel (0,0).
        # Starts at (0,0); we translate the canvas (not re-warp history) when
        # new content would fall outside current bounds.
        self._canvas_origin = np.array([0.0, 0.0])
        self._canvas: Optional[np.ndarray] = None      # accumulated RGB mosaic
        self._heat_canvas: Optional[np.ndarray] = None  # accumulated heatmap (float32)
        self._heat_weight: Optional[np.ndarray] = None  # blend weight accumulator

        self._keyframe_gray: Optional[np.ndarray] = None
        self._keyframe_H: np.ndarray = np.eye(3, dtype=np.float64)

    # ------------------------------------------------------------------
    # Core: per-frame homography
    # ------------------------------------------------------------------

    def update(self, fid: int, frame_rgb: np.ndarray) -> np.ndarray:
        with self._lock:
            if fid in self._fid_to_H:
                return self._fid_to_H[fid]

            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)

            if self._prev_gray is None:
                H_global = np.eye(3, dtype=np.float64)
                self._keyframe_gray = gray
                self._keyframe_H = H_global.copy()
            else:
                is_keyframe = (fid % self.cfg.keyframe_interval == 0)
                if is_keyframe:
                    self._keyframe_gray = gray
                    self._keyframe_H = self._H_cumulative.copy()

                H_step = self._estimate_step(self._keyframe_gray, gray)
                H_global = self._keyframe_H @ H_step

                if not self._is_valid_homography(H_global):
                    log.warning("mosaic: fid %s keyframe match failed, trying frame-to-frame", fid)
                    H_step = self._estimate_step(self._prev_gray, gray)
                    H_global = self._H_cumulative @ H_step
                    if not self._is_valid_homography(H_global):
                        log.warning("mosaic: fid %s both mathes failed, holding position", fid)
                        H_global = self._H_cumulative.copy()

            self._H_cumulative = H_global.copy()
            self._fid_to_H[fid] = H_global
            self._last_fid_seen = fid
            self._prev_gray = gray

            if self.cfg.build_canvas:
                self._accumulate_canvas(frame_rgb, H_global)

            return H_global

    def _estimate_step(self, gray_prev: np.ndarray, gray_curr: np.ndarray) -> np.ndarray:
        kp1, des1 = self._orb.detectAndCompute(gray_prev, None)
        kp2, des2 = self._orb.detectAndCompute(gray_curr, None)

        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            log.warning("mosaic: insufficient features, assuming no motion")
            return np.eye(3, dtype=np.float64)

        knn = self._bf.knnMatch(des1, des2, k=2)
        good = [m for m, n in knn if m.distance < 0.75 * n.distance]
        good = sorted(good, key=lambda m: m.distance)[: self.cfg.match_count]

        if len(good) < self.cfg.min_matches:
            log.warning("mosaic: insufficient matches (%d), assuming no motion", len(good))
            return np.eye(3, dtype=np.float64)

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, self.cfg.ransac_thresh)
        if not self._is_valid_homography(H):
            log.warning("mosaic: rejected degenerate homography, assuming no motion")
            return np.eye(3, dtype=np.float64)
        return H

    # ------------------------------------------------------------------
    # Projection helpers (used by kg.py to set node.attributes["global_pose"])
    # ------------------------------------------------------------------

    @staticmethod
    def project_point(px: float, py: float, H: np.ndarray) -> tuple[float, float]:
        pt = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, H.astype(np.float32))[0][0]
        return float(out[0]), float(out[1])

    @classmethod
    def project_bbox_xyxy(cls, x1, y1, x2, y2, H: np.ndarray):
        corners = np.array([[[x1, y1]], [[x2, y1]], [[x2, y2]], [[x1, y2]]], dtype=np.float32)
        out = cv2.perspectiveTransform(corners, H.astype(np.float32)).reshape(-1, 2)
        gx1, gy1 = out[:, 0].min(), out[:, 1].min()
        gx2, gy2 = out[:, 0].max(), out[:, 1].max()
        return float(gx1), float(gy1), float(gx2), float(gy2)

    # ------------------------------------------------------------------
    # Canvas accumulation (for visualization / heatmap blending)
    # ------------------------------------------------------------------

    def _ensure_canvas_bounds(self, H_global: np.ndarray):
        """Grow/translate the canvas so this frame's warped extent fits."""
        x1, y1, x2, y2 = self.project_bbox_xyxy(0, 0, self.frame_w, self.frame_h, H_global)

        if self._canvas is None:
            w = int(x2 - x1) + 2 * self.cfg.canvas_pad
            h = int(y2 - y1) + 2 * self.cfg.canvas_pad
            self._canvas_origin = np.array([x1 - self.cfg.canvas_pad, y1 - self.cfg.canvas_pad])
            self._canvas = np.zeros((h, w, 3), dtype=np.uint8)
            self._heat_canvas = np.zeros((h, w), dtype=np.float32)
            self._heat_weight = np.zeros((h, w), dtype=np.float32)
            return

        cx1, cy1 = self._canvas_origin
        cy2 = cy1 + self._canvas.shape[0]
        cx2 = cx1 + self._canvas.shape[1]

        pad_left = max(0, int(cx1 - (x1 - self.cfg.canvas_pad)))
        pad_top = max(0, int(cy1 - (y1 - self.cfg.canvas_pad)))
        pad_right = max(0, int((x2 + self.cfg.canvas_pad) - cx2))
        pad_bottom = max(0, int((y2 + self.cfg.canvas_pad) - cy2))

        if pad_left or pad_top or pad_right or pad_bottom:
            new_h = self._canvas.shape[0] + pad_top + pad_bottom
            new_w = self._canvas.shape[1] + pad_left + pad_right
            new_canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
            new_heat = np.zeros((new_h, new_w), dtype=np.float32)
            new_weight = np.zeros((new_h, new_w), dtype=np.float32)

            new_canvas[pad_top:pad_top + self._canvas.shape[0],
                       pad_left:pad_left + self._canvas.shape[1]] = self._canvas
            new_heat[pad_top:pad_top + self._heat_canvas.shape[0],
                      pad_left:pad_left + self._heat_canvas.shape[1]] = self._heat_canvas
            new_weight[pad_top:pad_top + self._heat_weight.shape[0],
                       pad_left:pad_left + self._heat_weight.shape[1]] = self._heat_weight

            self._canvas = new_canvas
            self._heat_canvas = new_heat
            self._heat_weight = new_weight
            self._canvas_origin = self._canvas_origin - np.array([pad_left, pad_top])

    def _global_to_canvas_H(self, H_global: np.ndarray) -> np.ndarray:
        T = np.array([
            [1, 0, -self._canvas_origin[0]],
            [0, 1, -self._canvas_origin[1]],
            [0, 0, 1],
        ], dtype=np.float64)
        return T @ H_global

    def _accumulate_canvas(self, frame_rgb: np.ndarray, H_global: np.ndarray):
        self._ensure_canvas_bounds(H_global)
        H_canvas = self._global_to_canvas_H(H_global)
        h, w = self._canvas.shape[:2]

        warped = cv2.warpPerspective(frame_rgb, H_canvas.astype(np.float32), (w, h))
        mask = cv2.warpPerspective(
            np.ones(frame_rgb.shape[:2], dtype=np.uint8) * 255,
            H_canvas.astype(np.float32), (w, h),
        ) > 0

        alpha = self.cfg.canvas_blend_alpha
        self._canvas[mask] = (
            (1 - alpha) * self._canvas[mask] + alpha * warped[mask]
        ).astype(np.uint8)

    def accumulate_heatmap(self, fid: int, heatmap: np.ndarray):
        """
        Reproject a per-frame heatmap (H x W float, same frame size as
        frame_hw) into the global canvas and blend it in. Call after
        update(fid, ...) has already been called for this fid.
        """
        with self._lock:
            H_global = self._fid_to_H.get(fid)
            if H_global is None or self._canvas is None:
                log.warning("accumulate_heatmap: no homography/canvas yet for fid %s", fid)
                return
            H_canvas = self._global_to_canvas_H(H_global)
            h, w = self._heat_canvas.shape

            warped_heat = cv2.warpPerspective(
                heatmap.astype(np.float32), H_canvas.astype(np.float32), (w, h)
            )
            warped_mask = cv2.warpPerspective(
                np.ones(heatmap.shape, dtype=np.float32), H_canvas.astype(np.float32), (w, h)
            )

            # running weighted average per-pixel (more frames seeing a spot
            # -> more confident priority estimate there)
            self._heat_canvas += warped_heat * warped_mask
            self._heat_weight += warped_mask

    def get_global_heatmap(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._heat_canvas is None:
                return None
            w = np.maximum(self._heat_weight, 1e-6)
            return self._heat_canvas / w
        
    def get_priority_map(
        self,
        kg_graph: "nx.DiGraph",
        kg_weight: float = 0.6,
        heatmap_weight: float = 0.4,
    ) -> Optional[np.ndarray]:
        with self._lock:
            if self._canvas is None:
                return None

            h, w = self._canvas.shape[:2]
            kg_layer = np.zeros((h, w), dtype=np.float32)

            for node_id, data in kg_graph.nodes(data=True):
                gp = data.get("global_pose")
                if gp is None:
                    continue
                gx, gy = gp["centroid"]
                cx = int(gx - self._canvas_origin[0])
                cy = int(gy - self._canvas_origin[1])
                if not (0 <= cx < w and 0 <= cy < h):
                    continue
                priority = float(data.get("priority", 0.0))
                confidence = float(data.get("confidence", 1.0))
                radius = max(10, int(data.get("mask_area", 0.01) * min(h, w)))
                cv2.circle(kg_layer, (cx, cy), radius,
                        float(priority * confidence), thickness=-1)

            if kg_layer.max() > 0:
                kg_layer = kg_layer / kg_layer.max()

            heat = self.get_global_heatmap()
            if heat is not None:
                if heat.shape != (h, w):
                    heat = cv2.resize(heat, (w, h))
                if heat.max() > 0:
                    heat = heat / heat.max()
            else:
                heat = np.zeros((h, w), dtype=np.float32)

            return np.clip(
                kg_weight * kg_layer + heatmap_weight * heat,
                0.0, 1.0
            ).astype(np.float32)

    def get_canvas(self) -> Optional[np.ndarray]:
        with self._lock:
            return None if self._canvas is None else self._canvas.copy()

    @staticmethod
    def _is_valid_homography(H: Optional[np.ndarray], max_translation: float = 400.0) -> bool:
        if H is None:
            return False
        det = np.linalg.det(H[:2, :2])
        if not (0.1 < abs(det) < 10.0):
            return False
        if abs(H[2, 0]) > 1e-3 or abs(H[2, 1]) > 5e-3:
            return False
        # reject implausibly large translations between frames
        tx, ty = H[0, 2], H[1, 2]
        if abs(tx) > max_translation or abs(ty) > max_translation:
            return False
        return True