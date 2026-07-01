"""
Generates heatmap
"""

import numpy as np
import cv2

def build_heatmap(masks, priorities, image_hw, max_priority_scale=1.0):
    H, W = image_hw
    heatmap = np.zeros((H, W), dtype=np.float32)
    for mask, p in zip(masks, priorities):
        mask = mask.astype(np.float32)
        mask_bin = (mask > 0.5).astype(np.uint8)
        if mask_bin.sum() == 0:
            continue
        dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 3)
        if dist.max() > 1e-6:
            dist /= dist.max()          # per-mask shape normalization is fine, this is local
        dist = np.power(dist, 0.6)
        dist = cv2.GaussianBlur(dist, (0, 0), sigmaX=15)
        heatmap += dist * p             # p is already priority in [0,1] — this IS the scale
    # do NOT renormalize by this frame's max — clip to the known valid range instead
    return np.clip(heatmap, 0.0, max_priority_scale)