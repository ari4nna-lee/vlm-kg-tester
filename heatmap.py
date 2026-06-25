"""
Generates heatmap
"""

import numpy as np
import cv2

def build_heatmap(masks, priorities, image_hw):
    H, W = image_hw
    heatmap = np.zeros((H, W), dtype=np.float32)

    for mask, p in zip(masks, priorities):

        mask = mask.astype(np.float32)

        print(mask.shape, (H, W))

        mask_bin = (mask > 0.5).astype(np.uint8)

        if mask_bin.sum() == 0:
            continue

        dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, 3)

        if dist.max() > 1e-6:
            dist /= dist.max()

        dist = np.power(dist, 0.6)
        dist = cv2.GaussianBlur(dist, (0, 0), sigmaX=15)

        heatmap += dist * p

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    return heatmap