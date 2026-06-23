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

        blurred = cv2.GaussianBlur(mask, (31, 31), 0)
        heatmap += blurred * p

    heatmap = heatmap / (np.max(heatmap) + 1e-6)
    return heatmap