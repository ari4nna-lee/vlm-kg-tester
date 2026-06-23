"""
Generates heatmap
"""

import numpy as np

def build_heatmap(masks, priorities, image_hw):
    H, W = image_hw
    heatmap = np.zeros((H, W), dtype=np.float32)

    for mask, p in zip(masks, priorities):
        heatmap += mask.astype(np.float32) * p

    heatmap = heatmap / (np.max(heatmap) + 1e-6) 
    return heatmap