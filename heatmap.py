"""
Generates heatmap
"""

import numpy as np
import cv2

def build_heatmap(masks, priorities, image_hw):
    H, W = image_hw
    heatmap = np.zeros((H, W), dtype=np.float32)

    for mask, p in zip(masks, priorities):
        mask_bin = (mask > 0).astype(np.uint8)

        # Distance transform: each pixel gets its distance to the nearest 0
        # This creates a smooth gradient that peaks at the object's "center of mass"
        # and naturally follows the shape of the segmented object
        dist = cv2.distanceTransform(mask_bin, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # Normalize so the peak of this object = 1.0 before weighting
        dist = dist / (dist.max() + 1e-6)

        # Now blur — on a smooth gradient this produces organic blobs
        # Use a large sigma relative to your object size; sigmaX=0 lets OpenCV choose
        sigma = max(H, W) // 20          # ~5% of image width — tune this
        k = sigma * 6 + 1                # kernel must be odd
        blurred = cv2.GaussianBlur(dist, (k, k), sigmaX=sigma)

        heatmap += blurred * p

    heatmap = heatmap / (np.max(heatmap) + 1e-6)
    return heatmap