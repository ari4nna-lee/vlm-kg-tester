import cv2
import numpy as np

def overlay_masks(frame, masks, alpha=0.4):
    overlay = frame.copy()

    for m in masks:
        mask = m.mask_array if hasattr(m, "mask_array") else m["mask"]
        color = np.random.randint(0, 255, (3,), dtype=np.uint8)

        overlay[mask] = (overlay[mask] * (1 - alpha) + color * alpha).astype(np.uint8)

    return overlay

