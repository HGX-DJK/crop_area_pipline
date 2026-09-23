import rasterio
import numpy as np
import time

# Let's inspect the actual cropland mask from the previous run if it exists
import os
import cv2
from scipy import ndimage

print("Testing OpenCV distance transform and watershed performance on a 3000x3000 synthetic plain...")
t0 = time.time()
test_mask = np.ones((3000, 3000), dtype=np.uint8)
# Add some empty borders
test_mask[0, :] = 0
test_mask[-1, :] = 0
test_mask[:, 0] = 0
test_mask[:, -1] = 0

# cv2 distanceTransform
dist = cv2.distanceTransform(test_mask, cv2.DIST_L2, 5)
t_dist = time.time() - t0
print(f"Distance transform on 3000x3000 completed in {t_dist:.3f} s, max dist = {np.max(dist):.1f}")

# Notice what dist looks like on a 3000x3000 square:
# It increases steadily from 1 at border to 1500 at the center!
# If we look for local maxima (dist == dilated) with kernel size 15 (450m),
# what are the local peaks?
win = 15
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
dilated = cv2.dilate(dist, kernel)
peaks = (dist == dilated) & (dist >= 2.5) & (test_mask > 0)
num_peaks, _ = cv2.connectedComponents(peaks.astype(np.uint8))
print(f"Num peaks on pure square: {num_peaks - 1}")
