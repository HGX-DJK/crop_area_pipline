import sys, os
sys.path.insert(0, os.path.abspath('.'))
from scratch.test_watershed_on_gradient import markers, cropland
import numpy as np
import cv2

u = np.unique(markers[markers > 0])
print(f"Parcels to vectorize: {len(u)}")

for p in [3, 5, 7, 9, 10]:
    mask = (markers == p).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: continue
    cnt = contours[0]
    # approximate polygon
    epsilon = 0.01 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    print(f"Parcel {p}: num_raw_points={len(cnt)}, approx_polygon_vertices={len(approx)}")
    print(f"  Sample approx points: {approx[:, 0, :].tolist()[:6]}")
