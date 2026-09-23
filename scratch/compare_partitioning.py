import numpy as np
import cv2
from scipy import ndimage

# Create a 500x500 square mask
mask = np.ones((500, 500), dtype=np.uint8)
mask[0, :] = 0; mask[-1, :] = 0; mask[:, 0] = 0; mask[:, -1] = 0

# Grid seeds every 50 pixels (10x10 = 100 seeds)
grid_mask = np.zeros_like(mask, dtype=bool)
grid_mask[25::50, 25::50] = True
seeds = grid_mask & (mask > 0)
num_seeds, markers = cv2.connectedComponents(seeds.astype(np.uint8), connectivity=8)

# Method 1: What we currently do: watershed on (255 - dist)
padded = np.pad(mask, 1, mode="constant", constant_values=0)
dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
dist_norm = (dist / np.max(dist) * 255.0).astype(np.uint8)
landscape_bgr = cv2.cvtColor(255 - dist_norm, cv2.COLOR_GRAY2BGR)

m1 = markers.copy().astype(np.int32)
cv2.watershed(landscape_bgr, m1)
m1[mask == 0] = 0

# Method 2: Voronoi partitioning (distance transform from markers)
# Or watershed on morphological gradient (edges)
# In OpenCV, cv2.distanceTransform on (seeds == 0) with labels gives exact Voronoi cells!
_, labels = cv2.distanceTransformWithLabels((seeds == 0).astype(np.uint8), cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
m2 = labels.copy()
m2[mask == 0] = 0

# Check parcels in Method 1 vs Method 2
print("=== Method 1 (Watershed on 255 - dist) ===")
u1, c1 = np.unique(m1[m1 > 0], return_counts=True)
print(f"Parcels: {len(u1)}, counts min={np.min(c1)}, max={np.max(c1)}, mean={np.mean(c1)}")
# Check bounding box of a few parcels
for p in u1[:5]:
    coords = np.argwhere(m1 == p)
    h = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    w = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    print(f"  Parcel {p}: shape=({h}, {w}), ratio={h/w:.2f}")

print("\n=== Method 2 (Voronoi Distance Transform) ===")
u2, c2 = np.unique(m2[m2 > 0], return_counts=True)
print(f"Parcels: {len(u2)}, counts min={np.min(c2)}, max={np.max(c2)}, mean={np.mean(c2)}")
for p in u2[:5]:
    coords = np.argwhere(m2 == p)
    h = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    w = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    print(f"  Parcel {p}: shape=({h}, {w}), ratio={h/w:.2f}")
