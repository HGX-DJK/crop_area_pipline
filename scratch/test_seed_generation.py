import numpy as np
import cv2
from scipy import ndimage

# Patch (500, 200)
mask = np.ones((500, 200), dtype=np.uint8)
mask[0:10, :] = 0; mask[-10:, :] = 0; mask[:, 0:10] = 0; mask[:, -10:] = 0

spatial_res = 30.0
pixel_area_m2 = 900.0
comp_pixels = int(np.sum(mask))
comp_area_m2 = comp_pixels * pixel_area_m2
max_area_m2 = 800000.0
target_parcel_m2 = max(float(max_area_m2) * 0.5, 300_000.0)

grid_step = max(5, int(round(np.sqrt(target_parcel_m2) / spatial_res)))
print(f"grid_step = {grid_step} pixels ({grid_step * spatial_res} meters)")

# Generate isolated 1-pixel grid seeds inside the mask
# Use an offset so they are nicely centered
grid_mask = np.zeros_like(mask, dtype=bool)
grid_mask[grid_step // 2 :: grid_step, grid_step // 2 :: grid_step] = True
grid_seeds = grid_mask & (mask > 0)

# Check if there are natural distance peaks
padded = np.pad(mask, 1, mode="constant", constant_values=0)
dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
win = max(3, int(round(180.0 / spatial_res)))
if win % 2 == 0: win += 1
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
dilated = cv2.dilate(dist, kernel)
threshold_height = max(1.5, min(2.5, np.max(dist) * 0.15))
local_peaks = (dist == dilated) & (dist >= threshold_height) & (mask > 0)

# For any local peaks, replace each connected peak component with its single centroid pixel!
num_raw_peaks, peak_labels = cv2.connectedComponents(local_peaks.astype(np.uint8), connectivity=8)
peak_centroids_mask = np.zeros_like(mask, dtype=bool)
for lab in range(1, num_raw_peaks):
    coords = np.argwhere(peak_labels == lab)
    c_r, c_c = int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1]))
    if mask[c_r, c_c] > 0:
        peak_centroids_mask[c_r, c_c] = True

print(f"Number of peak centroids: {np.sum(peak_centroids_mask)}")
print(f"Number of grid seeds: {np.sum(grid_seeds)}")

# If oversized, grid seeds guarantee spatial subdivision without elongated ridges!
# Combine: if a grid seed is too close to a peak centroid, keep one.
# Or simply use grid_seeds when comp_area_m2 > max_area_m2!
seeds = grid_seeds.copy()

num_seeds, seed_markers = cv2.connectedComponents(seeds.astype(np.uint8), connectivity=8)
print(f"Total isolated seeds: {num_seeds - 1}")

# Voronoi partitioning
_, nearest_idx = ndimage.distance_transform_edt(seed_markers == 0, return_indices=True)
voronoi = seed_markers[nearest_idx[0], nearest_idx[1]]
voronoi[mask == 0] = 0

u = np.unique(voronoi[voronoi > 0])
ratios = []
for p in u:
    coords = np.argwhere(voronoi == p)
    h = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    w = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    ratios.append(h / w)

print(f"Voronoi with isolated grid seeds:")
print(f"  Parcels: {len(u)}, mean ratio: {np.mean(ratios):.2f}, min: {np.min(ratios):.2f}, max: {np.max(ratios):.2f}")
print("  First 10 parcel shapes (h, w):")
for p in u[:10]:
    coords = np.argwhere(voronoi == p)
    h = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    w = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    print(f"    P{p}: ({h}, {w}), ratio: {h/w:.2f}")
