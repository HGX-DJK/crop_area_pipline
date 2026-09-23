import numpy as np
import cv2
from scipy import ndimage

dummy_large = np.ones((600, 300), dtype=np.uint8)
dummy_large[0:5, :] = 0; dummy_large[-5:, :] = 0; dummy_large[:, 0:5] = 0; dummy_large[:, -5:] = 0

spatial_res = 30.0
pixel_area_m2 = 900.0
comp_pixels = int(np.sum(dummy_large))
comp_area_m2 = comp_pixels * pixel_area_m2
max_area_m2 = 800000.0

target_parcel_m2 = max(float(max_area_m2) * 0.5, 300_000.0)
grid_step = max(5, int(round(np.sqrt(target_parcel_m2) / spatial_res)))

# Step 1: Detect natural distance peaks
padded = np.pad(dummy_large, 1, mode="constant", constant_values=0)
dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
win = max(3, int(round(180.0 / spatial_res)))
if win % 2 == 0: win += 1
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
dilated = cv2.dilate(dist, kernel)
threshold_height = max(1.5, min(2.5, np.max(dist) * 0.15))
local_peaks = (dist == dilated) & (dist >= threshold_height) & (dummy_large > 0)

# Step 2: Grid seeds
grid_mask = np.zeros_like(dummy_large, dtype=bool)
grid_mask[grid_step // 2 :: grid_step, grid_step // 2 :: grid_step] = True
grid_seeds = grid_mask & (dummy_large > 0)

# Step 3: Peak centroids
num_raw_peaks, peak_labels = cv2.connectedComponents(local_peaks.astype(np.uint8), connectivity=8)
peak_centroids_mask = np.zeros_like(dummy_large, dtype=bool)
for lab in range(1, num_raw_peaks):
    coords = np.argwhere(peak_labels == lab)
    c_r, c_c = int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1]))
    if dummy_large[c_r, c_c] > 0:
        peak_centroids_mask[c_r, c_c] = True

# For homogeneous plain, use grid seeds to guarantee compact 2D parcels
seeds = grid_seeds | peak_centroids_mask

num_seeds, seed_markers = cv2.connectedComponents(seeds.astype(np.uint8), connectivity=8)

# Voronoi partitioning via EDT
_, nearest_idx = ndimage.distance_transform_edt(seed_markers == 0, return_indices=True)
voronoi = seed_markers[nearest_idx[0], nearest_idx[1]]
voronoi[dummy_large == 0] = 0

u = np.unique(voronoi[voronoi > 0])
ratios = []
for p in u:
    coords = np.argwhere(voronoi == p)
    h = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    w = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    ratios.append(h / w)

print(f"Voronoi subdivision on dummy_large (600x300):")
print(f"  Num parcels: {len(u)}, Mean aspect ratio: {np.mean(ratios):.2f}, Max aspect ratio: {np.max(ratios):.2f}, Min: {np.min(ratios):.2f}")
