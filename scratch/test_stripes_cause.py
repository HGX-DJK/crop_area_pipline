import numpy as np
import cv2
from scipy import ndimage

# Suppose we have a patch of shape (500, 200) of cropland
mask = np.ones((500, 200), dtype=np.uint8)
# Add some borders
mask[0:10, :] = 0; mask[-10:, :] = 0; mask[:, 0:10] = 0; mask[:, -10:] = 0

comp_pixels = int(np.sum(mask))
spatial_res = 30.0
pixel_area_m2 = 900.0
comp_area_m2 = comp_pixels * pixel_area_m2
max_area_m2 = 800000.0

padded_mask = np.pad(mask, pad_width=1, mode="constant", constant_values=0)
dist = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)[1:-1, 1:-1]
win = max(3, int(round(180.0 / 30.0)))
if win % 2 == 0: win += 1
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
dilated = cv2.dilate(dist, kernel)

threshold_height = max(1.5, min(2.5, np.max(dist) * 0.15))
local_peaks = (dist == dilated) & (dist >= threshold_height) & (mask > 0)
num_peaks, peak_markers = cv2.connectedComponents(local_peaks.astype(np.uint8), connectivity=8)
num_peaks -= 1

target_parcel_m2 = max(float(max_area_m2) * 0.5, 300_000.0)
expected_parcels = int(round(comp_area_m2 / target_parcel_m2))

if comp_area_m2 > float(max_area_m2) and num_peaks < expected_parcels * 0.5 and expected_parcels > 1:
    grid_step = max(5, int(round(np.sqrt(target_parcel_m2) / 30.0)))
    grid_mask = np.zeros_like(mask, dtype=bool)
    grid_mask[grid_step // 2 :: grid_step, grid_step // 2 :: grid_step] = True
    grid_seeds = grid_mask & (mask > 0)
    all_seeds = local_peaks | grid_seeds
    num_peaks, peak_markers = cv2.connectedComponents(all_seeds.astype(np.uint8), connectivity=8)
    num_peaks -= 1
else:
    all_seeds = local_peaks

# Method 2: Voronoi Partitioning from seeds
# For each pixel in mask, assign it to the nearest seed marker
# Using cv2.distanceTransformWithLabels on (all_seeds == 0)
# But wait, distanceTransformWithLabels gives the index of the nearest seed point!
# If we use EDT on peak_markers > 0:
_, nearest_idx = ndimage.distance_transform_edt(peak_markers == 0, return_indices=True)
voronoi_markers = peak_markers[nearest_idx[0], nearest_idx[1]]
voronoi_markers[mask == 0] = 0

u_vor = np.unique(voronoi_markers[voronoi_markers > 0])
ratios_vor = []
for p in u_vor:
    coords = np.argwhere(voronoi_markers == p)
    h = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    w = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    ratios_vor.append(h / w)

print(f"Voronoi from seeds:")
print(f"  Parcels: {len(u_vor)}, mean ratio: {np.mean(ratios_vor):.2f}, min: {np.min(ratios_vor):.2f}, max: {np.max(ratios_vor):.2f}")
print("  Sample parcel shapes (h, w):")
for p in u_vor[:10]:
    coords = np.argwhere(voronoi_markers == p)
    h = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    w = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    print(f"    P{p}: ({h}, {w}), ratio: {h/w:.2f}")
