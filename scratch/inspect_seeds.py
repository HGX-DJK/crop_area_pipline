import sys, os
sys.path.insert(0, os.path.abspath('.'))
import rasterio
import numpy as np
import cv2
from scipy import ndimage

with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)

cropland_binary = (mask > 0).astype(np.uint8)
structure_cross = ndimage.generate_binary_structure(2, 1)
cleaned = ndimage.binary_opening(cropland_binary, structure=structure_cross).astype(np.uint8)
labeled_array, num_features = ndimage.label(cleaned, structure=structure_cross)
slices = ndimage.find_objects(labeled_array)

sl = slices[7] # Component #8
sub_binary_mask = (labeled_array[sl] == 8).astype(np.uint8)

# Now run the peak and grid generation:
padded_mask = np.pad(sub_binary_mask, pad_width=1, mode="constant", constant_values=0)
dist = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)[1:-1, 1:-1]
max_dist = float(np.max(dist))
win = max(3, int(round(180.0 / 30.0)))
if win % 2 == 0: win += 1
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
dilated = cv2.dilate(dist, kernel)
threshold_height = max(1.5, min(2.5, max_dist * 0.15))
local_peaks = (dist == dilated) & (dist >= threshold_height) & (sub_binary_mask > 0)

target_parcel_m2 = 400000.0
grid_step = max(5, int(round(np.sqrt(target_parcel_m2) / 30.0))) # 21
grid_mask = np.zeros_like(sub_binary_mask, dtype=bool)
grid_mask[grid_step // 2 :: grid_step, grid_step // 2 :: grid_step] = True
grid_seeds = grid_mask & (sub_binary_mask > 0)
all_seeds = local_peaks | grid_seeds

num_peaks, peak_markers = cv2.connectedComponents(all_seeds.astype(np.uint8), connectivity=8)

# Check seeds in rows 400..500, cols (2670..2730 - 2551) = (119..179)
seeds_patch = all_seeds[400:500, 119:179]
print("Seeds count in patch (rows 400..500, cols 119..179):", np.sum(seeds_patch))
print("Where are the seeds in this patch?")
coords = np.argwhere(seeds_patch)
print("Seed coordinates (r, c):", coords)
