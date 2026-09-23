import sys, os
sys.path.insert(0, os.path.abspath('.'))
import rasterio
import numpy as np
import cv2
from scipy import ndimage
from src.parcel_segmenter import ParcelSegmenter

with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)

cropland_binary = (mask > 0).astype(np.uint8)
structure_cross = ndimage.generate_binary_structure(2, 1)
cleaned = ndimage.binary_opening(cropland_binary, structure=structure_cross).astype(np.uint8)
labeled_array, num_features = ndimage.label(cleaned, structure=structure_cross)
slices = ndimage.find_objects(labeled_array)

sl = slices[7] # Component #8
sub_mask = (labeled_array[sl] == 8).astype(np.uint8)
print(f"Component #8 mask shape: {sub_mask.shape}, sum={np.sum(sub_mask)}")

segmenter = ParcelSegmenter({'segmentation': {'max_parcel_area_m2': 800000.0}})
res = segmenter._subdivide_oversized_component(sub_mask)

u, counts = np.unique(res[res > 0], return_counts=True)
print(f"Subdivided into {len(u)} parcels.")

# Check shapes of individual parcels
for p_id in u[:20]:
    coords = np.argwhere(res == p_id)
    r_min, c_min = np.min(coords, axis=0)
    r_max, c_max = np.max(coords, axis=0)
    w = c_max - c_min + 1
    h = r_max - r_min + 1
    print(f"Parcel {p_id}: shape=({h}, {w}), ratio={h/w:.2f}, count={np.sum(res==p_id)}")
