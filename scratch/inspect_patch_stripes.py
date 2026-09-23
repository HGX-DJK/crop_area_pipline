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

sl = slices[7]
sub_labeled = labeled_array[sl]
local_mask = (sub_labeled == 8).astype(np.uint8)

segmenter = ParcelSegmenter({'segmentation': {'max_parcel_area_m2': 800000.0}})
res = segmenter._subdivide_oversized_component(local_mask)

patch = res[400:700, 100:300]
u_p = [p for p in np.unique(patch) if p > 0]
print("Unique parcel IDs in patch:", len(u_p))

for pid in u_p[:15]:
    m = (res == pid)
    coords = np.argwhere(m)
    r_len = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    c_len = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    print(f"Parcel {pid}: row_span={r_len}, col_span={c_len}, ratio={r_len/c_len:.2f}")
