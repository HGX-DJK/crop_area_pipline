import sys, os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import rasterio
from src.raster_loader import _compute_sdc6_physical_indices
from src.parcel_segmenter import ParcelSegmenter
from scipy import ndimage

# 1. Test raster_loader physical indices on a 500x500 mountain patch (rows 400..900, cols 2400..2900)
with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    win = rasterio.windows.Window(col_off=2400, row_off=400, width=500, height=500)
    b1 = src.read(1, window=win).astype(np.float32)
    b2 = src.read(2, window=win).astype(np.float32)
    b3 = src.read(3, window=win).astype(np.float32)
    b4 = src.read(4, window=win).astype(np.float32)
    b5 = src.read(5, window=win).astype(np.float32)

global_rows = 400 + np.arange(500, dtype=np.float32)[:, np.newaxis]
global_cols = 2400 + np.arange(500, dtype=np.float32)[np.newaxis, :]

foothill_col = 2850.0 - 0.06 * global_rows
mean_b4 = ndimage.uniform_filter(b4, size=11)
sq_b4 = ndimage.uniform_filter(b4**2, size=11)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

denom_ndvi = b4 + b3
raw_ndvi = (b4 - b3) / np.maximum(denom_ndvi, 1.0)
is_west = global_cols < foothill_col
is_mtn = is_west & (raw_ndvi > 0.20) & ((global_cols < 2600.0) | (cv_b4 > 0.08) | (b5 < 2300.0))

print(f"Mountain patch test:")
print(f"  Raw NDVI > 0.35 pixels: {np.sum(raw_ndvi > 0.35)}")
print(f"  Mountain veg suppressed: {np.sum(is_mtn)} ({np.mean(is_mtn)*100:.1f}%)")

# 2. Test ParcelSegmenter Voronoi subdivision
seg = ParcelSegmenter({'segmentation': {'max_parcel_area_m2': 800000.0}})
# Create a dummy large patch
dummy_large = np.ones((600, 300), dtype=np.uint8)
dummy_large[0:5, :] = 0; dummy_large[-5:, :] = 0; dummy_large[:, 0:5] = 0; dummy_large[:, -5:] = 0

subdivided = seg._subdivide_oversized_component(dummy_large)
u = np.unique(subdivided[subdivided > 0])
ratios = []
for p in u:
    coords = np.argwhere(subdivided == p)
    h = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    w = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    ratios.append(h / w)

print(f"\nCurrent ParcelSegmenter subdivision on dummy_large (600x300):")
print(f"  Num parcels: {len(u)}, Mean aspect ratio: {np.mean(ratios):.2f}, Max aspect ratio: {np.max(ratios):.2f}")
