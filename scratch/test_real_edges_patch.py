import rasterio
import numpy as np
import cv2
from scipy import ndimage

# Load satellite bands around the user patch: r=2408, c=3029
with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    r, c = 2408, 3029
    win = rasterio.windows.Window(col_off=c-100, row_off=r-100, width=200, height=200)
    b3 = src.read(3, window=win).astype(np.float32) # Red
    b4 = src.read(4, window=win).astype(np.float32) # NIR
    b5 = src.read(5, window=win).astype(np.float32) # SWIR1

# Classification mask
with rasterio.open('output/crop_classification_map.tif') as src_m:
    win = rasterio.windows.Window(col_off=c-100, row_off=r-100, width=200, height=200)
    crop_mask = src_m.read(1, window=win)

cropland = (crop_mask == 1).astype(np.uint8)

# 1. Real Edge Detection in Farmland:
# In real satellite farmland, boundaries are visible as sharp gradients in NIR/Red/SWIR
# Multi-spectral edge magnitude:
grad_b4 = cv2.morphologyEx(b4, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
grad_b3 = cv2.morphologyEx(b3, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
grad_b5 = cv2.morphologyEx(b5, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

denom = b4 + b3
ndvi = np.zeros_like(b3)
v = denom > 0
ndvi[v] = (b4[v] - b3[v]) / denom[v]
grad_ndvi = cv2.morphologyEx(ndvi, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

# Composite edge magnitude
edge_norm = (grad_b4 / np.maximum(b4, 1.0) * 0.4 + grad_ndvi * 0.6)
# Roads / boundaries are strong edges
thresh = np.percentile(edge_norm[cropland > 0], 82)
real_edges = (edge_norm > thresh) & (cropland > 0)
# Thin edges to skeleton or 1-pixel width
real_edges_closed = ndimage.binary_dilation(real_edges, structure=np.ones((2, 2)))

# 2. Carve cropland with real edges!
cropland_carved = cropland.copy()
cropland_carved[real_edges_closed] = 0

# Opening with cross
cropland_clean = ndimage.binary_opening(cropland_carved, structure=ndimage.generate_binary_structure(2, 1))

# Connected components
lbl, num = ndimage.label(cropland_clean, structure=ndimage.generate_binary_structure(2, 1))
print(f"Number of natural components separated by real edges: {num}")
counts = np.bincount(lbl.ravel())
print(f"Top 10 component sizes (acres/mu): {np.sort(counts[1:])[-10:][::-1] * 900.0 / 666.67}")
