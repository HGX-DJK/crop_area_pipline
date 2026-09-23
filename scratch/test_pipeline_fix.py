import rasterio
import numpy as np
import cv2
from scipy import ndimage
import time

print("Loading data for Sacramento Valley slice...")
with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b1 = src.read(1).astype(np.float32)
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)
    h, w = b1.shape

    ndvi = (b4 - b3) / np.maximum(b4 + b3, 1e-4)
    ndbi = (b5 - b4) / np.maximum(b5 + b4, 1e-4)
    b5_b3 = b5 / np.maximum(b3, 1.0)
    mndwi = (b2 - b5) / np.maximum(b2 + b5, 1e-4)

    # 1. Urban envelope detection
    impervious_core = (
        (b1 > 1000.0) |
        ((b1 > 480.0) & (b5_b3 < 1.85) & (ndvi < 0.40)) |
        ((ndbi > -0.02) & (b5_b3 < 1.75)) |
        ((b5_b3 < 1.55) & (ndvi < 0.35))
    )
    dens = ndimage.uniform_filter(impervious_core.astype(np.float32), size=21)
    urban_cand = dens >= 0.12
    urban_closed = ndimage.binary_closing(urban_cand, structure=np.ones((7, 7)))
    lbl_u, num_u = ndimage.label(urban_closed, structure=ndimage.generate_binary_structure(2, 2))
    counts_u = np.bincount(lbl_u.ravel())
    large_u = np.zeros_like(urban_closed, dtype=bool)
    for i in range(1, num_u + 1):
        if counts_u[i] >= 200:
            large_u[lbl_u == i] = True
    final_urban_mask = ndimage.binary_dilation(large_u, structure=np.ones((7, 7)))

    # 2. Foothill / Mountain Mask
    global_rows = np.arange(h, dtype=np.float32)[:, np.newaxis]
    global_cols = np.arange(w, dtype=np.float32)[np.newaxis, :]
    foothill_col = 2550.0 + 0.05 * global_rows
    mean_b4 = ndimage.uniform_filter(b4, size=11)
    sq_b4 = ndimage.uniform_filter(b4**2, size=11)
    cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)
    is_mountain_zone = (global_cols < foothill_col) & (cv_b4 > 0.10)
    is_mountain_veg = is_mountain_zone & (ndvi > 0.25) & ((cv_b4 > 0.12) | (b5 < 2250.0) | (b5 < b4 * 1.05))
    is_water = (mndwi > -0.08) | ((mndwi > -0.15) & (ndvi < 0.20)) | ((b4 < 600.0) & (b2 > b4))

    cropland_mask = (ndvi >= 0.18) & (~final_urban_mask) & (~is_mountain_veg) & (~is_water)

print("\n--- Testing Edge-Guided Cropland Parcel Segmentation ---")
# Boundary gradient: farm roads, canals, ditches, field edges
grad_x = cv2.Sobel(ndvi, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(ndvi, cv2.CV_32F, 0, 1, ksize=3)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
# High spectral contrast boundaries (farm tracks / different crops)
strong_edges = (grad_mag > 0.35) & (cropland_mask)

# Separate adjacent parcels along strong spectral contrast edges
cropland_split = cropland_mask.copy().astype(np.uint8)
cropland_split[strong_edges] = 0

# Morphological opening to disconnect narrow 1-pixel bridges
structure_cross = ndimage.generate_binary_structure(2, 1)
cleaned = ndimage.binary_opening(cropland_split, structure=structure_cross).astype(np.uint8)

# Connected components
lbl, num_comp = ndimage.label(cleaned, structure=structure_cross)
print(f"Number of initial components after edge separation: {num_comp}")

# Check sizes of components
counts = np.bincount(lbl.ravel())
areas_mu = counts[1:num_comp+1] * 900.0 / 666.67
sorted_areas = np.sort(areas_mu)[::-1]
print(f"Top 5 component areas before watershed (万亩): {[round(a/10000.0, 2) for a in sorted_areas[:5]]}")
