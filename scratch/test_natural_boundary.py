import rasterio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b4 = src.read(4).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b5 = src.read(5).astype(np.float32)
    b2 = src.read(2).astype(np.float32)

step = 10
s_b4 = b4[::step, ::step]
s_b3 = b3[::step, ::step]
s_b5 = b5[::step, ::step]
s_b2 = b2[::step, ::step]
h, w = s_b4.shape
cols = np.tile(np.arange(w, dtype=np.int32), (h, 1))

ndvi = (s_b4 - s_b3) / np.maximum(s_b4 + s_b3, 1e-4)
mndwi = (s_b2 - s_b5) / np.maximum(s_b2 + s_b5, 1e-4)

# Multi-scale roughness:
# Size 7 on 10x downsampled corresponds to ~2km window
mean_b4 = ndimage.uniform_filter(s_b4, size=7)
sq_b4 = ndimage.uniform_filter(s_b4**2, size=7)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

# Local roughness:
mean_local = ndimage.uniform_filter(s_b4, size=3)
sq_local = ndimage.uniform_filter(s_b4**2, size=3)
cv_local = np.sqrt(np.maximum(sq_local - mean_local**2, 0.0)) / np.maximum(mean_local, 1.0)

# Water
is_water = (mndwi > -0.08) | ((mndwi > -0.15) & (ndvi < 0.20)) | ((s_b4 < 600.0) & (s_b2 > s_b4))

# Mountain forest:
# 1. High elevation/rugged terrain (cv_b4 > 0.18) with vegetation
# 2. Or steep shaded slope (cv_local > 0.22) with vegetation
# 3. Or deep mountain conifer (ndvi > 0.60 & b3 < 400 & b5 < 1500)
is_mountain_forest = (
    ((cv_b4 > 0.18) & (ndvi > 0.32) & (s_b5 < 2100.0)) |
    ((cv_local > 0.22) & (ndvi > 0.28) & (s_b5 < 2200.0)) |
    ((ndvi > 0.60) & (s_b3 < 420.0) & (s_b5 < 1500.0) & (cols < 260))
)

# Potential cropland: ndvi > 0.18 and not water and not mountain forest
pot_crop = (ndvi >= 0.18) & (~is_water) & (~is_mountain_forest)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(is_mountain_forest, cmap='gray_r')
axes[0].set_title("Identified Mountain Forest / Wild Vegetation")

axes[1].imshow(pot_crop, cmap='YlOrBr')
axes[1].set_title("Remaining Cropland Candidates")

plt.tight_layout()
plt.savefig("scratch/natural_boundary_test.png", dpi=150)
print("Saved scratch/natural_boundary_test.png")
