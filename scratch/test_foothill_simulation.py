import rasterio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b1 = src.read(1).astype(np.float32)
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

step = 5  # 5x downsampled (150m pixels) for high-fidelity test
s_b1 = b1[::step, ::step]
s_b2 = b2[::step, ::step]
s_b3 = b3[::step, ::step]
s_b4 = b4[::step, ::step]
s_b5 = b5[::step, ::step]

h, w = s_b4.shape
rows, cols = np.indices((h, w), dtype=np.float32)
# Scale back to full resolution coordinate space
full_rows = rows * step
full_cols = cols * step

ndvi = (s_b4 - s_b3) / np.maximum(s_b4 + s_b3, 1e-4)
mndwi = (s_b2 - s_b5) / np.maximum(s_b2 + s_b5, 1e-4)

# Multi-scale roughness
mean_b4 = ndimage.uniform_filter(s_b4, size=5)
sq_b4 = ndimage.uniform_filter(s_b4**2, size=5)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

# Physical masks
is_water = (mndwi > -0.08) | ((mndwi > -0.15) & (ndvi < 0.20)) | ((s_b4 < 600.0) & (s_b2 > s_b4))
b5_b3_ratio = s_b5 / np.maximum(s_b3, 1.0)
is_urban = (ndvi < 0.22) & (b5_b3_ratio < 1.75) & (s_b5 < 1950.0)

# Mountain boundary following the actual foothill line
foothill_col = 2550.0 + 0.05 * full_rows
is_mountain_zone = (full_cols < foothill_col) & (cv_b4 > 0.10)

is_mountain_veg = is_mountain_zone & (ndvi > 0.25) & ((cv_b4 > 0.12) | (s_b5 < 2250.0) | (s_b5 < s_b4 * 1.05))

ndvi_proc = ndvi.copy()
ndvi_proc[is_water] = -0.05
ndvi_proc[is_urban] = 0.15
ndvi_proc[is_mountain_veg] = 0.12

# Cropland prediction (NDVI >= 0.18)
is_cropland = (ndvi_proc >= 0.18)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(np.stack([s_b4/2500.0, s_b3/1500.0, s_b2/1500.0], axis=-1).clip(0, 1))
axes[0].set_title("NIR False Color")

axes[1].imshow(is_cropland, cmap='YlOrBr')
axes[1].set_title("Cropland Extraction Result (Smooth Foothill Guard)")

plt.tight_layout()
plt.savefig("scratch/foothill_simulation_result.png", dpi=150)
print("Saved scratch/foothill_simulation_result.png")
