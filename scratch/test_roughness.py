import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

valid = (b4 > 0) & (b3 > 0)
ndvi = np.where(b4 + b3 > 0, (b4 - b3) / (b4 + b3), 0.0)

# Local CV (5x5)
mean_b4 = ndimage.uniform_filter(b4, size=5)
sq_b4 = ndimage.uniform_filter(b4**2, size=5)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

# Currently, we have 2,191,405 crop pixels in the West (cols < 2000)
# and 2,586,281 in the East (cols >= 2000)
with rasterio.open('output/crop_classification_map.tif') as src:
    current_crop = (src.read(1) == 1)

west_crop = current_crop & (b4[:, :2000].shape[1] > 0) # mask for west
west_crop_mask = np.zeros_like(current_crop)
west_crop_mask[:, :2000] = current_crop[:, :2000]

east_crop_mask = np.zeros_like(current_crop)
east_crop_mask[:, 2000:] = current_crop[:, 2000:]

print(f"Current West crop pixels: {np.sum(west_crop_mask):,}")
print(f"Current East crop pixels: {np.sum(east_crop_mask):,}")

# What happens if we apply roughness filter: cv_b4 > 0.26 in high-NDVI areas?
is_rugged = (cv_b4 > 0.26)
print(f"\nWest crop pixels removed by roughness filter (cv_b4 > 0.26): {np.sum(west_crop_mask & is_rugged):,} ({np.mean(is_rugged[west_crop_mask]):.1%})")
print(f"East crop pixels removed by roughness filter (should be small!): {np.sum(east_crop_mask & is_rugged):,} ({np.mean(is_rugged[east_crop_mask]):.1%})")
