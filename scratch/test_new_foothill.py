import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b1 = src.read(1).astype(np.float32)
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

h, w = b4.shape
global_rows, global_cols = np.indices((h, w), dtype=np.float32)

mean_b4 = ndimage.uniform_filter(b4, size=11)
sq_b4 = ndimage.uniform_filter(b4**2, size=11)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

denom_ndvi = b4 + b3
ndvi = np.zeros_like(b3)
valid_ndvi = denom_ndvi > 0
ndvi[valid_ndvi] = (b4[valid_ndvi] - b3[valid_ndvi]) / denom_ndvi[valid_ndvi]

# New Foothill boundary:
# At row 0: col 2850
# At row 3660: col 2630
foothill_col = 2850.0 - 0.06 * global_rows

# West of foothill line:
is_west_zone = global_cols < foothill_col
# Mountain vegetation: either deep in Coast Range (col < 2600) with vegetation,
# or in the foothill zone (col < foothill_col) with topographic roughness cv_b4 > 0.08
is_mountain_veg = is_west_zone & (ndvi > 0.20) & ((global_cols < 2600) | (cv_b4 > 0.08))

print("Total mountain veg pixels identified:", np.sum(is_mountain_veg))

# Check the troublesome patch at (-122.08, 38.65) -> Row=723, Col=2669
patch_mtn = is_mountain_veg[723-50:723+50, 2669-50:2669+50]
print(f"Mountain veg in patch (100x100 around Berryessa/Coast Range): {np.sum(patch_mtn)} / 10000 ({np.mean(patch_mtn)*100:.1f}%)")

# Check eastern plain farmland around Winters/Davis: Row=1000, Col=3200 (100x100)
patch_plain = is_mountain_veg[1000-50:1000+50, 3200-50:3200+50]
print(f"Mountain veg in eastern plain (100x100 around Davis): {np.sum(patch_plain)} / 10000 ({np.mean(patch_plain)*100:.1f}%)")

# Check Vacaville / Fairfield plain: Row=2000, Col=3000 (100x100)
patch_solano = is_mountain_veg[2000-50:2000+50, 3000-50:3000+50]
print(f"Mountain veg in Solano plain (100x100): {np.sum(patch_solano)} / 10000 ({np.mean(patch_solano)*100:.1f}%)")
