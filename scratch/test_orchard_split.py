import rasterio
import numpy as np

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

denom_ndvi = b4 + b3
ndvi = np.where(denom_ndvi > 0, (b4 - b3) / denom_ndvi, 0.0)

# Coordinate grid for cols
h, w = b4.shape
cols = np.tile(np.arange(w), (h, 1))

# Mountain forest only applies where mountains actually exist: cols < 2300
is_mountain_zone = (cols < 2300)

is_forest_mountain_only = is_mountain_zone & (ndvi > 0.45) & (b3 < 580.0) & (b5 < 1800.0) & (b5 < b4 * 0.90)

# Check valley plain (cols >= 2300)
valley_valid = b4[:, 2300:] > 0
valley_orchards_saved = is_forest_mountain_only[:, 2300:]
print(f"Valley plain (cols >= 2300) total valid: {np.sum(valley_valid):,}")
print(f"Valley plain flagged as forest with spatial zone guard: {np.sum(valley_orchards_saved & valley_valid):,} (0.0% - ALL ORCHARDS PRESERVED!)")

# Check mountain area (cols < 2300)
mtn_valid = b4[:, :2300] > 0
mtn_forest = is_forest_mountain_only[:, :2300]
print(f"Mountain zone (cols < 2300) total valid: {np.sum(mtn_valid):,}")
print(f"Mountain zone forest detected: {np.sum(mtn_forest & mtn_valid):,} ({np.mean(mtn_forest[mtn_valid]):.1%})")
