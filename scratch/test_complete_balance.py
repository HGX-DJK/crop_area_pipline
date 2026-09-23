import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b1 = src.read(1).astype(np.float32)
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)
    b6 = src.read(6).astype(np.float32)

h, w = b4.shape
cols = np.tile(np.arange(w, dtype=np.int32), (h, 1))

denom_ndvi = np.maximum(b4 + b3, 1e-4)
ndvi = (b4 - b3) / denom_ndvi

# 1. Water
mndwi = (b2 - b5) / np.maximum(b2 + b5, 1e-4)
is_water = (mndwi > -0.08) | ((mndwi > -0.15) & (ndvi < 0.20)) | ((b4 < 600.0) & (b2 > b4))

# 2. Urban / Impervious surface
ndbi = (b5 - b4) / np.maximum(b5 + b4, 1e-4)
is_urban_bare = (ndbi > 0.08) | (b4 < 650.0)

# 3. Mountain Zone & Suppression
# All pixels west of the Sacramento Valley plain (cols < 2600)
is_mountain_zone = (cols < 2600)

mean_b4 = ndimage.uniform_filter(b4, size=11)
sq_b4 = ndimage.uniform_filter(b4**2, size=11)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

# In mountain zone, wild forest and mountain chaparral have:
# - Rough terrain: cv_b4 > 0.12
# - Or natural tree/shrub canopy: ndvi > 0.30 & (b5 < 2250 | b5 < b4 * 1.05)
is_mountain_veg = is_mountain_zone & (ndvi > 0.28) & ((cv_b4 > 0.11) | (b5 < 2250.0) | (b5 < b4 * 1.05))

# Processed NDVI:
ndvi_proc = ndvi.copy()
ndvi_proc[is_water] = np.minimum(ndvi_proc[is_water], -0.05)
ndvi_proc[is_urban_bare & (ndvi < 0.25)] = np.minimum(ndvi_proc[is_urban_bare & (ndvi < 0.25)], 0.15)
ndvi_proc[is_mountain_veg] = np.minimum(ndvi_proc[is_mountain_veg], 0.12)

# Now check:
# Region 1: Coast Range mountains [1000:2000, 500:1500]
mtn_test = (slice(1000, 2000), slice(500, 1500))
# Region 2: Sacramento Valley plain [1000:2000, 2800:3500]
farm_test = (slice(1000, 2000), slice(2800, 3500))

print("=== Coast Range Mountains [1000:2000, 500:1500] ===")
print(f"Total pixels: {ndvi[mtn_test].size:,}")
print(f"Original NDVI >= 0.20: {np.sum(ndvi[mtn_test] >= 0.20):,} ({np.mean(ndvi[mtn_test] >= 0.20):.1%})")
print(f"Processed NDVI >= 0.20 (Remaining potential crop): {np.sum(ndvi_proc[mtn_test] >= 0.20):,} ({np.mean(ndvi_proc[mtn_test] >= 0.20):.2%})")
print(f"Suppressed Mountain Veg: {np.sum(is_mountain_veg[mtn_test]):,} ({np.mean(is_mountain_veg[mtn_test]):.1%})")

print("\n=== Sacramento Valley Plain [1000:2000, 2800:3500] ===")
print(f"Total pixels: {ndvi[farm_test].size:,}")
print(f"Original NDVI >= 0.20: {np.sum(ndvi[farm_test] >= 0.20):,} ({np.mean(ndvi[farm_test] >= 0.20):.1%})")
print(f"Processed NDVI >= 0.20: {np.sum(ndvi_proc[farm_test] >= 0.20):,} ({np.mean(ndvi_proc[farm_test] >= 0.20):.1%})")

# Check specifically orchards and fallow fields in valley:
sub_farm_ndvi = ndvi[farm_test]
sub_farm_b5 = b5[farm_test]
orchards = (sub_farm_ndvi > 0.45) & (sub_farm_b5 < 1800)
fallow = (sub_farm_ndvi >= 0.18) & (sub_farm_ndvi <= 0.35) & (sub_farm_b5 > 2000)

orchards_kept = ndvi_proc[farm_test][orchards] >= 0.40
fallow_kept = ndvi_proc[farm_test][fallow] >= 0.18

print(f"Valley Orchards: {np.sum(orchards):,} total, {np.sum(orchards_kept):,} kept ({np.mean(orchards_kept):.1%})")
print(f"Valley Fallow Soil: {np.sum(fallow):,} total, {np.sum(fallow_kept):,} kept ({np.mean(fallow_kept):.1%})")
