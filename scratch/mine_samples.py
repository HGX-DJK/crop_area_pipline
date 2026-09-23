import rasterio
import numpy as np
import pandas as pd
from scipy import ndimage

files = [
    'data/satellite_tifs/CSDC30_10SEH_20150103.tif',
    'data/satellite_tifs/CSDC30_10SEH_20150106.tif',
    'data/satellite_tifs/CSDC30_10SEH_20150109.tif'
]

# Read bands from first date
with rasterio.open(files[0]) as s:
    b1 = s.read(1).astype(np.float32)
    b2 = s.read(2).astype(np.float32)
    b3 = s.read(3).astype(np.float32)
    b4 = s.read(4).astype(np.float32)
    b5 = s.read(5).astype(np.float32)
    b6 = s.read(6).astype(np.float32)

valid = (b1 > 0) & (b2 > 0) & (b3 > 0) & (b4 > 0) & (b5 > 0) & (b6 > 0)

# Compute NDVI for all 3 dates
doy_ndvis = {}
doys = [3, 6, 9]
for d, f in zip(doys, files):
    with rasterio.open(f) as s:
        r_b3 = s.read(3).astype(np.float32)
        r_b4 = s.read(4).astype(np.float32)
        d_ndvi = np.where(r_b4 + r_b3 > 0, (r_b4 - r_b3) / (r_b4 + r_b3), 0.0)
        doy_ndvis[d] = d_ndvi

ndvi = doy_ndvis[3]
mndwi = np.where(b2 + b5 > 0, (b2 - b5) / (b2 + b5), 0.0)
b5_b3_ratio = b5 / np.maximum(b3, 1.0)

# Local roughness
mean_b4 = ndimage.uniform_filter(b4, size=11)
sq_b4 = ndimage.uniform_filter(b4**2, size=11)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

h, w = b4.shape
y_coords, x_coords = np.indices((h, w))

# Physical Masks matching RasterLoader:
is_water = (mndwi > -0.08) | ((mndwi > -0.15) & (ndvi < 0.20)) | ((b4 < 600.0) & (b2 > b4))
is_urban = (ndvi < 0.22) & (b5_b3_ratio < 1.75) & (b5 < 1950.0)
foothill_col = 2550.0 + 0.05 * y_coords
is_mountain_zone = (x_coords < foothill_col) & (cv_b4 > 0.10)
is_mountain_veg = is_mountain_zone & (ndvi > 0.25) & ((cv_b4 > 0.12) | (b5 < 2250.0) | (b5 < b4 * 1.05))

# Apply same physical suppression as raster_loader to sample feature space
for d in doys:
    doy_ndvis[d][is_water] = np.minimum(doy_ndvis[d][is_water], -0.05)
    doy_ndvis[d][is_urban] = np.minimum(doy_ndvis[d][is_urban], 0.15)
    doy_ndvis[d][is_mountain_veg] = np.minimum(doy_ndvis[d][is_mountain_veg], 0.12)

# Candidate pools:
# Class 0 (Non-Crop):
water_mask = valid & is_water
mountain_veg_mask = valid & is_mountain_veg
urban_mask = valid & is_urban
mountain_bare_mask = valid & is_mountain_zone & (ndvi < 0.20) & (cv_b4 > 0.15)

# Class 1 (Cropland in Sacramento Valley, cols >= 2600):
# 1. Active green crop (winter wheat, alfalfa)
crop_active_mask = valid & (x_coords >= 2600) & (cv_b4 < 0.12) & (ndvi > 0.45) & (b5 > 1950) & (b3 > 550) & (b5 > b4 * 0.88)
# 2. Commercial Orchards / Vineyards (flat terrain, tree canopy in valley plain)
crop_orchard_mask = valid & (x_coords >= 2600) & (cv_b4 < 0.14) & (ndvi > 0.45) & (b5 < 1850)
# 3. Fallow agricultural soil (ploughed fields prepared for spring)
crop_fallow_mask = valid & (x_coords >= 2600) & (cv_b4 < 0.08) & (ndvi >= 0.18) & (ndvi <= 0.35) & (b5 > 2000) & (b5_b3_ratio > 1.8)

print("Candidate pools:")
print(f"  Water: {np.sum(water_mask):,}")
print(f"  Mountain Veg (Suppressed): {np.sum(mountain_veg_mask):,}")
print(f"  Urban: {np.sum(urban_mask):,}")
print(f"  Crop Active: {np.sum(crop_active_mask):,}")
print(f"  Crop Orchard: {np.sum(crop_orchard_mask):,}")
print(f"  Crop Fallow: {np.sum(crop_fallow_mask):,}")

np.random.seed(42)
def sample_indices(mask, n):
    idx = np.where(mask.ravel())[0]
    if len(idx) > n:
        return np.random.choice(idx, size=n, replace=False)
    return idx

s_water = sample_indices(water_mask, 200)
s_mtn_veg = sample_indices(mountain_veg_mask, 500)
s_urban = sample_indices(urban_mask, 200)
s_mtn_bare = sample_indices(mountain_bare_mask, 100)

s_crop_active = sample_indices(crop_active_mask, 350)
s_crop_orchard = sample_indices(crop_orchard_mask, 350)
s_crop_fallow = sample_indices(crop_fallow_mask, 300)

c0_indices = np.concatenate([s_water, s_mtn_veg, s_urban, s_mtn_bare])
c1_indices = np.concatenate([s_crop_active, s_crop_orchard, s_crop_fallow])

print(f"Total sampled Class 0 (Non-Cropland): {len(c0_indices)}")
print(f"Total sampled Class 1 (Cropland): {len(c1_indices)}")

records = []
pid = 1
for idx in c0_indices:
    r, c = divmod(idx, w)
    rec = {"point_id": f"P{pid:04d}", "label": 0, "crop_name": "非耕地"}
    for d in doys:
        rec[f"doy_{d}"] = round(float(doy_ndvis[d][r, c]), 4)
    records.append(rec)
    pid += 1

for idx in c1_indices:
    r, c = divmod(idx, w)
    rec = {"point_id": f"P{pid:04d}", "label": 1, "crop_name": "耕地"}
    for d in doys:
        rec[f"doy_{d}"] = round(float(doy_ndvis[d][r, c]), 4)
    records.append(rec)
    pid += 1

df_samples = pd.DataFrame(records)
df_samples.to_csv("data/sample_training_points.csv", index=False, encoding="utf-8")
print(f"\n[DONE] Successfully saved {len(df_samples)} empirical training points to data/sample_training_points.csv")
print(df_samples.groupby('label').mean(numeric_only=True))
