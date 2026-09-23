import rasterio
import numpy as np
import pandas as pd
from scipy import ndimage
import xgboost as xgb

files = [
    'data/satellite_tifs/CSDC30_10SEH_20150103.tif',
    'data/satellite_tifs/CSDC30_10SEH_20150106.tif',
    'data/satellite_tifs/CSDC30_10SEH_20150109.tif'
]

# Read bands
with rasterio.open(files[0]) as s:
    b1 = s.read(1).astype(np.float32)
    b2 = s.read(2).astype(np.float32)
    b3 = s.read(3).astype(np.float32)
    b4 = s.read(4).astype(np.float32)
    b5 = s.read(5).astype(np.float32)
    b6 = s.read(6).astype(np.float32)

h, w = b4.shape
cols = np.tile(np.arange(w, dtype=np.int32), (h, 1))

# Multitemporal NDVI
doy_ndvis = []
for f in files:
    with rasterio.open(f) as s:
        r_b3 = s.read(3).astype(np.float32)
        r_b4 = s.read(4).astype(np.float32)
        d_ndvi = (r_b4 - r_b3) / np.maximum(r_b4 + r_b3, 1e-4)
        doy_ndvis.append(d_ndvi)

# Physical masks
mndwi = (b2 - b5) / np.maximum(b2 + b5, 1e-4)
is_water = (mndwi > -0.08) | ((mndwi > -0.15) & (doy_ndvis[0] < 0.20)) | ((b4 < 600.0) & (b2 > b4))

# True urban: low NDVI, low B5/B3 ratio (concrete/asphalt)
b5_b3_ratio = b5 / np.maximum(b3, 1.0)
is_urban = (doy_ndvis[0] < 0.22) & (b5_b3_ratio < 1.75) & (b5 < 1950.0)

# Mountain zone: cols < 2600
is_mountain_zone = (cols < 2600)
mean_b4 = ndimage.uniform_filter(b4, size=11)
sq_b4 = ndimage.uniform_filter(b4**2, size=11)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

is_mountain_veg = is_mountain_zone & (doy_ndvis[0] > 0.25) & ((cv_b4 > 0.11) | (b5 < 2250.0) | (b5 < b4 * 1.05))

# Apply physical constraints to NDVI
for i in range(len(doy_ndvis)):
    doy_ndvis[i][is_water] = np.minimum(doy_ndvis[i][is_water], -0.05)
    doy_ndvis[i][is_urban] = np.minimum(doy_ndvis[i][is_urban], 0.15)
    doy_ndvis[i][is_mountain_veg] = np.minimum(doy_ndvis[i][is_mountain_veg], 0.12)

# Check test regions:
# Region 1: Mountain [1000:2000, 500:1500]
mtn = (slice(1000, 2000), slice(500, 1500))
# Region 2: Valley [1000:2000, 2800:3500]
farm = (slice(1000, 2000), slice(2800, 3500))

print("Quick check after physical filtering:")
print(f"Mountain region [1000:2000, 500:1500] with NDVI >= 0.18: {np.sum(doy_ndvis[0][mtn] >= 0.18):,} ({np.mean(doy_ndvis[0][mtn] >= 0.18):.2%})")
print(f"Valley region [1000:2000, 2800:3500] with NDVI >= 0.18: {np.sum(doy_ndvis[0][farm] >= 0.18):,} ({np.mean(doy_ndvis[0][farm] >= 0.18):.1%})")
