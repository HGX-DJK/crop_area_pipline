import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

ndvi = np.where(b4 + b3 > 0, (b4 - b3)/(b4 + b3), 0.0)

# Mountain region: rows 500:2500, cols 200:2200
mtn = (slice(500, 2500), slice(200, 2200))

# Valley plain region: rows 500:2500, cols 2600:3500
farm = (slice(500, 2500), slice(2600, 3500))

# Let's inspect roughness with window size 11 (approx 330m)
mean_b4 = ndimage.uniform_filter(b4, size=11)
sq_b4 = ndimage.uniform_filter(b4**2, size=11)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

print("In Mountain zone (cols 200:2200):")
print(f"Total valid pixels: {np.sum(b4[mtn] > 0):,}")
print(f"NDVI > 0.30: {np.sum(ndvi[mtn] > 0.30):,}")
print(f"cv_b4 > 0.12: {np.sum(cv_b4[mtn] > 0.12):,}")
print(f"cv_b4 > 0.15: {np.sum(cv_b4[mtn] > 0.15):,}")

# What if we define mountain natural vegetation in is_mountain_zone:
# Any vegetation in the mountain zone that has CV_B4 > 0.12 or is dense tree cover:
# Dense tree: ndvi > 0.40 and b5 < 2400
# Mountain shrub: ndvi > 0.25 and cv_b4 > 0.14
suppress_mtn = (ndvi[mtn] > 0.30) & ((cv_b4[mtn] > 0.12) | (b5[mtn] < 2200.0) | (b5[mtn] < b4[mtn] * 1.05))
print(f"Mountain vegetation suppressed under new rule: {np.sum(suppress_mtn):,} ({np.mean(suppress_mtn):.1%})")

# Check if this rule would hurt valley plain (farm) if applied there (it won't because is_mountain_zone guards it!):
suppress_farm = (ndvi[farm] > 0.30) & ((cv_b4[farm] > 0.12) | (b5[farm] < 2200.0) | (b5[farm] < b4[farm] * 1.05))
print(f"In valley plain (without zone guard): {np.sum(suppress_farm):,} would be suppressed.")
print("With zone guard (cols < 2450): Valley plain is 100% immune from mountain suppression!")
