import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b1 = src.read(1).astype(np.float32)
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

sub_m = mask[1000:2000, 500:1500]
sub_b1 = b1[1000:2000, 500:1500]
sub_b2 = b2[1000:2000, 500:1500]
sub_b3 = b3[1000:2000, 500:1500]
sub_b4 = b4[1000:2000, 500:1500]
sub_b5 = b5[1000:2000, 500:1500]

ndvi = (sub_b4 - sub_b3) / np.maximum(sub_b4 + sub_b3, 1e-4)

false_mtn = (sub_m == 1)
print(f"Total false mountain pixels in [1000:2000, 500:1500]: {np.sum(false_mtn):,}")
print(f"NDVI of false mountain: mean={ndvi[false_mtn].mean():.3f}, min={ndvi[false_mtn].min():.3f}, max={ndvi[false_mtn].max():.3f}")
print(f"B3: mean={sub_b3[false_mtn].mean():.1f}")
print(f"B4: mean={sub_b4[false_mtn].mean():.1f}")
print(f"B5: mean={sub_b5[false_mtn].mean():.1f}")
print(f"B5/B4 ratio: mean={(sub_b5[false_mtn]/sub_b4[false_mtn]).mean():.3f}")

# Check why they weren't suppressed by is_natural_forest in raster_loader:
# Look at raster_loader logic:
# is_forest = is_mountain_zone & (ndvi > 0.45) & (b3 < 580.0) & (b5 < 1800.0) & (b5 < b4 * 0.90)
# is_mountain_shrub = is_mountain_zone & (ndvi > 0.30) & (b3 < 500.0) & (b5 < 1650.0)
# cv_b4 = ...
# is_rugged_mountain = is_mountain_zone & (ndvi > 0.30) & (cv_b4 > 0.28) & (b5 < 2150.0) & (b3 < 620.0)

mean_b4 = ndimage.uniform_filter(sub_b4, size=5)
sq_b4 = ndimage.uniform_filter(sub_b4**2, size=5)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

print(f"CV_B4 of false mountain: mean={cv_b4[false_mtn].mean():.3f}, min={cv_b4[false_mtn].min():.3f}, max={cv_b4[false_mtn].max():.3f}")
print(f"How many have cv_b4 > 0.15: {np.mean(cv_b4[false_mtn] > 0.15):.1%}")
print(f"How many have cv_b4 > 0.20: {np.mean(cv_b4[false_mtn] > 0.20):.1%}")
print(f"How many have cv_b4 > 0.28: {np.mean(cv_b4[false_mtn] > 0.28):.1%}")
print(f"NDVI > 0.45: {np.mean(ndvi[false_mtn] > 0.45):.1%}")
print(f"NDVI between 0.18 and 0.45: {np.mean((ndvi[false_mtn] >= 0.18) & (ndvi[false_mtn] <= 0.45)):.1%}")
