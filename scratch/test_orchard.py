import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)
    b6 = src.read(6).astype(np.float32)

valid = (b1 > 0) & (b4 > 0) if 'b1' in dir() else (b4 > 0) & (b3 > 0)
denom_ndvi = b4 + b3
ndvi = np.where(denom_ndvi > 0, (b4 - b3) / denom_ndvi, 0.0)

# Local roughness
mean_b4 = ndimage.uniform_filter(b4, size=5)
sq_b4 = ndimage.uniform_filter(b4**2, size=5)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

# Mountain sample: rows 500:2000, cols 500:2000
# Valley plain sample: rows 500:2000, cols 2800:3500
mtn = (slice(500, 2000), slice(500, 2000))
farm = (slice(500, 2000), slice(2800, 3500))

# 1. Old forest rule (without CV check):
is_forest_old = (ndvi > 0.45) & (b3 < 580.0) & (b5 < 1800.0) & (b5 < b4 * 0.90)

# 2. New forest rule (only in rugged terrain CV > 0.20 or far west):
# In flat valley terrain (CV < 0.18), green tree canopies are commercial orchards, NOT wild forest!
is_forest_new = (ndvi > 0.45) & (b3 < 580.0) & (b5 < 1800.0) & (b5 < b4 * 0.90) & (cv_b4 > 0.20)

print("Mountain area forest detection:")
print(f"  Old rule: {np.sum(is_forest_old[mtn]):,} ({np.mean(is_forest_old[mtn]):.1%})")
print(f"  New rule (with CV>0.20): {np.sum(is_forest_new[mtn]):,} ({np.mean(is_forest_new[mtn]):.1%})")

print("\nValley plain area (Orchards misclassified as forest):")
print(f"  Old rule (orchards killed!): {np.sum(is_forest_old[farm]):,} ({np.mean(is_forest_old[farm]):.1%})")
print(f"  New rule (orchards saved!): {np.sum(is_forest_new[farm]):,} ({np.mean(is_forest_new[farm]):.1%})")

# Look at fallow farmland in valley plain:
# Pixels in valley plain with NDVI between 0.18 and 0.28:
farm_fallow = (ndvi[farm] >= 0.18) & (ndvi[farm] < 0.28) & (b5[farm] > 2000)
print(f"\nValley plain fallow cropland pixels (NDVI 0.18-0.28): {np.sum(farm_fallow):,} ({np.mean(farm_fallow):.1%})")
