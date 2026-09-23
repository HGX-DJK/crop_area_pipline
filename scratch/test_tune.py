import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

denom_ndvi = b4 + b3
ndvi = np.where(denom_ndvi > 0, (b4 - b3) / denom_ndvi, 0.0)

mean_b4 = ndimage.uniform_filter(b4, size=5)
sq_b4 = ndimage.uniform_filter(b4**2, size=5)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

# Mountain sample: rows 500:2000, cols 500:2000
# Valley plain sample: rows 500:2000, cols 2800:3500
mtn = (slice(500, 2000), slice(500, 2000))
farm = (slice(500, 2000), slice(2800, 3500))

# Test rugged mountain vegetation condition:
# Mountain grass/oak/brush: high NDVI (>0.35), high roughness (cv_b4 > 0.28), and lower SWIR (b5 < 2200)
is_rugged = (ndvi > 0.32) & (cv_b4 > 0.26) & (b5 < 2200.0)

print(f"Mountain region rugged detected: {np.sum(is_rugged[mtn]):,} ({np.mean(is_rugged[mtn]):.1%})")
print(f"Valley plain FALSE POSITIVE: {np.sum(is_rugged[farm]):,} ({np.mean(is_rugged[farm]):.2%})")
