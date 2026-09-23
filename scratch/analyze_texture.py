import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

ndvi = (b4 - b3) / np.maximum(b4 + b3, 1.0)

# Local standard deviation (texture) of NIR (B4) and SWIR1 (B5)
# In rough mountain terrain, ridges and valleys cause huge local brightness variation.
# In flat agricultural fields, fields are uniform within 3x3 to 5x5 windows.
win = 5
mean_b4 = ndimage.uniform_filter(b4, size=win)
sq_b4 = ndimage.uniform_filter(b4**2, size=win)
std_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0))
# Relative texture (coefficient of variation CV = std / mean)
cv_b4 = std_b4 / np.maximum(mean_b4, 1.0)

# Mountain sample: rows 500:2000, cols 500:2000
mtn_cv = cv_b4[500:2000, 500:2000]
# Valley plain sample: rows 500:2000, cols 2800:3500
farm_cv = cv_b4[500:2000, 2800:3500]

print("Texture (Local Coefficient of Variation 5x5) Comparison:")
print(f"Mountain: mean={mtn_cv.mean():.3f}, 10%={np.percentile(mtn_cv, 10):.3f}, 50%={np.median(mtn_cv):.3f}, 90%={np.percentile(mtn_cv, 90):.3f}")
print(f"Valley Plain: mean={farm_cv.mean():.3f}, 10%={np.percentile(farm_cv, 10):.3f}, 50%={np.median(farm_cv):.3f}, 90%={np.percentile(farm_cv, 90):.3f}")

# Also check Sobel gradient magnitude (edges/ridges)
sx = ndimage.sobel(b4, axis=0)
sy = ndimage.sobel(b4, axis=1)
grad_mag = np.hypot(sx, sy) / np.maximum(b4, 1.0)
mtn_grad = grad_mag[500:2000, 500:2000]
farm_grad = grad_mag[500:2000, 2800:3500]
print("\nGradient Magnitude Comparison:")
print(f"Mountain: mean={mtn_grad.mean():.3f}, 50%={np.median(mtn_grad):.3f}")
print(f"Valley Plain: mean={farm_grad.mean():.3f}, 50%={np.median(farm_grad):.3f}")
