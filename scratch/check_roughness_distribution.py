import rasterio
import numpy as np
from scipy import ndimage
from pyproj import Transformer

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b1 = src.read(1).astype(np.float32)
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

h, w = b4.shape
ndvi = (b4 - b3) / np.maximum(b4 + b3, 1.0)
lswi = (b4 - b5) / np.maximum(b4 + b5, 1.0)

# Multi-scale texture roughness (topographic ruggedness):
# Mountains have severe shadow/illumination variation across slopes in Band 4 and Band 5
mean_b4 = ndimage.uniform_filter(b4, size=15)
sq_b4 = ndimage.uniform_filter(b4**2, size=15)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

# Check CV_B4 in the true mountains (e.g. Berryessa region: row 500, col 1500; row 700, col 2600)
# vs true agricultural plain (row 500, col 3200; row 1200, col 3200; row 2000, col 3200)
print("True Mountain samples (cv_b4, ndvi, b5):")
for r, c in [(300, 1500), (500, 2000), (700, 2600), (700, 2669), (500, 2650), (1200, 2600)]:
    print(f"  (r={r}, c={c}): cv_b4={cv_b4[r,c]:.3f}, ndvi={ndvi[r,c]:.3f}, b5={b5[r,c]:.1f}, b4={b4[r,c]:.1f}")

print("\nTrue Agricultural Plain samples:")
for r, c in [(500, 3100), (500, 3300), (1000, 3100), (1200, 3300), (1500, 3100), (2000, 3100)]:
    print(f"  (r={r}, c={c}): cv_b4={cv_b4[r,c]:.3f}, ndvi={ndvi[r,c]:.3f}, b5={b5[r,c]:.1f}, b4={b4[r,c]:.1f}")
