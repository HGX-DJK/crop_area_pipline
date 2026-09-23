import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

h, w = b4.shape

# Compute multi-scale spatial variance (texture roughness):
# Mountains have high texture roughness because of ravines, ridges, and slopes.
# Alluvial plain has low texture roughness because the land is flat.
mean_b4 = ndimage.uniform_filter(b4, size=15)
sq_b4 = ndimage.uniform_filter(b4**2, size=15)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

# For each row, find the easternmost column where roughness > 0.25 (i.e. the foot of the mountain)
mountain_edge_cols = []
for r in range(0, h, 100):
    # Scan from west (col 1000) to east (col 3200)
    rough = cv_b4[r, 1000:3200]
    # Smooth roughness along column
    rough_s = ndimage.uniform_filter1d(rough, size=30)
    # Find points where rough_s is high (> 0.22)
    high_idx = np.where(rough_s > 0.22)[0]
    if len(high_idx) > 0:
        edge = 1000 + high_idx[-1]
    else:
        edge = 2000
    mountain_edge_cols.append((r, edge))
    print(f"Row {r:4d}: Easternmost mountain foot at Col {edge:4d}")
