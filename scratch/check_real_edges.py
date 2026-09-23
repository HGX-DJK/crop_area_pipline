import rasterio
import numpy as np
import cv2
from scipy import ndimage

# Load satellite bands around Dixon / Davis farmland (e.g. rows 1500..2000, cols 2900..3400)
with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    win = rasterio.windows.Window(col_off=2900, row_off=1500, width=500, height=500)
    b1 = src.read(1, window=win).astype(np.float32)
    b2 = src.read(2, window=win).astype(np.float32)
    b3 = src.read(3, window=win).astype(np.float32)
    b4 = src.read(4, window=win).astype(np.float32)
    b5 = src.read(5, window=win).astype(np.float32)

# NDVI
denom = b4 + b3
ndvi = np.zeros_like(b3)
v = denom > 0
ndvi[v] = (b4[v] - b3[v]) / denom[v]

# Multi-spectral edge magnitude (Sobel / Scharr on NIR, Red, SWIR1, NDVI)
grad_b4_x = cv2.Sobel(b4, cv2.CV_32F, 1, 0, ksize=3)
grad_b4_y = cv2.Sobel(b4, cv2.CV_32F, 0, 1, ksize=3)
mag_b4 = np.sqrt(grad_b4_x**2 + grad_b4_y**2)

grad_ndvi_x = cv2.Sobel(ndvi, cv2.CV_32F, 1, 0, ksize=3)
grad_ndvi_y = cv2.Sobel(ndvi, cv2.CV_32F, 0, 1, ksize=3)
mag_ndvi = np.sqrt(grad_ndvi_x**2 + grad_ndvi_y**2)

print("Edge statistics in plain farmland:")
print(f"  B4 grad mean: {np.mean(mag_b4):.1f}, 75th pct: {np.percentile(mag_b4, 75):.1f}, 90th pct: {np.percentile(mag_b4, 90):.1f}")
print(f"  NDVI grad mean: {np.mean(mag_ndvi):.3f}, 75th pct: {np.percentile(mag_ndvi, 75):.3f}, 90th pct: {np.percentile(mag_ndvi, 90):.3f}")

# Roads and field boundaries typically have strong gradient or lower NDVI / different brightness
# If we detect field boundaries (edges) using Canny / Sobel thresholding:
edge_thresh = np.percentile(mag_b4, 85)
field_edges = (mag_b4 > edge_thresh) | (mag_ndvi > 0.15)
print(f"Field edge pixels: {np.sum(field_edges)} / {field_edges.size} ({np.mean(field_edges)*100:.1f}%)")
