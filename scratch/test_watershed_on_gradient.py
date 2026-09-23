import rasterio
import numpy as np
import cv2
from scipy import ndimage

# Load satellite bands around user patch: r=2408, c=3029
with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    r, c = 2408, 3029
    win = rasterio.windows.Window(col_off=c-100, row_off=r-100, width=200, height=200)
    b3 = src.read(3, window=win).astype(np.float32) # Red
    b4 = src.read(4, window=win).astype(np.float32) # NIR
    b5 = src.read(5, window=win).astype(np.float32) # SWIR1

with rasterio.open('output/crop_classification_map.tif') as src_m:
    win = rasterio.windows.Window(col_off=c-100, row_off=r-100, width=200, height=200)
    cropland = (src_m.read(1, window=win) == 1).astype(np.uint8)

# 1. Compute multi-spectral gradient landscape (high at roads/ditches, low in field interiors)
gx = cv2.Sobel(b4, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(b4, cv2.CV_32F, 0, 1, ksize=3)
mag_b4 = np.sqrt(gx**2 + gy**2)

denom = b4 + b3
ndvi = np.zeros_like(b3)
v = denom > 0
ndvi[v] = (b4[v] - b3[v]) / denom[v]
gx_n = cv2.Sobel(ndvi, cv2.CV_32F, 1, 0, ksize=3)
gy_n = cv2.Sobel(ndvi, cv2.CV_32F, 0, 1, ksize=3)
mag_ndvi = np.sqrt(gx_n**2 + gy_n**2)

# Gradient magnitude image (0..255 uint8)
grad_norm = (mag_b4 / np.maximum(np.percentile(mag_b4, 98), 1.0) * 128.0 + 
             mag_ndvi / np.maximum(np.percentile(mag_ndvi, 98), 1e-4) * 127.0)
grad_uint8 = np.clip(grad_norm, 0, 255).astype(np.uint8)

# 2. Extract natural seeds (regional minima of gradient or peaks of distance transform to edges)
padded = np.pad(cropland, 1, mode="constant", constant_values=0)
dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]

# Peaks of distance transform with minimum distance ~ 180m (6 pixels)
win = 7
dilated = ndimage.maximum_filter(dist, size=win)
peaks = (dist == dilated) & (dist >= 2.0) & (cropland > 0)

# Centroids of peaks
num_raw, peak_labels = cv2.connectedComponents(peaks.astype(np.uint8))
peak_centroids = np.zeros_like(cropland, dtype=bool)
for lab in range(1, num_raw):
    coords = np.argwhere(peak_labels == lab)
    c_r, c_c = int(np.mean(coords[:, 0])), int(np.mean(coords[:, 1]))
    if cropland[c_r, c_c] > 0:
        peak_centroids[c_r, c_c] = True

num_seeds, markers = cv2.connectedComponents(peak_centroids.astype(np.uint8))
print(f"Number of natural field seeds in 200x200 patch: {num_seeds - 1}")

# 3. Watershed on GRADIENT IMAGE (not distance transform, not grid!)
landscape_bgr = cv2.cvtColor(grad_uint8, cv2.COLOR_GRAY2BGR)
cv2.watershed(landscape_bgr, markers)
markers[markers <= 0] = 0
markers[cropland == 0] = 0

# Check parcels
u = np.unique(markers[markers > 0])
print(f"Extracted natural parcels: {len(u)}")
for p in u[:10]:
    coords = np.argwhere(markers == p)
    h = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    w = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    area_mu = len(coords) * 900.0 / 666.67
    print(f"  Parcel {p}: area={area_mu:.1f} 亩, bbox=({h}x{w} px = {h*30}mx{w*30}m), aspect={max(h,w)/min(h,w):.2f}")
