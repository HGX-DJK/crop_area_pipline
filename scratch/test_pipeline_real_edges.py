import rasterio
import numpy as np
import cv2
from scipy import ndimage
import json
from shapely.geometry import shape

print("1. Loading satellite data and classification mask...")
with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)

with rasterio.open('output/crop_classification_map.tif') as src_m:
    cropland = (src_m.read(1) == 1).astype(np.uint8)

# 2. Compute multi-spectral edge map (Band 4 NIR + NDVI gradient)
denom = b4 + b3
ndvi = np.zeros_like(b3)
v = denom > 0
ndvi[v] = (b4[v] - b3[v]) / denom[v]

gx_b4 = cv2.Sobel(b4, cv2.CV_32F, 1, 0, ksize=3)
gy_b4 = cv2.Sobel(b4, cv2.CV_32F, 0, 1, ksize=3)
mag_b4 = np.sqrt(gx_b4**2 + gy_b4**2)

gx_ndvi = cv2.Sobel(ndvi, cv2.CV_32F, 1, 0, ksize=3)
gy_ndvi = cv2.Sobel(ndvi, cv2.CV_32F, 0, 1, ksize=3)
mag_ndvi = np.sqrt(gx_ndvi**2 + gy_ndvi**2)

mag_b4_norm = mag_b4 / np.maximum(b4, 1.0)
combined_edge = mag_b4_norm * 0.4 + mag_ndvi * 0.6

# Detect real physical boundaries (roads, canals, field edges) inside cropland
crop_edges = combined_edge[cropland > 0]
thresh = float(np.percentile(crop_edges, 80))
is_edge = (combined_edge > thresh) & (cropland > 0)

struct_cross = ndimage.generate_binary_structure(2, 1)
is_edge_dilated = ndimage.binary_dilation(is_edge, structure=struct_cross)

# 3. Carve cropland with real physical boundaries
cropland_carved = cropland.copy()
cropland_carved[is_edge_dilated] = 0

cropland_clean = ndimage.binary_opening(cropland_carved, structure=struct_cross).astype(np.uint8)
cropland_clean = ndimage.binary_fill_holes(cropland_clean).astype(np.uint8)

# 4. Connected components
lbl, num_features = ndimage.label(cropland_clean, structure=struct_cross)
print(f"Candidate components separated by real physical edges: {num_features}")

# 5. Check user patch: r=2408, c=3029 (200x200 patch)
sub_lbl = lbl[2408-100:2408+100, 3029-100:3029+100]
u_patch = np.unique(sub_lbl[sub_lbl > 0])
print(f"Parcels in user patch (formerly octagons): {len(u_patch)}")

# Check shapes of parcels in this patch
for p in u_patch[:8]:
    m = (sub_lbl == p)
    coords = np.argwhere(m)
    h = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    w = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    area_mu = len(coords) * 900.0 / 666.67
    aspect = max(h, w) / max(min(h, w), 1)
    print(f"  Field {p}: area={area_mu:.1f} 亩, bbox={h}x{w} px ({h*30}m x {w*30}m), aspect={aspect:.2f}")

