import rasterio
import numpy as np
import cv2
from scipy import ndimage

print("Loading satellite bands and crop mask...")
with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

with rasterio.open('output/crop_classification_map.tif') as src_m:
    cropland_mask = (src_m.read(1) == 1).astype(np.uint8)

print(f"Total cropland pixels: {np.sum(cropland_mask)}")

# 1. Compute multi-spectral edge gradient (NIR, Red, NDVI)
denom = b4 + b3
ndvi = np.zeros_like(b3)
v = denom > 0
ndvi[v] = (b4[v] - b3[v]) / denom[v]

# Sobel gradient on NIR and NDVI
gx_b4 = cv2.Sobel(b4, cv2.CV_32F, 1, 0, ksize=3)
gy_b4 = cv2.Sobel(b4, cv2.CV_32F, 0, 1, ksize=3)
mag_b4 = np.sqrt(gx_b4**2 + gy_b4**2)

gx_ndvi = cv2.Sobel(ndvi, cv2.CV_32F, 1, 0, ksize=3)
gy_ndvi = cv2.Sobel(ndvi, cv2.CV_32F, 0, 1, ksize=3)
mag_ndvi = np.sqrt(gx_ndvi**2 + gy_ndvi**2)

# Normalize gradients
mag_b4_norm = mag_b4 / np.maximum(b4, 1.0)
combined_edge = mag_b4_norm * 0.4 + mag_ndvi * 0.6

# Detect real physical boundaries inside or bordering cropland:
# Boundaries have significantly higher gradient than the homogeneous interior of fields
crop_edges = combined_edge[cropland_mask > 0]
thresh = float(np.percentile(crop_edges, 80))
is_edge = (combined_edge > thresh) & (cropland_mask > 0)

# Dilate edge by 1 pixel cross to form clean boundary cuts
struct_cross = ndimage.generate_binary_structure(2, 1)
is_edge_dilated = ndimage.binary_dilation(is_edge, structure=struct_cross)

# 2. Cut cropland with real physical boundaries
cropland_carved = cropland_mask.copy()
cropland_carved[is_edge_dilated] = 0

# Morphological opening to cut thin diagonal connections
cropland_clean = ndimage.binary_opening(cropland_carved, structure=struct_cross).astype(np.uint8)

# Fill small interior holes (e.g. tractor tracks or small noise inside fields)
cropland_clean = ndimage.binary_fill_holes(cropland_clean).astype(np.uint8)

# 3. Connected components labeling (4-connectivity to keep fields cleanly separated by 1-pixel roads)
lbl, num_features = ndimage.label(cropland_clean, structure=struct_cross)
print(f"Naturally delineated agricultural fields: {num_features}")

counts = np.bincount(lbl.ravel())
sizes_mu = counts[1:num_features+1] * 900.0 / 666.67
print(f"Fields > 10 亩: {np.sum(sizes_mu >= 10.0)}")
print(f"Fields > 50 亩: {np.sum(sizes_mu >= 50.0)}")
print(f"Fields > 100 亩: {np.sum(sizes_mu >= 100.0)}")
print(f"Fields > 500 亩: {np.sum(sizes_mu >= 500.0)}")
print(f"Max field size: {np.max(sizes_mu):.1f} 亩")

# Check aspect ratio of top 20 fields
sorted_idx = np.argsort(-sizes_mu)
slices = ndimage.find_objects(lbl)
ratios = []
for idx in sorted_idx[:30]:
    comp_id = idx + 1
    sl = slices[idx]
    if sl is None: continue
    sub = (lbl[sl] == comp_id)
    coords = np.argwhere(sub)
    h = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    w = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    aspect = max(h, w) / max(min(h, w), 1)
    ratios.append(aspect)
    print(f"Field {comp_id}: area={sizes_mu[idx]:.1f} 亩, bbox=({h}x{w} px = {h*30}mx{w*30}m), aspect={aspect:.2f}")

print(f"\nTop 30 fields average aspect ratio: {np.mean(ratios):.2f}")
