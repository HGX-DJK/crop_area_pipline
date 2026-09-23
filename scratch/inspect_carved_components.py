import rasterio
import numpy as np
import cv2
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    r, c = 2408, 3029
    win = rasterio.windows.Window(col_off=c-100, row_off=r-100, width=200, height=200)
    b3 = src.read(3, window=win).astype(np.float32)
    b4 = src.read(4, window=win).astype(np.float32)
    b5 = src.read(5, window=win).astype(np.float32)

with rasterio.open('output/crop_classification_map.tif') as src_m:
    win = rasterio.windows.Window(col_off=c-100, row_off=r-100, width=200, height=200)
    cropland = (src_m.read(1, window=win) == 1).astype(np.uint8)

# Edge detection
grad_b4 = cv2.morphologyEx(b4, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
denom = b4 + b3
ndvi = np.zeros_like(b3)
v = denom > 0
ndvi[v] = (b4[v] - b3[v]) / denom[v]
grad_ndvi = cv2.morphologyEx(ndvi, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

edge_norm = (grad_b4 / np.maximum(b4, 1.0) * 0.4 + grad_ndvi * 0.6)
thresh = np.percentile(edge_norm[cropland > 0], 80)
real_edges = (edge_norm > thresh) & (cropland > 0)
real_edges_dilated = ndimage.binary_dilation(real_edges, structure=np.ones((2, 2)))

cropland_carved = cropland.copy()
cropland_carved[real_edges_dilated] = 0
cropland_clean = ndimage.binary_opening(cropland_carved, structure=ndimage.generate_binary_structure(2, 1))

lbl, num = ndimage.label(cropland_clean, structure=ndimage.generate_binary_structure(2, 1))
print(f"Total components: {num}")

for lab in range(1, min(15, num + 1)):
    m = (lbl == lab)
    coords = np.argwhere(m)
    if len(coords) < 10: continue
    h = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    w = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    area_mu = len(coords) * 900.0 / 666.67
    print(f"Component {lab}: area={area_mu:.1f} 亩, h={h} px ({h*30}m), w={w} px ({w*30}m), aspect={max(h,w)/min(h,w):.2f}")
