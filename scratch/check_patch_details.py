import rasterio
import numpy as np
import cv2

# Look at Sacramento Valley farmland (e.g., around Dixon / Winters: rows 1600..1900, cols 2900..3200)
with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    win = rasterio.windows.Window(col_off=2900, row_off=1600, width=300, height=300)
    b3 = src.read(3, window=win).astype(np.float32) # Red
    b4 = src.read(4, window=win).astype(np.float32) # NIR
    b5 = src.read(5, window=win).astype(np.float32) # SWIR1

# Crop classification map
with rasterio.open('output/crop_classification_map.tif') as src_m:
    win = rasterio.windows.Window(col_off=2900, row_off=1600, width=300, height=300)
    crop_mask = src_m.read(1, window=win)

print("Classification mask in 300x300 farmland patch:")
print(f"  Cropland pixels: {np.sum(crop_mask == 1)} / {crop_mask.size} ({np.mean(crop_mask == 1)*100:.1f}%)")
print(f"  Non-crop pixels: {np.sum(crop_mask == 0)} / {crop_mask.size} ({np.mean(crop_mask == 0)*100:.1f}%)")

# Let's inspect the roads:
# Dirt roads and highways between fields have higher Red/SWIR reflectance, low NDVI, or sharp gradient!
# Let's check NDVI:
denom = b4 + b3
ndvi = np.zeros_like(b3)
v = denom > 0
ndvi[v] = (b4[v] - b3[v]) / denom[v]

# Sobel gradient on Red and NIR:
gx = cv2.Sobel(b4, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(b4, cv2.CV_32F, 0, 1, ksize=3)
edge_mag = np.sqrt(gx**2 + gy**2)

print(f"NDVI min={np.min(ndvi):.2f}, mean={np.mean(ndvi):.2f}, max={np.max(ndvi):.2f}")
print(f"Edge mag mean={np.mean(edge_mag):.1f}, 80th pct={np.percentile(edge_mag, 80):.1f}, 90th pct={np.percentile(edge_mag, 90):.1f}")
