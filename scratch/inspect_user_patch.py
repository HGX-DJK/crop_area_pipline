import rasterio
import numpy as np
import cv2

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    from pyproj import Transformer
    transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
    x, y = transformer.transform(-121.9625, 38.1935)
    r, c = src.index(x, y)
    print(f"(-121.9625, 38.1935) -> Row={r}, Col={c}")

    # Read a 200x200 window around this point
    win = rasterio.windows.Window(col_off=c-100, row_off=r-100, width=200, height=200)
    b3 = src.read(3, window=win).astype(np.float32)
    b4 = src.read(4, window=win).astype(np.float32)
    b5 = src.read(5, window=win).astype(np.float32)

# Compute NDVI
denom = b4 + b3
ndvi = np.zeros_like(b3)
v = denom > 0
ndvi[v] = (b4[v] - b3[v]) / denom[v]

print(f"Patch NIR: min={np.min(b4)}, max={np.max(b4)}")
print(f"Patch NDVI: min={np.min(ndvi):.2f}, mean={np.mean(ndvi):.2f}, max={np.max(ndvi):.2f}")

# Compute gradient edges on B4 and NDVI
grad_b4 = cv2.morphologyEx(b4, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
grad_ndvi = cv2.morphologyEx(ndvi, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))

print(f"Grad B4 85th pct: {np.percentile(grad_b4, 85):.1f}")
print(f"Grad NDVI 85th pct: {np.percentile(grad_ndvi, 85):.3f}")
