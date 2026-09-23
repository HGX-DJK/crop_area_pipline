import rasterio
import numpy as np
from scipy import ndimage
from pyproj import Transformer

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b4 = src.read(4).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

r = 723
mean_b4 = ndimage.uniform_filter(b4, size=11)
sq_b4 = ndimage.uniform_filter(b4**2, size=11)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

transformer = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)

print(f"{'Col':>6} | {'Lon':>9} | {'Lat':>8} | {'NDVI':>6} | {'SWIR1':>7} | {'CV_B4':>6}")
print("-" * 55)
for col_idx in range(2750, 3550, 50):
    x, y = src.xy(r, col_idx)
    lon, lat = transformer.transform(x, y)
    nv = (b4[r, col_idx] - b3[r, col_idx]) / (b4[r, col_idx] + b3[r, col_idx])
    cv = cv_b4[r, col_idx]
    sw = b5[r, col_idx]
    print(f"{col_idx:6d} | {lon:9.4f} | {lat:8.4f} | {nv:6.3f} | {sw:7.1f} | {cv:6.3f}")
