import os
import rasterio
import rasterio.env

proj_cand = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
if os.path.exists(proj_cand):
    rasterio.env.set_proj_data_search_path(proj_cand)
    os.environ["PROJ_LIB"] = proj_cand
    os.environ["PROJ_DATA"] = proj_cand

import numpy as np
from rasterio.warp import transform

with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b1 = src.read(1).astype(np.float32)
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)
    b6 = src.read(6).astype(np.float32)

vaca_xs, vaca_ys = transform('EPSG:4326', src.crs, [-121.99], [38.356])
fair_xs, fair_ys = transform('EPSG:4326', src.crs, [-122.04], [38.25])

v_row, v_col = ~src.transform * (vaca_xs[0], vaca_ys[0])
f_row, f_col = ~src.transform * (fair_xs[0], fair_ys[0])

v_row, v_col = int(v_row), int(v_col)
f_row, f_col = int(f_row), int(f_col)

print(f"Vacaville center: Row={v_row}, Col={v_col}")
print(f"Fairfield center: Row={f_row}, Col={f_col}")

# Inspect a 150x150 window around Vacaville
v_sl = (slice(v_row - 75, v_row + 75), slice(v_col - 75, v_col + 75))
# Inspect a 150x150 window around Fairfield
f_sl = (slice(f_row - 75, f_row + 75), slice(f_col - 75, f_col + 75))

v_mask = mask[v_sl]
f_mask = mask[f_sl]

print(f"\nVacaville urban window (150x150 = 22,500 pixels):")
print(f"  Cropland pixels: {np.sum(v_mask == 1):,} ({np.mean(v_mask == 1):.1%})")

print(f"\nFairfield urban window (150x150 = 22,500 pixels):")
print(f"  Cropland pixels: {np.sum(f_mask == 1):,} ({np.mean(f_mask == 1):.1%})")

# Let's inspect the spectral properties of the pixels in Vacaville that got classified as 1 (cropland):
v_crop = (v_mask == 1)
v_b1 = b1[v_sl][v_crop]
v_b2 = b2[v_sl][v_crop]
v_b3 = b3[v_sl][v_crop]
v_b4 = b4[v_sl][v_crop]
v_b5 = b5[v_sl][v_crop]
v_b6 = b6[v_sl][v_crop]
v_ndvi = (v_b4 - v_b3) / np.maximum(v_b4 + v_b3, 1e-4)
v_ndbi = (v_b5 - v_b4) / np.maximum(v_b5 + v_b4, 1e-4)
v_b5_b3 = v_b5 / np.maximum(v_b3, 1.0)

print(f"\nProperties of pixels in Vacaville classified as Cropland (N={np.sum(v_crop):,}):")
print(f"  NDVI: mean={v_ndvi.mean():.3f}, min={v_ndvi.min():.3f}, max={v_ndvi.max():.3f}")
print(f"  NDBI: mean={v_ndbi.mean():.3f}")
print(f"  B5/B3: mean={v_b5_b3.mean():.2f}")
print(f"  B4 (NIR): mean={v_b4.mean():.1f}")
print(f"  B5 (SWIR1): mean={v_b5.mean():.1f}")
