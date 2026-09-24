import sys, os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import rasterio, numpy as np
from scipy import ndimage

s0 = rasterio.open("crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif")
win = rasterio.windows.Window(3450, 3400, 100, 100) # x=3450, y=3400
b1 = s0.read(1, window=win).astype(np.float32)
b2 = s0.read(2, window=win).astype(np.float32)
b3 = s0.read(3, window=win).astype(np.float32)
b4 = s0.read(4, window=win).astype(np.float32)
b5 = s0.read(5, window=win).astype(np.float32)

px_r = 3464 - 3400
px_c = 3509 - 3450

for d_idx, path in enumerate([
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000621.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20001015.tif"
]):
    s = rasterio.open(path)
    b1 = s.read(1, window=win).astype(np.float32)
    b2 = s.read(2, window=win).astype(np.float32)
    b3 = s.read(3, window=win).astype(np.float32)
    b4 = s.read(4, window=win).astype(np.float32)
    b5 = s.read(5, window=win).astype(np.float32)
    ndvi_raw = (b4 - b3) / np.maximum(b4 + b3, 1.0)
    b5_b3_ratio = b5 / np.maximum(b3, 1.0)
    ndbi = (b5 - b4) / np.maximum(b5 + b4, 1e-4)
    imp = (
        (b1 > 1000.0) |
        ((b1 > 480.0) & (b5_b3_ratio < 1.85) & (ndvi_raw < 0.40)) |
        ((ndbi > -0.02) & (b5_b3_ratio < 1.75)) |
        ((b5_b3_ratio < 1.55) & (ndvi_raw < 0.35))
    )
    dens = ndimage.uniform_filter(imp.astype(np.float32), size=21)
    cand = dens >= 0.12
    print(f"Date {d_idx} at P0001: B1={b1[px_r, px_c]:.0f}, B4={b4[px_r, px_c]:.0f}, B3={b3[px_r, px_c]:.0f}, Raw NDVI={ndvi_raw[px_r, px_c]:.3f}, imp={imp[px_r, px_c]}, dens={dens[px_r, px_c]:.3f}, cand={cand[px_r, px_c]}")

