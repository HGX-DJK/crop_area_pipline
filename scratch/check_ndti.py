import rasterio
import numpy as np

sorted_files = [
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000621.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20001015.tif"
]

with rasterio.open(sorted_files[0]) as s:
    print(f"Band count in SDC30: {s.count}")
    b5 = s.read(5, window=rasterio.windows.Window(2700, 1800, 100, 100)).astype(float)
    b6 = s.read(6, window=rasterio.windows.Window(2700, 1800, 100, 100)).astype(float)
    ndti = (b5 - b6) / np.maximum(b5 + b6, 1.0)
    print(f"Valley NDTI (March): mean={np.mean(ndti):.3f}, range=[{np.min(ndti):.3f}, {np.max(ndti):.3f}]")

with rasterio.open(sorted_files[0]) as s:
    b5_m = s.read(5, window=rasterio.windows.Window(3400, 3400, 100, 100)).astype(float)
    b6_m = s.read(6, window=rasterio.windows.Window(3400, 3400, 100, 100)).astype(float)
    ndti_m = (b5_m - b6_m) / np.maximum(b5_m + b6_m, 1.0)
    print(f"Mountain NDTI (March): mean={np.mean(ndti_m):.3f}, range=[{np.min(ndti_m):.3f}, {np.max(ndti_m):.3f}]")
