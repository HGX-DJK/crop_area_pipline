import rasterio
import numpy as np

files = [
    'data/satellite_tifs/CSDC30_10SEH_20150103.tif',
    'data/satellite_tifs/CSDC30_10SEH_20150106.tif',
    'data/satellite_tifs/CSDC30_10SEH_20150109.tif'
]

ndvis = []
for f in files:
    with rasterio.open(f) as s:
        b3 = s.read(3).astype(np.float32)
        b4 = s.read(4).astype(np.float32)
        ndvi = np.where(b4 + b3 > 0, (b4 - b3) / (b4 + b3), 0.0)
        ndvis.append(ndvi)

ndvi_stack = np.stack(ndvis, axis=0) # (3, H, W)
ndvi_std = np.std(ndvi_stack, axis=0)
ndvi_diff_max = np.max(ndvi_stack, axis=0) - np.min(ndvi_stack, axis=0)

print(f"Max NDVI difference across 6 days: mean={ndvi_diff_max.mean():.4f}, median={np.median(ndvi_diff_max):.4f}, 90%={np.percentile(ndvi_diff_max, 90):.4f}")
print("Sample values at center:")
for i, f in enumerate(files):
    print(f"  Date {i+1}: mean NDVI={ndvis[i].mean():.3f}")
