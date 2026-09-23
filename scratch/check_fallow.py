import rasterio
import numpy as np

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

farm_test = (slice(1000, 2000), slice(2800, 3500))
sub_b3 = b3[farm_test]
sub_b4 = b4[farm_test]
sub_b5 = b5[farm_test]

ndvi = (sub_b4 - sub_b3) / np.maximum(sub_b4 + sub_b3, 1e-4)
ndbi = (sub_b5 - sub_b4) / np.maximum(sub_b5 + sub_b4, 1e-4)

# Fallow soil: ndvi between 0.18 and 0.35, b5 > 2000
fallow = (ndvi >= 0.18) & (ndvi <= 0.35) & (sub_b5 > 2000)

print(f"Fallow total: {np.sum(fallow):,}")
print(f"NDBI of fallow: mean={ndbi[fallow].mean():.3f}, min={ndbi[fallow].min():.3f}, max={ndbi[fallow].max():.3f}")
print(f"B4 of fallow: mean={sub_b4[fallow].mean():.1f}, min={sub_b4[fallow].min():.1f}")
print(f"How many fallow have ndbi > 0.08: {np.sum(ndbi[fallow] > 0.08):,} ({np.mean(ndbi[fallow] > 0.08):.1%})")
print(f"How many fallow have ndbi > 0.15: {np.sum(ndbi[fallow] > 0.15):,} ({np.mean(ndbi[fallow] > 0.15):.1%})")
print(f"How many fallow have ndbi > 0.20: {np.sum(ndbi[fallow] > 0.20):,} ({np.mean(ndbi[fallow] > 0.20):.1%})")
