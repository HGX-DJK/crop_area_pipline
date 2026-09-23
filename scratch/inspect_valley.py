import rasterio
import numpy as np

with rasterio.open('output/crop_classification_map.tif') as src:
    crop_mask = src.read(1)

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

ndvi = np.where(b4 + b3 > 0, (b4 - b3) / (b4 + b3), 0.0)

# Check Sacramento Valley farmland region (rows 1000:2000, cols 2800:3500)
sub_ndvi = ndvi[1000:2000, 2800:3500]
sub_crop = crop_mask[1000:2000, 2800:3500]

print("Sacramento Valley sample (1000x700 pixels, ~30km x 21km):")
print(f"Total pixels: {sub_crop.size:,}")
print(f"Classified as Crop (1): {np.sum(sub_crop == 1):,} ({np.mean(sub_crop == 1):.1%})")
print(f"Classified as Non-Crop (0): {np.sum(sub_crop == 0):,} ({np.mean(sub_crop == 0):.1%})")

# Look at the pixels in this valley that are classified as 0 (Non-Crop):
sub_c0_ndvi = sub_ndvi[sub_crop == 0]
print(f"Non-crop pixels NDVI in valley:")
print(f"  mean={sub_c0_ndvi.mean():.3f}, min={sub_c0_ndvi.min():.3f}, max={sub_c0_ndvi.max():.3f}")
print(f"  NDVI < 0.20 (bare/urban/water): {np.mean(sub_c0_ndvi < 0.20):.1%}")
print(f"  NDVI 0.20-0.35 (fallow/dormant orchard): {np.mean((sub_c0_ndvi >= 0.20) & (sub_c0_ndvi < 0.35)):.1%}")
print(f"  NDVI > 0.35 (green): {np.mean(sub_c0_ndvi >= 0.35):.1%}")
