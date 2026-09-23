import rasterio
import numpy as np

with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

ndvi = np.where(b4 + b3 > 0, (b4 - b3)/(b4 + b3), 0.0)

# 1. Check Sacramento Valley central farmland: rows 1000:2000, cols 2800:3500
sub_m = mask[1000:2000, 2800:3500]
sub_ndvi = ndvi[1000:2000, 2800:3500]
sub_b5 = b5[1000:2000, 2800:3500]

print("=== Sacramento Valley Plain Region (Rows 1000:2000, Cols 2800:3500) ===")
print(f"Total region pixels: {sub_m.size:,}")
print(f"Total Cropland extracted: {np.sum(sub_m == 1):,} ({np.mean(sub_m == 1):.1%})")

# Specifically check orchards: NDVI > 0.45, B5 < 1800
orchard_pixels = (sub_ndvi > 0.45) & (sub_b5 < 1800)
orchard_extracted = np.sum((sub_m == 1) & orchard_pixels)
print(f"Orchard pixels in valley (NDVI > 0.45, B5 < 1800): {np.sum(orchard_pixels):,}")
print(f"  -> Extracted as Cropland: {orchard_extracted:,} ({orchard_extracted/max(1, np.sum(orchard_pixels)):.1%})")

# Specifically check fallow cropland: NDVI 0.18-0.35, B5 > 2000
fallow_pixels = (sub_ndvi >= 0.18) & (sub_ndvi <= 0.35) & (sub_b5 > 2000)
fallow_extracted = np.sum((sub_m == 1) & fallow_pixels)
print(f"Fallow pixels in valley (NDVI 0.18-0.35, B5 > 2000): {np.sum(fallow_pixels):,}")
print(f"  -> Extracted as Cropland: {fallow_extracted:,} ({fallow_extracted/max(1, np.sum(fallow_pixels)):.1%})")

# 2. Check Mountain area to make sure no forest leakage: rows 1000:2000, cols 500:1500
sub_mtn = mask[1000:2000, 500:1500]
print("\n=== Coast Range Mountain Region (Rows 1000:2000, Cols 500:1500) ===")
print(f"Total mountain pixels: {sub_mtn.size:,}")
print(f"Cropland pixels in deep mountains: {np.sum(sub_mtn == 1):,} ({np.mean(sub_mtn == 1):.2%})")

# 3. Check overall image stats
print("\n=== Full Scene Statistics ===")
print(f"Total Cropland pixels: {np.sum(mask == 1):,} ({np.mean(mask == 1):.1%})")
