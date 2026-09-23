import rasterio
import numpy as np

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b1 = src.read(1).astype(np.float32)
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)
    b6 = src.read(6).astype(np.float32)

# Davis / Dixon / Vacaville urban core (Vacaville is around row 1700-1800, col 2800-2900)
# Fairfield urban core is around row 2200-2300, col 2400-2500
urban_slice = (slice(1700, 1800), slice(2800, 2900))
# Agricultural bare soil (row 1200-1400, col 3100-3300)
soil_slice = (slice(1200, 1400), slice(3100, 3300))

ndvi_urban = (b4[urban_slice] - b3[urban_slice]) / np.maximum(b4[urban_slice] + b3[urban_slice], 1e-4)
ndvi_soil = (b4[soil_slice] - b3[soil_slice]) / np.maximum(b4[soil_slice] + b3[soil_slice], 1e-4)

soil_mask = (ndvi_soil >= 0.18) & (ndvi_soil <= 0.32) & (b5[soil_slice] > 2000)
urban_mask = (ndvi_urban < 0.20)

print(f"Soil pixels: N={np.sum(soil_mask)}")
print(f"  B1: {b1[soil_slice][soil_mask].mean():.1f}")
print(f"  B2: {b2[soil_slice][soil_mask].mean():.1f}")
print(f"  B3: {b3[soil_slice][soil_mask].mean():.1f}")
print(f"  B4: {b4[soil_slice][soil_mask].mean():.1f}")
print(f"  B5: {b5[soil_slice][soil_mask].mean():.1f}")
print(f"  B5/B3: {(b5[soil_slice][soil_mask]/np.maximum(b3[soil_slice][soil_mask], 1)).mean():.2f}")
print(f"  B4/B1: {(b4[soil_slice][soil_mask]/np.maximum(b1[soil_slice][soil_mask], 1)).mean():.2f}")

print(f"\nUrban pixels: N={np.sum(urban_mask)}")
print(f"  B1: {b1[urban_slice][urban_mask].mean():.1f}")
print(f"  B2: {b2[urban_slice][urban_mask].mean():.1f}")
print(f"  B3: {b3[urban_slice][urban_mask].mean():.1f}")
print(f"  B4: {b4[urban_slice][urban_mask].mean():.1f}")
print(f"  B5: {b5[urban_slice][urban_mask].mean():.1f}")
print(f"  B5/B3: {(b5[urban_slice][urban_mask]/np.maximum(b3[urban_slice][urban_mask], 1)).mean():.2f}")
print(f"  B4/B1: {(b4[urban_slice][urban_mask]/np.maximum(b1[urban_slice][urban_mask], 1)).mean():.2f}")
