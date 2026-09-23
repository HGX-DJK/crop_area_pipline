import rasterio
import numpy as np

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

ndvi = np.where(b4 + b3 > 0, (b4 - b3)/(b4 + b3), 0.0)

# Let's inspect columns 2300 to 2600 across rows 500 to 2500
# to see what is between column 2300 and 2600.
# Is it mountains, foothills, or farmland?
for r in [600, 1000, 1400, 1800, 2200]:
    print(f"\nRow {r}:")
    for c in [2300, 2400, 2500, 2600, 2700]:
        chunk_ndvi = ndvi[r:r+50, c:c+50]
        chunk_b5 = b5[r:r+50, c:c+50]
        chunk_b4 = b4[r:r+50, c:c+50]
        print(f"  Col {c}: NDVI={chunk_ndvi.mean():.2f} (std={chunk_ndvi.std():.2f}), B5={chunk_b5.mean():.0f}, B4={chunk_b4.mean():.0f}")
