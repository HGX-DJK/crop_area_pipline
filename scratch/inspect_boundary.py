import rasterio
import numpy as np

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)

ndvi = np.where(b4 + b3 > 0, (b4 - b3)/(b4 + b3), 0.0)

# The scene width is 3660 columns.
# Let's inspect columns 1800 to 2600 to see where the mountains end and where the plain starts!
# Let's check row 1500 (across the middle of the image):
print("Column profile at row 1500:")
for c in range(1800, 2700, 50):
    chunk_b4 = b4[1400:1600, c:c+50]
    std_b4 = np.std(chunk_b4)
    mean_b4 = np.mean(chunk_b4)
    cv = std_b4 / max(mean_b4, 1)
    print(f"Cols {c}-{c+50}: Mean B4={mean_b4:.1f}, Std={std_b4:.1f}, CV={cv:.3f}")
