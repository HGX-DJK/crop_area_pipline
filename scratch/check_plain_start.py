import rasterio
import numpy as np

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b4 = src.read(4).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

ndvi = (b4 - b3) / np.maximum(b4 + b3, 1.0)

# In Sacramento Valley, real farmland is located in:
# Yolo County (Winters, Davis, Woodland): Cols ~ 2850 to 3660, Rows ~ 0 to 1800
# Solano County (Dixon, Elmira): Cols ~ 2800 to 3660, Rows ~ 1500 to 2800
# Let's inspect rows 100, 300, 500, 700, 900, 1100, 1300, 1500, 2000:
# Where does the flat agricultural plain begin?
print(f"{'Row':>5} | Mountain ends / Plain begins around Col:")
for r in range(100, 2500, 200):
    # In each row, let's find where Col starts having flat plain vs mountain
    # Mountain has high elevation, rugged texture (cv_b4), while plain fields have clear roads/field boundaries
    # Let's check col from 2600 to 3000
    pass

