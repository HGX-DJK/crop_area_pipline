import rasterio
import numpy as np

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b4 = src.read(4).astype(np.float32)

from scipy import ndimage
mean_b4 = ndimage.uniform_filter(b4, size=15)
sq_b4 = ndimage.uniform_filter(b4**2, size=15)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

# Check across different rows:
for r in [500, 1000, 1500, 2000, 2500, 3000]:
    row_cv = cv_b4[r, :]
    # Where does CV drop below 0.15 permanently toward the east?
    smooth_cv = ndimage.uniform_filter1d(row_cv, size=50)
    flat_cols = np.where(smooth_cv < 0.18)[0]
    print(f"Row {r}: min flat col = {flat_cols.min() if len(flat_cols)>0 else 'None'}, flat count east of 2000 = {np.sum(smooth_cv[2000:] < 0.18)}")
