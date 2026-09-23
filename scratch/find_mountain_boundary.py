import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b4 = src.read(4).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b5 = src.read(5).astype(np.float32)
    h, w = b4.shape
    
    # Calculate roughness across the whole scene
    mean_b4 = ndimage.uniform_filter(b4, size=15)
    sq_b4 = ndimage.uniform_filter(b4**2, size=15)
    cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)
    
    # At different rows (from north to south), where does the roughness (mountain) end and plain begin?
    print("Finding boundary between mountain (rough) and plain (flat):")
    for r in range(200, 3600, 400):
        # Look across columns for this row
        cv_row = cv_b4[r, :]
        # Smooth cv_row with 50-pixel window
        cv_smooth = ndimage.uniform_filter(cv_row, size=50)
        # Find the transition where roughness drops below 0.08
        mountain_cols = np.where(cv_smooth > 0.08)[0]
        if len(mountain_cols) > 0:
            eastmost_mountain_col = np.max(mountain_cols)
        else:
            eastmost_mountain_col = 0
        print(f"Row {r:4d}: Eastmost rough mountain col = {eastmost_mountain_col}")
