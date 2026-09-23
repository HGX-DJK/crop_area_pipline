import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b4 = src.read(4).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

ndvi = np.where(b4 + b3 > 0, (b4 - b3)/(b4 + b3), 0.0)

# The user's screenshot media_1790132194598.png:
# Where is this orchard located?
# Let's search where in the image this pattern exists:
# The user's image has orchard with parallel tree rows.
# Commercial orchards in this tile are primarily around Winters, Vacaville, Dixon, Davis (Solano/Yolo counties).
# That is east of the Vaca Mountains (roughly east of UTM X 575,000 to 585,000, i.e., Cols >= 2600 to 2800).
# Let's inspect Cols across different latitudes!
for r in [500, 1000, 1500, 2000, 2500, 3000, 3500]:
    # Look at B5 along row r:
    # In mountains, B5 has high spatial frequency variance from shadows and ridges.
    # In agricultural plain, B5 has large homogeneous rectangular parcels.
    win_var = ndimage.uniform_filter(b5[r, :]**2, 21) - ndimage.uniform_filter(b5[r, :], 21)**2
    roughness = np.sqrt(np.maximum(win_var, 0)) / np.maximum(ndimage.uniform_filter(b5[r, :], 21), 1.0)
    # Find where the roughness changes
    print(f"Row {r}: cols 2400-2900 roughness:")
    for c in range(2400, 3000, 100):
        print(f"  Col {c}: {np.mean(roughness[c:c+100]):.3f}", end="")
    print()
