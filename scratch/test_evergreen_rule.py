import sys, os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import rasterio, numpy as np

sorted_files = [
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000621.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20001015.tif"
]
srcs = [rasterio.open(f) for f in sorted_files]

win_mtn = rasterio.windows.Window(3400, 3400, 256, 256) # Mountain
win_val = rasterio.windows.Window(2700, 1800, 256, 256) # Valley

def get_ndvis(win):
    res = []
    for s in srcs:
        b3 = s.read(3, window=win).astype(float)
        b4 = s.read(4, window=win).astype(float)
        res.append((b4 - b3) / np.maximum(b4 + b3, 1.0))
    return res

nd_mtn = get_ndvis(win_mtn)
nd_val = get_ndvis(win_val)

# Mountain: how many pixels have Date 0 >= 0.35 and Date 2 >= 0.55?
mtn_evergreen = (nd_mtn[0] >= 0.35) & (nd_mtn[2] >= 0.55)
val_evergreen = (nd_val[0] >= 0.35) & (nd_val[2] >= 0.55)

print(f"Mountain: {np.mean(mtn_evergreen)*100:.1f}% are evergreen canopy!")
print(f"Valley: {np.mean(val_evergreen)*100:.1f}% are evergreen canopy!")

for s in srcs: s.close()
