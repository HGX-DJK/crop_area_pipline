import rasterio
import numpy as np

src1 = rasterio.open('crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif')
src2 = rasterio.open('crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000621.tif')
src3 = rasterio.open('crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20001015.tif')

def get_vals(r, c):
    res = []
    for s in [src1, src2, src3]:
        b1 = float(s.read(1)[r, c])
        b2 = float(s.read(2)[r, c])
        b3 = float(s.read(3)[r, c])
        b4 = float(s.read(4)[r, c])
        b5 = float(s.read(5)[r, c])
        ndvi = (b4 - b3) / (b4 + b3 + 1e-5)
        lswi = (b4 - b5) / (b4 + b5 + 1e-5)
        res.append((round(ndvi, 3), round(lswi, 3), int(b1), int(b2), int(b3), int(b4), int(b5)))
    return res

print("Mountain Forest P0001 (r=2378, c=1229):", get_vals(2378, 1229))
print("Mountain Ridge 2      (r=2000, c=1000):", get_vals(2000, 1000))
print("Valley farmland       (r=3000, c=3000):", get_vals(3000, 3000))
print("Valley farmland 2     (r=2500, c=2500):", get_vals(2500, 2500))
