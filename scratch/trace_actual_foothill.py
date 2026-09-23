import rasterio
import numpy as np
from pyproj import Transformer

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b4 = src.read(4).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

transformer = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)

# For various rows, find the longitude of the mountain front
# The mountain front in Sacramento Valley:
# From Google Maps / USGS:
# At 38.85N (r=0): Mountain foot is at Longitude -122.03 (Easting ~584,000, col ~2800)
# At 38.65N (r=720): Mountain foot is at Longitude -122.02 (Easting ~585,000, col ~2830)
# At 38.52N (r=1200, Winters): Mountain foot is at Longitude -122.00 (Easting ~587,000, col ~2900)
# At 38.35N (r=1800, Vacaville): Mountain foot turns southwest toward Fairfield (col ~2600)
# At 38.25N (r=2200, Fairfield): Mountain foot is at col ~2450

for r in [100, 500, 1000, 1500, 2000, 2500]:
    for col in [2500, 2600, 2700, 2800, 2850]:
        x, y = src.xy(r, col)
        lon, lat = transformer.transform(x, y)
        print(f"r={r:4d}, col={col:4d} -> lon={lon:8.4f}, lat={lat:7.4f}")
    print()
