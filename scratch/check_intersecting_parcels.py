import rasterio
import numpy as np
import json
from shapely.geometry import shape, box

# Load vectorized_parcels.geojson and find all parcels intersecting this 200x200 patch
with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    r, c = 2408, 3029
    left, top = src.xy(r - 100, c - 100)
    right, bottom = src.xy(r + 100, c + 100)

from pyproj import Transformer
transformer = Transformer.from_crs(src.crs, 'EPSG:4326', always_xy=True)
lon1, lat1 = transformer.transform(left, bottom)
lon2, lat2 = transformer.transform(right, top)
print(f"Bounding box in WGS84: ({lon1:.4f}, {lat1:.4f}) to ({lon2:.4f}, {lat2:.4f})")

patch_box = box(min(lon1, lon2), min(lat1, lat2), max(lon1, lon2), max(lat1, lat2))

with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    geojson = json.load(f)

intersecting = []
for feat in geojson['features']:
    geom = shape(feat['geometry'])
    if geom.intersects(patch_box):
        intersecting.append(feat)

print(f"Number of parcels intersecting this patch: {len(intersecting)}")
for feat in intersecting:
    p = feat['properties']
    print(f"  {p['parcel_id']}: area={p['area_mu']} 亩, centroid=({p['center_lon']:.4f}, {p['center_lat']:.4f}), num_coords={len(feat['geometry']['coordinates'][0])}")
