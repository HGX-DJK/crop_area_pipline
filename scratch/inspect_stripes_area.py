import rasterio
import numpy as np

with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)

sub = mask[336:659, 2650:2750]
print("Shape:", sub.shape)

# Let's inspect the parcels in this area from vectorized_parcels.geojson
import json
from shapely.geometry import shape, Point
from pyproj import Transformer
transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)

with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find all parcels whose bbox intersects rows 336:659, cols 2650:2750
# Convert bbox to UTM
left, top = src.xy(336, 2650)
right, bottom = src.xy(659, 2750)
from shapely.geometry import box
area_box = box(left, bottom, right, top)

# Transform geojson features to UTM and check intersection
transformer_to_utm = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)

hits = []
for feat in data['features']:
    geom = shape(feat['geometry'])
    # sample point
    lon, lat = geom.centroid.x, geom.centroid.y
    ux, uy = transformer_to_utm.transform(lon, lat)
    pt = Point(ux, uy)
    if area_box.contains(pt):
        hits.append(feat['properties'])

print(f"Parcels in this box: {len(hits)}")
for h in hits:
    print(f"  {h['parcel_id']}: area={h['area_mu']} 亩, centroid=({h.get('center_lon')}, {h.get('center_lat')})")
