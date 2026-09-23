import json

with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

features = data['features']
print(f"Total features: {len(features)}")
for feat in features[:10]:
    props = feat['properties']
    print(f"Parcel ID: {props.get('parcel_id')}, Area mu: {props.get('area_mu')}, ha: {props.get('area_ha')}, dominant: {props.get('dominant_crop')}")

# Check if P0001 contains Vacaville or Fairfield
from shapely.geometry import shape, Point
p1_geom = shape(features[0]['geometry'])
# Vacaville approx: -121.9877, 38.3566
# Fairfield approx: -122.0400, 38.2494
pt_vaca = Point(-121.9877, 38.3566)
pt_fair = Point(-122.0400, 38.2494)

print(f"P0001 contains Vacaville: {p1_geom.contains(pt_vaca)}")
print(f"P0001 contains Fairfield: {p1_geom.contains(pt_fair)}")
print(f"P0001 bbox: {p1_geom.bounds}")
