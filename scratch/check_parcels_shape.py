import json
from shapely.geometry import shape

with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total features: {len(data['features'])}")

# Check first 20 features
for i in range(20):
    feat = data['features'][i]
    props = feat['properties']
    geom = shape(feat['geometry'])
    minx, miny, maxx, maxy = geom.bounds
    width_deg = maxx - minx
    height_deg = maxy - miny
    print(f"#{i+1} {props['parcel_id']}: area={props['area_mu']} 亩, "
          f"bbox=({minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}), "
          f"width={width_deg:.4f}, height={height_deg:.4f}, ratio={height_deg/max(width_deg, 1e-6):.1f}")
