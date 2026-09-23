import json
from shapely.geometry import shape

with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Find parcels where num exterior points is small (like 8 points = octagon) or check sample parcels
for feat in data['features'][:20]:
    geom = shape(feat['geometry'])
    pid = feat['properties']['parcel_id']
    coords = list(geom.exterior.coords) if geom.geom_type == 'Polygon' else list(geom.geoms[0].exterior.coords)
    print(f"{pid}: num_coords={len(coords)}, area={feat['properties']['area_mu']} 亩, centroid=({feat['properties']['center_lon']:.4f}, {feat['properties']['center_lat']:.4f})")
