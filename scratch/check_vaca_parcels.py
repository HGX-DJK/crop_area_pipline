import json
from shapely.geometry import shape, Point

with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

vaca_pts = {
    'Downtown': Point(-121.9877, 38.3566),
    'North Res': Point(-121.9600, 38.3700),
    'South Res': Point(-121.9500, 38.3400),
}

for feat in data['features']:
    pid = feat['properties']['parcel_id']
    if pid in ['P0614', 'P1347']:
        geom = shape(feat['geometry'])
        print(f"{pid}: area={feat['properties']['area_mu']} 亩, bounds={geom.bounds}")
        for name, pt in vaca_pts.items():
            if geom.contains(pt):
                print(f"  Contains {name}!")
