import json
from shapely.geometry import shape, box

with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Berryessa bbox
berryessa_bbox = box(-122.25, 38.45, -122.05, 38.70)

hits = []
for feat in data['features']:
    geom = shape(feat['geometry'])
    if geom.intersects(berryessa_bbox):
        hits.append((feat['properties'], geom))

print(f"Features in Berryessa mountain area: {len(hits)}")
for props, geom in hits[:15]:
    minx, miny, maxx, maxy = geom.bounds
    w = maxx - minx
    h = maxy - miny
    print(f"{props['parcel_id']}: area={props['area_mu']} 亩, bounds=({minx:.4f}, {miny:.4f}, {maxx:.4f}, {maxy:.4f}), w={w:.4f}, h={h:.4f}")
