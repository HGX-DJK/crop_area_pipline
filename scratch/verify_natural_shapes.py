import json
from shapely.geometry import shape, box
import numpy as np

with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

print(f"Total features in geojson: {len(data['features'])}")

# Check user patch: r=2408, c=3029 -> around lon=-121.9625, lat=38.1935
user_box = box(-121.98, 38.175, -121.94, 38.21)

patch_parcels = []
for feat in data['features']:
    geom = shape(feat['geometry'])
    if geom.intersects(user_box):
        patch_parcels.append(feat)

print(f"\nParcels intersecting user screenshot area: {len(patch_parcels)}")
for feat in patch_parcels[:12]:
    p = feat['properties']
    geom = shape(feat['geometry'])
    coords = np.array(geom.exterior.coords) if geom.geom_type == 'Polygon' else np.array(geom.geoms[0].exterior.coords)
    minx, miny, maxx, maxy = geom.bounds
    w_m = (maxx - minx) * 111320.0 * np.cos(np.radians(38.5))
    h_m = (maxy - miny) * 110574.0
    aspect = max(w_m, h_m) / max(min(w_m, h_m), 1.0)
    print(f"  {p['parcel_id']}: area={p['area_mu']:.1f} 亩, bbox={w_m:.0f}m x {h_m:.0f}m, aspect={aspect:.2f}, num_vertices={len(coords)}, compactness={p['compactness']}")

# Check overall aspect ratio distribution across all 3000 parcels
aspects = []
for feat in data['features']:
    geom = shape(feat['geometry'])
    minx, miny, maxx, maxy = geom.bounds
    w_m = (maxx - minx) * 111320.0 * np.cos(np.radians(38.5))
    h_m = (maxy - miny) * 110574.0
    aspects.append(max(w_m, h_m) / max(min(w_m, h_m), 1.0))

print("\nOverall statistics across 3000 parcels:")
print(f"  Mean aspect ratio: {np.mean(aspects):.2f}")
print(f"  Median aspect ratio: {np.median(aspects):.2f}")
print(f"  Parcels with aspect <= 2.5: {np.mean(np.array(aspects) <= 2.5)*100:.1f}%")
