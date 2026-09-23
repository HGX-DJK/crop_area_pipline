import json
from shapely.geometry import shape, Point

with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

features = data['features']
print(f"Total features: {len(features)}")

# Key city coordinates
vaca_pts = [
    Point(-121.9877, 38.3566), # Downtown
    Point(-121.9600, 38.3700), # North residential
    Point(-121.9500, 38.3400), # South residential
]
fair_pts = [
    Point(-122.0400, 38.2494), # Downtown
    Point(-122.0600, 38.2600), # West residential
    Point(-122.0200, 38.2700), # Northeast residential
]

vaca_hits = 0
fair_hits = 0
for idx, feat in enumerate(features):
    geom = shape(feat['geometry'])
    for pt in vaca_pts:
        if geom.contains(pt):
            vaca_hits += 1
            print(f"Warning: Parcel {feat['properties']['parcel_id']} contains Vacaville point!")
    for pt in fair_pts:
        if geom.contains(pt):
            fair_hits += 1
            print(f"Warning: Parcel {feat['properties']['parcel_id']} contains Fairfield point!")

print(f"Total Vacaville hits across all 3000 parcels: {vaca_hits}")
print(f"Total Fairfield hits across all 3000 parcels: {fair_hits}")

areas = [f['properties']['area_mu'] for f in features]
print(f"Min area: {min(areas):.1f} 亩")
print(f"Median area: {sorted(areas)[len(areas)//2]:.1f} 亩")
print(f"Mean area: {sum(areas)/len(areas):.1f} 亩")
print(f"Max area: {max(areas):.1f} 亩")
