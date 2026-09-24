import json
import numpy as np

data = json.load(open('crop_area_pipeline/output/vectorized_parcels.geojson', encoding='utf-8'))
features = data['features']
print(f"Total exported parcels: {len(features)}")
areas = [f['properties']['area_mu'] for f in features]
print(f"Total parcel area: {sum(areas):.1f} mu ({sum(areas)/10000:.2f} wan mu)")
print(f"Area stats: min={min(areas):.1f}, max={max(areas):.1f}, mean={np.mean(areas):.1f}, median={np.median(areas):.1f}")
compactness = [f['properties']['compactness'] for f in features]
print(f"Mean compactness: {np.mean(compactness):.3f}")

print("\nTop 15 parcels:")
for f in features[:15]:
    p = f['properties']
    print(f"  {p['parcel_id']}: area={p['area_mu']:.1f} mu, lon={p['center_lon']:.4f}, lat={p['center_lat']:.4f}, comp={p['compactness']:.3f}, crop={p['crop_name']}")
