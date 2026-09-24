import json
from shapely.geometry import shape, Point

with open("crop_area_pipeline/output/vectorized_parcels.geojson", "r", encoding="utf-8") as f:
    data = json.load(f)

# The user screenshot media_1790221442516.png was taken at the old mountain peak (around lon 113.02, lat 24.36)
old_mtn_pt = Point(113.0236, 24.3648)

hit_parcels = []
for feat in data["features"]:
    geom = shape(feat["geometry"])
    if geom.contains(old_mtn_pt) or geom.distance(old_mtn_pt) < 0.01: # within ~1km
        hit_parcels.append(feat["properties"]["parcel_id"])

print(f"Number of parcels near/on the old false mountain peak: {len(hit_parcels)}")
if hit_parcels:
    print(f"  Parcels: {hit_parcels}")
else:
    print("  SUCCESS: Old false mountain peak is 100% clean! Zero parcels on the mountain!")
