import sys, os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import json
from shapely.geometry import shape, Point

with open("crop_area_pipeline/output/vectorized_parcels.geojson", "r", encoding="utf-8") as f:
    data = json.load(f)

# Detected primary urban center around r=300, c=1520
# In WGS84:
import rasterio
from src.utils.geo_utils import utm_to_wgs84
s0 = rasterio.open("crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif")
ux, uy = s0.xy(300, 1520)
ulon, ulat = utm_to_wgs84(ux, uy, zone=49)
urban_pt = Point(ulon, ulat)

hit_parcels = []
for feat in data["features"]:
    geom = shape(feat["geometry"])
    if geom.contains(urban_pt) or geom.distance(urban_pt) < 0.005: # within ~500m
        hit_parcels.append(feat["properties"]["parcel_id"])

print(f"Urban center (lon={ulon:.4f}, lat={ulat:.4f}):")
print(f"Number of parcels in urban core: {len(hit_parcels)}")
