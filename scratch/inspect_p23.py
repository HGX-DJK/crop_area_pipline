import json
import rasterio
import numpy as np
from shapely.geometry import shape

with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

for feat in data['features']:
    if feat['properties']['parcel_id'] == 'P0023':
        geom = shape(feat['geometry'])
        print("P0023 geom type:", geom.geom_type)
        print("P0023 bounds:", geom.bounds)
        print("P0023 area:", feat['properties']['area_mu'])
        # Print coordinates of exterior
        coords = list(geom.exterior.coords) if geom.geom_type == 'Polygon' else list(geom.geoms[0].exterior.coords)
        print(f"Num exterior coords: {len(coords)}")
        print("Sample coords:", coords[:10])
