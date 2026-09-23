import json
import rasterio
import numpy as np

with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    data = json.load(f)

for feat in data['features']:
    pid = feat['properties']['parcel_id']
    if pid in ['P0023', 'P0030', 'P0066']:
        print(f"{pid}: {feat['properties']}")

# Let's inspect parcel_attribute_table.csv
import csv
with open('output/vectorized_parcels_attribute_table.csv', 'r', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row['parcel_id'] in ['P0023', 'P0030', 'P0066']:
            print(f"Row {row['parcel_id']}: row={row.get('centroid_row')}, col={row.get('centroid_col')}, area_mu={row.get('area_mu')}")
