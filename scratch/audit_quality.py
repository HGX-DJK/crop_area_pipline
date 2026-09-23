import rasterio
import numpy as np
import json

with rasterio.open('output/crop_classification_map.tif') as src:
    arr = src.read(1)
    vals, counts = np.unique(arr, return_counts=True)
    print("Classification raster stats:")
    for v, c in zip(vals, counts):
        print(f"  Class {v}: {c:,} pixels ({c/arr.size:.2%})")
    
    h, w = arr.shape
    west_crop = np.sum(arr[:, :2000] == 1)
    east_crop = np.sum(arr[:, 2000:] == 1)
    print(f"\nSpatial distribution of Class 1 (Cropland):")
    print(f"  West (Cols 0-2000, Coast Range/Mountains): {west_crop:,} pixels ({west_crop / (h*2000):.2%})")
    print(f"  East (Cols 2000-3660, Sacramento Valley/Plain): {east_crop:,} pixels ({east_crop / (h*(w-2000)):.2%})")

# Check the GeoJSON parcels
with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    gj = json.load(f)

features = gj.get('features', [])
print(f"\nVectorized GeoJSON parcels count: {len(features)}")
lons = [f['properties']['center_lon'] for f in features]
lats = [f['properties']['center_lat'] for f in features]
areas = [f['properties']['area_mu'] for f in features]
print(f"Parcels center lon: min={min(lons):.3f}, max={max(lons):.3f}, median={np.median(lons):.3f}")
print(f"Parcels area (mu): min={min(areas):.1f}, median={np.median(areas):.1f}, max={max(areas):.1f}, sum={sum(areas):.1f}")

# How many parcels are in the West (lon < -122.2) vs East (lon >= -122.2)?
west_parcels = [f for f in features if f['properties']['center_lon'] < -122.2]
east_parcels = [f for f in features if f['properties']['center_lon'] >= -122.2]
print(f"Parcels in West (lon < -122.2, mountains/valleys): {len(west_parcels)}")
print(f"Parcels in East (lon >= -122.2, central valley): {len(east_parcels)}")
