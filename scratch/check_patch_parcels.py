import rasterio
import numpy as np
from scipy import ndimage
import json
from shapely.geometry import shape, box

with rasterio.open('output/crop_classification_map.tif') as src:
    win = rasterio.windows.Window(col_off=2900, row_off=1600, width=300, height=300)
    sub_crop = src.read(1, window=win)
    # Get geographic bounds of this patch
    left, top = src.xy(1600, 2900)
    right, bottom = src.xy(1900, 3200)

print(f"Patch bounds: UTM X=({left}, {right}), Y=({bottom}, {top})")

# Connected components in sub_crop directly
lbl, num = ndimage.label(sub_crop > 0, structure=ndimage.generate_binary_structure(2, 1))
print(f"Connected components in this 300x300 patch (4-conn): {num}")
counts = np.bincount(lbl.ravel())
print(f"Top 5 component sizes (pixels): {np.sort(counts[1:])[-5:][::-1]}")
print(f"In 亩: {np.sort(counts[1:])[-5:][::-1] * 900.0 / 666.67}")

# Check what parcels were extracted in this patch from vectorized_parcels.geojson
with open('output/vectorized_parcels.geojson', 'r', encoding='utf-8') as f:
    geojson = json.load(f)

from pyproj import Transformer
t = Transformer.from_crs('EPSG:4326', 'EPSG:32610', always_xy=True)
patch_box = box(left, bottom, right, top)

hits = []
for feat in geojson['features']:
    props = feat['properties']
    ux, uy = props['center_utm_x'], props['center_utm_y']
    from shapely.geometry import Point
    if patch_box.contains(Point(ux, uy)):
        hits.append(props)

print(f"\nParcels extracted in this 300x300 patch: {len(hits)}")
for h in hits[:10]:
    print(f"  {h['parcel_id']}: area={h['area_mu']:.1f} 亩, compactness={h['compactness']}, center=({h['center_lon']:.4f}, {h['center_lat']:.4f})")
