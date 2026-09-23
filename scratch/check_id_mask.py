import sys, os
sys.path.insert(0, os.path.abspath('.'))
import rasterio
import numpy as np
from src.parcel_segmenter import ParcelSegmenter

with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)

segmenter = ParcelSegmenter({'segmentation': {'max_parcel_area_m2': 800000.0, 'max_export_parcels': 3000}})
parcel_id_mask, metadata = segmenter.segment_parcels(mask)

# Check the slice around rows 400..500, cols 2670..2730
patch = parcel_id_mask[400:500, 2670:2730]
print("Unique parcel IDs in patch (rows 400..500, cols 2670..2730):")
print(np.unique(patch))

# Let's print the IDs along a horizontal line at row 450:
print("Row 450 parcel IDs across cols 2670..2730:")
print(parcel_id_mask[450, 2670:2730])

# And along row 480:
print("Row 480 parcel IDs across cols 2670..2730:")
print(parcel_id_mask[480, 2670:2730])
