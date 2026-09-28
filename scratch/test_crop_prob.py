import rasterio
import numpy as np

# Let's inspect the current crop_classification_map.tif
with rasterio.open("crop_area_pipeline/output/crop_classification_map.tif") as src:
    w = src.width
    h = src.height
    print(f"Raster dimensions: {h} x {w}")
