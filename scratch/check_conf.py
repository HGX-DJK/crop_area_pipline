import numpy as np
import rasterio

with rasterio.open("crop_area_pipeline/output/crop_classification_map.tif") as src:
    mask = src.read(1)

print(f"Mask values: {np.unique(mask)}")
print(f"Mask sum: {np.sum(mask > 0)}")
