import rasterio, numpy as np

with rasterio.open("crop_area_pipeline/output/crop_classification_map.tif") as src:
    mask = src.read(1)
    print("Crop pixels:", np.sum(mask == 1), f"({np.sum(mask == 1)*900/666.6667:,.1f} 亩)")

# 检查一下 P0001 (r=3464, c=3509)
print("Pixel at mountain (3464, 3509):", mask[3464, 3509])
print("Pixel at valley (1800, 2600):", mask[1800, 2600])
