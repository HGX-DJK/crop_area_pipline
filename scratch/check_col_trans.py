import rasterio
import numpy as np

with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)

for r in [350, 400, 450, 500, 550, 600, 650]:
    row_data = mask[r, 2500:2850]
    ones_idx = np.where(row_data == 1)[0]
    if len(ones_idx) > 0:
        c_min = 2500 + ones_idx[0]
        c_max = 2500 + ones_idx[-1]
        print(f"Row {r}: 1s start at col {c_min}, end at col {c_max} (width = {c_max - c_min + 1})")
    else:
        print(f"Row {r}: no 1s")
