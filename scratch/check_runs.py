import rasterio
import numpy as np

with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)

# Check the mask at row 500, from col 2500 to 2800
row_slice = mask[500, 2500:2800]
print("Row 500 values from col 2500 to 2800 (summary of runs):")
# Find runs of 0s and 1s
runs = []
cur_val = row_slice[0]
cur_len = 0
for v in row_slice:
    if v == cur_val:
        cur_len += 1
    else:
        runs.append((cur_val, cur_len))
        cur_val = v
        cur_len = 1
runs.append((cur_val, cur_len))

print("Runs of (value, length):", runs)
