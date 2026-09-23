import rasterio
import numpy as np

with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)
    print("Crop classification mask shape:", mask.shape)
    print("Unique values:", np.unique(mask, return_counts=True))
    
    # Check if there are vertical stripes in crop_classification_map.tif:
    # A vertical stripe pattern means column sums or row-by-row differences
    col_sums = np.sum(mask == 1, axis=0)
    print("Nonzero columns count:", np.sum(col_sums > 0))
    
    # Check around the mountain area on the west (Cols 0 to 2500, Rows 1000 to 2500)
    west_crop = mask[1000:2500, 0:2500]
    print("West mountain area crop pixels count:", np.sum(west_crop == 1))
    
    # Check a 200x200 patch in the west mountain area
    patch = mask[1500:1700, 1000:1200]
    print("Patch shape:", patch.shape, "Crop pixels:", np.sum(patch == 1))
    # Print the sum across rows (horizontal profile)
    print("Column profile of patch (sum over rows for each col):", np.sum(patch == 1, axis=0)[:30])
