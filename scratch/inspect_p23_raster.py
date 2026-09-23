import rasterio
import numpy as np

# Load crop_classification_map.tif
with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)
    
from pyproj import Transformer
transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)

# For P0023: bounds=(-122.0729, 38.6674, -122.0666, 38.7545)
x1, y1 = transformer.transform(-122.0729, 38.6674)
x2, y2 = transformer.transform(-122.0666, 38.7545)
r1, c1 = src.index(x1, y1)
r2, c2 = src.index(x2, y2)
print(f"P0023 in raster rows: {r2}..{r1}, cols: {c1}..{c2}")
# Width in cols:
print(f"Col span: {c2 - c1}, Row span: {r1 - r2}")

# Now let's see what the mask looks like in this rectangle:
sub_crop = mask[r2:r1, c1:c2]
print(f"Mask shape: {sub_crop.shape}")
print(f"Total pixels: {sub_crop.size}, Cropland pixels: {np.sum(sub_crop == 1)}")

# Check columns of sub_crop:
print("Row sums across columns (how many 1s in each column):")
print(np.sum(sub_crop == 1, axis=0))
