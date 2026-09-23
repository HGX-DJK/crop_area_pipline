import rasterio
import numpy as np
from pyproj import Transformer

with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)
    
    # Lat/Lon to row/col
    transformer = Transformer.from_crs('EPSG:4326', src.crs, always_xy=True)
    x, y = transformer.transform(-122.08, 38.65)
    r, c = src.index(x, y)
    print(f"(-122.08, 38.65) is at Row={r}, Col={c}")
    
    # Check a 100x100 patch around this point in crop_classification_map.tif
    patch = mask[r-50:r+50, c-50:c+50]
    print("Crop classification mask in this area (100x100):")
    print(f"  Crop pixels: {np.sum(patch == 1)} / {patch.size} ({np.mean(patch==1)*100:.1f}%)")
    
    # Print the patch as ASCII art (0 as '.', 1 as '#')
    for row_idx in range(0, 100, 5):
        line = "".join(['#' if patch[row_idx, col_idx] == 1 else '.' for col_idx in range(0, 100, 2)])
        print(line)
