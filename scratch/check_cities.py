import rasterio
import numpy as np

# Load classification result if exists in output
import os
for fname in ['output/vectorized_parcels.geojson', 'output/crop_classification_mask.tif']:
    if os.path.exists(fname):
        print(f"Found {fname}")

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    vaca_r, vaca_c = 1806, 2948
    fair_r, fair_c = 2204, 2800
    
    # Check current urban mask logic in raster_loader:
    # is_urban = (ndvi < 0.22) & (b5_b3_ratio < 1.75) & (b5 < 1950.0)
    for name, r, c in [('Vacaville', vaca_r, vaca_c), ('Fairfield', fair_r, fair_c)]:
        win = rasterio.windows.Window(c - 75, r - 75, 150, 150)
        b2 = src.read(2, window=win).astype(np.float32)
        b3 = src.read(3, window=win).astype(np.float32)
        b4 = src.read(4, window=win).astype(np.float32)
        b5 = src.read(5, window=win).astype(np.float32)
        
        ndvi = (b4 - b3) / np.maximum(b4 + b3, 1e-4)
        ndbi = (b5 - b4) / np.maximum(b5 + b4, 1e-4)
        b5_b3 = b5 / np.maximum(b3, 1.0)
        
        old_is_urban = (ndvi < 0.22) & (b5_b3 < 1.75) & (b5 < 1950.0)
        print(f"\n{name} (150x150 window, 22500 pixels):")
        print(f"  Old is_urban detected: {np.sum(old_is_urban)} / {old_is_urban.size} ({np.sum(old_is_urban)/old_is_urban.size*100:.1f}%)")
        print(f"  Pixels with ndvi >= 0.22: {np.sum(ndvi >= 0.22)} ({np.sum(ndvi >= 0.22)/old_is_urban.size*100:.1f}%)")
        print(f"  NDBI > -0.05: {np.sum(ndbi > -0.05)} ({np.sum(ndbi > -0.05)/old_is_urban.size*100:.1f}%)")
        print(f"  NDBI > 0.00: {np.sum(ndbi > 0.00)} ({np.sum(ndbi > 0.00)/old_is_urban.size*100:.1f}%)")
