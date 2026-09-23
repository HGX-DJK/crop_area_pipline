import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    vaca_r, vaca_c = 1806, 2948
    fair_r, fair_c = 2204, 2800
    farm_r, farm_c = 1800, 3200
    
    # Read a larger region covering Vacaville, Fairfield, and Farmland
    # Window from row 1500 to 2500, col 2500 to 3500 (1000x1000)
    win = rasterio.windows.Window(2500, 1500, 1000, 1000)
    b1 = src.read(1, window=win).astype(np.float32)
    b2 = src.read(2, window=win).astype(np.float32)
    b3 = src.read(3, window=win).astype(np.float32)
    b4 = src.read(4, window=win).astype(np.float32)
    b5 = src.read(5, window=win).astype(np.float32)
    b6 = src.read(6, window=win).astype(np.float32)
    
    ndvi = (b4 - b3) / np.maximum(b4 + b3, 1e-4)
    ndbi = (b5 - b4) / np.maximum(b5 + b4, 1e-4)
    mndwi = (b2 - b5) / np.maximum(b2 + b5, 1e-4)
    b5_b3 = b5 / np.maximum(b3, 1.0)
    
    # Pure impervious surface (buildings, concrete, asphalt roads, commercial roofs):
    # High blue (B1 > 460) or NDBI > 0 or (B5/B3 < 1.8 and NDVI < 0.25)
    # Let's inspect impervious indicators
    impervious_core = (
        ((ndbi > -0.02) & (ndvi < 0.35)) |
        ((b1 > 480.0) & (b5_b3 < 2.2) & (ndvi < 0.40)) |
        ((b5_b3 < 1.7) & (ndvi < 0.30)) |
        ((b1 > 520.0) & (ndvi < 0.35))
    )
    
    # Now check impervious density in a 15x15 window (450m x 450m neighborhood, typical urban block)
    urban_density = ndimage.uniform_filter(impervious_core.astype(np.float32), size=15)
    
    # Check at Vacaville, Fairfield, Farmland
    # Vacaville in this window: r = 1806 - 1500 = 306, c = 2948 - 2500 = 448
    # Fairfield in this window: r = 2204 - 1500 = 704, c = 2800 - 2500 = 300
    # Farmland in this window:  r = 1800 - 1500 = 300, c = 3200 - 2500 = 700
    
    print("Vacaville urban density (100x100 box around center):",
          np.mean(urban_density[306-50:306+50, 448-50:448+50]))
    print("Fairfield urban density (100x100 box around center):",
          np.mean(urban_density[704-50:704+50, 300-50:300+50]))
    print("Farmland urban density (100x100 box around center):",
          np.mean(urban_density[300-50:300+50, 700-50:700+50]))
    print("Farmland max urban density in box:",
          np.max(urban_density[300-50:300+50, 700-50:700+50]))
