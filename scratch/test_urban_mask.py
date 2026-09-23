import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    vaca_r, vaca_c = 1806, 2948
    fair_r, fair_c = 2204, 2800
    dixon_r, dixon_c = 1800, 3200
    
    # Read a 1200x1200 region covering Vacaville, Fairfield, Dixon
    win = rasterio.windows.Window(2400, 1400, 1200, 1200)
    b1 = src.read(1, window=win).astype(np.float32)
    b2 = src.read(2, window=win).astype(np.float32)
    b3 = src.read(3, window=win).astype(np.float32)
    b4 = src.read(4, window=win).astype(np.float32)
    b5 = src.read(5, window=win).astype(np.float32)
    b6 = src.read(6, window=win).astype(np.float32)
    
    ndvi = (b4 - b3) / np.maximum(b4 + b3, 1e-4)
    ndbi = (b5 - b4) / np.maximum(b5 + b4, 1e-4)
    b5_b3 = b5 / np.maximum(b3, 1.0)
    
    # Step 1: Detect core impervious pixels (buildings, roads, concrete, commercial, asphalt)
    # Bright surfaces (roofs, concrete): B1 > 1000 or (B1 > 500 & B5_B3 < 2.0)
    # Standard impervious: NDBI > -0.04 & NDVI < 0.35 & B5_B3 < 2.2
    # Industrial/commercial: NDBI > 0.0 & B5_B3 < 2.2
    core_impervious = (
        (b1 > 1100.0) |
        ((b1 > 500.0) & (b5_b3 < 2.0) & (ndvi < 0.40)) |
        ((ndbi > -0.03) & (ndvi < 0.32) & (b5_b3 < 2.2)) |
        ((ndbi > 0.02) & (b5_b3 < 2.0))
    )
    
    # Step 2: Urban cluster expansion using morphological closing / dilation
    # A residential block is 100m - 300m wide (3-10 pixels @ 30m)
    # Use disk/rect structure of size ~11 (approx 330m)
    struct = ndimage.generate_binary_structure(2, 2)
    # Closing bridges between buildings and streets across residential lawns
    closed_urban = ndimage.binary_closing(core_impervious, structure=np.ones((9, 9)))
    
    # Require connected urban components to have at least a certain size (e.g. > 50 pixels = ~45000 m² = 4.5 ha)
    # to avoid flagging lone farmhouses or small rural sheds
    lbl, num = ndimage.label(closed_urban, structure=struct)
    counts = np.bincount(lbl.ravel())
    large_urban = np.zeros_like(closed_urban)
    for i in range(1, num + 1):
        if counts[i] > 60:  # > 5.4 ha
            large_urban[lbl == i] = True
            
    # Buffer the large urban clusters slightly (by 2 pixels = 60m) to fully cover suburban edges
    urban_envelope = ndimage.binary_dilation(large_urban, structure=np.ones((5, 5)))
    
    # Let's inspect coverage:
    # Vacaville center in window: r = 1806 - 1400 = 406, c = 2948 - 2400 = 548
    # Fairfield center in window: r = 2204 - 1400 = 804, c = 2800 - 2400 = 400
    # Dixon farmland in window:   r = 1800 - 1400 = 400, c = 3200 - 2400 = 800
    
    def check_region(r, c, name, size=80):
        sub = urban_envelope[r-size//2:r+size//2, c-size//2:c+size//2]
        print(f"{name:<20} urban envelope coverage: {np.mean(sub)*100:.1f}% ({np.sum(sub)}/{sub.size})")

    print("=== Testing Urban Mask Formulation ===")
    check_region(406, 548, "Vacaville Center")
    check_region(804, 400, "Fairfield Center")
    check_region(400, 800, "Dixon Farmland")
    check_region(100, 700, "Winters Farmland")
