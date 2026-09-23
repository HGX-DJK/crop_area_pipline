import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    print(f"Reading bands from {src.name}...")
    b1 = src.read(1).astype(np.float32)
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)
    b6 = src.read(6).astype(np.float32)
    h, w = b1.shape

    ndvi = (b4 - b3) / np.maximum(b4 + b3, 1e-4)
    ndbi = (b5 - b4) / np.maximum(b5 + b4, 1e-4)
    mndwi = (b2 - b5) / np.maximum(b2 + b5, 1e-4)
    b5_b3 = b5 / np.maximum(b3, 1.0)
    
    # Let's inspect impervious indicators across the scene:
    # 1. High-confidence built-up impervious pixels (rooftops, concrete, asphalt roads, runways, commercial):
    # - High Blue reflectance: concrete / metal roofs / commercial buildings have B1 > 600 or (B1 > 450 & B5_B3 < 1.9)
    # - NDBI > -0.02 with flat spectral slope (B5_B3 < 1.85)
    # - Urban flat spectrum: B5_B3 < 1.65 (agricultural soil is > 2.0 because of mineral absorption)
    impervious_core = (
        (b1 > 1000.0) |
        ((b1 > 480.0) & (b5_b3 < 1.85) & (ndvi < 0.40)) |
        ((ndbi > -0.02) & (b5_b3 < 1.75)) |
        ((b5_b3 < 1.55) & (ndvi < 0.35) & (mndwi < -0.05))
    )
    
    print(f"Total impervious core pixels: {np.sum(impervious_core)} ({np.sum(impervious_core)/(h*w)*100:.2f}%)")
    
    # 2. To cover residential areas (lawns, yards, urban street trees):
    # We use spatial density of impervious core.
    # An urban neighborhood (residential, commercial, industrial) has a dense grid of impervious surfaces (streets + houses).
    # In a 15x15 window (450m x 450m), what is the impervious density in Vacaville/Fairfield vs Farmland?
    dens = ndimage.uniform_filter(impervious_core.astype(np.float32), size=17)
    
    # Check densities at key spots:
    spots = {
        'Vacaville Downtown': (1806, 2948),
        'Vacaville North Res': (1770, 2940),
        'Vacaville South Res': (1840, 2960),
        'Fairfield Downtown': (2204, 2800),
        'Fairfield West Res': (2180, 2750),
        'Fairfield Industrial': (2250, 2830),
        'Travis AFB Runway': (2100, 2950),
        'Dixon Town Center': (1630, 3330),
        'Dixon Farmland': (1800, 3200),
        'Winters Farmland': (1500, 3100),
        'Yolo Farmland': (1600, 3300),
        'Davis Farmland': (1300, 3400),
    }
    
    print("\nImpervious density in 17x17 window:")
    for name, (r, c) in spots.items():
        print(f"  {name:<22}: density = {dens[r, c]:.3f}, core = {impervious_core[r, c]}")
