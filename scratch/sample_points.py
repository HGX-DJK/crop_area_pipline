import rasterio
import numpy as np

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    points = {
        'Vacaville Downtown': (1806, 2948),
        'Vacaville Residential N': (1770, 2940),
        'Vacaville Residential S': (1840, 2960),
        'Vacaville I-80 Commercial': (1820, 2920),
        'Fairfield Downtown': (2204, 2800),
        'Fairfield Residential W': (2180, 2750),
        'Fairfield Industrial': (2250, 2830),
        'Farmland Dixon Active': (1800, 3200),
        'Farmland Dixon Fallow/Bare': (1850, 3300),
        'Farmland Winters Orchard': (1500, 3100),
        'Farmland Yolo Wheat': (1600, 3300),
        'Suisun Marsh Wetland': (2450, 2900),
    }
    
    print(f"{'Location':<26} | {'B1':>5} {'B2':>5} {'B3':>5} {'B4':>5} {'B5':>5} {'B6':>5} | {'NDVI':>6} {'NDBI':>6} {'B5/B3':>6} {'B1/B3':>6}")
    print("-" * 85)
    for name, (r, c) in points.items():
        vals = [src.read(b, window=rasterio.windows.Window(c, r, 1, 1))[0, 0] for b in range(1, 7)]
        b1, b2, b3, b4, b5, b6 = [float(v) for v in vals]
        ndvi = (b4 - b3) / max(b4 + b3, 1e-4)
        ndbi = (b5 - b4) / max(b5 + b4, 1e-4)
        b5_b3 = b5 / max(b3, 1.0)
        b1_b3 = b1 / max(b3, 1.0)
        print(f"{name:<26} | {b1:5.0f} {b2:5.0f} {b3:5.0f} {b4:5.0f} {b5:5.0f} {b6:5.0f} | {ndvi:6.3f} {ndbi:6.3f} {b5_b3:6.2f} {b1_b3:6.2f}")
