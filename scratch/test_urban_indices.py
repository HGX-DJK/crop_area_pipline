import rasterio
import numpy as np

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    points = {
        'Vacaville Downtown': (1806, 2948),
        'Vacaville Comm I-80': (1820, 2920),
        'Vacaville Res S': (1840, 2960),
        'Vacaville Res N (lawn)': (1770, 2940),
        'Fairfield Downtown': (2204, 2800),
        'Fairfield Industrial': (2250, 2830),
        'Dixon Active Crop': (1800, 3200),
        'Dixon Fallow Field': (1850, 3300),
        'Winters Orchard 1': (1500, 3100),
        'Winters Orchard 2': (1450, 3150),
        'Yolo Wheat Field': (1600, 3300),
    }
    
    print(f"{'Location':<24} | {'NDVI':>6} {'NDBI':>6} {'NDBI-NDVI':>9} {'UI(B6,B4)':>10} {'B5/B4':>6} {'B5/B3':>6} {'B6/B4':>6}")
    print("-" * 85)
    for name, (r, c) in points.items():
        vals = [src.read(b, window=rasterio.windows.Window(c, r, 1, 1))[0, 0] for b in range(1, 7)]
        b1, b2, b3, b4, b5, b6 = [float(v) for v in vals]
        ndvi = (b4 - b3) / max(b4 + b3, 1e-4)
        ndbi = (b5 - b4) / max(b5 + b4, 1e-4)
        ui = (b6 - b4) / max(b6 + b4, 1e-4)
        b5_b4 = b5 / max(b4, 1.0)
        b5_b3 = b5 / max(b3, 1.0)
        b6_b4 = b6 / max(b4, 1.0)
        print(f"{name:<24} | {ndvi:6.3f} {ndbi:6.3f} {ndbi-ndvi:9.3f} {ui:10.3f} {b5_b4:6.2f} {b5_b3:6.2f} {b6_b4:6.2f}")
