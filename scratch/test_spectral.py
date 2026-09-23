import rasterio
import numpy as np

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    vaca_r, vaca_c = 1806, 2948
    fair_r, fair_c = 2204, 2800
    # True farmland east of Vacaville (towards Dixon): row=1800, col=3200
    farm_r, farm_c = 1800, 3200
    
    def get_stats(r, c, name, size=80):
        win = rasterio.windows.Window(c - size//2, r - size//2, size, size)
        b1 = src.read(1, window=win).astype(np.float32)
        b2 = src.read(2, window=win).astype(np.float32)
        b3 = src.read(3, window=win).astype(np.float32)
        b4 = src.read(4, window=win).astype(np.float32)
        b5 = src.read(5, window=win).astype(np.float32)
        b6 = src.read(6, window=win).astype(np.float32)
        
        ndvi = (b4 - b3) / np.maximum(b4 + b3, 1e-4)
        ndbi = (b5 - b4) / np.maximum(b5 + b4, 1e-4) # Built-up index
        mndwi = (b2 - b5) / np.maximum(b2 + b5, 1e-4)
        b5_b4 = b5 / np.maximum(b4, 1.0)
        b5_b3 = b5 / np.maximum(b3, 1.0)
        
        print(f"=== {name} (size {size}x{size}) ===")
        print(f"  B1 (Blue):    mean={np.mean(b1):.1f}")
        print(f"  B2 (Green):   mean={np.mean(b2):.1f}")
        print(f"  B3 (Red):     mean={np.mean(b3):.1f}")
        print(f"  B4 (NIR):     mean={np.mean(b4):.1f}")
        print(f"  B5 (SWIR1):   mean={np.mean(b5):.1f}")
        print(f"  B6 (SWIR2):   mean={np.mean(b6):.1f}")
        print(f"  NDVI:         mean={np.mean(ndvi):.3f}, 10-90%=[{np.percentile(ndvi, 10):.3f}, {np.percentile(ndvi, 90):.3f}]")
        print(f"  NDBI (SWIR1-NIR)/(SWIR1+NIR): mean={np.mean(ndbi):.3f}, 10-90%=[{np.percentile(ndbi, 10):.3f}, {np.percentile(ndbi, 90):.3f}]")
        print(f"  NDBI > -0.05: {np.mean(ndbi > -0.05)*100:.1f}%")
        print(f"  NDBI > -0.10: {np.mean(ndbi > -0.10)*100:.1f}%")
        print(f"  B5/B4:        mean={np.mean(b5_b4):.3f}, 10-90%=[{np.percentile(b5_b4, 10):.3f}, {np.percentile(b5_b4, 90):.3f}]")
        print(f"  B5/B3:        mean={np.mean(b5_b3):.3f}, 10-90%=[{np.percentile(b5_b3, 10):.3f}, {np.percentile(b5_b3, 90):.3f}]")

    get_stats(vaca_r, vaca_c, "Vacaville City")
    get_stats(fair_r, fair_c, "Fairfield City")
    get_stats(farm_r, farm_c, "Dixon Farmland")
    # Also check orchard / bare cropland
    get_stats(1500, 3100, "Winters / Solano Farmland")
