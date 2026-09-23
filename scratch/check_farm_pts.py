import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b1 = src.read(1).astype(np.float32)
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)
    
    ndvi = (b4 - b3) / np.maximum(b4 + b3, 1e-4)
    ndbi = (b5 - b4) / np.maximum(b5 + b4, 1e-4)
    b5_b3 = b5 / np.maximum(b3, 1.0)
    
    # 1. Core impervious surfaces
    impervious_core = (
        (b1 > 1000.0) |
        ((b1 > 480.0) & (b5_b3 < 1.85) & (ndvi < 0.40)) |
        ((ndbi > -0.02) & (b5_b3 < 1.75)) |
        ((b5_b3 < 1.55) & (ndvi < 0.35))
    )
    
    # 2. Density filter: 21x21 window (~630m x 630m)
    dens = ndimage.uniform_filter(impervious_core.astype(np.float32), size=21)
    
    # Candidate urban zone where impervious density is >= 0.12 (at least 12% buildings/streets)
    urban_candidate = dens >= 0.12
    
    # Morphological closing (7x7 = 210m) to fill internal neighborhood parks/lawns
    urban_closed = ndimage.binary_closing(urban_candidate, structure=np.ones((7, 7)))
    
    # Filter out small disconnected patches (keep only significant towns/settlements >= 200 pixels ≈ 18 ha)
    lbl, num = ndimage.label(urban_closed, structure=ndimage.generate_binary_structure(2, 2))
    counts = np.bincount(lbl.ravel())
    large_urban = np.zeros_like(urban_closed, dtype=bool)
    for i in range(1, num + 1):
        if counts[i] >= 200:
            large_urban[lbl == i] = True
            
    final_urban_mask = ndimage.binary_dilation(large_urban, structure=np.ones((7, 7)))

farm_pts = [
    (1800, 3200, "Dixon Open Farmland"),
    (1500, 3100, "Winters Orchard Farmland"),
    (1600, 3300, "Yolo Plain Farmland"),
    (1700, 3400, "South Yolo Farmland"),
    (1850, 3350, "Solano Open Farmland"),
    (1630, 3330, "Dixon Town Center"),
    (1806, 2948, "Vacaville Downtown"),
    (2204, 2800, "Fairfield Downtown"),
]

for r, c, name in farm_pts:
    print(f"{name:<24}: Urban mask = {final_urban_mask[r, c]}")
