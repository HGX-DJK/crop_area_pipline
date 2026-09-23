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
    
    # Morphological closing (5x5 = 150m) to fill internal neighborhood parks/lawns
    urban_closed = ndimage.binary_closing(urban_candidate, structure=np.ones((7, 7)))
    
    # Filter out small disconnected patches (keep only significant towns/settlements >= 200 pixels ≈ 18 ha)
    lbl, num = ndimage.label(urban_closed, structure=ndimage.generate_binary_structure(2, 2))
    counts = np.bincount(lbl.ravel())
    large_urban = np.zeros_like(urban_closed, dtype=bool)
    for i in range(1, num + 1):
        if counts[i] >= 200:
            large_urban[lbl == i] = True
            
    # Buffer by 3 pixels (90m) to completely wrap city borders and roadside tree belts
    final_urban_mask = ndimage.binary_dilation(large_urban, structure=np.ones((7, 7)))
    
    # Check coverage on Vacaville & Fairfield boxes:
    # Upper box (Vacaville): Row 1650 to 1950, Col 2850 to 3050
    # Lower box (Fairfield): Row 2100 to 2350, Col 2700 to 2900
    # Farmland: Row 1500 to 1900, Col 3100 to 3500
    vaca_box = final_urban_mask[1650:1950, 2850:3050]
    fair_box = final_urban_mask[2100:2350, 2700:2900]
    farm_box = final_urban_mask[1500:1900, 3100:3500]
    
    print("=== Verification of Urban Envelope Detection ===")
    print(f"Vacaville box coverage: {np.mean(vaca_box)*100:.1f}% ({np.sum(vaca_box)}/{vaca_box.size})")
    print(f"Fairfield box coverage: {np.mean(fair_box)*100:.1f}% ({np.sum(fair_box)}/{fair_box.size})")
    print(f"Farmland box coverage:  {np.mean(farm_box)*100:.1f}% ({np.sum(farm_box)}/{farm_box.size})")
