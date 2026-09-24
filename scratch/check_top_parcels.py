import numpy as np, rasterio
from scipy import ndimage

with rasterio.open("crop_area_pipeline/output/crop_classification_map.tif") as src:
    crop_mask = src.read(1)

lbl, n = ndimage.label(crop_mask > 0, structure=ndimage.generate_binary_structure(2, 2))
cnt = np.bincount(lbl.ravel())[1:]
areas = cnt * 900.0 / 666.6667
valid = areas >= 6.0
sorted_areas = np.sort(areas[valid])[::-1]
print(f"Total valid parcels: {len(sorted_areas):,}")
print(f"Top 3000 area: {np.sum(sorted_areas[:3000]):,.1f} 亩 ({np.sum(sorted_areas[:3000])/np.sum(sorted_areas)*100:.1f}%)")
print(f"Top 5000 area: {np.sum(sorted_areas[:5000]):,.1f} 亩 ({np.sum(sorted_areas[:5000])/np.sum(sorted_areas)*100:.1f}%)")
print(f"All valid area: {np.sum(sorted_areas):,.1f} 亩")
