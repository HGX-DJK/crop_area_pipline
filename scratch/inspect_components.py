import rasterio
import numpy as np
from scipy import ndimage

with rasterio.open('output/crop_classification_map.tif') as src:
    mask = src.read(1)

cropland_binary = (mask > 0).astype(np.uint8)
structure_cross = ndimage.generate_binary_structure(2, 1)
cleaned = ndimage.binary_opening(cropland_binary, structure=structure_cross).astype(np.uint8)
labeled_array, num_features = ndimage.label(cleaned, structure=structure_cross)

counts = np.bincount(labeled_array.ravel())
oversized = np.where(counts[1:] * 900.0 > 800000.0)[0] + 1
print(f"Number of oversized components: {len(oversized)}")
slices = ndimage.find_objects(labeled_array)

for i, cid in enumerate(oversized[:10]):
    sl = slices[cid - 1]
    sub = labeled_array[sl] == cid
    print(f"Component #{cid}: shape={sub.shape}, pixels={np.sum(sub)}, area_mu={np.sum(sub)*900/666.67:.1f}, "
          f"rows={sl[0].start}..{sl[0].stop}, cols={sl[1].start}..{sl[1].stop}")
