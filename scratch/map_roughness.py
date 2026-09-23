import rasterio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b4 = src.read(4).astype(np.float32)

step = 10
sub_b4 = b4[::step, ::step]

# Test multi-scale roughness
mean_b4 = ndimage.uniform_filter(sub_b4, size=7)
sq_b4 = ndimage.uniform_filter(sub_b4**2, size=7)
cv = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
im1 = axes[0].imshow(sub_b4, cmap='gray')
axes[0].set_title("NIR Band 4 (Downsampled 10x)")
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(cv, cmap='inferno', vmin=0, vmax=0.4)
axes[1].set_title("Texture Roughness CV(B4)")
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig("scratch/roughness_map.png", dpi=150)
print("Saved scratch/roughness_map.png")
