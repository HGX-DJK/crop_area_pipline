import rasterio
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b4 = src.read(4).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

step = 10
s_b4 = b4[::step, ::step]
s_b3 = b3[::step, ::step]
s_b5 = b5[::step, ::step]
ndvi = (s_b4 - s_b3) / np.maximum(s_b4 + s_b3, 1e-4)

# Topographic roughness at regional scale (~600m - 1.5km)
# On downsampled (300m pixels), size=5 is 1.5km
mean_b4 = ndimage.uniform_filter(s_b4, size=5)
sq_b4 = ndimage.uniform_filter(s_b4**2, size=5)
cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

# Rugged mountain terrain: cv_b4 > 0.18
is_rugged = cv_b4 > 0.18

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
axes[0].imshow(s_b4, cmap='gray')
axes[0].set_title("NIR B4")

axes[1].imshow(cv_b4, cmap='inferno', vmin=0, vmax=0.4)
axes[1].set_title("Regional Topographic CV")

axes[2].imshow(is_rugged, cmap='coolwarm')
axes[2].set_title("Rugged Mountain Mask (CV > 0.18)")

plt.tight_layout()
plt.savefig("scratch/topographic_mask.png", dpi=150)
print("Saved scratch/topographic_mask.png")
