import rasterio
import numpy as np

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b1 = src.read(1).astype(np.float32)
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)
    b6 = src.read(6).astype(np.float32)

h, w = b4.shape

# Let's save a thumbnail visualization of True Color (B3, B2, B1) and False Color (B4, B3, B2)
# so we can see exactly where mountains, plains, bays, cities, and lakes are located!
import matplotlib.pyplot as plt

step = 10
tc = np.stack([b3[::step, ::step], b2[::step, ::step], b1[::step, ::step]], axis=-1)
tc = np.clip(tc / 1500.0, 0, 1)

fc = np.stack([b4[::step, ::step], b3[::step, ::step], b2[::step, ::step]], axis=-1)
fc = np.clip(fc / 2500.0, 0, 1)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
axes[0].imshow(tc)
axes[0].set_title("True Color (B3, B2, B1)")
axes[0].set_xlabel("Col / 10")
axes[0].set_ylabel("Row / 10")

axes[1].imshow(fc)
axes[1].set_title("False Color NIR (B4, B3, B2)")
axes[1].set_xlabel("Col / 10")
axes[1].set_ylabel("Row / 10")

plt.tight_layout()
plt.savefig("scratch/scene_overview.png", dpi=150)
print("Saved scratch/scene_overview.png")
