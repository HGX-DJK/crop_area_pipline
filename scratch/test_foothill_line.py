import rasterio
import numpy as np
import matplotlib.pyplot as plt

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b4 = src.read(4).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b2 = src.read(2).astype(np.float32)

step = 10
s_b4 = b4[::step, ::step]
s_b3 = b3[::step, ::step]
s_b2 = b2[::step, ::step]
fc = np.stack([s_b4/2500.0, s_b3/1500.0, s_b2/1500.0], axis=-1)
fc = np.clip(fc, 0, 1)

h, w, _ = fc.shape
rows = np.arange(h)
# Test foothill line: col = 255 + 0.05 * row
cols_line = 255 + 0.05 * rows

fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(fc)
ax.plot(cols_line, rows, color='yellow', linewidth=2.5, label='Foothill Boundary Line (Coast Range vs Sacramento Valley)')
ax.set_title("Foothill Demarcation Line on NIR False Color")
ax.legend(loc='lower left')

plt.tight_layout()
plt.savefig("scratch/foothill_alignment.png", dpi=150)
print("Saved scratch/foothill_alignment.png")
