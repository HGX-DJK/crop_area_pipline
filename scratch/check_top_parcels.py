import glob
import rasterio
import numpy as np
import pandas as pd

df = pd.read_csv("crop_area_pipeline/output/vectorized_parcels_attribute_table.csv")
files = sorted(glob.glob("crop_area_pipeline/data/satellite_tifs/*.tif"))

with rasterio.open(files[0]) as ref:
    inv_t = ~ref.transform

for i in range(5):
    row_data = df.iloc[i]
    pid = row_data["parcel_id"]
    x, y = row_data["center_utm_x"], row_data["center_utm_y"]
    c, r = inv_t * (x, y)
    r, c = int(r), int(c)
    
    ndvis = []
    for f in files:
        with rasterio.open(f) as src:
            b3 = src.read(3, window=((r-2, r+3), (c-2, c+3))).astype(np.float32)
            b4 = src.read(4, window=((r-2, r+3), (c-2, c+3))).astype(np.float32)
            nd = (b4 - b3) / (b4 + b3 + 1e-6)
            ndvis.append(round(float(np.mean(nd)), 3))
    print(f"Parcel {pid} at (r={r}, c={c}), area={row_data['area_mu']:.1f}亩, NDVIs across 4 dates: {ndvis}, min={min(ndvis)}, max={max(ndvis)}, range={max(ndvis)-min(ndvis):.3f}")
