import json
import numpy as np
import pandas as pd
import rasterio

# 1. 检查属性表
df = pd.read_csv("crop_area_pipeline/output/vectorized_parcels_attribute_table.csv")
print("=== Parcels Summary ===")
print("Total parcels:", len(df))
print("Total area (mu):", df["area_mu"].sum())
print("Mean area:", df["area_mu"].mean())
print("Area quartiles:")
print(df["area_mu"].describe())

# 2. 检查遥感影像真实像元值分布
sorted_files = [
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000621.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20001015.tif"
]

with rasterio.open("crop_area_pipeline/output/crop_classification_map.tif") as src:
    crop_mask = src.read(1)
    print("\n=== Classification Map ===")
    print("Crop mask shape:", crop_mask.shape)
    print("Cropland pixels (1):", np.sum(crop_mask == 1), f"({np.sum(crop_mask == 1) / crop_mask.size * 100:.2f}%)")
    print("Cropland area (mu, 30m):", np.sum(crop_mask == 1) * 900.0 / 666.6667)

# 3. 检查 P0001 ~ P0005 的地理坐标对应的光谱值
with rasterio.open(sorted_files[0]) as s0, rasterio.open(sorted_files[1]) as s1, rasterio.open(sorted_files[2]) as s2:
    print("\n=== Inspect Top 5 Parcels Center Spectra ===")
    for i in range(min(5, len(df))):
        row = df.iloc[i]
        lon, lat = row["center_lon"], row["center_lat"]
        # convert lon/lat to row, col
        # The GeoTIFF CRS:
        py_r, px_c = s0.index(row["center_utm_x"], row["center_utm_y"])
        print(f"Parcel {row['parcel_id']}: ({lon:.4f}, {lat:.4f}), pixel (r={py_r}, c={px_c}), area={row['area_mu']:.1f} 亩")
        
        # Read 3 dates B1~B5
        b_d0 = s0.read(window=rasterio.windows.Window(px_c, py_r, 1, 1)).flatten().astype(float)
        b_d1 = s1.read(window=rasterio.windows.Window(px_c, py_r, 1, 1)).flatten().astype(float)
        b_d2 = s2.read(window=rasterio.windows.Window(px_c, py_r, 1, 1)).flatten().astype(float)
        
        # SDC bands: 1:Blue, 2:Green, 3:Red, 4:NIR, 5:SWIR1, 6:SWIR2
        def get_indices(b):
            if len(b) < 5 or b[3] + b[2] == 0: return 0, 0, 0
            ndvi = (b[3] - b[2]) / (b[3] + b[2])
            lswi = (b[3] - b[4]) / (b[3] + b[4])
            gcvi = (b[3] / max(b[1], 1.0)) - 1.0
            return ndvi, lswi, gcvi
            
        ndvi0, lswi0, gcvi0 = get_indices(b_d0)
        ndvi1, lswi1, gcvi1 = get_indices(b_d1)
        ndvi2, lswi2, gcvi2 = get_indices(b_d2)
        print(f"  NDVI: [{ndvi0:.3f}, {ndvi1:.3f}, {ndvi2:.3f}]")
        print(f"  LSWI: [{lswi0:.3f}, {lswi1:.3f}, {lswi2:.3f}]")
        print(f"  GCVI: [{gcvi0:.3f}, {gcvi1:.3f}, {gcvi2:.3f}]")
        print(f"  NIR (B4): [{b_d0[3]:.0f}, {b_d1[3]:.0f}, {b_d2[3]:.0f}]")
        print(f"  Red (B3): [{b_d0[2]:.0f}, {b_d1[2]:.0f}, {b_d2[2]:.0f}]")
