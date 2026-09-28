import os
import glob
import pandas as pd
import numpy as np
import rasterio

print("=== 1. 地块属性统计审计 ===")
df_parcels = pd.read_csv("crop_area_pipeline/output/vectorized_parcels_attribute_table.csv")
print(f"总导出主力地块数: {len(df_parcels)}")
print(f"累计面积: {df_parcels['area_mu'].sum():.1f} 亩 ({df_parcels['area_mu'].sum() / 1500:.2f} 平方公里)")
print(f"面积统计 (亩):")
print(df_parcels["area_mu"].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.90, 0.95, 0.99]))

print("\n地块规模分档统计:")
print(df_parcels["area_tier"].value_counts())

print("\n农机适宜度分级统计:")
print(df_parcels["machinery_suitability"].value_counts())

print("\n=== 2. 分类栅格图 (crop_classification_map.tif) 审计 ===")
with rasterio.open("crop_area_pipeline/output/crop_classification_map.tif") as src:
    w = src.width
    h = src.height
    arr = src.read(1)
    total_pix = w * h
    crop_pix = np.sum(arr > 0)
    print(f"栅格尺寸: {h} 行 × {w} 列 (共 {total_pix:,} 像元)")
    print(f"有效耕地像元数: {crop_pix:,} 像元 (占比: {crop_pix / total_pix * 100:.2f}%)")
    print(f"总计像元面积: {crop_pix * 900 / 666.6667:.1f} 亩")

print("\n=== 3. 为什么还存在 > 3000 亩的超大地块？===")
oversized = df_parcels[df_parcels["area_mu"] > 3000]
print(f"> 3000 亩的地块数量: {len(oversized)}")
for idx, r in oversized.head(5).iterrows():
    print(f"  - {r['parcel_id']}: {r['area_mu']:.1f} 亩, 紧凑度={r['compactness']:.3f}, 周长={r['perimeter_m']:.1f}m, 中心=({r['center_lon']:.4f}, {r['center_lat']:.4f})")

print("\n=== 4. 联合国无偏评估报告审计 ===")
if os.path.exists("crop_area_pipeline/output/acreage_statistics_report.csv"):
    df_stat = pd.read_csv("crop_area_pipeline/output/acreage_statistics_report.csv")
    print(df_stat)
