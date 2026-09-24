import sys, os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import rasterio, numpy as np
import yaml
from scipy import ndimage
import cv2

with open("crop_area_pipeline/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

with rasterio.open("crop_area_pipeline/output/crop_classification_map.tif") as src:
    crop_mask = src.read(1)

total_crop_pixels = np.sum(crop_mask == 1)
total_crop_mu = total_crop_pixels * 900.0 / 666.6667
print(f"Classification Map Cropland: {total_crop_pixels:,} pixels ({total_crop_mu:,.1f} 亩)")

sorted_files = [
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000621.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20001015.tif"
]

# 在 6月 (夏季生长高峰) 影像上提取真正的农田机耕路与水渠网络
with rasterio.open(sorted_files[1]) as s1:
    b3 = s1.read(3).astype(np.float32)
    b4 = s1.read(4).astype(np.float32)

denom = b4 + b3
ndvi_june = np.zeros_like(b3)
v = denom > 0
ndvi_june[v] = (b4[v] - b3[v]) / denom[v]

gx = cv2.Sobel(ndvi_june, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(ndvi_june, cv2.CV_32F, 0, 1, ksize=3)
mag_ndvi = np.sqrt(gx**2 + gy**2)

# 真实机耕路/水渠：田块与道路边缘交界处有清晰突变 (梯度 > 0.20)，且道路本身植被低 (NDVI < 0.55)
# 采用细线边缘（不膨胀），仅切开相邻不同农田，不蚕食田块内部
true_roads = (mag_ndvi > 0.22) & (ndvi_june < 0.55) & (crop_mask > 0)
print(f"Detected true field roads/canals: {np.sum(true_roads):,} pixels ({np.sum(true_roads)*900/666.6667:,.1f} 亩)")

cropland_binary = (crop_mask > 0).astype(np.uint8)
cropland_binary[true_roads] = 0

# 连通域标记
lbl, num_parcels = ndimage.label(cropland_binary, structure=ndimage.generate_binary_structure(2, 2))
counts = np.bincount(lbl.ravel())[1:]
areas_mu = counts * 900.0 / 666.6667

# 过滤最小合法地块 (>= 6 亩，4000 m2)
valid_mask = areas_mu >= 6.0
valid_parcels = np.sum(valid_mask)
valid_area_mu = np.sum(areas_mu[valid_mask])

print(f"\n=== Optimized Road-Carving Results ===")
print(f"Total valid parcels (>= 6 亩): {valid_parcels:,}")
print(f"Total parcel area: {valid_area_mu:,.1f} 亩 (占分类掩膜 {valid_area_mu / total_crop_mu * 100:.1f}%)")
print(f"Average parcel size: {valid_area_mu / valid_parcels:.1f} 亩")

# 查看前 10 大地块
top10_idx = np.argsort(areas_mu[valid_mask])[::-1][:10]
top10_areas = areas_mu[valid_mask][top10_idx]
print(f"Top 10 parcel sizes (亩): {[round(x, 1) for x in top10_areas]}")
