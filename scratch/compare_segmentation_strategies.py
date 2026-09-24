import sys, os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import rasterio, numpy as np
import yaml
from scipy import ndimage

with open("crop_area_pipeline/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

with rasterio.open("crop_area_pipeline/output/crop_classification_map.tif") as src:
    crop_mask = src.read(1)

total_crop_pixels = np.sum(crop_mask == 1)
total_crop_mu = total_crop_pixels * 900.0 / 666.6667
print(f"Total crop pixels in classification map: {total_crop_pixels:,} ({total_crop_mu:,.1f} 亩)")

# 比较不同的分割方案：
# 方案 A: 当前默认 (包含 80% 分位数边缘雕刻 + binary_opening)
# 方案 B: 仅使用结构开运算 (不使用粗暴的 80% 边缘雕刻)
# 方案 C: 保持真实边界完整性 (直接基于连通域 + 超大田块智能细分，保留真实田面面积)

# 方案 A:
from src.raster_loader import RasterLoader
sorted_files = [
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000621.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20001015.tif"
]
loader = RasterLoader(config)
edge_mask_curr = loader.compute_spectral_edge_mask(sorted_files[0], crop_mask, percentile_thresh=80.0)

binary_curr = (crop_mask > 0).astype(np.uint8)
binary_curr[edge_mask_curr] = 0
struct_cross = ndimage.generate_binary_structure(2, 1)
cleaned_curr = ndimage.binary_opening(binary_curr, structure=struct_cross)
lbl_curr, n_curr = ndimage.label(cleaned_curr, structure=ndimage.generate_binary_structure(2, 2))
cnt_curr = np.bincount(lbl_curr.ravel())[1:]
valid_curr = cnt_curr * 900.0 / 666.6667 >= 6.0 # >= 6 亩
print(f"\n方案 A (当前默认):")
print(f"  识别地块数 (>=6亩): {np.sum(valid_curr):,}")
print(f"  地块总面积: {np.sum(cnt_curr[valid_curr]) * 900.0 / 666.6667:,.1f} 亩 (占分类面积 {np.sum(cnt_curr[valid_curr])/total_crop_pixels*100:.1f}%)")

# 方案 B: 真实农田边界保真（不使用 80% 强行破损，仅使用真实自然边缘 + 连通域）
lbl_clean, n_clean = ndimage.label(crop_mask > 0, structure=ndimage.generate_binary_structure(2, 2))
cnt_clean = np.bincount(lbl_clean.ravel())[1:]
valid_clean = cnt_clean * 900.0 / 666.6667 >= 6.0
print(f"\n方案 B (自然真实连通域，>=6亩):")
print(f"  识别地块数: {np.sum(valid_clean):,}")
print(f"  地块总面积: {np.sum(cnt_clean[valid_clean]) * 900.0 / 666.6667:,.1f} 亩 (占分类面积 {np.sum(cnt_clean[valid_clean])/total_crop_pixels*100:.1f}%)")

# 方案 C: 科学物理边缘（利用夏季高峰景 B4/NDVI 的真实显著梯度 > 固定绝对阈值，而非百分位数一刀切）
# 在夏季 peak green 影像上计算真实的田埂/道路边缘
with rasterio.open(sorted_files[1]) as s1: # June 21
    b3 = s1.read(3).astype(np.float32)
    b4 = s1.read(4).astype(np.float32)
denom = b4 + b3
ndvi = np.zeros_like(b3)
v = denom > 0
ndvi[v] = (b4[v] - b3[v]) / denom[v]

import cv2
gx = cv2.Sobel(ndvi, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(ndvi, cv2.CV_32F, 0, 1, ksize=3)
grad_ndvi = np.sqrt(gx**2 + gy**2)

# 真实机耕路/水渠：其 NDVI 梯度显著高于田块内部（例如梯度 > 0.25）
real_roads = (grad_ndvi > 0.25) & (ndvi < 0.50) & (crop_mask > 0)
binary_c = (crop_mask > 0).astype(np.uint8)
binary_c[real_roads] = 0
lbl_c, n_c = ndimage.label(binary_c, structure=ndimage.generate_binary_structure(2, 2))
cnt_c = np.bincount(lbl_c.ravel())[1:]
valid_c = cnt_c * 900.0 / 666.6667 >= 6.0
print(f"\n方案 C (夏季真机耕路切分 + 物理边界保真):")
print(f"  识别地块数 (>=6亩): {np.sum(valid_c):,}")
print(f"  地块总面积: {np.sum(cnt_c[valid_c]) * 900.0 / 666.6667:,.1f} 亩 (占分类面积 {np.sum(cnt_c[valid_c])/total_crop_pixels*100:.1f}%)")
