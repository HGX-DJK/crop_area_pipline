import sys
import os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import rasterio
import numpy as np
import yaml
from scipy import ndimage
from src.time_series_builder import TimeSeriesBuilder
from src.crop_classifier import CropClassifier
from src.raster_loader import RasterLoader, _compute_sdc6_physical_indices

with open("crop_area_pipeline/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 确保随机种子完全固定
np.random.seed(config.get("classification", {}).get("random_state", 42))

ts_builder = TimeSeriesBuilder(config)
ts_builder.is_sdc6_dual = True
classifier = CropClassifier(config)
classifier.train_with_samples(
    "crop_area_pipeline/data/sample_training_points.csv",
    ts_builder=ts_builder,
    target_t=3,
    doy_list=[77, 173, 289]
)

sorted_files = [
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000621.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20001015.tif"
]

# 分块评估整个栅格，并统计不同区域的作物分类情况
block_size = 1024
with rasterio.open(sorted_files[0]) as ref:
    h, w = ref.height, ref.width

srcs = [rasterio.open(f) for f in sorted_files]

crop_mask = np.zeros((h, w), dtype=np.uint8)
ndvi_stack = np.zeros((h, w, 3), dtype=np.float32)

for r in range(0, h, block_size):
    bh = min(block_size, h - r)
    for c in range(0, w, block_size):
        bw = min(block_size, w - c)
        win = rasterio.windows.Window(c, r, bw, bh)
        slices = []
        for i_s, s in enumerate(srcs):
            b1 = s.read(1, window=win).astype(np.float32)
            b2 = s.read(2, window=win).astype(np.float32)
            b3 = s.read(3, window=win).astype(np.float32)
            b4 = s.read(4, window=win).astype(np.float32)
            b5 = s.read(5, window=win).astype(np.float32)
            ndvi, lswi, gcvi, cv_b4 = _compute_sdc6_physical_indices(b1, b2, b3, b4, b5, r, c)
            slices.extend([ndvi, lswi, gcvi, cv_b4])
            ndvi_stack[r:r+bh, c:c+bw, i_s] = ndvi
        win_cube = np.stack(slices, axis=-1)
        feats = ts_builder.extract_phenological_features(win_cube)
        mask, _ = classifier.predict_raster_cube(feats)
        crop_mask[r:r+bh, c:c+bw] = mask

for s in srcs: s.close()

total_crop = int(np.sum(crop_mask == 1))
print(f"Total Raw Predicted Crop Pixels: {total_crop:,} ({total_crop/crop_mask.size*100:.2f}%)")
print(f"Total Area: {total_crop * 900.0 / 666.6667:,.1f} mu")

# 统计分析预测为耕地的像元在 3 个时相的 NDVI 特征
crop_ndvis = ndvi_stack[crop_mask == 1] # (N, 3)
nd_m3 = crop_ndvis[:, 0]
nd_m6 = crop_ndvis[:, 1]
nd_m10 = crop_ndvis[:, 2]
nd_max = np.max(crop_ndvis, axis=1)
nd_min = np.min(crop_ndvis, axis=1)
nd_range = nd_max - nd_min

print("\n--- Crop Pixels Phenology Distribution ---")
print(f"March NDVI (T0): mean={np.mean(nd_m3):.3f}, median={np.median(nd_m3):.3f}, <0.35 ratio={np.mean(nd_m3 < 0.35)*100:.1f}%")
print(f"June  NDVI (T1): mean={np.mean(nd_m6):.3f}, median={np.median(nd_m6):.3f}, >0.55 ratio={np.mean(nd_m6 > 0.55)*100:.1f}%")
print(f"Oct   NDVI (T2): mean={np.mean(nd_m10):.3f}, median={np.median(nd_m10):.3f}, <0.45 ratio={np.mean(nd_m10 < 0.45)*100:.1f}%")
print(f"NDVI Range (Max-Min): mean={np.mean(nd_range):.3f}, median={np.median(nd_range):.3f}, >0.25 ratio={np.mean(nd_range > 0.25)*100:.1f}%")

# 检查常绿木本植被混入情况：
# 森林特征：三期全部高绿度 (March >= 0.40, June >= 0.70, Oct >= 0.60) 且年内变幅小 (Range < 0.25)
evergreen_forest_leak = (nd_m3 >= 0.38) & (nd_m6 >= 0.65) & (nd_m10 >= 0.55) & (nd_range < 0.28)
print(f"\nEvergreen Forest-like Pixels among Predicted Crop: {np.sum(evergreen_forest_leak):,} ({np.mean(evergreen_forest_leak)*100:.1f}%)")

# 检查低矮草坪/稀疏杂草混入情况：三期 NDVI 均偏低 (Max < 0.35)
low_veg_leak = (nd_max < 0.35)
print(f"Low Vegetation / Non-Crop Leak among Predicted Crop: {np.sum(low_veg_leak):,} ({np.mean(low_veg_leak)*100:.1f}%)")

# 真实农田农艺物候特征：
# 必有播种/翻耕期 (nd_min <= 0.35 或 nd_m3 <= 0.35) 或显著生长期变幅 (nd_range >= 0.25)
true_agronomic_crop = (nd_max >= 0.35) & ((nd_min <= 0.35) | (nd_range >= 0.25)) & (~evergreen_forest_leak)
print(f"\nTrue Agronomic Crop Pixels: {np.sum(true_agronomic_crop):,} ({np.mean(true_agronomic_crop)*100:.1f}%)")
print(f"True Agronomic Crop Area: {np.sum(true_agronomic_crop) * 900.0 / 666.6667:,.1f} mu (约 {np.sum(true_agronomic_crop) * 900.0 / 666.6667 / 10000:.2f} 万亩)")
