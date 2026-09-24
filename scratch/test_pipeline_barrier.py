import sys
import os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import rasterio
import numpy as np
import yaml
from src.time_series_builder import TimeSeriesBuilder
from src.crop_classifier import CropClassifier
from src.raster_loader import _compute_sdc6_physical_indices

with open("crop_area_pipeline/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 固定随机种子
np.random.seed(42)

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

block_size = 1024
with rasterio.open(sorted_files[0]) as ref:
    h, w = ref.height, ref.width

srcs = [rasterio.open(f) for f in sorted_files]

crop_mask = np.zeros((h, w), dtype=np.uint8)
crop_mask_barrier = np.zeros((h, w), dtype=np.uint8)

for r in range(0, h, block_size):
    bh = min(block_size, h - r)
    for c in range(0, w, block_size):
        bw = min(block_size, w - c)
        win = rasterio.windows.Window(c, r, bw, bh)
        slices = []
        for s in srcs:
            b1 = s.read(1, window=win).astype(np.float32)
            b2 = s.read(2, window=win).astype(np.float32)
            b3 = s.read(3, window=win).astype(np.float32)
            b4 = s.read(4, window=win).astype(np.float32)
            b5 = s.read(5, window=win).astype(np.float32)
            ndvi, lswi, gcvi, cv_b4 = _compute_sdc6_physical_indices(b1, b2, b3, b4, b5, r, c)
            slices.extend([ndvi, lswi, gcvi, cv_b4])
        win_cube = np.stack(slices, axis=-1)
        feats = ts_builder.extract_phenological_features(win_cube)
        mask, _ = classifier.predict_raster_cube(feats)
        crop_mask[r:r+bh, c:c+bw] = mask

        # 应用物理农艺屏障
        t_eff = 3
        nd_t0 = feats[:, :, 0]
        nd_t1 = feats[:, :, 1]
        nd_t2 = feats[:, :, 2]
        nd_max = feats[:, :, t_eff]
        nd_min = feats[:, :, t_eff + 1]
        nd_range = feats[:, :, t_eff + 2]

        is_crop = (mask == 1)
        # 1. 峰值绿度达到农田标准 (>= 0.35)
        # 2. 具有显著的农田物候动态变化 (翻耕/低谷 <= 0.35 或 变幅 >= 0.25)
        # 3. 排除常绿木本森林: (nd_t0 >= 0.38) & (nd_t1 >= 0.65) & (nd_t2 >= 0.55) & (nd_range < 0.28)
        # 4. 排除平坦草坪: (nd_range < 0.18) & (nd_max < 0.65)
        is_forest = (nd_t0 >= 0.38) & (nd_t1 >= 0.65) & (nd_t2 >= 0.55) & (nd_range < 0.28)
        is_lawn = (nd_range < 0.18) & (nd_max < 0.65)
        has_pheno_cycle = (nd_min <= 0.35) | (nd_range >= 0.25)

        valid_crop = is_crop & (nd_max >= 0.35) & has_pheno_cycle & (~is_forest) & (~is_lawn)
        crop_mask_barrier[r:r+bh, c:c+bw] = valid_crop.astype(np.uint8)

for s in srcs: s.close()

p_raw = int(np.sum(crop_mask == 1))
p_bar = int(np.sum(crop_mask_barrier == 1))
area_raw = p_raw * 900.0 / 666.6667
area_bar = p_bar * 900.0 / 666.6667

print(f"Raw Crop: {p_raw:,} pixels ({p_raw/crop_mask.size*100:.2f}%), Area: {area_raw:,.1f} mu ({area_raw/10000:.2f} wan mu)")
print(f"With Agronomic Barrier: {p_bar:,} pixels ({p_bar/crop_mask.size*100:.2f}%), Area: {area_bar:,.1f} mu ({area_bar/10000:.2f} wan mu)")
print(f"Eliminated False Positives: {p_raw - p_bar:,} pixels ({(p_raw - p_bar) * 900.0 / 666.6667 / 10000:.2f} wan mu)")

# 专项检查关键区域：
# 1. 高山森林 P0001 (r=3464, c=3509)
win_mtn = crop_mask_barrier[3450:3480, 3500:3530]
print(f"P0001 Mountain Forest 30x30 crop count: {np.sum(win_mtn)}")

# 2. 城镇居民区 (r=300, c=1520)
win_urb = crop_mask_barrier[290:320, 1510:1540]
print(f"Urban Settlement 30x30 crop count: {np.sum(win_urb)}")

# 3. 河谷主产区 (r=1800, c=2600)
win_val = crop_mask_barrier[1780:1820, 2580:2620]
print(f"River Valley Farmland 40x40 crop count: {np.sum(win_val)} ({np.mean(win_val)*100:.1f}%)")
