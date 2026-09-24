import sys
import os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import numpy as np
import pandas as pd
import yaml
import rasterio
from src.time_series_builder import TimeSeriesBuilder
from src.crop_classifier import CropClassifier

# 1. 训练模型
with open("crop_area_pipeline/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

ts_builder = TimeSeriesBuilder(config)
ts_builder.is_sdc6_dual = True
classifier = CropClassifier(config)
classifier.train_with_samples(
    "crop_area_pipeline/data/sample_training_points.csv",
    ts_builder=ts_builder,
    target_t=3,
    doy_list=[77, 173, 289]
)

# 2. 读取 P0001 处的真实数据立方体像元 (r=3464, c=3509)
sorted_files = [
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000621.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20001015.tif"
]
from src.raster_loader import RasterLoader, _compute_sdc6_physical_indices

# 读取一个 32x32 窗口围绕 P0001
win = rasterio.windows.Window(3500, 3450, 32, 32)
srcs = [rasterio.open(f) for f in sorted_files]
bands_list = []
for s in srcs:
    b1 = s.read(1, window=win).astype(np.float32)
    b2 = s.read(2, window=win).astype(np.float32)
    b3 = s.read(3, window=win).astype(np.float32)
    b4 = s.read(4, window=win).astype(np.float32)
    b5 = s.read(5, window=win).astype(np.float32)
    ndvi, lswi, gcvi, cv_b4 = _compute_sdc6_physical_indices(b1, b2, b3, b4, b5, 0, 0)
    bands_list.extend([ndvi, lswi, gcvi, cv_b4])
for s in srcs: s.close()

patch_cube = np.stack(bands_list, axis=-1) # (32, 32, 12)
feats = ts_builder.extract_phenological_features(patch_cube) # (32, 32, 29)
pred_mask, conf_map = classifier.predict_raster_cube(feats)

# 查看中心像元 (14, 9) 即 (3464, 3509)
r_rel = 3464 - 3450
c_rel = 3509 - 3500
center_feat = feats[r_rel, c_rel:c_rel+1, :].reshape(1, -1)
center_prob = classifier.model.predict_proba(center_feat)
print(f"P0001 center pixel (r=3464, c=3509):")
print(f"  Prediction: {pred_mask[r_rel, c_rel]}, Prob: {center_prob[0]}")
print(f"  Patch crop pixel percentage: {np.mean(pred_mask) * 100:.1f}%")

# 检查中心像元的各个特征具体数值
feat_names = [
    "ndvi_1", "ndvi_2", "ndvi_3",
    "ndvi_max", "ndvi_min", "ndvi_range", "ndvi_std",
    "grad_max", "grad_min", "grad_mean",
    "early_slope", "mid_slope", "late_drop",
    "paddy_dip", "paddy_surge", "paddy_v_index",
    "lswi_max", "lswi_min", "lswi_mean", "lswi_std", "lswi_diff",
    "gcvi_max", "gcvi_mean",
    "cv_mean", "cv_max",
    "fourier_amp1", "fourier_phase1", "fourier_amp2", "fourier_phase2"
]
print("\nP0001 Features vs Training Sample Means:")
f_df = pd.DataFrame({
    "feat": feat_names[:feats.shape[-1]],
    "P0001_val": center_feat[0]
})
print(f_df)
