import glob
import sys
sys.path.insert(0, "crop_area_pipeline")
import numpy as np
import pandas as pd
import yaml
from src.raster_loader import RasterLoader
from src.time_series_builder import TimeSeriesBuilder
from src.crop_classifier import CropClassifier
from src.prior_fusion import PriorReferenceFusion

with open("crop_area_pipeline/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

loader = RasterLoader(cfg)
files = sorted(glob.glob("crop_area_pipeline/data/satellite_tifs/*.tif"))
sorted_files, geo_info, doy_list, _ = loader.get_multitemporal_metadata(files)

ts_builder = TimeSeriesBuilder(cfg)
ts_builder.is_sdc6_dual = True

cube = loader.load_multitemporal_thumbnail(files, max_dim=1200)

# Build features for thumbnail
feat = ts_builder.extract_phenological_features(cube)

# Train baseline classifier
clf = CropClassifier(cfg)
clf.train_with_samples("crop_area_pipeline/data/sample_training_points.csv", ts_builder=ts_builder, target_t=4, doy_list=doy_list)

crop_mask, prob = clf.predict_raster_cube(feat)

print(f"原始训练集下 P0001 (森林大山) 本地预测均值: {np.mean(prob[230:240, 1080:1090]):.4f}")
print(f"原始训练集下 真实梯田 本地预测均值: {np.mean(prob[155:165, 178:188]):.4f}")

# Prior fusion
fusion = PriorReferenceFusion(cfg)
prior = fusion.load_or_generate_prior_map(geo_info, cube, ts_builder)
fused_prob = fusion.fuse_prediction_with_prior(prob, prior)

print(f"融合后 P0001 (森林大山) 概率: {np.mean(fused_prob[230:240, 1080:1090]):.4f}")
print(f"融合后 真实梯田 概率: {np.mean(fused_prob[155:165, 178:188]):.4f}")
print(f"融合后 P0001 是否被完全剔除 (有效耕地像元数): {np.sum(fused_prob[230:240, 1080:1090] >= 0.50)}")
print(f"融合后 真实梯田 留存耕地像元数: {np.sum(fused_prob[155:165, 178:188] >= 0.50)}")
