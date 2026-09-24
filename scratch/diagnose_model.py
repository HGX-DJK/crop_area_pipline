import sys
import os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import numpy as np
import pandas as pd
import yaml
import rasterio
from src.time_series_builder import TimeSeriesBuilder
from src.crop_classifier import CropClassifier
from src.raster_loader import RasterLoader

# 1. 检查 sample_training_points.csv 中的样点特征
df = pd.read_csv("crop_area_pipeline/data/sample_training_points.csv", comment="#", encoding="utf-8")
print("=== Sample Points Breakdown ===")
print(df.groupby(["label", "crop_name"]).size())

print("\n=== Mean NDVI by Class in Training Data ===")
doy_cols = [c for c in df.columns if c.startswith("doy_")]
print(df.groupby(["label", "crop_name"])[doy_cols].mean())

# 2. 检查实际训练出的分类器特征重要性
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

print("\n=== Classifier Trained ===")
if hasattr(classifier.model, "feature_importances_"):
    importances = classifier.model.feature_importances_
    # What are the feature names?
    # In extract_phenological_features:
    # 3 ts_ndvi + ndvi_max + ndvi_min + ndvi_range + ndvi_std (4) + 
    # grad_max, grad_min, grad_mean (3) + 
    # early_slope, mid_slope, late_drop (3) + 
    # paddy_flooding_dip, paddy_rebound_surge, paddy_v_index (3) + 
    # lswi_feats (5) + gcvi_feats (2) + cv_feats (2) + fourier (4)
    # Total = 3 + 4 + 3 + 3 + 3 + 5 + 2 + 2 + 4 = 29 features
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
    fi_df = pd.DataFrame({"feature": feat_names[:len(importances)], "importance": importances})
    fi_df = fi_df.sort_values("importance", ascending=False)
    print("Top 15 Most Important Features in XGBoost:")
    print(fi_df.head(15))
