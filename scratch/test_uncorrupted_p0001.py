import sys, os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import rasterio, numpy as np
import yaml
from src.time_series_builder import TimeSeriesBuilder
from src.crop_classifier import CropClassifier

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

# Raw spectra for P0001 (without 0.10 crush):
# NDVI: [0.392, 0.799, 0.755]
# LSWI: [0.122, 0.311, 0.306]
# GCVI: [0.938, 4.647, 4.162]
# CV_B4: [0.113, 0.167, 0.125]
ts_raw = np.array([
    [0.392, 0.122, 0.938, 0.113,
     0.799, 0.311, 4.647, 0.167,
     0.755, 0.306, 4.162, 0.125]
], dtype=np.float32)

feats = ts_builder.extract_phenological_features(ts_raw)
pred = classifier.model.predict(feats)
prob = classifier.model.predict_proba(feats)
print("Without urban crush (RAW SPECTRA):")
print(f"Prediction: {pred[0]} (0=Non-Crop, 1=Crop)")
print(f"Probabilities [Non-Crop, Crop]: {prob[0]}")
