import sys, os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import rasterio, numpy as np
import yaml
from scipy import ndimage
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

sorted_files = [
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000621.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20001015.tif"
]

def compute_clean_indices(b1, b2, b3, b4, b5):
    denom_ndvi = b4 + b3
    ndvi = np.zeros_like(b3)
    valid_ndvi = denom_ndvi > 0
    ndvi[valid_ndvi] = (b4[valid_ndvi] - b3[valid_ndvi]) / denom_ndvi[valid_ndvi]

    denom_lswi = b4 + b5
    lswi = np.zeros_like(b4)
    valid_lswi = denom_lswi > 0
    lswi[valid_lswi] = (b4[valid_lswi] - b5[valid_lswi]) / denom_lswi[valid_lswi]

    gcvi = np.zeros_like(b2)
    valid_gcvi = b2 > 0
    gcvi[valid_gcvi] = (b4[valid_gcvi] / b2[valid_gcvi]) - 1.0
    gcvi = np.clip(gcvi, -2.0, 10.0)

    mean_b4 = ndimage.uniform_filter(b4, size=11)
    sq_b4 = ndimage.uniform_filter(b4**2, size=11)
    cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)
    
    mndwi = (b2 - b5) / np.maximum(b2 + b5, 1e-4)
    is_water = (mndwi > 0.10) & (ndvi < 0.10)
    ndvi[is_water] = -0.5
    lswi[is_water] = 0.5
    gcvi[is_water] = -1.0
    
    return ndvi, lswi, gcvi, cv_b4

win_mtn = rasterio.windows.Window(3400, 3400, 256, 256)
win_val = rasterio.windows.Window(2700, 1800, 256, 256)

srcs = [rasterio.open(f) for f in sorted_files]

def get_region_stats(win):
    slices = []
    raw_cvs = []
    raw_ndvis = []
    for s in srcs:
        b1 = s.read(1, window=win).astype(np.float32)
        b2 = s.read(2, window=win).astype(np.float32)
        b3 = s.read(3, window=win).astype(np.float32)
        b4 = s.read(4, window=win).astype(np.float32)
        b5 = s.read(5, window=win).astype(np.float32)
        ndvi, lswi, gcvi, cv_b4 = compute_clean_indices(b1, b2, b3, b4, b5)
        slices.extend([ndvi, lswi, gcvi, cv_b4])
        raw_cvs.append(cv_b4)
        raw_ndvis.append(ndvi)
    win_cube = np.stack(slices, axis=-1)
    feats = ts_builder.extract_phenological_features(win_cube)
    mask, _ = classifier.predict_raster_cube(feats)
    
    mean_cv = np.mean(np.stack(raw_cvs, axis=-1), axis=-1)
    nd_arr = np.stack(raw_ndvis, axis=-1)
    nd_max = np.max(nd_arr, axis=-1)
    nd_min = np.min(nd_arr, axis=-1)
    
    crop = (mask == 1) & (nd_max >= 0.35) & ((nd_min <= 0.32) | (nd_max - nd_min >= 0.28))
    return crop, mean_cv

crop_mtn, cv_mtn = get_region_stats(win_mtn)
crop_val, cv_val = get_region_stats(win_val)

print(f"Mountain region: mean CV of crop candidates = {np.mean(cv_mtn[crop_mtn]):.4f}, mean CV of non-crop = {np.mean(cv_mtn[~crop_mtn]):.4f}")
print(f"Valley region: mean CV of true crops = {np.mean(cv_val[crop_val]):.4f}")

# What if we require cv_mean <= 0.050 (agricultural flatness)?
crop_mtn_flat = crop_mtn & (cv_mtn <= 0.050)
crop_val_flat = crop_val & (cv_val <= 0.050)
print(f"After CV <= 0.050 filter:")
print(f"  Mountain cropland pct: {np.mean(crop_mtn_flat)*100:.2f}% (was {np.mean(crop_mtn)*100:.2f}%)")
print(f"  Valley cropland pct: {np.mean(crop_val_flat)*100:.2f}% (was {np.mean(crop_val)*100:.2f}%)")

for s in srcs: s.close()
