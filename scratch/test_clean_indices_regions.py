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

sorted_files = [
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000621.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20001015.tif"
]
from scipy import ndimage

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
    
    return ndvi, lswi, gcvi, cv_b4

# 测试 5 个 256x256 的区域：
# 1. 山地森林区 (围绕 P0001)
# 2. 河谷农田区
# 3. 城镇/居民区
# 4. 水库水体区
# 5. 丘陵过渡带
test_windows = [
    ("Mountain Forest (P0001)", rasterio.windows.Window(3400, 3400, 256, 256)),
    ("Valley Center (Farmland)", rasterio.windows.Window(1800, 2600, 256, 256)),
    ("North Valley (Farmland)", rasterio.windows.Window(2700, 1800, 256, 256)),
    ("Ridge Top", rasterio.windows.Window(1000, 1000, 256, 256)),
]

srcs = [rasterio.open(f) for f in sorted_files]
for name, win in test_windows:
    slices = []
    for s in srcs:
        b1 = s.read(1, window=win).astype(np.float32)
        b2 = s.read(2, window=win).astype(np.float32)
        b3 = s.read(3, window=win).astype(np.float32)
        b4 = s.read(4, window=win).astype(np.float32)
        b5 = s.read(5, window=win).astype(np.float32)
        ndvi, lswi, gcvi, cv_b4 = compute_clean_indices(b1, b2, b3, b4, b5)
        slices.extend([ndvi, lswi, gcvi, cv_b4])
    win_cube = np.stack(slices, axis=-1)
    feats = ts_builder.extract_phenological_features(win_cube)
    mask, conf = classifier.predict_raster_cube(feats)
    crop_pct = np.mean(mask) * 100
    mean_conf = np.mean(conf[mask == 1]) if np.any(mask == 1) else 0.0
    print(f"Region: {name:25s} -> Cropland Pct: {crop_pct:5.2f}%, Mean Crop Conf: {mean_conf*100:5.1f}%")

for s in srcs: s.close()
