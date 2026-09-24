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

# 寻找高反射率城镇建筑中心 (高 B1, 高 B5/B3, 低 NDVI)
with rasterio.open(sorted_files[1]) as s1: # June
    b1_sample = s1.read(1, window=rasterio.windows.Window(0, 0, 3600, 3600), out_shape=(360, 360))
    b3_sample = s1.read(3, window=rasterio.windows.Window(0, 0, 3600, 3600), out_shape=(360, 360))
    b4_sample = s1.read(4, window=rasterio.windows.Window(0, 0, 3600, 3600), out_shape=(360, 360))
    ndvi_s = (b4_sample - b3_sample) / np.maximum(b4_sample + b3_sample, 1.0)
    # 城镇特征：NDVI < 0.25 且 B1 较高
    urban_s = (ndvi_s < 0.25) & (b1_sample > 400)
    dens_u = ndimage.uniform_filter(urban_s.astype(float), size=5)
    max_idx = np.unravel_index(np.argmax(dens_u), dens_u.shape)
    urban_r = max_idx[0] * 10
    urban_c = max_idx[1] * 10
    print(f"Detected primary urban center around r={urban_r}, c={urban_c}, urban density={dens_u[max_idx]:.2f}")

# 测试该城镇中心 128x128 窗口
win_u = rasterio.windows.Window(urban_c, urban_r, 128, 128)
srcs = [rasterio.open(f) for f in sorted_files]
slices = []
for s in srcs:
    b1 = s.read(1, window=win_u).astype(np.float32)
    b2 = s.read(2, window=win_u).astype(np.float32)
    b3 = s.read(3, window=win_u).astype(np.float32)
    b4 = s.read(4, window=win_u).astype(np.float32)
    b5 = s.read(5, window=win_u).astype(np.float32)
    ndvi, lswi, gcvi, cv_b4 = compute_clean_indices(b1, b2, b3, b4, b5)
    slices.extend([ndvi, lswi, gcvi, cv_b4])
win_cube = np.stack(slices, axis=-1)
feats = ts_builder.extract_phenological_features(win_cube)
mask_u, conf_u = classifier.predict_raster_cube(feats)
for s in srcs: s.close()

print(f"Urban Window Cropland Ratio: {np.mean(mask_u) * 100:.2f}% (Expected near 0%)")
