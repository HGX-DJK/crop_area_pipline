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
    
    # 真正的水体掩膜：只压制纯水体 (NDVI < 0 且 MNDWI > 0)
    mndwi = (b2 - b5) / np.maximum(b2 + b5, 1e-4)
    is_water = (mndwi > 0.10) & (ndvi < 0.10)
    ndvi[is_water] = -0.5
    lswi[is_water] = 0.5
    gcvi[is_water] = -1.0
    
    return ndvi, lswi, gcvi, cv_b4

# 采用分块方式统计全景像元数
block_size = 1024
with rasterio.open(sorted_files[0]) as ref:
    h, w = ref.height, ref.width

srcs = [rasterio.open(f) for f in sorted_files]
total_crop_pixels = 0
total_valid_pixels = 0

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
            ndvi, lswi, gcvi, cv_b4 = compute_clean_indices(b1, b2, b3, b4, b5)
            slices.extend([ndvi, lswi, gcvi, cv_b4])
        win_cube = np.stack(slices, axis=-1)
        feats = ts_builder.extract_phenological_features(win_cube)
        mask, _ = classifier.predict_raster_cube(feats)
        total_crop_pixels += np.sum(mask == 1)
        total_valid_pixels += mask.size

for s in srcs: s.close()

cropland_mu = total_crop_pixels * 900.0 / 666.6667
cropland_ratio = total_crop_pixels / total_valid_pixels * 100
print(f"Whole Scene Clean Cropland Pixels: {total_crop_pixels:,} ({cropland_ratio:.2f}%)")
print(f"Whole Scene Clean Cropland Area: {cropland_mu:,.1f} 亩 ({cropland_mu/10000:.2f} 万亩)")
