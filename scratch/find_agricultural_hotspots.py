import sys
import os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import rasterio
import numpy as np
import yaml
from scipy import ndimage
from src.time_series_builder import TimeSeriesBuilder
from src.crop_classifier import CropClassifier
from src.raster_loader import _compute_sdc6_physical_indices

with open("crop_area_pipeline/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

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
    profile = ref.profile.copy()

srcs = [rasterio.open(f) for f in sorted_files]
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

        t_eff = 3
        nd_t0 = feats[:, :, 0]
        nd_t1 = feats[:, :, 1]
        nd_t2 = feats[:, :, 2]
        nd_max = feats[:, :, t_eff]
        nd_min = feats[:, :, t_eff + 1]
        nd_range = feats[:, :, t_eff + 2]

        is_crop = (mask == 1)
        is_forest = (nd_t0 >= 0.38) & (nd_t1 >= 0.65) & (nd_t2 >= 0.55) & (nd_range < 0.28)
        is_lawn = (nd_range < 0.18) & (nd_max < 0.65)
        has_pheno_cycle = (nd_min <= 0.35) | (nd_range >= 0.25)

        valid_crop = is_crop & (nd_max >= 0.35) & has_pheno_cycle & (~is_forest) & (~is_lawn)
        crop_mask_barrier[r:r+bh, c:c+bw] = valid_crop.astype(np.uint8)

for s in srcs: s.close()

# 找出作物最密集的 10 个 100x100 局部热点区域
density_map = ndimage.uniform_filter(crop_mask_barrier.astype(np.float32), size=100)
max_density = np.max(density_map)
print(f"Max 100x100 crop density: {max_density * 100:.1f}%")

# 打印热点位置
from skimage.feature import peak_local_max
peaks = peak_local_max(density_map, min_distance=200, num_peaks=10)
print("\nTop 10 Agricultural Hotspots (River Valleys / Plains):")
with rasterio.open(sorted_files[0]) as ref:
    from pyproj import Transformer
    transformer = Transformer.from_crs(ref.crs, "EPSG:4326", always_xy=True)
    for idx, (pr, pc) in enumerate(peaks, 1):
        x, y = ref.xy(pr, pc)
        lon, lat = transformer.transform(x, y)
        d = density_map[pr, pc] * 100.0
        print(f"  Hotspot {idx:2d}: (r={pr:4d}, c={pc:4d}) -> Lon: {lon:.4f}°, Lat: {lat:.4f}°, Farmland Density: {d:4.1f}%")
