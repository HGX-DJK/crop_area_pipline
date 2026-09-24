import sys
import os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import rasterio
import numpy as np
import yaml
from src.time_series_builder import TimeSeriesBuilder
from src.crop_classifier import CropClassifier
from src.raster_loader import RasterLoader, _compute_sdc6_physical_indices
from src.parcel_segmenter import ParcelSegmenter

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

srcs = [rasterio.open(f) for f in sorted_files]
crop_mask_barrier = np.zeros((h, w), dtype=np.uint8)
conf_map = np.zeros((h, w), dtype=np.float32)

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
        mask, conf = classifier.predict_raster_cube(feats)

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
        conf_map[r:r+bh, c:c+bw] = np.where(valid_crop, conf, 1.0)

for s in srcs: s.close()

# 计算真实道路与田埂边缘掩膜 (使用第 1 景即 6 月夏收夏种旺季)
loader = RasterLoader(config)
edge_mask = loader.compute_spectral_edge_mask(sorted_files[1], crop_mask_barrier)

# 运行地块分割
segmenter = ParcelSegmenter(config)
parcel_id_mask, parcel_metadata = segmenter.segment_parcels(crop_mask_barrier, conf_map, edge_mask=edge_mask)

total_parcels = len(parcel_metadata)
total_area_mu = sum(p["area_mu"] for p in parcel_metadata)
areas = [p["area_mu"] for p in parcel_metadata]
compactness = [p.get("compactness", 0.0) for p in parcel_metadata]

print("\n=== 地块分割评估结果 ===")
print(f"有效农田地块数量: {total_parcels}")
print(f"地块累计净耕地面积: {total_area_mu:,.1f} 亩 ({total_area_mu/10000:.2f} 万亩)")
if areas:
    print(f"面积统计: 最小={min(areas):.1f} 亩, 最大={max(areas):.1f} 亩, 平均={np.mean(areas):.1f} 亩, 中位数={np.median(areas):.1f} 亩")
    print(f"平均紧凑度: {np.mean(compactness):.3f}")

print("\n前 10 大地块概览:")
for i in range(min(10, len(parcel_metadata))):
    p = parcel_metadata[i]
    print(f"  {p['parcel_id']}: 面积={p['area_mu']:.1f} 亩, 紧凑度={p.get('compactness',0):.3f}, 像元数={p['pixel_count']}")
