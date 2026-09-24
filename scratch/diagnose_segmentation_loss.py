import sys, os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import rasterio
import numpy as np
import pandas as pd
from scipy import ndimage
import yaml

with open("crop_area_pipeline/config.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

# 读取分类栅格
with rasterio.open("crop_area_pipeline/output/crop_classification_map.tif") as src:
    crop_mask = src.read(1)

total_crop_pixels = int(np.sum(crop_mask == 1))
total_crop_mu = total_crop_pixels * 900.0 / 666.6667
print(f"Total crop pixels in classification map: {total_crop_pixels:,} ({total_crop_mu:,.1f} 亩)")

# 分析连通域分布（在未腐蚀前）
labeled_raw, num_features = ndimage.label(crop_mask == 1, structure=ndimage.generate_binary_structure(2, 2))
sizes_raw = np.bincount(labeled_raw.ravel())[1:]
sizes_mu = sizes_raw * 900.0 / 666.6667
print(f"\nRaw Connected Components (Before Erosion):")
print(f"  Total components: {num_features:,}")
print(f"  Total area: {np.sum(sizes_mu):,.1f} 亩")
print(f"  Components < 6 亩 (min_parcel_area 4000m2): {np.sum(sizes_mu < 6.0):,} (Total area: {np.sum(sizes_mu[sizes_mu < 6.0]):,.1f} 亩)")
print(f"  Components 6~30 亩: {np.sum((sizes_mu >= 6.0) & (sizes_mu < 30.0)):,} (Total area: {np.sum(sizes_mu[(sizes_mu >= 6.0) & (sizes_mu < 30.0)]):,.1f} 亩)")
print(f"  Components 30~300 亩: {np.sum((sizes_mu >= 30.0) & (sizes_mu < 300.0)):,} (Total area: {np.sum(sizes_mu[(sizes_mu >= 30.0) & (sizes_mu < 300.0)]):,.1f} 亩)")
print(f"  Components > 300 亩: {np.sum(sizes_mu >= 300.0):,} (Total area: {np.sum(sizes_mu[sizes_mu >= 300.0]):,.1f} 亩)")

# 模拟 parcel_segmenter 的各个处理阶段，查看面积是在哪一步被消耗的：
from src.parcel_segmenter import ParcelSegmenter
segmenter = ParcelSegmenter(config)

# 1. 边缘腐蚀
if segmenter.apply_erosion:
    eroded = ndimage.binary_erosion(crop_mask == 1, structure=np.ones((3, 3)))
    eroded_pixels = int(np.sum(eroded))
    print(f"\nAfter 3x3 Binary Erosion:")
    print(f"  Remaining pixels: {eroded_pixels:,} ({eroded_pixels * 900 / 666.6667:,.1f} 亩)")
    print(f"  Eaten pixels: {total_crop_pixels - eroded_pixels:,} ({(total_crop_pixels - eroded_pixels) * 900 / 666.6667:,.1f} 亩, {(total_crop_pixels - eroded_pixels)/total_crop_pixels*100:.1f}%)")

# 2. 真实物理边缘雕刻 (edge_mask)
sorted_files = [
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000621.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20001015.tif"
]
from src.raster_loader import RasterLoader
loader = RasterLoader(config)
edge_mask = loader.extract_optical_edges_stream(sorted_files, block_size=1024)
edge_pixels = int(np.sum(edge_mask & (crop_mask == 1)))
print(f"\nEdge Mask inside Cropland:")
print(f"  Carved edge pixels: {edge_pixels:,} ({edge_pixels * 900 / 666.6667:,.1f} 亩)")
