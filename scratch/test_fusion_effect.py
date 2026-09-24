import glob
import sys
sys.path.insert(0, "crop_area_pipeline")
import numpy as np
from src.raster_loader import RasterLoader
from src.prior_fusion import PriorReferenceFusion

files = sorted(glob.glob("crop_area_pipeline/data/satellite_tifs/*.tif"))
loader = RasterLoader()
cube = loader.load_multitemporal_thumbnail(files, max_dim=1200)

ts_ndvi = cube[:, :, 0::2]
ts_lswi = cube[:, :, 1::2]
mean_ndvi = np.mean(ts_ndvi, axis=2)
min_ndvi = np.min(ts_ndvi, axis=2)
max_ndvi = np.max(ts_ndvi, axis=2)
range_ndvi = max_ndvi - min_ndvi

prior = np.full((1220, 1220), 0.5, dtype=np.float32)

# 1. 水体/深阴影
prior[(mean_ndvi < 0.25) | (min_ndvi <= 0.0)] = 0.01

# 2. 常绿山地森林
is_dense_forest = (
    (min_ndvi >= 0.55) |
    ((mean_ndvi >= 0.65) & (min_ndvi >= 0.48)) |
    ((ts_ndvi[:, :, 2] > 0.75) & (ts_ndvi[:, :, 3] > 0.60) & (ts_lswi[:, :, 2] > 0.20))
)
prior[is_dense_forest] = 0.01

# 3. 真实梯田增强
nd1, nd2, nd3, nd4 = ts_ndvi[:, :, 0], ts_ndvi[:, :, 1], ts_ndvi[:, :, 2], ts_ndvi[:, :, 3]
lw1, lw2, lw3, lw4 = ts_lswi[:, :, 0], ts_lswi[:, :, 1], ts_lswi[:, :, 2], ts_lswi[:, :, 3]

is_mountain_terrace = (
    (nd1 > 0.40) & (nd1 < 0.55) & (nd2 > 0.55) & (nd3 > 0.80) &
    (lw2 < 0.10) & (range_ndvi >= 0.25)
)
prior[is_mountain_terrace] = 0.95

print("--- 优化后先验评估 ---")
print(f"被判定为非耕地森林/水体 (prior <= 0.05): {np.mean(prior <= 0.05)*100:.2f}%")
print(f"被判定为高确信耕地 (prior >= 0.85): {np.mean(prior >= 0.85)*100:.2f}%")

p0001_prior = prior[230:240, 1080:1090]
print(f"P0001 森林大山区域 prior 均值: {np.mean(p0001_prior):.4f}, 最大值: {np.max(p0001_prior):.4f}")

terrace_prior = prior[155:165, 178:188]
print(f"红框真实山地梯田区域 prior 均值: {np.mean(terrace_prior):.4f}, 最小值: {np.min(terrace_prior):.4f}")
