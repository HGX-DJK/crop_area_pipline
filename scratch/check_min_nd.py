import glob
import sys
sys.path.insert(0, "crop_area_pipeline")
import yaml
import numpy as np
from src.raster_loader import RasterLoader

with open("crop_area_pipeline/config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

loader = RasterLoader(cfg)
files = sorted(glob.glob("crop_area_pipeline/data/satellite_tifs/*.tif"))
cube = loader.load_multitemporal_thumbnail(files, max_dim=1200)

ndvis = cube[:, :, 0::2]
min_nd = np.min(ndvis, axis=2)
mean_nd = np.mean(ndvis, axis=2)

print(f"全图像元数: {min_nd.size}")
print(f"全年最低 NDVI >= 0.52 的像元占比 (典型森林): {np.mean(min_nd >= 0.52)*100:.1f}%")
print(f"全年最低 NDVI >= 0.55 的像元占比 (深山茂密森林): {np.mean(min_nd >= 0.55)*100:.1f}%")
print(f"全年最低 NDVI >= 0.58 的像元占比: {np.mean(min_nd >= 0.58)*100:.1f}%")
print(f"全年最低 NDVI >= 0.60 的像元占比: {np.mean(min_nd >= 0.60)*100:.1f}%")

# Check Red box terrace (rows 470~495, cols 535~560 in full 3660x3660 -> in 1220: r=156~165, c=178~186)
# Let's check full min_nd at red box
red_min = min_nd[int(470/3):int(495/3), int(535/3):int(560/3)]
print(f"红框真实山地梯田的 min_nd 均值: {np.mean(red_min):.3f}")

# Check P0001 (row=704, col=3250 in 3660 -> r=234, c=1083 in 1220)
p0001_min = min_nd[230:240, 1080:1090]
print(f"P0001 森林大山的 min_nd 均值: {np.mean(p0001_min):.3f}")
