import glob
import sys
sys.path.insert(0, "crop_area_pipeline")
import numpy as np
from src.raster_loader import RasterLoader

files = sorted(glob.glob("crop_area_pipeline/data/satellite_tifs/*.tif"))
loader = RasterLoader()
cube = loader.load_multitemporal_thumbnail(files, max_dim=1200)

for (r, c) in [(230, 1087), (230, 1088), (231, 1087), (231, 1088), (231, 1089)]:
    nd = cube[r, c, 0::2]
    lw = cube[r, c, 1::2]
    print(f"Pixel ({r}, {c}): NDVI={nd}, LSWI={lw}")
