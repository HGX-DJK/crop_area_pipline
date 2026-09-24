import rasterio
import numpy as np

with rasterio.open("crop_area_pipeline/output/crop_classification_map.tif") as src:
    # Read window around row=704, col=3250 (the old mountain P0001)
    w_old = src.read(1, window=((695, 715), (3240, 3260)))
    print(f"旧 P0001 森林大山 (row=704, col=3250) 20x20 窗口中的有效耕地像元数: {np.sum(w_old > 0)} / {w_old.size}")

    # Read window around row=480, col=550 (the red box genuine terrace)
    w_terrace = src.read(1, window=((475, 485), (545, 555)))
    print(f"真实山地梯田 (row=480, col=550) 10x10 窗口中的有效耕地像元数: {np.sum(w_terrace > 0)} / {w_terrace.size}")
