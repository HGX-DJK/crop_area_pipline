import sys, os
sys.path.insert(0, os.path.abspath("crop_area_pipeline"))
import rasterio, numpy as np

sorted_files = [
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000317.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20000621.tif",
    "crop_area_pipeline/data/satellite_tifs/SDC30_V003_49RFH_20001015.tif"
]
srcs = [rasterio.open(f) for f in sorted_files]
win_val = rasterio.windows.Window(2700, 1800, 256, 256)
win_mtn = rasterio.windows.Window(3400, 3400, 256, 256)

def inspect_win_spectra(win, name):
    b1_list, b3_list, b4_list, b5_list = [], [], [], []
    for s in srcs:
        b1_list.append(s.read(1, window=win).astype(float))
        b3_list.append(s.read(3, window=win).astype(float))
        b4_list.append(s.read(4, window=win).astype(float))
        b5_list.append(s.read(5, window=win).astype(float))
    
    ndvis = [(b4 - b3) / np.maximum(b4 + b3, 1.0) for b3, b4 in zip(b3_list, b4_list)]
    lswis = [(b4 - b5) / np.maximum(b4 + b5, 1.0) for b4, b5 in zip(b4_list, b5_list)]
    
    print(f"\n=== {name} Spectra ===")
    print(f"Mean NDVI: Date 0 (March)={np.mean(ndvis[0]):.3f}, Date 1 (June)={np.mean(ndvis[1]):.3f}, Date 2 (Oct)={np.mean(ndvis[2]):.3f}")
    print(f"Mean LSWI: Date 0 (March)={np.mean(lswis[0]):.3f}, Date 1 (June)={np.mean(lswis[1]):.3f}, Date 2 (Oct)={np.mean(lswis[2]):.3f}")
    print(f"Mean B4: Date 0={np.mean(b4_list[0]):.0f}, Date 1={np.mean(b4_list[1]):.0f}, Date 2={np.mean(b4_list[2]):.0f}")
    print(f"Mean B5: Date 0={np.mean(b5_list[0]):.0f}, Date 1={np.mean(b5_list[1]):.0f}, Date 2={np.mean(b5_list[2]):.0f}")

inspect_win_spectra(win_val, "Valley Farmland Window")
inspect_win_spectra(win_mtn, "Mountain Forest Window")

for s in srcs: s.close()
