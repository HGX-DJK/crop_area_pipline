import rasterio
import numpy as np

with rasterio.open('output/crop_classification_map.tif') as src:
    crop_mask = src.read(1)

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)

ndvi = np.where(b4 + b3 > 0, (b4 - b3) / (b4 + b3), 0.0)

sub_ndvi = ndvi[1000:2000, 2800:3500]
sub_crop = crop_mask[1000:2000, 2800:3500]
sub_b2 = b2[1000:2000, 2800:3500]
sub_b3 = b3[1000:2000, 2800:3500]
sub_b4 = b4[1000:2000, 2800:3500]
sub_b5 = b5[1000:2000, 2800:3500]

# Pixels in valley that have NDVI > 0.40 but were classified as 0 (Non-Crop)
missed = (sub_crop == 0) & (sub_ndvi > 0.40)
print(f"Valley pixels with NDVI > 0.40 that were MISSED (classified as 0): {np.sum(missed):,} ({np.mean(missed):.1%})")

if np.sum(missed) > 0:
    print(f"Missed pixels properties:")
    print(f"  NDVI: mean={sub_ndvi[missed].mean():.3f}, min={sub_ndvi[missed].min():.3f}, max={sub_ndvi[missed].max():.3f}")
    print(f"  B3 (Red): mean={sub_b3[missed].mean():.1f}, min={sub_b3[missed].min():.1f}")
    print(f"  B4 (NIR): mean={sub_b4[missed].mean():.1f}, min={sub_b4[missed].min():.1f}")
    print(f"  B5 (SWIR1): mean={sub_b5[missed].mean():.1f}, min={sub_b5[missed].min():.1f}")
    print(f"  B5/B4 ratio: mean={(sub_b5[missed]/sub_b4[missed]).mean():.3f}")
    
    # Did they trigger is_forest or is_rugged_mountain?
    from scipy import ndimage
    mean_b4 = ndimage.uniform_filter(sub_b4, size=5)
    sq_b4 = ndimage.uniform_filter(sub_b4**2, size=5)
    cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)
    
    trig_forest = (sub_ndvi[missed] > 0.45) & (sub_b3[missed] < 580.0) & (sub_b5[missed] < 1800.0) & (sub_b5[missed] < sub_b4[missed] * 0.90)
    trig_rugged = (sub_ndvi[missed] > 0.30) & (cv_b4[missed] > 0.28) & (sub_b5[missed] < 2150.0) & (sub_b3[missed] < 620.0)
    
    print(f"  Triggered is_forest: {np.sum(trig_forest):,} ({np.mean(trig_forest):.1%})")
    print(f"  Triggered is_rugged: {np.sum(trig_rugged):,} ({np.mean(trig_rugged):.1%})")
