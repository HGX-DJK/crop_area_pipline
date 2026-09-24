import glob
import rasterio
import numpy as np

tifs = sorted(glob.glob("crop_area_pipeline/data/satellite_tifs/*.tif"))
print(f"Found {len(tifs)} tifs")

# Inspect P0001 around row=704, col=3250
for f in tifs:
    with rasterio.open(f) as src:
        w = src.read(window=((700, 710), (3245, 3255))).astype(np.float32)
        red = w[2].mean()
        nir = w[3].mean()
        swir1 = w[4].mean()
        swir2 = w[5].mean()
        ndvi = (nir - red) / (nir + red + 1e-6)
        lswi = (nir - swir1) / (nir + swir1 + 1e-6)
        date_str = f[-12:-4]
        print(f"{date_str}: Red={red:.1f}, NIR={nir:.1f}, SWIR1={swir1:.1f}, SWIR2={swir2:.1f}, NDVI={ndvi:.3f}, LSWI={lswi:.3f}")
