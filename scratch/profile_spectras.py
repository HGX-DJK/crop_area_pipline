import rasterio
import numpy as np

with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b1 = src.read(1).astype(np.float32)
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)
    b6 = src.read(6).astype(np.float32)

ndvi = np.where(b4 + b3 > 0, (b4 - b3)/(b4 + b3), 0.0)
mndwi = np.where(b2 + b5 > 0, (b2 - b5)/(b2 + b5), 0.0)
ndbi = np.where(b5 + b4 > 0, (b5 - b4)/(b5 + b4), 0.0)
lswi = np.where(b4 + b5 > 0, (b4 - b5)/(b4 + b5), 0.0)

# Sample 1: Mountain woodlands (Coast Range, row 800:1500, col 500:1500, green pixels ndvi > 0.40)
mtn_mask = (ndvi > 0.40)
mtn_slice = (slice(800, 1500), slice(500, 1500))
mtn_pixels = mtn_mask[mtn_slice]

# Sample 2: Valley commercial orchards (row 800:1500, col 2800:3400, tree pixels ndvi > 0.45, b5 < 1800)
orchard_mask = (ndvi > 0.45) & (b5 < 1800)
orchard_slice = (slice(800, 1500), slice(2800, 3400))
orchard_pixels = orchard_mask[orchard_slice]

# Sample 3: Valley active winter crops (row 800:1500, col 2800:3400, high green ndvi > 0.50, b5 > 2000)
crop_mask = (ndvi > 0.50) & (b5 > 2000)
crop_slice = (slice(800, 1500), slice(2800, 3400))
crop_pixels = crop_mask[crop_slice]

# Sample 4: Valley fallow soil (row 800:1500, col 2800:3400, ndvi 0.18-0.32, b5 > 2000)
fallow_mask = (ndvi >= 0.18) & (ndvi <= 0.32) & (b5 > 2000)
fallow_slice = (slice(800, 1500), slice(2800, 3400))
fallow_pixels = fallow_mask[fallow_slice]

def print_stats(name, b_list, slc, p_mask):
    print(f"\n--- {name} (N={np.sum(p_mask):,}) ---")
    b_names = ['B1(Blue)', 'B2(Green)', 'B3(Red)', 'B4(NIR)', 'B5(SWIR1)', 'B6(SWIR2)', 'NDVI', 'LSWI', 'B5/B4']
    vals = [
        b_list[0][slc][p_mask],
        b_list[1][slc][p_mask],
        b_list[2][slc][p_mask],
        b_list[3][slc][p_mask],
        b_list[4][slc][p_mask],
        b_list[5][slc][p_mask],
        ndvi[slc][p_mask],
        lswi[slc][p_mask],
        (b_list[4][slc][p_mask] / np.maximum(b_list[3][slc][p_mask], 1.0))
    ]
    for n, v in zip(b_names, vals):
        print(f"  {n:12s}: mean={v.mean():.3f}, std={v.std():.3f}, min={np.percentile(v, 5):.3f}, p50={np.median(v):.3f}, p95={np.percentile(v, 95):.3f}")

b_all = [b1, b2, b3, b4, b5, b6]
print_stats("Mountain Woodland/Forest (Coast Range)", b_all, mtn_slice, mtn_pixels)
print_stats("Valley Orchard (Sacramento Valley)", b_all, orchard_slice, orchard_pixels)
print_stats("Valley Active Green Crop", b_all, crop_slice, crop_pixels)
print_stats("Valley Fallow Soil", b_all, fallow_slice, fallow_pixels)
