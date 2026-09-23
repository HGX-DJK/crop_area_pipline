import rasterio
import numpy as np
import cv2
from scipy import ndimage
import time

print("Loading data for Sacramento Valley slice...")
with rasterio.open('data/satellite_tifs/CSDC30_10SEH_20150103.tif') as src:
    b1 = src.read(1).astype(np.float32)
    b2 = src.read(2).astype(np.float32)
    b3 = src.read(3).astype(np.float32)
    b4 = src.read(4).astype(np.float32)
    b5 = src.read(5).astype(np.float32)
    h, w = b1.shape

    ndvi = (b4 - b3) / np.maximum(b4 + b3, 1e-4)
    ndbi = (b5 - b4) / np.maximum(b5 + b4, 1e-4)
    b5_b3 = b5 / np.maximum(b3, 1.0)
    mndwi = (b2 - b5) / np.maximum(b2 + b5, 1e-4)

    # 1. Urban envelope detection
    impervious_core = (
        (b1 > 1000.0) |
        ((b1 > 480.0) & (b5_b3 < 1.85) & (ndvi < 0.40)) |
        ((ndbi > -0.02) & (b5_b3 < 1.75)) |
        ((b5_b3 < 1.55) & (ndvi < 0.35))
    )
    dens = ndimage.uniform_filter(impervious_core.astype(np.float32), size=21)
    urban_cand = dens >= 0.12
    urban_closed = ndimage.binary_closing(urban_cand, structure=np.ones((7, 7)))
    lbl_u, num_u = ndimage.label(urban_closed, structure=ndimage.generate_binary_structure(2, 2))
    counts_u = np.bincount(lbl_u.ravel())
    large_u = np.zeros_like(urban_closed, dtype=bool)
    for i in range(1, num_u + 1):
        if counts_u[i] >= 200:
            large_u[lbl_u == i] = True
    final_urban_mask = ndimage.binary_dilation(large_u, structure=np.ones((7, 7)))

    # 2. Foothill / Mountain Mask
    global_rows = np.arange(h, dtype=np.float32)[:, np.newaxis]
    global_cols = np.arange(w, dtype=np.float32)[np.newaxis, :]
    foothill_col = 2550.0 + 0.05 * global_rows
    mean_b4 = ndimage.uniform_filter(b4, size=11)
    sq_b4 = ndimage.uniform_filter(b4**2, size=11)
    cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)
    is_mountain_zone = (global_cols < foothill_col) & (cv_b4 > 0.10)
    is_mountain_veg = is_mountain_zone & (ndvi > 0.25) & ((cv_b4 > 0.12) | (b5 < 2250.0) | (b5 < b4 * 1.05))
    is_water = (mndwi > -0.08) | ((mndwi > -0.15) & (ndvi < 0.20)) | ((b4 < 600.0) & (b2 > b4))

    cropland_mask = (ndvi >= 0.18) & (~final_urban_mask) & (~is_mountain_veg) & (~is_water)

# Boundary gradient: farm roads, canals, ditches, field edges
grad_x = cv2.Sobel(ndvi, cv2.CV_32F, 1, 0, ksize=3)
grad_y = cv2.Sobel(ndvi, cv2.CV_32F, 0, 1, ksize=3)
grad_mag = np.sqrt(grad_x**2 + grad_y**2)
strong_edges = (grad_mag > 0.35) & (cropland_mask)

cropland_split = cropland_mask.copy().astype(np.uint8)
cropland_split[strong_edges] = 0

structure_cross = ndimage.generate_binary_structure(2, 1)
cleaned = ndimage.binary_opening(cropland_split, structure=structure_cross).astype(np.uint8)
lbl, num_comp = ndimage.label(cleaned, structure=structure_cross)

# Now test subdividing any component with area > 1200 亩 (800,000 m²)
subdivide_thresh_m2 = 800000.0
target_parcel_m2 = 400000.0 # ~600 亩
pixel_res = 30.0
pixel_area_m2 = 900.0
max_pixels = int(round(subdivide_thresh_m2 / pixel_area_m2))

counts = np.bincount(lbl.ravel())
oversized_cids = np.where(counts[1:num_comp+1] > max_pixels)[0] + 1
print(f"Number of oversized components (> 1200 亩): {len(oversized_cids)}")

slices = ndimage.find_objects(lbl)
next_label = num_comp + 1

t0 = time.time()
for loop_idx, cid in enumerate(oversized_cids):
    sl = slices[cid - 1]
    if sl is None:
        continue
    sub_labeled = lbl[sl]
    local_mask = (sub_labeled == cid).astype(np.uint8)
    comp_pixels = int(np.sum(local_mask))
    if comp_pixels <= max_pixels:
        continue

    # Distance transform
    padded = np.pad(local_mask, 1, mode="constant", constant_values=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    
    # Grid seeding for large plains
    grid_step = max(6, int(round(np.sqrt(target_parcel_m2) / pixel_res)))
    grid_mask = np.zeros_like(local_mask, dtype=bool)
    grid_mask[grid_step//2::grid_step, grid_step//2::grid_step] = True
    seeds = (grid_mask & (local_mask > 0))
    
    # Also add natural peaks if any
    win = max(5, int(round(180.0 / pixel_res)))
    if win % 2 == 0: win += 1
    dilated = cv2.dilate(dist, cv2.getStructuringElement(cv2.MORPH_RECT, (win, win)))
    peaks = (dist == dilated) & (dist >= 2.5) & (local_mask > 0)
    all_seeds = seeds | peaks
    
    num_seeds, markers = cv2.connectedComponents(all_seeds.astype(np.uint8), connectivity=8)
    num_seeds -= 1
    if num_seeds <= 1:
        continue

    # Watershed on inverted distance
    dist_norm = (dist / max(np.max(dist), 1e-4) * 255.0).astype(np.uint8)
    landscape_bgr = cv2.cvtColor(255 - dist_norm, cv2.COLOR_GRAY2BGR)
    markers_in = markers.astype(np.int32)
    cv2.watershed(landscape_bgr, markers_in)
    markers_in[markers_in <= 0] = 0
    markers_in[local_mask == 0] = 0

    unassigned = (markers_in == 0) & (local_mask > 0)
    if np.any(unassigned):
        _, nearest_idx = ndimage.distance_transform_edt(markers_in == 0, return_indices=True)
        markers_in[unassigned] = markers_in[nearest_idx[0][unassigned], nearest_idx[1][unassigned]]

    u_sub = np.unique(markers_in[markers_in > 0])
    for idx_s, s_id in enumerate(u_sub):
        m_s = (markers_in == s_id)
        target_lbl = cid if idx_s == 0 else next_label
        sub_labeled[m_s] = target_lbl
        if target_lbl == next_label:
            next_label += 1

print(f"Watershed subdivision took {time.time() - t0:.2f} s")
final_num = next_label - 1
final_counts = np.bincount(lbl.ravel())
final_areas_mu = final_counts[1:final_num+1] * pixel_area_m2 / 666.67
# Filter small noise < 5.4 亩 (4 pixels)
valid_areas = final_areas_mu[final_areas_mu >= 5.4]
sorted_final = np.sort(valid_areas)[::-1]
print(f"Total valid parcels: {len(sorted_final)}")
print(f"Top 10 parcel areas (亩): {[round(a, 1) for a in sorted_final[:10]]}")
