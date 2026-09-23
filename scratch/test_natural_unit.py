import numpy as np
from scipy import ndimage

def subdivide_oversized_component(sub_binary_mask: np.ndarray, spatial_res=10.0, max_area_m2=5000.0, min_peak_distance_m=None) -> np.ndarray:
    pixel_area_m2 = spatial_res * spatial_res
    if min_peak_distance_m is None:
        min_peak_distance_m = max(60.0, float(spatial_res) * 4.0)

    comp_pixels = int(np.sum(sub_binary_mask))
    comp_area_m2 = comp_pixels * pixel_area_m2
    if comp_area_m2 <= float(max_area_m2):
        return sub_binary_mask.astype(np.int32)

    padded_mask = np.pad(sub_binary_mask, pad_width=1, mode="constant", constant_values=0)
    try:
        import cv2
        dist = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)[1:-1, 1:-1]
    except Exception:
        dist = ndimage.distance_transform_edt(padded_mask).astype(np.float32)[1:-1, 1:-1]

    max_dist = float(np.max(dist)) if dist.size > 0 else 0.0
    if max_dist < 2.0:
        return sub_binary_mask.astype(np.int32)

    win = max(3, int(round(min_peak_distance_m / max(float(spatial_res), 1.0))))
    if win % 2 == 0:
        win += 1

    dilated = ndimage.maximum_filter(dist, size=win)
    threshold_height = max(1.5, min(2.5, max_dist * 0.15))
    local_peaks = (dist == dilated) & (dist >= threshold_height) & (sub_binary_mask > 0)

    peak_labels, num_raw_peaks = ndimage.label(local_peaks, structure=ndimage.generate_binary_structure(2, 2))
    peak_centroids_mask = np.zeros_like(sub_binary_mask, dtype=bool)
    if num_raw_peaks > 0:
        slices_peaks = ndimage.find_objects(peak_labels)
        for lab_idx, p_sl in enumerate(slices_peaks, start=1):
            if p_sl is not None:
                coords = np.argwhere(peak_labels[p_sl] == lab_idx)
                if len(coords) > 0:
                    c_r = int(np.mean(coords[:, 0])) + p_sl[0].start
                    c_c = int(np.mean(coords[:, 1])) + p_sl[1].start
                    if sub_binary_mask[c_r, c_c] > 0:
                        peak_centroids_mask[c_r, c_c] = True

    # Natural seeds only - NO artificial square grid!
    seeds = peak_centroids_mask
    if np.sum(seeds) <= 1:
        return sub_binary_mask.astype(np.int32)

    seed_markers, num_seeds = ndimage.label(seeds, structure=ndimage.generate_binary_structure(2, 2))
    if num_seeds <= 1:
        return sub_binary_mask.astype(np.int32)

    try:
        _, nearest_idx = ndimage.distance_transform_edt(seed_markers == 0, return_indices=True)
        markers = seed_markers[nearest_idx[0], nearest_idx[1]]
        markers[sub_binary_mask == 0] = 0
        return markers
    except Exception as e:
        return sub_binary_mask.astype(np.int32)

# Test 1: oversized cluster with neck
mask1 = np.zeros((100, 100), dtype=np.uint8)
mask1[10:45, 10:45] = 1
mask1[55:90, 55:90] = 1
mask1[40:60, 40:60] = 1
res1 = subdivide_oversized_component(mask1)
u1 = np.unique(res1[res1 > 0])
print(f"Test 1 (cluster with neck): parcels={len(u1)}")
assert len(u1) >= 2, "Test 1 failed!"

# Test 2: single homogeneous square
mask2 = np.zeros((80, 80), dtype=np.uint8)
mask2[20:60, 20:60] = 1
res2 = subdivide_oversized_component(mask2)
u2 = np.unique(res2[res2 > 0])
print(f"Test 2 (single square): parcels={len(u2)}")
assert len(u2) == 1, "Test 2 failed!"

print("All tests passed successfully!")
