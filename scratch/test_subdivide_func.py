import numpy as np
from scipy import ndimage

def subdivide_oversized_component(sub_binary_mask: np.ndarray, spatial_res=30.0, max_area_m2=800000.0, min_peak_distance_m=None) -> np.ndarray:
    pixel_area_m2 = spatial_res * spatial_res
    if min_peak_distance_m is None:
        min_peak_distance_m = max(180.0, float(spatial_res) * 5.0)

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

    target_parcel_m2 = max(float(max_area_m2) * 0.5, 300_000.0)
    expected_parcels = int(round(comp_area_m2 / target_parcel_m2))

    if expected_parcels > 1:
        grid_step = max(5, int(round(np.sqrt(target_parcel_m2) / max(float(spatial_res), 1.0))))
        grid_mask = np.zeros_like(sub_binary_mask, dtype=bool)
        grid_mask[grid_step // 2 :: grid_step, grid_step // 2 :: grid_step] = True
        grid_seeds = grid_mask & (sub_binary_mask > 0)
        seeds = grid_seeds | peak_centroids_mask
    else:
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

# Test on 600x300
arr = np.ones((600, 300), dtype=np.uint8)
arr[0:5, :] = 0; arr[-5:, :] = 0; arr[:, 0:5] = 0; arr[:, -5:] = 0
res = subdivide_oversized_component(arr)
u = np.unique(res[res > 0])
ratios = []
for p in u:
    coords = np.argwhere(res == p)
    h = np.max(coords[:, 0]) - np.min(coords[:, 0]) + 1
    w = np.max(coords[:, 1]) - np.min(coords[:, 1]) + 1
    ratios.append(h / w)

print(f"Test on 600x300: parcels={len(u)}, mean ratio={np.mean(ratios):.2f}, max ratio={np.max(ratios):.2f}")
