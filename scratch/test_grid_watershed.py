import numpy as np
import cv2
from scipy import ndimage
import time

def subdivide_oversized_smart(mask, pixel_res=30.0, max_area_m2=800000.0, target_area_m2=400000.0):
    """
    Subdivide an oversized connected component into realistic agricultural parcels.
    1. If natural distance peaks exist, use them.
    2. If the component is still too large (or flat with few peaks), supplement with grid seeds.
    3. Run watershed to produce clean, realistic agricultural field boundaries.
    """
    pixel_area_m2 = pixel_res * pixel_res
    total_pixels = int(np.sum(mask))
    total_area_m2 = total_pixels * pixel_area_m2
    
    if total_area_m2 <= max_area_m2:
        return mask.astype(np.int32)
        
    padded = np.pad(mask, 1, mode="constant", constant_values=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    
    # 1. Look for natural peaks
    min_dist_m = max(180.0, pixel_res * 6.0)
    win = max(3, int(round(min_dist_m / pixel_res)))
    if win % 2 == 0:
        win += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
    dilated = cv2.dilate(dist, kernel)
    peaks = (dist == dilated) & (dist >= 2.0) & (mask > 0)
    
    num_peaks, peak_markers = cv2.connectedComponents(peaks.astype(np.uint8), connectivity=8)
    num_peaks -= 1
    
    # Check if number of peaks is sufficient for the total area
    expected_parcels = max(2, int(round(total_area_m2 / target_area_m2)))
    
    if num_peaks < expected_parcels * 0.5:
        # Distance transform alone did not find enough peaks (flat alluvial plain / monster blob)
        # Generate regularly spaced seeds across the interior of the mask
        # Spacing D in pixels: sqrt(target_area_m2) / pixel_res
        grid_step = max(5, int(round(np.sqrt(target_area_m2) / pixel_res)))
        
        # Sample points on grid where mask is 1 and dist is not too close to edge
        grid_mask = np.zeros_like(mask, dtype=bool)
        grid_mask[::grid_step, ::grid_step] = True
        grid_seeds = grid_mask & (mask > 0) & (dist >= 1.5)
        
        # Merge natural peaks and grid seeds
        all_seeds = peaks | grid_seeds
        num_peaks, peak_markers = cv2.connectedComponents(all_seeds.astype(np.uint8), connectivity=8)
        num_peaks -= 1
        
    if num_peaks <= 1:
        return mask.astype(np.int32)
        
    # Run watershed
    # Inverted distance or gradient as watershed landscape
    # Note: cv2.watershed expects a 3-channel uint8 image
    # For watershed on distance transform, we use 255 - normalized(dist)
    dist_norm = (dist / max(np.max(dist), 1e-4) * 255.0).astype(np.uint8)
    landscape_bgr = cv2.cvtColor(255 - dist_norm, cv2.COLOR_GRAY2BGR)
    
    markers = peak_markers.astype(np.int32)
    cv2.watershed(landscape_bgr, markers)
    markers[markers <= 0] = 0
    markers[mask == 0] = 0
    
    # Fill any boundary gaps left by watershed
    unassigned = (markers == 0) & (mask > 0)
    if np.any(unassigned):
        _, nearest_idx = ndimage.distance_transform_edt(markers == 0, return_indices=True)
        markers[unassigned] = markers[nearest_idx[0][unassigned], nearest_idx[1][unassigned]]
        
    return markers

# Test on 2000x2000 plain
test_mask = np.ones((2000, 2000), dtype=np.uint8)
test_mask[0, :] = 0; test_mask[-1, :] = 0; test_mask[:, 0] = 0; test_mask[:, -1] = 0

t0 = time.time()
res = subdivide_oversized_smart(test_mask, pixel_res=30.0, max_area_m2=800000.0, target_area_m2=400000.0)
t_elapsed = time.time() - t0

u_parcels = np.unique(res[res > 0])
print(f"2000x2000 (4,000,000 pixels = 540,000 亩) split into {len(u_parcels)} parcels in {t_elapsed:.3f} s!")
sizes = [np.sum(res == u) * 900.0 / 666.67 for u in u_parcels[:10]]
print(f"Sample parcel areas (亩): {[round(s, 1) for s in sizes]}")
