import numpy as np
import cv2
from scipy import ndimage
import time

def test_grid():
    test_mask = np.ones((1000, 1000), dtype=np.uint8)
    test_mask[0, :] = 0; test_mask[-1, :] = 0; test_mask[:, 0] = 0; test_mask[:, -1] = 0
    
    pixel_res = 30.0
    pixel_area_m2 = 900.0
    target_area_m2 = 400000.0 # ~600 亩
    grid_step = max(5, int(round(np.sqrt(target_area_m2) / pixel_res))) # ~21 pixels = 630m
    
    # Place seeds at regular offset so seeds don't align right at 0
    grid_mask = np.zeros_like(test_mask, dtype=bool)
    grid_mask[grid_step//2::grid_step, grid_step//2::grid_step] = True
    seeds = grid_mask & (test_mask > 0)
    
    num_seeds, markers = cv2.connectedComponents(seeds.astype(np.uint8), connectivity=8)
    num_seeds -= 1
    
    # Distance transform
    padded = np.pad(test_mask, 1, mode="constant", constant_values=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, 5)[1:-1, 1:-1]
    dist_norm = (dist / max(np.max(dist), 1e-4) * 255.0).astype(np.uint8)
    landscape_bgr = cv2.cvtColor(255 - dist_norm, cv2.COLOR_GRAY2BGR)
    
    markers_in = markers.astype(np.int32)
    cv2.watershed(landscape_bgr, markers_in)
    markers_in[markers_in <= 0] = 0
    markers_in[test_mask == 0] = 0
    
    unassigned = (markers_in == 0) & (test_mask > 0)
    if np.any(unassigned):
        _, nearest_idx = ndimage.distance_transform_edt(markers_in == 0, return_indices=True)
        markers_in[unassigned] = markers_in[nearest_idx[0][unassigned], nearest_idx[1][unassigned]]
        
    u_parcels, counts = np.unique(markers_in[markers_in > 0], return_counts=True)
    areas_mu = counts * pixel_area_m2 / 666.67
    print(f"Num parcels: {len(u_parcels)}")
    print(f"Min area: {np.min(areas_mu):.1f} 亩, Mean area: {np.mean(areas_mu):.1f} 亩, Max area: {np.max(areas_mu):.1f} 亩")
    print(f"10th-90th percentile: [{np.percentile(areas_mu, 10):.1f}, {np.percentile(areas_mu, 90):.1f}] 亩")

test_grid()
