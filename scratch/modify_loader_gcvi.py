import re

with open('src/raster_loader.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Add GCVI calculation
calc_target = """    # --- LSWI ---
    denom_lswi = b4 + b5
    lswi = np.zeros_like(b4)
    valid_lswi = denom_lswi > 0
    lswi[valid_lswi] = (b4[valid_lswi] - b5[valid_lswi]) / denom_lswi[valid_lswi]"""

calc_replacement = """    # --- LSWI ---
    denom_lswi = b4 + b5
    lswi = np.zeros_like(b4)
    valid_lswi = denom_lswi > 0
    lswi[valid_lswi] = (b4[valid_lswi] - b5[valid_lswi]) / denom_lswi[valid_lswi]

    # --- GCVI (Green Chlorophyll Vegetation Index) ---
    gcvi = np.zeros_like(b2)
    valid_gcvi = b2 > 0
    gcvi[valid_gcvi] = (b4[valid_gcvi] / b2[valid_gcvi]) - 1.0
    gcvi = np.clip(gcvi, -2.0, 10.0)"""

content = content.replace(calc_target, calc_replacement)

# 2. Add GCVI physical suppression
suppress_target = """    ndvi[is_mountain_veg] = np.minimum(ndvi[is_mountain_veg], 0.10)
    lswi[is_mountain_veg] = np.minimum(lswi[is_mountain_veg], -0.05)

    return ndvi, lswi"""

suppress_replacement = """    ndvi[is_mountain_veg] = np.minimum(ndvi[is_mountain_veg], 0.10)
    lswi[is_mountain_veg] = np.minimum(lswi[is_mountain_veg], -0.05)
    gcvi[is_mountain_veg] = np.minimum(gcvi[is_mountain_veg], 0.0)
    
    gcvi[is_water_wetland] = np.minimum(gcvi[is_water_wetland], -0.5)
    gcvi[final_urban_mask] = np.minimum(gcvi[final_urban_mask], 0.0)

    return ndvi, lswi, gcvi"""
content = content.replace(suppress_target, suppress_replacement)

# 3. Update callers
content = content.replace('ndvi, lswi = _compute_sdc6_physical_indices(b1, b2, b3, b4, b5, global_rows, global_cols)', 'ndvi, lswi, gcvi = _compute_sdc6_physical_indices(b1, b2, b3, b4, b5, global_rows, global_cols)')

band_slice_target = """                            band_slices.append(ndvi)
                            band_slices.append(lswi)"""
band_slice_replacement = """                            band_slices.append(ndvi)
                            band_slices.append(lswi)
                            band_slices.append(gcvi)"""
content = content.replace(band_slice_target, band_slice_replacement)

# thumbnail
content = content.replace('thumbnail, _ = _compute_sdc6_physical_indices(b1, b2, b3, b4, b5, thumb_rows, thumb_cols)', 'thumbnail, _, _ = _compute_sdc6_physical_indices(b1, b2, b3, b4, b5, thumb_rows, thumb_cols)')

# Add load_optical_for_slic
slic_method = """    def load_optical_for_slic(self, tif_dir_or_list):
        \"\"\"
        加载全分辨率的光学底图（RGB或假彩色）用于SLIC超像素分割
        \"\"\"
        if isinstance(tif_dir_or_list, list) and len(tif_dir_or_list) > 0:
            file_path = tif_dir_or_list[-1] # 使用最近一期的影像
        else:
            files = [os.path.join(tif_dir_or_list, f) for f in os.listdir(tif_dir_or_list) if f.endswith(('.tif', '.tiff'))]
            file_path = sorted(files)[-1]
            
        with rasterio.open(file_path) as src:
            # 读取 Red, Green, Blue (B3, B2, B1 for SDC30)
            b3 = src.read(3).astype(np.float32)
            b2 = src.read(2).astype(np.float32)
            b1 = src.read(1).astype(np.float32)
            
            # 标准化到 0-255
            def stretch(band):
                p2, p98 = np.percentile(band, (2, 98))
                stretched = np.clip((band - p2) / (p98 - p2 + 1e-5) * 255.0, 0, 255)
                return stretched.astype(np.uint8)
                
            rgb = np.dstack([stretch(b3), stretch(b2), stretch(b1)])
            return rgb

"""
content = content.replace('class RasterLoader:', slic_method + 'class RasterLoader:')

with open('src/raster_loader.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated raster_loader.py")
