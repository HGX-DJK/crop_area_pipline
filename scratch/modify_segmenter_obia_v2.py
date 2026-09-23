import re

with open('src/parcel_segmenter.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_lines = []
skip = False
for i, line in enumerate(lines):
    if "def segment_parcels(self, crop_classified_mask" in line:
        new_lines.append("""    def segment_parcels(self, crop_classified_mask: np.ndarray, conf_map: np.ndarray = None, edge_mask: np.ndarray = None, optical_image: np.ndarray = None):
        \"\"\"
        执行 OBIA (面向对象图像分析) 级别的农田地块分割，引入 SLIC 超像素技术。
        \"\"\"
        rows, cols = crop_classified_mask.shape
        if optical_image is None:
            self.logger.warning("未提供光学底图，回退至连通域合并")
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(crop_classified_mask > 0)
        else:
            self.logger.info("  [OBIA] 正在对高分辨率光学底图执行 SLIC 超像素过分割...")
            try:
                from skimage.segmentation import slic
                target_superpixel_size_m2 = 1000.0
                pixels_per_sp = max(10, int(target_superpixel_size_m2 / self.pixel_area_m2))
                n_segments = int((rows * cols) / pixels_per_sp)
                n_segments = min(max(n_segments, 1000), 500000)
                segments_slic = slic(optical_image, n_segments=n_segments, compactness=20, sigma=1, start_label=1)
                self.logger.info(f"  [OBIA] SLIC 生成 {len(np.unique(segments_slic))} 个超像素碎片。正在与模型分类结果融合...")
                
                flat_segments = segments_slic.ravel()
                flat_crop = (crop_classified_mask > 0).ravel().astype(np.int32)
                if edge_mask is not None:
                    flat_crop[edge_mask.ravel()] = 0
                
                import numpy as np
                sp_totals = np.bincount(flat_segments)
                sp_crops = np.bincount(flat_segments, weights=flat_crop)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    crop_ratio = sp_crops / sp_totals
                is_sp_crop = crop_ratio > 0.5
                if len(is_sp_crop) > 0:
                    is_sp_crop[0] = False
                
                obia_mask = is_sp_crop[segments_slic]
                self.logger.info("  [OBIA] 正在执行区域生长连通域提取...")
                from scipy import ndimage
                labeled_array, num_features = ndimage.label(obia_mask)
            except Exception as e:
                self.logger.warning(f"SLIC 失败 ({e})。回退至连通域...")
                from scipy import ndimage
                labeled_array, num_features = ndimage.label(crop_classified_mask > 0)
        
        self.logger.info(f"  [地块分割] OBIA 初步合并生成 {num_features} 个独立田块。")\n""")
        skip = True
    elif skip and "labeled_array, num_features = ndimage.label(cleaned)" in line:
        skip = False
    elif not skip:
        new_lines.append(line)

with open('src/parcel_segmenter.py', 'w', encoding='utf-8') as f:
    f.writelines(new_lines)
print("Updated parcel_segmenter.py")
