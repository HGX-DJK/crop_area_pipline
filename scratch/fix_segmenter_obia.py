import re

with open('src/parcel_segmenter.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update signature
sig_target = "    def segment_parcels(self, crop_classified_mask: np.ndarray, conf_map: np.ndarray = None, edge_mask: np.ndarray = None):"
sig_replacement = "    def segment_parcels(self, crop_classified_mask: np.ndarray, conf_map: np.ndarray = None, edge_mask: np.ndarray = None, optical_image: np.ndarray = None):"
content = content.replace(sig_target, sig_replacement)

# 2. Inject OBIA logic before morphological steps
obia_injection = """
        rows, cols = crop_classified_mask.shape
        # =========================================================================
        # OBIA 升维：如果提供了高分辨率光学底图，执行 SLIC 超像素分割
        # =========================================================================
        obia_mask = None
        if optical_image is not None:
            self.logger.info("  [OBIA] 正在对高分辨率光学底图执行 SLIC 超像素过分割...")
            try:
                from skimage.segmentation import slic
                target_superpixel_size_m2 = 1000.0
                pixels_per_sp = max(10, int(target_superpixel_size_m2 / self.pixel_area_m2))
                n_segments = int((rows * cols) / pixels_per_sp)
                n_segments = min(max(n_segments, 1000), 500000)
                
                segments_slic = slic(optical_image, n_segments=n_segments, compactness=20, sigma=1, start_label=1)
                self.logger.info(f"  [OBIA] SLIC 生成了 {len(np.unique(segments_slic))} 个超像素碎片。正在与预测掩膜进行深度融合...")
                
                flat_segments = segments_slic.ravel()
                flat_crop = (crop_classified_mask > 0).ravel().astype(np.int32)
                
                if edge_mask is not None:
                    flat_crop[edge_mask.ravel()] = 0
                
                sp_totals = np.bincount(flat_segments)
                sp_crops = np.bincount(flat_segments, weights=flat_crop)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    crop_ratio = sp_crops / sp_totals
                
                is_sp_crop = crop_ratio > 0.5
                if len(is_sp_crop) > 0:
                    is_sp_crop[0] = False
                
                obia_mask = is_sp_crop[segments_slic].astype(np.uint8)
                self.logger.info("  [OBIA] 超像素融合完成。")
            except Exception as e:
                self.logger.warning(f"  [OBIA] SLIC 处理失败 ({e})。自动降级到像素级连通域模式...")

        # -------------------------------------------------------------------------
"""
content = content.replace("        rows, cols = crop_classified_mask.shape", obia_injection)

# 3. Use OBIA mask instead of binary_opening if obia is active
morph_target = """        # 3. ̬ѧʴErosion뿪㣨OpeningжϸСŽ
        if self.apply_erosion:"""
morph_replacement = """        # 3. 形态学腐蚀与开运算 (如果启用了 OBIA，则跳过腐蚀，保留完美边界)
        if obia_mask is not None:
            cleaned = obia_mask
        elif self.apply_erosion:"""
content = content.replace(morph_target, morph_replacement)

with open('src/parcel_segmenter.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated parcel_segmenter.py safely")
