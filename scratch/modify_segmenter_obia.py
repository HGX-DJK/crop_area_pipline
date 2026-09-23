import re

with open('src/parcel_segmenter.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Replace the signature and body of segment_parcels
target_sig = """    def segment_parcels(self, crop_classified_mask: np.ndarray, conf_map: np.ndarray = None, edge_mask: np.ndarray = None):
        \"\"\"
        ģͳ 0/1 չ 2D ̬ѧָ﹡зֿ֣֧شˮ»
        \"\"\""""

replacement_sig = """    def segment_parcels(self, crop_classified_mask: np.ndarray, conf_map: np.ndarray = None, edge_mask: np.ndarray = None, optical_image: np.ndarray = None):
        \"\"\"
        执行 OBIA (面向对象图像分析) 级别的农田地块分割，引入 SLIC 超像素技术。
        彻底替代传统的形态学腐蚀与分水岭算法，完美贴合物理边缘。
        \"\"\"
        rows, cols = crop_classified_mask.shape
        
        # fallback if optical_image is not provided
        if optical_image is None:
            self.logger.warning("未提供光学底图，回退至连通域合并模式 (Fallback to Connected Components)")
            from scipy import ndimage
            labeled_array, num_features = ndimage.label(crop_classified_mask > 0)
        else:
            self.logger.info("  [OBIA] 正在对高分辨率光学底图执行 SLIC 超像素过分割...")
            try:
                from skimage.segmentation import slic
                from skimage.measure import regionprops
                # 根据分辨率自适应设定 SLIC 的 n_segments
                target_superpixel_size_m2 = 1000.0 # 约 1.5 亩作为一个纯色碎块
                pixels_per_sp = max(10, int(target_superpixel_size_m2 / self.pixel_area_m2))
                n_segments = int((rows * cols) / pixels_per_sp)
                n_segments = min(max(n_segments, 1000), 500000) # 防止切得太多或太少
                
                # 运行 SLIC (使用 compactness=15，sigma=1 来平滑细微噪点)
                segments_slic = slic(optical_image, n_segments=n_segments, compactness=20, sigma=1, start_label=1)
                
                self.logger.info(f"  [OBIA] SLIC 分割完成，共生成 {len(np.unique(segments_slic))} 个超像素碎片。正在与模型分类结果进行多数投票融合...")
                
                # 计算每个超像素内的分类置信度和物理边界屏蔽
                # 使用 np.bincount 高效计算
                flat_segments = segments_slic.ravel()
                flat_crop = (crop_classified_mask > 0).ravel().astype(np.int32)
                
                # 消除边界 (如果有 edge_mask)
                if edge_mask is not None:
                    flat_crop[edge_mask.ravel()] = 0
                
                # 统计每个超像素内的总像素数和作为耕地的像素数
                sp_totals = np.bincount(flat_segments)
                sp_crops = np.bincount(flat_segments, weights=flat_crop)
                
                # 多数投票: 当超像素内 > 50% 的像素被预测为耕地，则该整个超像素被判定为耕地
                # 这样可以完美继承 SLIC 切割出来的物理边界
                with np.errstate(divide='ignore', invalid='ignore'):
                    crop_ratio = sp_crops / sp_totals
                
                # 判定超像素
                is_sp_crop = crop_ratio > 0.5
                is_sp_crop[0] = False # index 0 is unused in slic(start_label=1)
                
                # 映射回 2D 图像
                obia_mask = is_sp_crop[segments_slic]
                
                self.logger.info("  [OBIA] 正在执行区域生长连通域提取...")
                from scipy import ndimage
                labeled_array, num_features = ndimage.label(obia_mask)
                
            except ImportError:
                self.logger.warning("未安装 scikit-image，请运行 pip install scikit-image。回退至连通域模式...")
                from scipy import ndimage
                labeled_array, num_features = ndimage.label(crop_classified_mask > 0)
        
        # --- 后续流程保持一致：过滤细碎杂斑，保留主流地块 ---
        self.logger.info(f"  [地块分割] OBIA 初步合并生成 {num_features} 个独立田块。")"""

# Because of regex encoding issues, we will just patch it using Python
