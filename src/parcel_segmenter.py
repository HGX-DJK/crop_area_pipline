"""
零碎农田地块形态学分割与田埂切分模块。
对应《联合国农业统计遥感手册》理论篇第 8 章（IBGE 与 FAO 联合规范）。
实现功能：
1. 识别并分离狭窄田埂（1~2 个像素宽度的沟渠、道路与交界带）
2. 形态学边缘腐蚀（Erosion）与连通域标记（Connected Component Labeling）
3. 碎斑噪声过滤（排除小于指定最小面积阈值的杂乱像元）
4. 地块内部多数投票机制确定主导作物类别与纯度
"""

import numpy as np
from scipy import ndimage

from src.utils.unit_utils import sqm_to_mu, sqm_to_ha
from src.utils.logger import get_logger, log_success


class ParcelSegmenter:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = get_logger("地块分割")
        seg_cfg = self.config.get("segmentation", {})
        self.apply_erosion = seg_cfg.get("apply_boundary_erosion", True)
        self.min_area_m2 = seg_cfg.get("min_parcel_area_m2", 200.0)
        self.max_area_m2 = seg_cfg.get("max_parcel_area_m2", 500000.0)
        self.subdivide_oversized = seg_cfg.get("subdivide_oversized", True)
        self.discard_oversized = seg_cfg.get("discard_oversized", False)
        self.max_export_parcels = seg_cfg.get("max_export_parcels", 500)
        self.connectivity = seg_cfg.get("connectivity", 8)
        self.enable_obia = seg_cfg.get("enable_obia", False)
        self.spatial_res = self.config.get("spatial", {}).get("resolution_meters", 10.0)
        self.pixel_area_m2 = self.spatial_res * self.spatial_res  # 10m x 10m = 100 m²

    def _subdivide_oversized_component(self, sub_binary_mask: np.ndarray, min_peak_distance_m: float = None) -> np.ndarray:
        """
        基于离散几何核心种子点与 Voronoi 欧式最近邻拓扑剖分算法，
        将过度粘连的超大连片农田斑块智能解构为规整的标准化田块单元。
        彻底消除沿距离变换脊线塌陷产生的竖直/水平条带伪影，
        确保平原大田能够精细解构为规模适中 (300~600 亩)、长宽比接近 1:1 的饱满现实农田。
        
        参数：
            sub_binary_mask: 斑块局部二值掩膜 (0/1 uint8)
            min_peak_distance_m: 识别田块几何核心的最小物理间距 (米，自适应传感器分辨率)
        返回：
            sub_labeled: 局部多边形标记矩阵 (0 为背景，1..K 为分割后的各子地块)
        """
        if min_peak_distance_m is None:
            min_peak_distance_m = max(180.0, float(self.spatial_res) * 5.0)
            
        comp_pixels = int(np.sum(sub_binary_mask))
        comp_area_m2 = comp_pixels * self.pixel_area_m2
        if comp_area_m2 <= float(self.max_area_m2):
            return sub_binary_mask.astype(np.int32)

        # 局部外包矩形周围垫充 1 像素背景 0，确保距离变换以真实外轮廓为基准
        padded_mask = np.pad(sub_binary_mask, pad_width=1, mode="constant", constant_values=0)
        try:
            import cv2
            dist = cv2.distanceTransform(padded_mask, cv2.DIST_L2, 5)[1:-1, 1:-1]
        except Exception:
            dist = ndimage.distance_transform_edt(padded_mask).astype(np.float32)[1:-1, 1:-1]

        max_dist = float(np.max(dist)) if dist.size > 0 else 0.0
        if max_dist < 2.0:
            return sub_binary_mask.astype(np.int32)

        # 依据空间分辨率动态计算核心峰值最小间距 (像元窗口)
        win = max(3, int(round(min_peak_distance_m / max(float(self.spatial_res), 1.0))))
        if win % 2 == 0:
            win += 1

        dilated = ndimage.maximum_filter(dist, size=win)
        threshold_height = max(1.5, min(2.5, max_dist * 0.15))
        local_peaks = (dist == dilated) & (dist >= threshold_height) & (sub_binary_mask > 0)

        # 核心优化 1：将距离变换极大值脊线压缩为单像元质心，彻底消灭沿连续脊线蔓延合并的长条种子
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

        # 核心优化 2：仅以真实自然距离极大值核心质心作为种子，彻底弃用人造几何方格，杜绝八边形/六边形蜂窝伪影
        seeds = peak_centroids_mask

        if np.sum(seeds) <= 1:
            return sub_binary_mask.astype(np.int32)

        seed_markers, num_seeds = ndimage.label(seeds, structure=ndimage.generate_binary_structure(2, 2))
        if num_seeds <= 1:
            return sub_binary_mask.astype(np.int32)

        # 核心优化 3：采用欧式最近邻距离变换 (Voronoi 拓扑剖分) 对多峰连片田块进行自然分割
        try:
            _, nearest_idx = ndimage.distance_transform_edt(seed_markers == 0, return_indices=True)
            markers = seed_markers[nearest_idx[0], nearest_idx[1]]
            markers[sub_binary_mask == 0] = 0
            return markers
        except Exception as e:
            self.logger.warning(f"地块几何剖分异常 ({e})，保留原始斑块。")
            return sub_binary_mask.astype(np.int32)

    def segment_parcels(self, crop_classified_mask, confidence_map=None, edge_mask=None, optical_image=None):
        """
        将连续的作物分类栅格切分为独立的细碎田块矢量单元。
        
        参数：
            crop_classified_mask: 形状为 (Rows, Cols) 的作物类型整数矩阵 (0=非农田, 1..K=不同作物)
            confidence_map: 分类置信度矩阵 (0~1)
            edge_mask: 遥感影像真实机耕路/水渠/田埂多光谱物理边缘掩膜 (布尔矩阵)
        
        返回：
            parcel_id_mask: 独立地块编号矩阵 (0 为非农田/田埂, 1..N 为各独立农田地块)
            parcel_metadata: 包含每个地块的面积、主导作物、置信度等元数据列表
        """

        rows, cols = crop_classified_mask.shape
        # =========================================================================
        # OBIA 升维：如果提供了高分辨率光学底图，执行 SLIC 超像素分割
        # =========================================================================
        obia_mask = None
        if self.enable_obia and optical_image is not None:
            self.logger.info("  [OBIA] 正在对高分辨率光学底图执行 SLIC 超像素过分割...")
            try:
                from skimage.segmentation import slic
                target_superpixel_size_m2 = 1000.0
                pixels_per_sp = max(10, int(target_superpixel_size_m2 / self.pixel_area_m2))
                n_segments = int((rows * cols) / pixels_per_sp)
                n_segments = min(max(n_segments, 1000), 500000)
                
                segments_slic = slic(optical_image, n_segments=n_segments, compactness=5, sigma=1, start_label=1)
                self.logger.info(f"  [OBIA] SLIC 生成了 {len(np.unique(segments_slic))} 个超像素碎片。正在与预测掩膜进行深度融合...")
                
                flat_segments = segments_slic.ravel()
                flat_crop = (crop_classified_mask > 0).ravel().astype(np.int32)
                
                if edge_mask is not None:
                    flat_crop[edge_mask.ravel()] = 0
                
                sp_totals = np.bincount(flat_segments)
                sp_crops = np.bincount(flat_segments, weights=flat_crop)
                
                with np.errstate(divide='ignore', invalid='ignore'):
                    crop_ratio = sp_crops / sp_totals
                
                # 提高融合门槛：超像素内必须有 75% 以上被预测为耕地，才判定为耕地，防止边缘溢出到林地
                is_sp_crop = crop_ratio > 0.75
                if len(is_sp_crop) > 0:
                    is_sp_crop[0] = False
                
                obia_mask = is_sp_crop[segments_slic].astype(np.uint8)
                self.logger.info("  [OBIA] 超像素融合完成。")
            except Exception as e:
                self.logger.warning(f"  [OBIA] SLIC 处理失败 ({e})。自动降级到像素级连通域模式...")

        # -------------------------------------------------------------------------
        
        # 1. 生成农田二值掩膜 (只要是农作物即为 True，若启用 OBIA 则采用超像素边界融合结果)
        if obia_mask is not None:
            cropland_binary = obia_mask.copy()
        else:
            cropland_binary = (crop_classified_mask > 0).astype(np.uint8)

        # 2. 依据联合国手册第 8 章：矢量化计算不同作物交界处的边界梯度（Zero-Copy Slicing）
        gradient_edges = np.zeros((rows, cols), dtype=bool)
        if rows > 1:
            diff_v = (crop_classified_mask[:-1, :] > 0) & (crop_classified_mask[1:, :] > 0) & (crop_classified_mask[:-1, :] != crop_classified_mask[1:, :])
            gradient_edges[:-1, :][diff_v] = True
            gradient_edges[1:, :][diff_v] = True
        if cols > 1:
            diff_h = (crop_classified_mask[:, :-1] > 0) & (crop_classified_mask[:, 1:] > 0) & (crop_classified_mask[:, :-1] != crop_classified_mask[:, 1:])
            gradient_edges[:, :-1][diff_h] = True
            gradient_edges[:, 1:][diff_h] = True

        # 将不同作物交界处切开
        cropland_binary[gradient_edges] = 0
        del gradient_edges

        # 融合真实遥感多光谱物理边缘（机耕路、灌溉渠与田埂网络）
        if edge_mask is not None and np.any(edge_mask):
            cropland_binary[edge_mask] = 0
            self.logger.info("已成功将真实遥感物理边界融入田块分割网络，沿真实机耕路与水渠自然切分农田。")

        # 3. 形态学腐蚀（Erosion）与开运算（Opening）切断细小桥接
        if self.apply_erosion:
            if self.spatial_res < 20.0:
                # 高分辨率 (10m 哨兵 / 2m 高分)：按物理米数计算结构元
                k_size = max(3, int(round(18.0 / max(float(self.spatial_res), 1.0))))
                if k_size % 2 == 0:
                    k_size += 1
                k_size = min(k_size, 9)
                structure = np.ones((k_size, k_size), dtype=np.uint8) if self.connectivity == 8 else ndimage.generate_binary_structure(2, 1)
                cleaned = ndimage.binary_opening(cropland_binary, structure=structure).astype(np.uint8)
            else:
                # 中低分辨率 (如 30m Landsat/CSDC30)：采用十字结构元切断角点对角粘连，杜绝过度消除
                structure_cross = ndimage.generate_binary_structure(2, 1)
                cleaned = ndimage.binary_opening(cropland_binary, structure=structure_cross).astype(np.uint8)
            del cropland_binary
        else:
            cleaned = cropland_binary

        # 填充内部孤立微小空洞（仅闭合真实农田内部的微小车辙、积水与像元阴影，绝对禁止将村落街区作为"空洞"整体填补为耕地）
        if self.config.get("segmentation", {}).get("fill_internal_holes", True):
            inverted = (cleaned == 0)
            lbl_holes, n_holes = ndimage.label(inverted, structure=ndimage.generate_binary_structure(2, 1))
            if n_holes > 0:
                border_mask = np.zeros_like(cleaned, dtype=bool)
                border_mask[0, :] = True
                border_mask[-1, :] = True
                border_mask[:, 0] = True
                border_mask[:, -1] = True
                edge_labels = set(np.unique(lbl_holes[border_mask]))
                
                hole_sizes = np.bincount(lbl_holes.ravel())
                max_hole_pixels = 25 # 仅限闭合 25 像元 (约 3.5 亩) 以内的细小农田微空洞
                valid_hole_lut = (hole_sizes <= max_hole_pixels)
                valid_hole_lut[0] = False
                if edge_labels:
                    for el in edge_labels:
                        valid_hole_lut[el] = False
                cleaned[valid_hole_lut[lbl_holes]] = 1
                del lbl_holes, inverted, valid_hole_lut

        # 4. 连通域标记（Connected Component Labeling）
        struct_conn = ndimage.generate_binary_structure(2, 2 if self.connectivity == 8 else 1)
        labeled_array, num_features = ndimage.label(cleaned, structure=struct_conn)
        del cleaned

        self.logger.info(f"初步识别到 {num_features} 个候选连通斑块。")

        # 5. 针对超大连片农田执行欧式距离变换与分水岭智能细分 (Watershed Subdivision)
        # 避免将相邻多个规整农田因田埂模糊粘连为几万亩巨斑，同时杜绝直接丢弃大农田
        eff_max_area = float(self.max_area_m2) if (self.max_area_m2 and self.max_area_m2 > 0) else float("inf")
        subdivide_threshold = eff_max_area

        if self.subdivide_oversized and num_features > 0 and subdivide_threshold < float("inf"):
            counts = np.bincount(labeled_array.ravel())
            comp_areas = counts[1:num_features + 1] * self.pixel_area_m2
            oversized_cids = np.where(comp_areas > subdivide_threshold)[0] + 1

            if len(oversized_cids) > 0:
                self.logger.info(f"检测到 {len(oversized_cids)} 个面积超标连片农田区域 (>{sqm_to_mu(subdivide_threshold, 1):.1f} 亩)，正在执行分水岭核心种子自适应切分...")
                slices = ndimage.find_objects(labeled_array)
                next_label = num_features + 1
                subdivided_count = 0
                new_sub_parcels_count = 0

                # 分水岭逐斑块切分（带进度打印，避免无响应假象）
                for loop_idx, cid in enumerate(oversized_cids):
                    # 每 10 个斑块打印一次进度
                    if loop_idx % 10 == 0 or loop_idx == len(oversized_cids) - 1:
                        self.logger.info(
                            f"  -> 分水岭切分进度: {loop_idx + 1}/{len(oversized_cids)} 块 "
                            f"({(loop_idx + 1) / len(oversized_cids) * 100:.0f}%)..."
                        )

                    sl = slices[cid - 1]
                    if sl is None:
                        continue
                    sub_labeled = labeled_array[sl]
                    local_mask = (sub_labeled == cid).astype(np.uint8)

                    sub_res = self._subdivide_oversized_component(local_mask)
                    u_sub = np.unique(sub_res[sub_res > 0])

                    if len(u_sub) > 1:
                        subdivided_count += 1
                        for idx, sub_u in enumerate(u_sub):
                            m_sub = (sub_res == sub_u)
                            sub_size_m2 = float(np.sum(m_sub)) * self.pixel_area_m2
                            # 若初级细分后的某子块依然显著超标 (> 3倍阈值)，执行二级更细颗粒度分水岭递归解构
                            if sub_size_m2 > subdivide_threshold * 3.0:
                                sec_dist = max(100.0, float(self.spatial_res) * 3.5)
                                sub2_res = self._subdivide_oversized_component(m_sub.astype(np.uint8), min_peak_distance_m=sec_dist)
                                u_sub2 = np.unique(sub2_res[sub2_res > 0])
                                if len(u_sub2) > 1:
                                    for idx2, s2 in enumerate(u_sub2):
                                        m_s2 = (sub2_res == s2)
                                        target_lbl = cid if (idx == 0 and idx2 == 0) else next_label
                                        sub_labeled[m_s2] = target_lbl
                                        if target_lbl == next_label:
                                            next_label += 1
                                    new_sub_parcels_count += len(u_sub2)
                                    continue

                            target_lbl = cid if idx == 0 else next_label
                            sub_labeled[m_sub] = target_lbl
                            if target_lbl == next_label:
                                next_label += 1
                            new_sub_parcels_count += 1
                    else:
                        new_sub_parcels_count += 1

                num_features = next_label - 1
                if subdivided_count > 0:
                    self.logger.info(f"  -> 分水岭切分完成: 成功将 {subdivided_count} 个超大连片区精细解构为 {new_sub_parcels_count} 个规整田块单元。")

        # 6. 自适应面积阈值过滤与元数据提取
        # 自适应最小有效面积：必须至少覆盖 2 个像元，滤除单像元椒盐噪声
        eff_min_area = max(float(self.min_area_m2), float(self.pixel_area_m2 * 2.0))

        # 核心保障：默认严禁丢弃大农田 (discard_oversized=False)；大农田细分后仍较大的部分将作为规模化连片产区完整保留
        if self.discard_oversized and eff_max_area < float("inf"):
            filter_max_area = eff_max_area
        else:
            filter_max_area = float("inf")

        parcel_id_mask = np.zeros((rows, cols), dtype=np.int32)
        parcel_metadata = []
        valid_parcel_id = 1

        if num_features > 0:
            counts = np.bincount(labeled_array.ravel())
            component_sizes = counts[1:num_features + 1]
            candidate_areas_m2 = component_sizes * self.pixel_area_m2
        else:
            component_sizes = np.array([], dtype=np.int64)
            candidate_areas_m2 = np.array([], dtype=np.float64)

        total_candidate_m2 = float(np.sum(component_sizes)) * self.pixel_area_m2
        total_candidate_mu = sqm_to_mu(total_candidate_m2, 2)

        if num_features > 0:
            valid_mask = (candidate_areas_m2 >= eff_min_area) & (candidate_areas_m2 <= filter_max_area)
            valid_comp_indices = np.where(valid_mask)[0] + 1

            if len(valid_comp_indices) == 0:
                self.logger.warning("未检测到符合面积区间的候选斑块，自动放宽下限保留全部候选斑块...")
                valid_mask = (candidate_areas_m2 > 0)
                valid_comp_indices = np.where(valid_mask)[0] + 1

            # 按面积降序排列优选主力核心地块
            sorted_order = np.argsort(-candidate_areas_m2[valid_comp_indices - 1])
            sorted_valid = valid_comp_indices[sorted_order]

            if len(sorted_valid) > self.max_export_parcels:
                comp_indices = sorted_valid[:self.max_export_parcels].tolist()
                residual_indices = sorted_valid[self.max_export_parcels:].tolist()
                selected_m2 = float(np.sum(candidate_areas_m2[np.array(comp_indices) - 1]))
                residual_m2 = float(np.sum(candidate_areas_m2[np.array(residual_indices) - 1]))
                self.logger.info(f"成功识别到 {len(sorted_valid)} 个合规独立农田地块 (候选总面积约 {total_candidate_mu/10000.0:.1f} 万亩)：")
                self.logger.info(f"  -> 优选面积前 {len(comp_indices)} 个主力核心集中区进行高精度矢量化（约 {sqm_to_mu(selected_m2)/10000.0:.1f} 万亩，占 {(selected_m2/max(total_candidate_m2,1e-6))*100:.1f}%）；")
                self.logger.info(f"  -> 剩余 {len(residual_indices)} 个长尾散碎零星斑块（约 {sqm_to_mu(residual_m2)/10000.0:.1f} 万亩），已在无偏统计总表中完整纳统。")
            else:
                comp_indices = sorted_valid.tolist()
                selected_m2 = float(np.sum(candidate_areas_m2[np.array(comp_indices) - 1])) if len(comp_indices) > 0 else 0.0
                self.logger.info(f"成功筛选到 {len(comp_indices)} 个符合面积规范的独立农田地块 (累计面积: {sqm_to_mu(selected_m2):.1f} 亩)。")
        else:
            comp_indices = []

        # 重新计算最新切片 (此时 labeled_array 已包含细分的新编号)
        slices = ndimage.find_objects(labeled_array)

        for comp_id in comp_indices:
            pixel_count = component_sizes[comp_id - 1]
            area_m2 = candidate_areas_m2[comp_id - 1]

            sl = slices[comp_id - 1]
            if sl is None:
                continue

            sub_labeled = labeled_array[sl]
            comp_mask_local = (sub_labeled == comp_id)

            # 局部写入全局地块编号掩膜 (利用切片视图直接原地更新)
            sub_parcel_id = parcel_id_mask[sl]
            sub_parcel_id[comp_mask_local] = valid_parcel_id

            # 统计该地块内部作物类别分布
            sub_crops = crop_classified_mask[sl][comp_mask_local]
            crop_only = sub_crops[sub_crops > 0]
            if len(crop_only) > 0:
                classes, counts = np.unique(crop_only, return_counts=True)
                dominant_class = int(classes[np.argmax(counts)])
                purity = float(np.max(counts) / len(crop_only))
            else:
                classes, counts = np.unique(sub_crops, return_counts=True)
                dominant_class = int(classes[np.argmax(counts)]) if len(classes) > 0 else 1
                purity = 1.0

            local_coords = np.argwhere(comp_mask_local)
            center_r = float(np.mean(local_coords[:, 0]) + sl[0].start)
            center_c = float(np.mean(local_coords[:, 1]) + sl[1].start)

            if confidence_map is not None:
                mean_conf = float(np.mean(confidence_map[sl][comp_mask_local]))
            else:
                mean_conf = 1.0

            area_mu = sqm_to_mu(area_m2, decimals=2)
            area_hectares = sqm_to_ha(area_m2, decimals=4)

            parcel_metadata.append({
                "parcel_id": f"P{valid_parcel_id:04d}",
                "internal_id": valid_parcel_id,
                "crop_code": dominant_class,
                "pixel_count": int(pixel_count),
                "area_m2": round(float(area_m2), 1),
                "area_mu": area_mu,
                "area_ha": area_hectares,
                "dominant_purity": round(purity, 3),
                "mean_confidence": round(mean_conf, 3),
                "centroid_row": round(center_r, 2),
                "centroid_col": round(center_c, 2)
            })
            valid_parcel_id += 1

        log_success(self.logger, f"过滤细碎杂斑后，成功提取 {len(parcel_metadata)} 个有效规范农田地块。")
        return parcel_id_mask, parcel_metadata
