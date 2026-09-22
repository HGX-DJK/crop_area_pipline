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
        self.spatial_res = self.config.get("spatial", {}).get("resolution_meters", 10.0)
        self.pixel_area_m2 = self.spatial_res * self.spatial_res  # 10m x 10m = 100 m²

    def _subdivide_oversized_component(self, sub_binary_mask: np.ndarray, min_peak_distance_m: float = 120.0) -> np.ndarray:
        """
        基于欧式距离变换与标记分水岭算法 (Distance Transform + Watershed)，
        将过度粘连的超大连片农田斑块智能切分为规整的标准化田块单元。
        如果斑块属于单一均质超大田块 (仅有单一峰值核心)，则保持整体不拆解。
        
        参数：
            sub_binary_mask: 斑块局部二值掩膜 (0/1 uint8)
            min_peak_distance_m: 识别田块几何核心的最小物理间距 (米，默认 120 米)
        返回：
            sub_labeled: 局部多边形标记矩阵 (0 为背景，1..K 为分割后的各子地块)
        """
        # 局部外包矩形周围垫充 1 像素背景 0，确保距离变换正确以真实外轮廓为基准（避免边界截断伪影）
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

        try:
            import cv2
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (win, win))
            dilated = cv2.dilate(dist, kernel)
        except Exception:
            dilated = ndimage.maximum_filter(dist, size=win)

        threshold_height = max(1.5, max_dist * 0.20)
        local_peaks = (dist == dilated) & (dist >= threshold_height) & (sub_binary_mask > 0)

        try:
            import cv2
            num_peaks, peak_markers = cv2.connectedComponents(local_peaks.astype(np.uint8), connectivity=8)
            num_peaks -= 1
        except Exception:
            peak_markers, num_peaks = ndimage.label(local_peaks)

        # 仅有 1 个或 0 个核心种子点时，说明是均质单体大田，不予过度拆解
        if num_peaks <= 1:
            return sub_binary_mask.astype(np.int32)

        try:
            import cv2
            sub_bgr = cv2.cvtColor(sub_binary_mask * 255, cv2.COLOR_GRAY2BGR)
            markers = peak_markers.astype(np.int32)
            cv2.watershed(sub_bgr, markers)
            markers[markers <= 0] = 0
            markers[sub_binary_mask == 0] = 0
            
            # 若分水岭边界缝隙导致少量像元未分配，执行就地近邻补齐
            unassigned = (markers == 0) & (sub_binary_mask > 0)
            if np.any(unassigned):
                _, nearest_idx = ndimage.distance_transform_edt(markers == 0, return_indices=True)
                markers[unassigned] = markers[nearest_idx[0][unassigned], nearest_idx[1][unassigned]]
            return markers
        except Exception as e:
            self.logger.warning(f"分水岭切分异常 ({e})，保留原始斑块。")
            return sub_binary_mask.astype(np.int32)

    def segment_parcels(self, crop_classified_mask, confidence_map=None):
        """
        将连续的作物分类栅格切分为独立的细碎田块矢量单元。
        
        参数：
            crop_classified_mask: 形状为 (Rows, Cols) 的作物类型整数矩阵 (0=非农田, 1..K=不同作物)
            confidence_map: 分类置信度矩阵 (0~1)
        
        返回：
            parcel_id_mask: 独立地块编号矩阵 (0 为非农田/田埂, 1..N 为各独立农田地块)
            parcel_metadata: 包含每个地块的面积、主导作物、置信度等元数据列表
        """
        rows, cols = crop_classified_mask.shape
        
        # 1. 生成农田二值掩膜 (只要是农作物即为 True)
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

        # 填充内部孤立空洞（保证农田地块拓扑严密完整）
        if self.config.get("segmentation", {}).get("fill_internal_holes", True):
            cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)

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

                for cid in oversized_cids:
                    sl = slices[cid - 1]
                    if sl is None:
                        continue
                    sub_labeled = labeled_array[sl]
                    local_mask = (sub_labeled == cid).astype(np.uint8)

                    sub_res = self._subdivide_oversized_component(local_mask)
                    u_sub = np.unique(sub_res[sub_res > 0])

                    if len(u_sub) > 1:
                        subdivided_count += 1
                        new_sub_parcels_count += len(u_sub)
                        for idx, sub_u in enumerate(u_sub):
                            m_sub = (sub_res == sub_u)
                            if idx == 0:
                                sub_labeled[m_sub] = cid
                            else:
                                sub_labeled[m_sub] = next_label
                                next_label += 1

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
