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


class ParcelSegmenter:
    def __init__(self, config=None):
        self.config = config or {}
        seg_cfg = self.config.get("segmentation", {})
        self.apply_erosion = seg_cfg.get("apply_boundary_erosion", True)
        self.min_area_m2 = seg_cfg.get("min_parcel_area_m2", 200.0)
        self.max_area_m2 = seg_cfg.get("max_parcel_area_m2", 500000.0)
        self.connectivity = seg_cfg.get("connectivity", 8)
        self.spatial_res = self.config.get("spatial", {}).get("resolution_meters", 10.0)
        self.pixel_area_m2 = self.spatial_res * self.spatial_res  # 10m x 10m = 100 m²

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

        # 2. 依据联合国手册第 8 章：计算不同作物交界处的边界梯度
        # 相邻不同作物或者作物与田埂之间，执行梯度检测防止误粘连
        gradient_edges = np.zeros((rows, cols), dtype=bool)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(np.roll(crop_classified_mask, dr, axis=0), dc, axis=1)
            # 若当前像元与相邻像元类别不同且均不为背景，则标记为分界线
            diff_crop = (crop_classified_mask > 0) & (shifted > 0) & (crop_classified_mask != shifted)
            gradient_edges = gradient_edges | diff_crop

        # 将不同作物交界处切开
        cropland_binary[gradient_edges] = 0

        # 3. 形态学腐蚀（Erosion）与开运算（Opening）切断细小桥接
        # 仅针对高分辨率影像（像元 < 20米）执行田埂切分；对于中低分辨率/宏观尺度栅格，单个像元已远大于真实田埂，跳过腐蚀以防过度消除
        if self.apply_erosion and self.spatial_res < 20.0:
            k_size = max(3, int(round(18.0 / max(float(self.spatial_res), 1.0))))
            if k_size % 2 == 0:
                k_size += 1
            k_size = min(k_size, 9)
            structure = np.ones((k_size, k_size), dtype=np.uint8) if self.connectivity == 8 else ndimage.generate_binary_structure(2, 1)
            cleaned = ndimage.binary_opening(cropland_binary, structure=structure).astype(np.uint8)
        else:
            cleaned = cropland_binary

        # 填充内部孤立空洞（保证农田地块拓扑严密完整）
        if self.config.get("segmentation", {}).get("fill_internal_holes", True):
            cleaned = ndimage.binary_fill_holes(cleaned).astype(np.uint8)

        # 4. 连通域标记（Connected Component Labeling）
        struct_conn = ndimage.generate_binary_structure(2, 2 if self.connectivity == 8 else 1)
        labeled_array, num_features = ndimage.label(cleaned, structure=struct_conn)

        print(f"[地块分割] 初步识别到 {num_features} 个候选连通斑块。")

        # 5. 自适应面积阈值过滤与元数据提取
        # 针对高分辨率影像 (10m 哨兵 / 2m 高分)，严格使用 200㎡ ~ 500,000㎡ 规整小农地块阈值；
        # 针对中低分辨率大尺度影像 (像元物理面积超过设定上限)，自适应缩放面积上下限，防止因单像元面积超标导致提取为 0
        if self.pixel_area_m2 > self.max_area_m2:
            eff_min_area = self.pixel_area_m2 * 1.0
            eff_max_area = self.pixel_area_m2 * 20000.0
            print(f"[地块分割自适应] 当前输入影像为宏观尺度遥感图 (像元跨度: {self.spatial_res:.1f}米，单像元面积: {self.pixel_area_m2*0.0015:.1f}亩)，")
            print(f"                 已自动将地块过滤上下限动态调整为 {eff_min_area*0.0015:.1f} ~ {eff_max_area*0.0015:.1f} 亩。")
        else:
            eff_min_area = self.min_area_m2
            eff_max_area = self.max_area_m2

        parcel_id_mask = np.zeros((rows, cols), dtype=np.int32)
        parcel_metadata = []
        valid_parcel_id = 1

        # 统计每个斑块的像素数量
        component_sizes = ndimage.sum(np.ones_like(labeled_array), labeled_array, range(1, num_features + 1))

        # 若候选斑块极多（如宏观全国图数千个），按面积从大到小优选前 500 个主力地块要素，确保秒级矢量化与 Web 地图极速加载
        comp_indices = list(range(1, num_features + 1))
        if num_features > 500:
            comp_indices = sorted(comp_indices, key=lambda cid: component_sizes[cid - 1], reverse=True)[:500]
            print(f"[地块分割] 候选斑块数量较多 ({num_features}个)，自动优选面积前 500 个主力农田区片进行高精度矢量化。")

        for comp_id in comp_indices:
            pixel_count = component_sizes[comp_id - 1]
            area_m2 = pixel_count * self.pixel_area_m2
            
            # 过滤面积过小或过大的异常斑块
            if area_m2 < eff_min_area or area_m2 > eff_max_area:
                continue

            comp_mask = (labeled_array == comp_id)
            parcel_id_mask[comp_mask] = valid_parcel_id

            # 统计该地块内部的像元作物类别分布（多数投票原则）
            crop_pixels = crop_classified_mask[comp_mask]
            classes, counts = np.unique(crop_pixels, return_counts=True)
            dominant_class = int(classes[np.argmax(counts)])
            purity = float(np.max(counts) / len(crop_pixels))

            # 计算地块中心点像素坐标
            coords = np.argwhere(comp_mask)
            center_r = float(np.mean(coords[:, 0]))
            center_c = float(np.mean(coords[:, 1]))

            # 计算平均置信度
            mean_conf = float(np.mean(confidence_map[comp_mask])) if confidence_map is not None else 1.0

            # 换算中国通用农业面积单位：1 亩 = 666.67 平方米 = 1/15 公顷
            area_mu = round(area_m2 * 0.0015, 2)
            area_hectares = round(area_m2 / 10000.0, 4)

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

        print(f"[地块分割] 过滤细碎杂斑后，成功提取 {len(parcel_metadata)} 个有效规范农田地块。")
        return parcel_id_mask, parcel_metadata
