"""
多时相遥感物候特征工程模块。
实现功能：
1. 构建多时相卫星影像时间序列（Sentinel-2 / Landsat 对齐）
2. 提取作物关键物候特征（生长峰值、生长季变幅、返青拔节速率、收割下降斜率）
3. 生成支持算法校验的标准测试栅格立方体（模拟零碎农田、狭窄田埂与背景地物）
"""

import numpy as np
import pandas as pd


class TimeSeriesBuilder:
    def __init__(self, config=None):
        self.config = config or {}
        self.spatial_cfg = self.config.get("spatial", {})
        self.resolution = self.spatial_cfg.get("resolution_meters", 10.0)
        self.doy_list = [80, 110, 140, 170, 200, 230, 260, 290]

    def smooth_time_series(self, ts, method="savgol"):
        """
        对多时相植被指数时序执行抗噪去云平滑滤波与缺测插补（依据联合国手册 SITS 标准）。
        1. 修复 NaN / NoData 缺测像元（线性插补）
        2. 剔除因薄云/阴影造成的瞬时突发低值负尖峰（上包络重建 Upper-Envelope Reconstruction）
        3. 执行 Savitzky-Golay 自适应时序滤波，保留真实作物生长峰值
        """
        ts_clean = ts.copy()
        
        # 1. 修复 NaN 缺测像元
        if np.isnan(ts_clean).any():
            for i in range(ts_clean.shape[0]):
                row = ts_clean[i]
                nans = np.isnan(row)
                if np.any(nans) and not np.all(nans):
                    x = np.where(~nans)[0]
                    y = row[~nans]
                    row[nans] = np.interp(np.where(nans)[0], x, y)
                elif np.all(nans):
                    row[:] = 0.0
                ts_clean[i] = row

        # 2. 识别并修复突发性薄云/阴影低值突降尖峰 (Cloud Spike Repair)
        t_len = ts_clean.shape[1]
        if t_len >= 3:
            for t in range(1, t_len - 1):
                prev_val = ts_clean[:, t - 1]
                next_val = ts_clean[:, t + 1]
                cur_val = ts_clean[:, t]
                # 若当前像元较前后时刻均出现大幅异常突降 (> 0.18)，判断为薄云遮挡
                cloud_spike = (cur_val < prev_val - 0.18) & (cur_val < next_val - 0.18)
                if np.any(cloud_spike):
                    ts_clean[cloud_spike, t] = (prev_val[cloud_spike] + next_val[cloud_spike]) / 2.0

        # 3. Savitzky-Golay 曲线平滑
        if t_len >= 4:
            try:
                from scipy.signal import savgol_filter
                win = min(5, t_len if t_len % 2 == 1 else t_len - 1)
                if win >= 3:
                    ts_clean = savgol_filter(ts_clean, window_length=win, polyorder=2, axis=1)
            except Exception:
                pass
        return ts_clean

    def extract_phenological_features(self, time_series_array):
        """
        从多时相 NDVI/EVI 序列中提炼物候指纹特征。
        参数：
            time_series_array: 形状为 (N, T) 或 (Height, Width, T) 的时序植被指数矩阵
        返回：
            feature_matrix: 融合了多时相原始值与提取物候因子的特征矩阵
        """
        if time_series_array.ndim == 1:
            ts = time_series_array.reshape(1, -1)
            t = ts.shape[1]
            is_3d = False
        elif time_series_array.ndim == 3:
            h, w, t = time_series_array.shape
            ts = time_series_array.reshape(-1, t)
            is_3d = True
        else:
            ts = time_series_array
            t = ts.shape[1]
            is_3d = False

        # 可选：时序去云抗噪平滑滤波（依据联合国手册 SITS 标准）
        if self.config.get("preprocessing", {}).get("apply_temporal_smoothing", False):
            ts = self.smooth_time_series(ts)

        # 1. 基础极值与波动统计
        ndvi_max = np.max(ts, axis=1, keepdims=True)
        ndvi_min = np.min(ts, axis=1, keepdims=True)
        ndvi_range = ndvi_max - ndvi_min
        ndvi_std = np.std(ts, axis=1, keepdims=True)

        # 2. 自适应计算时序动态梯度与物候斜率（动态适配任意时相数 T >= 2，杜绝固定索引硬编码）
        if t >= 2:
            grad = np.gradient(ts, axis=1)
            grad_max = np.max(grad, axis=1, keepdims=True)   # 最大暴发增长率（拔节/抽穗）
            grad_min = np.min(grad, axis=1, keepdims=True)   # 最大衰退下降率（成熟/收割）
            grad_mean = np.mean(np.abs(grad), axis=1, keepdims=True)  # 生长季活跃度
        else:
            grad_max = np.zeros((ts.shape[0], 1), dtype=np.float32)
            grad_min = np.zeros((ts.shape[0], 1), dtype=np.float32)
            grad_mean = np.zeros((ts.shape[0], 1), dtype=np.float32)

        # 自适应关键物候阶段差分斜率（划分为苗期增长、旺盛期、成熟衰落期）
        if t >= 4:
            i_early = max(1, t // 4)
            i_mid = max(i_early + 1, t // 2)
            i_late = min(t - 1, (3 * t) // 4)
            early_slope = (ts[:, [i_early]] - ts[:, [0]]) / max(1.0, float(i_early))
            mid_slope = (ts[:, [i_late]] - ts[:, [i_mid]]) / max(1.0, float(i_late - i_mid))
            late_drop = (ts[:, [-1]] - ts[:, [i_late]]) / max(1.0, float(t - 1 - i_late))
        else:
            early_slope = (ts[:, [-1]] - ts[:, [0]]) / max(1.0, float(t - 1)) if t > 1 else np.zeros_like(ndvi_max)
            mid_slope = np.zeros_like(early_slope)
            late_drop = np.zeros_like(early_slope)

        # 3. 组合全部特征向量：[原始全部时相NDVI, max, min, range, std, 动态梯度3维, 阶段斜率3维]
        features = np.hstack([
            ts,
            ndvi_max,
            ndvi_min,
            ndvi_range,
            ndvi_std,
            grad_max,
            grad_min,
            grad_mean,
            early_slope,
            mid_slope,
            late_drop
        ])

        if is_3d:
            return features.reshape(h, w, -1)
        return features

    def generate_synthetic_agricultural_landscape(self, rows=120, cols=120, random_seed=42):
        """
        生成逼真的零碎小农田块遥感多时相景观（用于算法基准测试与功能验证）。
        场景中包含：
        - 零碎、形状不规则的小农地块（玉米、小麦、大豆）
        - 1~2 个像素宽度的狭窄田埂与沟渠道路（测试田埂分割算法的核心挑战）
        - 林地、水体等非农田背景
        """
        np.random.seed(random_seed)
        ground_truth_mask = np.zeros((rows, cols), dtype=np.int32)
        parcel_id_mask = np.zeros((rows, cols), dtype=np.int32)

        # 初始化地块网格（模拟中国北方或南方丘陵细碎小田块，地块面积约 10~80 亩）
        parcel_counter = 1
        r_step = 18
        c_step = 22

        # 动态加载作物物候曲线（优先从物候曲线库中加载已定义的所有作物）
        import os
        import pandas as pd
        pheno_csv = self.config.get("paths", {}).get("phenology_curves", "data/sample_phenology_curves.csv")
        if not os.path.exists(pheno_csv) and not os.path.isabs(pheno_csv):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cand = os.path.join(project_root, pheno_csv)
            if os.path.exists(cand):
                pheno_csv = cand
        pheno_curves = {}
        if os.path.exists(pheno_csv):
            df_p = pd.read_csv(pheno_csv, comment="#")
            doy_cols = [c for c in df_p.columns if c.startswith("doy_")]
            if doy_cols:
                for _, row in df_p.iterrows():
                    cid = int(row["crop_id"])
                    pheno_curves[cid] = row[doy_cols].values.astype(np.float32)

        if not pheno_curves:
            pheno_curves = {
                0: np.array([0.18, 0.20, 0.22, 0.21, 0.23, 0.22, 0.20, 0.18], dtype=np.float32),
                1: np.array([0.15, 0.18, 0.21, 0.35, 0.68, 0.85, 0.58, 0.22], dtype=np.float32),
                2: np.array([0.48, 0.78, 0.82, 0.32, 0.18, 0.20, 0.19, 0.25], dtype=np.float32),
                3: np.array([0.16, 0.19, 0.22, 0.38, 0.62, 0.79, 0.49, 0.20], dtype=np.float32),
            }

        # 候选农作物集合（排除背景0）
        available_crops = [c for c in pheno_curves.keys() if c > 0]
        if not available_crops:
            available_crops = [1, 2, 3]

        for r in range(4, rows - 15, r_step):
            for c in range(4, cols - 15, c_step):
                # 随机微调地块大小与形状，模拟自然农田零碎边界
                h = np.random.randint(11, 16)
                w = np.random.randint(14, 20)
                
                # 动态自适应指派作物类别
                crop_class = int(np.random.choice(available_crops))
                
                r_end = min(r + h, rows - 2)
                c_end = min(c + w, cols - 2)
                
                # 填充地块内部
                ground_truth_mask[r:r_end, c:c_end] = crop_class
                parcel_id_mask[r:r_end, c:c_end] = parcel_counter
                parcel_counter += 1

        # 模拟自然狭窄田埂（在相邻地块间留出 1 像素宽的田埂/通道，属于 Class 0 背景）
        ridge_mask = np.zeros((rows, cols), dtype=bool)
        for r in range(1, rows - 1):
            for c in range(1, cols - 1):
                cur_pid = parcel_id_mask[r, c]
                if cur_pid > 0:
                    # 如果相邻存在不同地块或背景，则边缘有概率为田埂
                    neighbors = [parcel_id_mask[r-1, c], parcel_id_mask[r+1, c],
                                 parcel_id_mask[r, c-1], parcel_id_mask[r, c+1]]
                    if any(n != cur_pid for n in neighbors):
                        if np.random.rand() < 0.65:
                            ridge_mask[r, c] = True

        # 将田埂标记为背景 Class 0
        ground_truth_mask[ridge_mask] = 0
        parcel_id_mask[ridge_mask] = 0

        # 根据每类作物的物候基准生成多时相反射率数据立方体
        t_steps = len(next(iter(pheno_curves.values())))
        raster_cube = np.zeros((rows, cols, t_steps), dtype=np.float32)

        for cid, curve in pheno_curves.items():
            mask = (ground_truth_mask == cid)
            if not np.any(mask):
                continue
            # 添加空间自然异质性与遥感传感器高斯观测噪声
            noise = np.random.normal(0.0, 0.025, size=(rows, cols, t_steps))
            base_signal = curve.reshape(1, 1, t_steps)
            raster_cube[mask] = np.clip(base_signal + noise, 0.05, 0.95)[mask]

        return {
            "raster_cube": raster_cube,
            "ground_truth_mask": ground_truth_mask,
            "true_parcel_mask": parcel_id_mask,
            "total_parcels": parcel_counter - 1,
            "rows": rows,
            "cols": cols,
            "resolution": self.resolution
        }
