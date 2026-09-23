"""
多时相遥感物候特征工程模块。
实现功能：
1. 构建多时相卫星影像时间序列（Sentinel-2 / Landsat 对齐）
2. 提取作物关键物候特征（生长峰值、生长季变幅、返青拔节速率、收割下降斜率）
3. 生成支持算法校验的标准测试栅格立方体（模拟零碎农田、狭窄田埂与背景地物）
"""

import os
import numpy as np
import pandas as pd


class TimeSeriesBuilder:
    def __init__(self, config=None):
        self.config = config or {}
        self.spatial_cfg = self.config.get("spatial", {})
        self.resolution = self.spatial_cfg.get("resolution_meters", 10.0)
        self.doy_list = [80, 110, 140, 170, 200, 230, 260, 290]
        # 显式双通道标志：仅由 raster_loader 在确认 SDC6 数据时设为 True
        # 避免用 T%2==0 误判合成训练数据（T=8 也是偶数）
        self.is_sdc6_dual = False

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
        从多时相 NDVI/LSWI 序列中提炼物候指纹特征。
        参数：
            time_series_array: 形状为 (N, T) 或 (Height, Width, T) 的时序植被指数矩阵。
            SDC30 双通道模式：T 为偶数时，偶数列 = NDVI，奇数列 = LSWI（自动识别）。
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

        # 防御性 NaN / Inf 缺测值清洗与像元级安全过滤
        if np.isnan(ts).any() or np.isinf(ts).any():
            ts = np.nan_to_num(ts, nan=0.0, posinf=1.0, neginf=0.0)

        # 自适应数值范围归一化（兼容未缩放的 Sentinel-2 / Landsat L2A 地表反射率数据 0~10000）
        val_max = float(np.max(ts)) if ts.size > 0 else 0.0
        if val_max > 10.0:
            ts = ts / 10000.0

        # --- SDC30 双通道拆分：由 raster_loader 在加载 SDC6 数据时显式设置 is_sdc6_dual=True ---
        # 绝对不能使用 T%2==0 作为判断依据（合成训练数据 T=8 也是偶数会被误判）
        is_dual_channel = getattr(self, "is_sdc6_dual", False)
        if is_dual_channel:
            if ts.shape[1] % 4 == 0:
                ts_ndvi = ts[:, 0::4]
                ts_lswi = ts[:, 1::4]
                ts_gcvi = ts[:, 2::4]
                ts_cv = ts[:, 3::4]
            elif ts.shape[1] % 3 == 0:
                ts_ndvi = ts[:, 0::3]
                ts_lswi = ts[:, 1::3]
                ts_gcvi = ts[:, 2::3]
                ts_cv = None
            else:
                ts_ndvi = ts[:, 0::2]
                ts_lswi = ts[:, 1::2]
                ts_gcvi = None
                ts_cv = None
            ts_for_pheno = ts_ndvi
            t_eff = ts_ndvi.shape[1]
        else:
            ts_lswi = None
            ts_gcvi = None
            ts_cv = None
            ts_for_pheno = ts
            t_eff = t

        # 可选：时序去云抗噪平滑滤波（依据联合国手册 SITS 标准）
        if self.config.get("preprocessing", {}).get("apply_temporal_smoothing", False):
            ts_for_pheno = self.smooth_time_series(ts_for_pheno)

        # 1. 基础极值与波动统计（基于 NDVI）
        ndvi_max = np.max(ts_for_pheno, axis=1, keepdims=True)
        ndvi_min = np.min(ts_for_pheno, axis=1, keepdims=True)
        ndvi_range = ndvi_max - ndvi_min
        ndvi_std = np.std(ts_for_pheno, axis=1, keepdims=True)

        # 2. 自适应计算时序动态梯度与物候斜率（动态适配任意时相数 T >= 2，杜绝固定索引硬编码）
        if t_eff >= 2:
            grad = np.gradient(ts_for_pheno, axis=1)
            grad_max = np.max(grad, axis=1, keepdims=True)   # 最大暴发增长率（拔节/抽穗）
            grad_min = np.min(grad, axis=1, keepdims=True)   # 最大衰退下降率（成熟/收割）
            grad_mean = np.mean(np.abs(grad), axis=1, keepdims=True)  # 生长季活跃度
        else:
            grad_max = np.zeros((ts.shape[0], 1), dtype=np.float32)
            grad_min = np.zeros((ts.shape[0], 1), dtype=np.float32)
            grad_mean = np.zeros((ts.shape[0], 1), dtype=np.float32)

        # 自适应关键物候阶段差分斜率（划分为苗期增长、旺盛期、成熟衰落期）
        if t_eff >= 4:
            i_early = max(1, t_eff // 4)
            i_mid = max(i_early + 1, t_eff // 2)
            i_late = min(t_eff - 1, (3 * t_eff) // 4)
            early_slope = (ts_for_pheno[:, [i_early]] - ts_for_pheno[:, [0]]) / max(1.0, float(i_early))
            mid_slope = (ts_for_pheno[:, [i_late]] - ts_for_pheno[:, [i_mid]]) / max(1.0, float(i_late - i_mid))
            late_drop = (ts_for_pheno[:, [-1]] - ts_for_pheno[:, [i_late]]) / max(1.0, float(t_eff - 1 - i_late))
        else:
            early_slope = (ts_for_pheno[:, [-1]] - ts_for_pheno[:, [0]]) / max(1.0, float(t_eff - 1)) if t_eff > 1 else np.zeros_like(ndvi_max)
            mid_slope = np.zeros_like(early_slope)
            late_drop = np.zeros_like(early_slope)

        # 3. 针对水稻（Paddy Rice）的轻量水分与泡田期物候增强特征 (Flooding & Transplanting Signals)
        if t_eff >= 3:
            early_bound = max(2, (t_eff + 1) // 2)
            early_min = np.min(ts_for_pheno[:, :early_bound], axis=1, keepdims=True)
            paddy_flooding_dip = np.maximum(0.0, ts_for_pheno[:, [0]] - early_min)
            paddy_rebound_surge = np.maximum(0.0, ndvi_max - early_min)
            denom_dip = np.where(ts_for_pheno[:, [0]] + 0.05 > 0.02, ts_for_pheno[:, [0]] + 0.05, 0.05)
            denom_surge = np.where(ndvi_max + 0.05 > 0.02, ndvi_max + 0.05, 0.05)
            paddy_v_index = (paddy_flooding_dip / denom_dip) * (paddy_rebound_surge / denom_surge)
            paddy_v_index = np.nan_to_num(paddy_v_index, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            paddy_flooding_dip = np.zeros_like(ndvi_max)
            paddy_rebound_surge = np.zeros_like(ndvi_max)
            paddy_v_index = np.zeros_like(ndvi_max)

        # 4. LSWI 附加特征（SDC30 双通道模式：5维）
        # 荒漠干沙 LSWI≈-0.05 / 休耕农田 LSWI≈+0.002 / 活跃作物 LSWI≈+0.19
        # 补充 NDVI 在荒漠-农田边界的模糊区域判别力
        if ts_lswi is not None:
            lswi_max  = np.max(ts_lswi, axis=1, keepdims=True)
            lswi_min  = np.min(ts_lswi, axis=1, keepdims=True)
            lswi_mean = np.mean(ts_lswi, axis=1, keepdims=True)
            lswi_std  = np.std(ts_lswi, axis=1, keepdims=True)
            # LSWI-NDVI 差值：同向升高=活跃有水分作物，分歧=背景/稀疏植被
            lswi_ndvi_diff = lswi_mean - np.mean(ts_for_pheno, axis=1, keepdims=True)
            lswi_feats = np.hstack([lswi_max, lswi_min, lswi_mean, lswi_std, lswi_ndvi_diff])
        else:
            lswi_feats = np.zeros((ts.shape[0], 5), dtype=np.float32)

        # 6. GCVI 特征
        if ts_gcvi is not None:
            gcvi_max = np.max(ts_gcvi, axis=1, keepdims=True)
            gcvi_mean = np.mean(ts_gcvi, axis=1, keepdims=True)
            gcvi_feats = np.hstack([gcvi_max, gcvi_mean])
        else:
            gcvi_feats = np.zeros((ts.shape[0], 2), dtype=np.float32)
            
        # 7. CV 纹理特征
        if ts_cv is not None:
            cv_mean = np.mean(ts_cv, axis=1, keepdims=True)
            cv_max = np.max(ts_cv, axis=1, keepdims=True)
            cv_feats = np.hstack([cv_mean, cv_max])
        else:
            cv_feats = np.zeros((ts.shape[0], 2), dtype=np.float32)

        # 5. 组合全部特征向量：
        features = np.hstack([
            ts_for_pheno,
            ndvi_max,
            ndvi_min,
            ndvi_range,
            ndvi_std,
            grad_max,
            grad_min,
            grad_mean,
            early_slope,
            mid_slope,
            late_drop,
            paddy_flooding_dip,
            paddy_rebound_surge,
            paddy_v_index,
            lswi_feats,
            gcvi_feats,
            cv_feats
        ])

        if is_3d:
            return features.reshape(h, w, -1)
        return features

    @staticmethod
    def compute_lswi(nir, swir):
        """
        计算地表水分指数 LSWI (Land Surface Water Index) = (NIR - SWIR) / (NIR + SWIR)
        水稻移栽期 LSWI + 0.05 >= NDVI 是国际公认的水稻泡田淹水黄金判据 (Xiao et al., 2005)。
        """
        denom = nir + swir
        denom = np.where(denom == 0, 1e-6, denom)
        return (nir - swir) / denom

    @staticmethod
    def compute_ndwi(green, nir):
        """
        计算归一化水体指数 NDWI (Normalized Difference Water Index) = (Green - NIR) / (Green + NIR)
        """
        denom = green + nir
        denom = np.where(denom == 0, 1e-6, denom)
        return (green - nir) / denom

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
                4: np.array([0.15, 0.11, 0.42, 0.75, 0.84, 0.62, 0.25, 0.18], dtype=np.float32),  # 水稻 (5月插秧泡田低值0.11，8月抽穗高峰0.84)
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
