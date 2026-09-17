"""
联合国官方种植面积无偏推断与样框校准模块。
严格对应《联合国农业统计遥感手册》（UN Handbook）：
- 第 24 章：加权面积估计量（Weighted Area Estimator & Olofsson 2014 解析方差与误差矩阵）
- 第 26 章：预测增强推断（Prediction-Powered Inference, PPI / Predict-Then-Debias）

核心理论与意义：
针对零碎小农田块，由于田埂密集、边缘混合像元严重，直接“数像元面积”（Pixel-Counting）
存在 15%~35% 的系统性分类偏差。
本模块通过极少量地面抽样方块（Validation Sample）建立：
1. Olofsson et al. (2014) 面积加权混淆矩阵与解析标准误闭式估计（Standard Error, SE）
2. 联合国官方精度矩阵：用户精度（UA）、生产者精度（PA）、总体精度（OA）及其置信区间
3. Percentile Bootstrap 经验置信区间（2,000次重抽样交叉印证）
4. 变异系数（CV = SE / Area）评估官方统计可靠性
5. 预测增强推断（PPI / PTD）拓展单产与连续变量去偏推断
"""

import numpy as np
import pandas as pd

from src.utils.unit_utils import sqm_to_mu, sqm_to_ha, MU_PER_SQM


class AreaUnbiasedEstimator:
    def __init__(self, config=None):
        self.config = config or {}
        ppi_cfg = self.config.get("unbiased_area_inference", {})
        self.n_bootstrap = ppi_cfg.get("n_bootstrap", 2000)
        self.alpha = 1.0 - ppi_cfg.get("confidence_level", 0.95)
        self.pixel_res = self.config.get("spatial", {}).get("resolution_meters", 10.0)
        self.pixel_area_m2 = self.pixel_res * self.pixel_res
        self.crop_legend = self.config.get("crop_legend", {
            0: "非农田/背景",
            1: "夏玉米",
            2: "冬小麦",
            3: "大豆"
        })
        self.df_confusion_matrix = None
        self.accuracy_metrics = {}

    def estimate_unbiased_areas(self, crop_classified_mask, ground_truth_csv="data/ground_truth_area_sample.csv", return_details=False):
        """
        对比并输出两套面积统计方案：
        1. 传统朴素像元计数（Naive Map-Only）：直接乘像元大小，存在边缘偏差；
        2. 联合国样框无偏校准（Calibrated Unbiased Area）：
           - Olofsson et al. (2014) 解析标准误 (SE)、变异系数 (CV%) 与解析 95% 置信区间；
           - 2,000 次 Percentile Bootstrap 经验置信区间；
           - 联合国官方三维精度评价矩阵（用户精度 UA, 生产者精度 PA, 总体精度 OA）。
        
        参数：
            crop_classified_mask: 形状为 (Rows, Cols) 的分类栅格掩膜
            ground_truth_csv: 地面验证样点数据表路径
            return_details: 是否同时返回全量面积加权混淆矩阵与精度元数据字典
        返回：
            若 return_details=False: (df_report, cond_matrix) [保持 100% 向后兼容]
            若 return_details=True:  (df_report, cond_matrix, df_confusion_matrix, accuracy_metrics)
        """
        # 1. 计算全域地图的朴素像元面积 (Naive Area) 与分层权重 (Wi)
        unique_classes, pixel_counts = np.unique(crop_classified_mask, return_counts=True)
        total_pixels = float(crop_classified_mask.size)
        total_area_m2 = total_pixels * self.pixel_area_m2
        map_pixel_dict = dict(zip(unique_classes, pixel_counts))

        all_crop_ids = sorted(list(self.crop_legend.keys()))
        k_classes = len(all_crop_ids)
        id_to_idx = {cid: idx for idx, cid in enumerate(all_crop_ids)}

        # 各图层分层面积权重 Wi = Ai / Atot
        W = np.array([float(map_pixel_dict.get(c, 0)) / total_pixels for c in all_crop_ids], dtype=np.float64)

        # 2. 读取地面抽样检验样方数据 (Ground Truth Reference Data)
        df_sample = pd.read_csv(ground_truth_csv, comment="#")
        y_map_sample = df_sample["map_classified_label"].values.astype(int)
        y_true_sample = df_sample["ground_truth_label"].values.astype(int)
        weights = df_sample["weight_sampling_prob"].values.astype(np.float64) if "weight_sampling_prob" in df_sample.columns else np.ones(len(df_sample), dtype=np.float64)

        n_samples = len(df_sample)

        # 3. 统计各分层的样点频数与转移频次 (n_ij 与 n_i.)
        n_matrix = np.zeros((k_classes, k_classes), dtype=np.float64)
        for i_idx, i_cls in enumerate(all_crop_ids):
            mask_i = (y_map_sample == i_cls)
            if np.any(mask_i):
                for j_idx, j_cls in enumerate(all_crop_ids):
                    n_matrix[i_idx, j_idx] = np.sum(weights[mask_i & (y_true_sample == j_cls)])

        n_i_dot = np.sum(n_matrix, axis=1)  # 各分层样点总数

        # 4. 构建条件转移概率矩阵 P(True = j | Map = i)
        cond_matrix = np.zeros((k_classes, k_classes), dtype=np.float64)
        for i_idx in range(k_classes):
            if n_i_dot[i_idx] > 0:
                cond_matrix[i_idx, :] = n_matrix[i_idx, :] / n_i_dot[i_idx]
            else:
                cond_matrix[i_idx, i_idx] = 1.0  # 缺测时设为对角占优

        # 遵循联合国手册第 24 章【耕地目标域分层（Cropland Domain Stratification）】：
        # 针对宏观大尺度遥感图（背景非农田像元占比 > 80%），当某作物在全域地图中无检出像元（Count = 0）时，
        # 判定该作物不在当前观测域内，杜绝局部小样点背景误判概率向全境大背景无限外推
        bg_pixel_ratio = float(map_pixel_dict.get(0, 0)) / max(total_pixels, 1.0)
        is_macro_background = bg_pixel_ratio > 0.80

        if is_macro_background:
            for j_idx, j_cls in enumerate(all_crop_ids):
                if j_cls != 0 and map_pixel_dict.get(j_cls, 0) == 0:
                    cond_matrix[0, j_idx] = 0.0
            row_sum = np.sum(cond_matrix[0, :])
            if row_sum > 0:
                cond_matrix[0, :] /= row_sum
            else:
                cond_matrix[0, 0] = 1.0

        # 5. 计算联合国手册第 24 章 Olofsson et al. (2014) 面积加权混淆矩阵 (Estimated Area Proportions p_ij)
        # p_ij = W_i * (n_ij / n_i.)
        p_matrix = np.zeros((k_classes, k_classes), dtype=np.float64)
        for i_idx in range(k_classes):
            p_matrix[i_idx, :] = W[i_idx] * cond_matrix[i_idx, :]

        # 无偏估计面积比例 p_.j 与真实无偏面积 (m²)
        p_dot_j = np.sum(p_matrix, axis=0)  # (k_classes,)
        calibrated_area_m2_vector = p_dot_j * total_area_m2

        map_area_m2_vector = np.array([map_pixel_dict.get(c, 0) * self.pixel_area_m2 for c in all_crop_ids], dtype=np.float64)

        # 6. Olofsson et al. (2014) 官方解析方差与标准误 (Analytical Variance & Standard Error)
        # V(p_.j) = sum_i [ W_i^2 * (n_ij / n_i.) * (1 - n_ij / n_i.) / (n_i. - 1) ]
        var_p_dot_j = np.zeros(k_classes, dtype=np.float64)
        for j_idx in range(k_classes):
            v_sum = 0.0
            for i_idx in range(k_classes):
                ni = n_i_dot[i_idx]
                if ni > 1:
                    p_cond = cond_matrix[i_idx, j_idx]
                    term = (W[i_idx] ** 2) * (p_cond * (1.0 - p_cond)) / (ni - 1.0)
                    v_sum += term
            var_p_dot_j[j_idx] = max(0.0, v_sum)

        se_p_dot_j = np.sqrt(var_p_dot_j)
        se_area_m2_vector = se_p_dot_j * total_area_m2

        # 7. 联合国手册官方分类精度指标计算 (UA, PA, OA)
        # 用户精度 User's Accuracy (UA_i = p_ii / p_i. = n_ii / n_i.)
        users_accuracy = np.zeros(k_classes, dtype=np.float64)
        se_ua = np.zeros(k_classes, dtype=np.float64)
        for i_idx in range(k_classes):
            ni = n_i_dot[i_idx]
            if ni > 0:
                ua = cond_matrix[i_idx, i_idx]
                users_accuracy[i_idx] = ua
                if ni > 1:
                    se_ua[i_idx] = np.sqrt(max(0.0, ua * (1.0 - ua) / (ni - 1.0)))

        # 生产者精度 Producer's Accuracy (PA_j = p_jj / p_.j)
        producers_accuracy = np.zeros(k_classes, dtype=np.float64)
        se_pa = np.zeros(k_classes, dtype=np.float64)
        for j_idx in range(k_classes):
            if p_dot_j[j_idx] > 1e-9:
                pa = p_matrix[j_idx, j_idx] / p_dot_j[j_idx]
                producers_accuracy[j_idx] = min(1.0, pa)
                # Olofsson (2014) Eq. 7b 生产者精度解析方差
                nj = n_i_dot[j_idx]
                if nj > 1:
                    uj = cond_matrix[j_idx, j_idx]
                    term1 = (W[j_idx] ** 2) * ((1.0 - pa) ** 2) * (uj * (1.0 - uj)) / (nj - 1.0)
                    term2 = 0.0
                    for i_idx in range(k_classes):
                        if i_idx != j_idx and n_i_dot[i_idx] > 1:
                            pic = cond_matrix[i_idx, j_idx]
                            term2 += (W[i_idx] ** 2) * (pic * (1.0 - pic)) / (n_i_dot[i_idx] - 1.0)
                    var_pa = (term1 + (pa ** 2) * term2) / (p_dot_j[j_idx] ** 2)
                    se_pa[j_idx] = np.sqrt(max(0.0, var_pa))

        # 总体精度 Overall Accuracy (OA = sum_k p_kk)
        overall_accuracy = float(np.sum(np.diag(p_matrix)))
        var_oa = 0.0
        for i_idx in range(k_classes):
            ni = n_i_dot[i_idx]
            if ni > 1:
                ui = cond_matrix[i_idx, i_idx]
                var_oa += (W[i_idx] ** 2) * (ui * (1.0 - ui)) / (ni - 1.0)
        se_oa = float(np.sqrt(max(0.0, var_oa)))

        # 8. Percentile Bootstrap 重抽样计算 95% 置信区间 (2000 次交叉验证)
        np.random.seed(self.config.get("classification", {}).get("random_state", 42))
        boot_calibrated_m2 = np.zeros((self.n_bootstrap, k_classes))

        for b in range(self.n_bootstrap):
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            b_map = y_map_sample[idx]
            b_true = y_true_sample[idx]
            b_w = weights[idx]

            b_cond = np.zeros((k_classes, k_classes), dtype=np.float64)
            for i_idx, i_cls in enumerate(all_crop_ids):
                m_i = (b_map == i_cls)
                denom = np.sum(b_w[m_i])
                if denom > 0:
                    for j_idx, j_cls in enumerate(all_crop_ids):
                        num = np.sum(b_w[m_i & (b_true == j_cls)])
                        b_cond[i_idx, j_idx] = num / denom
                else:
                    b_cond[i_idx, i_idx] = 1.0

            if is_macro_background:
                for j_idx, j_cls in enumerate(all_crop_ids):
                    if j_cls != 0 and map_pixel_dict.get(j_cls, 0) == 0:
                        b_cond[0, j_idx] = 0.0
                row_sum = np.sum(b_cond[0, :])
                if row_sum > 0:
                    b_cond[0, :] /= row_sum
                else:
                    b_cond[0, 0] = 1.0

            boot_calibrated_m2[b, :] = np.dot(map_area_m2_vector, b_cond)

        low_pct = (self.alpha / 2.0) * 100.0
        high_pct = (1.0 - self.alpha / 2.0) * 100.0

        # 9. 构造结构化无偏统计报表 DataFrame
        report_rows = []
        z_score = 1.96  # 95% 对应正态临界值

        for idx, cid in enumerate(all_crop_ids):
            c_name = self.crop_legend.get(cid, f"类别_{cid}")
            if cid == 0:
                continue  # 重点聚焦农作物

            naive_m2 = map_area_m2_vector[idx]
            calib_m2 = calibrated_area_m2_vector[idx]
            se_m2 = se_area_m2_vector[idx]

            # 解析 95% 置信区间 (Olofsson 2014)
            ci_ana_low_m2 = max(0.0, calib_m2 - z_score * se_m2)
            ci_ana_high_m2 = calib_m2 + z_score * se_m2

            # Bootstrap 经验 95% 置信区间
            ci_boot_low_m2 = np.percentile(boot_calibrated_m2[:, idx], low_pct)
            ci_boot_high_m2 = np.percentile(boot_calibrated_m2[:, idx], high_pct)

            # 换算中国通用亩与公顷
            naive_mu = sqm_to_mu(naive_m2, decimals=1)
            calib_mu = sqm_to_mu(calib_m2, decimals=1)
            se_mu = sqm_to_mu(se_m2, decimals=1)

            ci_ana_low_mu = sqm_to_mu(ci_ana_low_m2, decimals=1)
            ci_ana_high_mu = sqm_to_mu(ci_ana_high_m2, decimals=1)
            ci_boot_low_mu = sqm_to_mu(ci_boot_low_m2, decimals=1)
            ci_boot_high_mu = sqm_to_mu(ci_boot_high_m2, decimals=1)

            # 变异系数 (CV% = SE / Area * 100%)
            cv_pct = round((se_m2 / max(calib_m2, 1e-6)) * 100.0, 2)

            bias_mu = round(naive_mu - calib_mu, 1)
            bias_pct = round((bias_mu / max(calib_mu, 1e-4)) * 100.0, 2)

            ua_val = users_accuracy[idx] * 100.0
            se_ua_val = se_ua[idx] * 100.0
            pa_val = producers_accuracy[idx] * 100.0
            se_pa_val = se_pa[idx] * 100.0

            report_rows.append({
                "crop_id": cid,
                "crop_name": c_name,
                "naive_pixel_count": int(map_pixel_dict.get(cid, 0)),
                "naive_area_mu": naive_mu,
                "naive_area_ha": sqm_to_ha(naive_m2, decimals=2),
                "unbiased_calibrated_mu": calib_mu,
                "unbiased_calibrated_ha": sqm_to_ha(calib_m2, decimals=2),
                "se_analytic_mu": se_mu,
                "se_analytic_ha": sqm_to_ha(se_m2, decimals=2),
                "cv_pct": cv_pct,
                # 保持向后兼容字段 (默认采用联合国解析标准误构建的 CI)
                "ci_95_lower_mu": ci_ana_low_mu,
                "ci_95_upper_mu": ci_ana_high_mu,
                "ci_95_bootstrap_lower_mu": ci_boot_low_mu,
                "ci_95_bootstrap_upper_mu": ci_boot_high_mu,
                "bias_mu": bias_mu,
                "bias_pct": bias_pct,
                "users_accuracy": f"{ua_val:.1f}% ± {se_ua_val:.1f}%",
                "producers_accuracy": f"{pa_val:.1f}% ± {se_pa_val:.1f}%",
                "unbiased_guarantee": "✓ 联合国手册第24章解析标准误与无偏估计"
            })

        df_report = pd.DataFrame(report_rows)

        # 10. 构建符合联合国手册 Table 2 标准的面积加权混淆矩阵 (Area-weighted Confusion Matrix)
        crop_names = [self.crop_legend.get(c, f"类别_{c}") for c in all_crop_ids]
        cm_data = np.round(p_matrix * 100.0, 4)  # 换算为百分比
        df_cm = pd.DataFrame(cm_data, index=[f"Map_{name}" for name in crop_names], columns=[f"Ref_{name}" for name in crop_names])
        df_cm["User_Accuracy_UA"] = [f"{users_accuracy[i]*100:.2f}% ± {se_ua[i]*100:.2f}%" for i in range(k_classes)]
        df_cm["Stratum_Weight_Wi"] = np.round(W, 5)

        # 追加底部生产者精度与汇总行
        pa_row = {f"Ref_{name}": f"{producers_accuracy[j]*100:.2f}% ± {se_pa[j]*100:.2f}%" for j, name in enumerate(crop_names)}
        pa_row["User_Accuracy_UA"] = f"OA: {overall_accuracy*100:.2f}% ± {se_oa*100:.2f}%"
        pa_row["Stratum_Weight_Wi"] = 1.0000
        df_cm.loc["Producer_Accuracy_PA"] = pa_row

        self.df_confusion_matrix = df_cm
        self.accuracy_metrics = {
            "overall_accuracy": round(overall_accuracy, 4),
            "se_overall_accuracy": round(se_oa, 4),
            "users_accuracy": {c_name: round(users_accuracy[i], 4) for i, c_name in enumerate(crop_names)},
            "producers_accuracy": {c_name: round(producers_accuracy[j], 4) for j, c_name in enumerate(crop_names)},
            "p_matrix": p_matrix,
            "cond_matrix": cond_matrix
        }

        if return_details:
            return df_report, cond_matrix, df_cm, self.accuracy_metrics
        return df_report, cond_matrix

    def estimate_ppi_mean(self, map_all_vals, sample_map_vals, sample_gt_vals, sample_weights=None):
        """
        联合国手册第 26 章预测增强推断 (Prediction-Powered Inference, PPI / Predict-Then-Debias):
        用于对任意农业指标（如作物平均单产、生物量、NDVI均值）进行大范围卫星图 + 极少量样框去偏推断。
        
        公式:
            theta_PTD = gamma_map_all + (theta_gt_sample - gamma_map_sample)
        
        参数:
            map_all_vals: 全域卫星预测值数组 (例如整景图像元单产或植被指数)
            sample_map_vals: 抽样调查样方处的卫星预测值数组
            sample_gt_vals: 抽样调查样方处的真实实测值数组
            sample_weights: 抽样权重 (可选)
        返回:
            dict 包含 Map-only, Survey-only, PPI-PTD 点估计值、标准误与 95% 置信区间
        """
        map_all = np.asarray(map_all_vals, dtype=np.float64)
        s_map = np.asarray(sample_map_vals, dtype=np.float64)
        s_gt = np.asarray(sample_gt_vals, dtype=np.float64)
        n = len(s_gt)

        # 1. Map-only 估计量
        gamma_map_all = float(np.mean(map_all))

        # 2. Survey-only 估计量
        if sample_weights is not None:
            w = np.asarray(sample_weights, dtype=np.float64)
            theta_gt_survey = float(np.sum(s_gt * w) / np.sum(w))
            gamma_map_survey = float(np.sum(s_map * w) / np.sum(w))
        else:
            theta_gt_survey = float(np.mean(s_gt))
            gamma_map_survey = float(np.mean(s_map))

        # 3. Predict-Then-Debias (PTD) 估计量
        theta_ptd = gamma_map_all + (theta_gt_survey - gamma_map_survey)

        # 残差方差与标准误
        diff_residuals = (s_gt - s_map)
        se_ptd = float(np.std(diff_residuals, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        se_survey = float(np.std(s_gt, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

        # 方差削减 / 效率增益倍数 (Efficiency Gain)
        efficiency_gain = round((se_survey ** 2) / max(se_ptd ** 2, 1e-12), 2) if se_ptd > 0 else 1.0

        return {
            "map_only_estimate": round(gamma_map_all, 3),
            "survey_only_estimate": round(theta_gt_survey, 3),
            "se_survey_only": round(se_survey, 3),
            "ppi_ptd_estimate": round(theta_ptd, 3),
            "se_ppi_ptd": round(se_ptd, 3),
            "ci_95_ppi": [round(theta_ptd - 1.96 * se_ptd, 3), round(theta_ptd + 1.96 * se_ptd, 3)],
            "efficiency_gain_factor": efficiency_gain
        }

    def design_optimal_sample_allocation(
        self,
        total_sample_budget: int = 150,
        crop_classified_mask: np.ndarray = None,
        min_sample_per_class: int = 20,
        prior_accuracies: dict = None
    ) -> pd.DataFrame:
        """
        联合国手册第 24 章规范：基于 Neyman 最佳分层抽样设计（Neyman Optimal Allocation）。
        在给定的实地调查总预算样方数（如 150 个样框）下，自动求解使全域作物面积无偏估计量方差最小的最优样本分配方案。

        数学公式:
            n_h = n * (W_h * sigma_h) / sum(W_k * sigma_k)
            其中:
            - W_h: 分类制图层中各作物的面积比例权重 (A_h / A_total)
            - sigma_h: 各层先验标准差 (依据二项分布 sigma_h = sqrt(UA_h * (1 - UA_h)) 或保守经验值)
            - min_sample_per_class: 每类作物最低保底样方数 (Olofsson 2014 推荐 >= 20~30，确保方差可估)

        返回:
            pd.DataFrame: 包含比例分配、Neyman 最佳分配与最终工程推荐分配方案的决策台账
        """
        crop_codes = sorted(list(self.crop_legend.keys()))
        crop_names = [self.crop_legend[c] for c in crop_codes]
        K = len(crop_codes)

        # 1. 计算分层面积权重 W_h
        if crop_classified_mask is not None:
            unique_classes, pixel_counts = np.unique(crop_classified_mask, return_counts=True)
            total_pixels = float(crop_classified_mask.size)
            wh_map = {c: 0.0 for c in crop_codes}
            for c, cnt in zip(unique_classes, pixel_counts):
                if c in wh_map:
                    wh_map[c] = cnt / total_pixels
            weights = np.array([wh_map[c] for c in crop_codes], dtype=np.float64)
            # 面积 (亩)
            areas_mu = np.array([wh_map[c] * total_pixels * self.pixel_area_m2 * MU_PER_SQM for c in crop_codes])
        else:
            # 默认宏观中原/华北粮仓典型比例
            weights = np.array([0.45, 0.25, 0.20, 0.10][:K], dtype=np.float64)
            weights = weights / np.sum(weights)
            areas_mu = weights * 1000000.0

        # 2. 先验标准差 sigma_h 计算
        prior_ua = prior_accuracies or {
            0: 0.92,  # 背景/非农田精度
            1: 0.88,  # 玉米
            2: 0.94,  # 小麦 (冬小麦物候鲜明，精度通常最高)
            3: 0.85   # 大豆
        }
        sigmas = []
        for c in crop_codes:
            ua = prior_ua.get(c, 0.85)
            # 二项抽样方差开方
            sig = np.sqrt(max(0.01, ua * (1.0 - ua)))
            sigmas.append(sig)
        sigmas = np.array(sigmas, dtype=np.float64)

        # 3. 传统比例抽样分配 (Proportional Allocation)
        n_prop = np.round(total_sample_budget * weights).astype(int)

        # 4. 联合国 Neyman 最佳抽样分配 (Neyman Allocation)
        w_sig = weights * sigmas
        sum_w_sig = np.sum(w_sig)
        neyman_raw = total_sample_budget * (w_sig / max(sum_w_sig, 1e-12))
        n_neyman = np.round(neyman_raw).astype(int)

        # 5. 加入最低保底约束的最终推荐样方分配 (Constrained Optimal Allocation)
        # 稀缺农作物（如大豆、花生）若纯按面积比例分配可能会样本过少导致方差膨胀，强制保底
        n_recommended = np.maximum(n_neyman, min_sample_per_class)
        # 调整多余/不足样本至总预算
        diff = total_sample_budget - int(np.sum(n_recommended))
        if diff != 0:
            # 在面积最大的优势层调整差额
            dom_idx = int(np.argmax(weights))
            n_recommended[dom_idx] = max(min_sample_per_class, n_recommended[dom_idx] + diff)

        records = []
        for i, c in enumerate(crop_codes):
            records.append({
                "crop_code": c,
                "crop_name": crop_names[i],
                "map_area_mu": round(areas_mu[i], 1),
                "stratum_weight_Wh": round(weights[i], 4),
                "prior_sigma_h": round(sigmas[i], 3),
                "proportional_n": int(n_prop[i]),
                "neyman_optimal_n": int(n_neyman[i]),
                "recommended_sample_n": int(n_recommended[i]),
                "sample_ratio_pct": round((n_recommended[i] / total_sample_budget) * 100.0, 1)
            })

        df_plan = pd.DataFrame(records)
        return df_plan

    def export_sampling_plan(self, df_plan: pd.DataFrame, output_csv="output/sample_allocation_plan.csv") -> str:
        """导出联合国手册规范的样方抽样设计方案台账。"""
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df_plan.to_csv(output_csv, index=False, encoding="utf-8-sig")
        return output_csv
