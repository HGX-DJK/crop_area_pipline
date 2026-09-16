"""
作物多时相时序分类模块。
基于随机森林（Random Forest）与梯度提升（Gradient Boosting）算法，
输入多时相物候特征矩阵，输出全域农作物种植分类图与置信度概率图。
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, cohen_kappa_score
from sklearn.model_selection import StratifiedKFold


class CropClassifier:
    def __init__(self, config=None):
        self.config = config or {}
        cls_cfg = self.config.get("classification", {})
        self.model_type = cls_cfg.get("model_type", "random_forest").lower()
        self.n_estimators = cls_cfg.get("n_estimators", 100)
        self.max_depth = cls_cfg.get("max_depth", 12)
        self.random_state = cls_cfg.get("random_state", 42)
        self.n_jobs = cls_cfg.get("n_jobs", -1)
        self.model = self._init_model()
        self.crop_legend = self.config.get("crop_legend", {
            0: "非农田/背景",
            1: "夏玉米",
            2: "冬小麦",
            3: "大豆"
        })
        self.is_trained = False
        self.metrics = {}

    def _init_model(self):
        if self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=min(self.max_depth, 6),
                random_state=self.random_state
            )
        else:
            return RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                class_weight="balanced",
                n_jobs=self.n_jobs
            )

    def train_with_samples(self, training_csv_path="data/sample_training_points.csv", ts_builder=None, target_t=None, doy_list=None):
        """
        基于外部抽样调查标定样本训练分类器，并执行 5 折交叉验证。
        若传入 ts_builder，则通过特征工程模块对标定样点执行相同的物候特征提取。
        - 具备自动时相自适应对齐功能（无论输入是 1 个时相、多时相还是全时序，均自动对齐特征空间）。
        """
        df = pd.read_csv(training_csv_path, comment="#")

        doy_cols = [c for c in df.columns if c.startswith("doy_")]
        sample_doys = [int(c.replace("doy_", "")) for c in doy_cols]

        if doy_cols and ts_builder is not None:
            raw_ts = df[doy_cols].values  # (N, 8)

            # 自适应特征时序维度对齐
            if target_t is not None and target_t != raw_ts.shape[1]:
                print(f"[分类器特征对齐] 输入影像时相数 (T={target_t}) 与标定样本基准 (T={raw_ts.shape[1]}) 不同，正在执行自适应对齐...")
                if target_t == 1:
                    # 单时相影像：匹配对应 DOY 或提取盛夏作物生长峰值期 (DOY 200)
                    if doy_list and len(doy_list) > 0:
                        closest_idx = int(np.argmin([abs(d - doy_list[0]) for d in sample_doys]))
                    else:
                        closest_idx = int(np.argmin([abs(d - 200) for d in sample_doys]))
                    ts_values = raw_ts[:, [closest_idx]]
                    print(f"  -> 自动匹配提取对应生长旺季 DOY {sample_doys[closest_idx]} 单时相物候特征")
                else:
                    # 多时相数量差异：沿时间轴执行物候曲线线性插值对齐
                    target_doys = doy_list if (doy_list and len(doy_list) == target_t) else np.linspace(sample_doys[0], sample_doys[-1], target_t)
                    aligned_list = []
                    for row in raw_ts:
                        interp_row = np.interp(target_doys, sample_doys, row)
                        aligned_list.append(interp_row)
                    ts_values = np.array(aligned_list, dtype=np.float32)
                    print(f"  -> 成功将标定样本插值对齐至目标 {target_t} 个生长时相")
            else:
                ts_values = raw_ts

            X = ts_builder.extract_phenological_features(ts_values)
            feature_cols = [f"feat_{i+1}" for i in range(X.shape[1])]
        else:
            feature_cols = [c for c in df.columns if c not in ["point_id", "label", "crop_name"]]
            X = df[feature_cols].values
        
        y = df["label"].values.astype(int)

        # 5 折交叉验证
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        y_cv_pred = np.zeros_like(y)

        for train_idx, val_idx in skf.split(X, y):
            fold_clf = self._init_model()
            fold_clf.fit(X[train_idx], y[train_idx])
            y_cv_pred[val_idx] = fold_clf.predict(X[val_idx])

        oa = accuracy_score(y, y_cv_pred)
        kappa = cohen_kappa_score(y, y_cv_pred)
        cm = confusion_matrix(y, y_cv_pred)

        self.metrics = {
            "overall_accuracy": round(float(oa), 4),
            "kappa_coefficient": round(float(kappa), 4),
            "confusion_matrix": cm.tolist(),
            "n_samples": len(df),
            "feature_names": feature_cols
        }

        # 训练全量最终模型
        self.model.fit(X, y)
        self.is_trained = True

        print(f"[分类器] 基于 {len(df)} 个样本完成 {self.model_type.upper()} 模型训练 (输入特征数: {X.shape[1]})。")
        print(f"[评估] 5折交叉验证总体精度 (OA): {oa * 100:.2f}%, Kappa系数: {kappa:.3f}")
        return self.metrics

    def fit_from_arrays(self, X_train, y_train):
        """直接从特征矩阵与标签数组进行训练。"""
        self.model.fit(X_train, y_train)
        self.is_trained = True
        return self

    def predict_raster_cube(self, feature_cube, batch_size=200000):
        """
        对整景三维多时相物候特征立方体 (Rows, Cols, Features) 执行像素级流式分批推断。
        采用分批切片（Batch Streaming）技术，内存开销恒定，彻底避免超大幅宽遥感影像导致的内存溢出 (OOM)。
        自动识别并安全处理 NoData / NaN / Inf 像元，确保极端遥感场景不报错。
        """
        if not self.is_trained:
            raise RuntimeError("分类器尚未训练，无法执行全域空间预测。")

        h, w, f = feature_cube.shape
        X_flat = feature_cube.reshape(-1, f)
        total_pixels = X_flat.shape[0]

        # 检查特征维度一致性
        if hasattr(self.model, "n_features_in_") and f != self.model.n_features_in_:
            raise ValueError(f"输入特征立方体维度 ({f}) 与分类器训练特征维度 ({self.model.n_features_in_}) 不匹配！请确保时相数与训练配置对齐。")

        # 识别有效像元与无效像元 (如 NoData / NaN / Inf)，彻底避免 sklearn predict 报错
        valid_mask = np.isfinite(X_flat).all(axis=1)
        valid_count = int(np.sum(valid_mask))

        preds_flat = np.zeros(total_pixels, dtype=np.int32)
        max_probs = np.zeros(total_pixels, dtype=np.float32)

        if valid_count > 0:
            valid_indices = np.where(valid_mask)[0]
            X_valid = X_flat[valid_indices]

            # 针对有效像元执行分批流式推断
            if valid_count <= batch_size:
                preds_flat[valid_indices] = self.model.predict(X_valid)
                probs_valid = self.model.predict_proba(X_valid)
                max_probs[valid_indices] = np.max(probs_valid, axis=1)
            else:
                for start_idx in range(0, valid_count, batch_size):
                    end_idx = min(start_idx + batch_size, valid_count)
                    chunk_X = X_valid[start_idx:end_idx]
                    chunk_indices = valid_indices[start_idx:end_idx]

                    chunk_pred = self.model.predict(chunk_X)
                    chunk_prob = self.model.predict_proba(chunk_X)

                    preds_flat[chunk_indices] = chunk_pred
                    max_probs[chunk_indices] = np.max(chunk_prob, axis=1)

        predicted_mask = preds_flat.reshape(h, w).astype(np.int32)
        confidence_map = max_probs.reshape(h, w).astype(np.float32)

        return predicted_mask, confidence_map
