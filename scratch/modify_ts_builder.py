import re

with open('src/time_series_builder.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Update channel splitting logic
split_target = """        is_dual_channel = getattr(self, "is_sdc6_dual", False)
        if is_dual_channel:
            ts_ndvi = ts[:, 0::2]   # ż = NDVI
            ts_lswi = ts[:, 1::2]   #  = LSWI
            ts_for_pheno = ts_ndvi
            t_eff = ts_ndvi.shape[1]
        else:
            ts_lswi = None
            ts_for_pheno = ts
            t_eff = t"""

split_replacement = """        is_dual_channel = getattr(self, "is_sdc6_dual", False)
        if is_dual_channel:
            # Check if it's actually 3 channels (NDVI, LSWI, GCVI)
            if ts.shape[1] % 3 == 0 and not (ts.shape[1] % 2 == 0 and ts.shape[1] % 3 != 0):
                ts_ndvi = ts[:, 0::3]
                ts_lswi = ts[:, 1::3]
                ts_gcvi = ts[:, 2::3]
                t_eff = ts_ndvi.shape[1]
            else:
                ts_ndvi = ts[:, 0::2]
                ts_lswi = ts[:, 1::2]
                ts_gcvi = None
                t_eff = ts_ndvi.shape[1]
            ts_for_pheno = ts_ndvi
        else:
            ts_lswi = None
            ts_gcvi = None
            ts_for_pheno = ts
            t_eff = t"""
content = content.replace(split_target, split_replacement)

# 2. Add GCVI features calculation
feat_target = """        # 6. LSWI ˮʪص
        if ts_lswi is not None:
            lswi_max = np.max(ts_lswi, axis=1, keepdims=True)
            lswi_min = np.min(ts_lswi, axis=1, keepdims=True)
            lswi_mean = np.mean(ts_lswi, axis=1, keepdims=True)
            lswi_std = np.std(ts_lswi, axis=1, keepdims=True)
            lswi_ndvi_diff = ndvi_max - lswi_max
            lswi_feats = np.hstack([lswi_max, lswi_min, lswi_mean, lswi_std, lswi_ndvi_diff])
        else:
            lswi_feats = np.zeros((n_samples, 5), dtype=np.float32)

        # ƴع
        features = np.hstack([
            ts_for_pheno,
            ndvi_max, ndvi_min, ndvi_range, ndvi_std,
            grad_max, grad_min, grad_mean,
            early_slope, mid_slope, late_drop,
            paddy_flooding, bimodal_score, persistent_green,
            lswi_feats
        ])"""

feat_replacement = """        # 6. LSWI ˮʪص
        if ts_lswi is not None:
            lswi_max = np.max(ts_lswi, axis=1, keepdims=True)
            lswi_min = np.min(ts_lswi, axis=1, keepdims=True)
            lswi_mean = np.mean(ts_lswi, axis=1, keepdims=True)
            lswi_std = np.std(ts_lswi, axis=1, keepdims=True)
            lswi_ndvi_diff = ndvi_max - lswi_max
            lswi_feats = np.hstack([lswi_max, lswi_min, lswi_mean, lswi_std, lswi_ndvi_diff])
        else:
            lswi_feats = np.zeros((n_samples, 5), dtype=np.float32)
            
        # 7. GCVI Ҷ̼
        if ts_gcvi is not None:
            gcvi_max = np.max(ts_gcvi, axis=1, keepdims=True)
            gcvi_mean = np.mean(ts_gcvi, axis=1, keepdims=True)
            gcvi_std = np.std(ts_gcvi, axis=1, keepdims=True)
            gcvi_feats = np.hstack([gcvi_max, gcvi_mean, gcvi_std])
        else:
            gcvi_feats = np.zeros((n_samples, 3), dtype=np.float32)

        # ƴع
        features = np.hstack([
            ts_for_pheno,
            ndvi_max, ndvi_min, ndvi_range, ndvi_std,
            grad_max, grad_min, grad_mean,
            early_slope, mid_slope, late_drop,
            paddy_flooding, bimodal_score, persistent_green,
            lswi_feats, gcvi_feats
        ])"""
content = content.replace(feat_target, feat_replacement)

with open('src/time_series_builder.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Updated time_series_builder.py")
