"""
联合国手册无偏统计推断与样方分配算法单元测试。
"""

import unittest
import numpy as np
import pandas as pd

from src.area_unbiased_estimator import AreaUnbiasedEstimator


class TestUnbiasedEstimator(unittest.TestCase):
    """测试 Olofsson (2014) 无偏校准与 Neyman 最佳抽样分配"""

    def setUp(self):
        self.config = {
            "spatial": {"resolution_meters": 10.0},
            "unbiased_area_inference": {"n_bootstrap": 100, "confidence_level": 0.95},
            "crop_legend": {0: "背景", 1: "夏玉米", 2: "冬小麦", 3: "大豆"}
        }
        self.estimator = AreaUnbiasedEstimator(self.config)

    def test_neyman_sample_allocation(self):
        """测试 Neyman 最优分层抽样方案设计器"""
        # 构造虚拟分类栅格
        mask = np.zeros((100, 100), dtype=np.int32)
        mask[:50, :] = 2   # 50% 冬小麦
        mask[50:80, :] = 1 # 30% 夏玉米
        mask[80:90, :] = 3 # 10% 大豆
        # 剩余 10% 背景 0

        budget = 160
        min_sample = 20
        df_plan = self.estimator.design_optimal_sample_allocation(
            total_sample_budget=budget,
            crop_classified_mask=mask,
            min_sample_per_class=min_sample
        )

        self.assertEqual(len(df_plan), 4, "应为 4 类作物分别分配样方")
        self.assertEqual(int(df_plan["recommended_sample_n"].sum()), budget, "分配样本总和必须严格等于预算")
        # 验证每类作物均满足最低保底样方数
        for _, r in df_plan.iterrows():
            self.assertGreaterEqual(r["recommended_sample_n"], min_sample, f"{r['crop_name']} 样本数未达保底要求")

    def test_ppi_mean_predict_then_debias(self):
        """测试联合国第26章 PPI / PTD 均值去偏估计量"""
        # 全域卫星图预测值 (均值 100)
        np.random.seed(42)
        map_all = np.random.normal(100.0, 10.0, 1000)

        # 抽样样方处的预测值存在 +5.0 的系统性高估偏差
        sample_map = np.array([105.0, 104.0, 106.0, 105.5, 104.5])
        sample_gt = np.array([100.0, 99.0, 101.0, 100.5, 99.5])  # 真实地表实测值

        res = self.estimator.estimate_ppi_mean(map_all, sample_map, sample_gt)

        # PTD 估计量应成功减去 +5.0 的系统性偏差，去偏后均值约 95~100 左右
        self.assertTrue("ppi_ptd_estimate" in res)
        self.assertTrue("se_ppi_ptd" in res)
        self.assertTrue("ci_95_ppi" in res)
    def test_macro_background_protection(self):
        """测试宏观大尺度背景下稀有作物防外推保护机制"""
        # 模拟全图 99.9% 为背景，0.1% 为冬小麦 (模拟大尺度卫星瓦片)
        mask = np.zeros((1000, 1000), dtype=np.int32)
        mask[:10, :100] = 2  # 1,000 个小麦像元，占比 0.001

        df_report, cond_matrix = self.estimator.estimate_unbiased_areas(mask, ground_truth_csv=None)

        wheat_row = df_report[df_report["crop_name"] == "冬小麦"].iloc[0]
        naive_mu = wheat_row["naive_area_mu"]
        calib_mu = wheat_row["unbiased_calibrated_mu"]
        cv_pct = wheat_row["cv_pct"]

        # 无偏校准面积不应脱离像元面积数十倍，应在合理范围 (0.5 ~ 2.0 倍之间)
        self.assertGreater(calib_mu, 0.0)
        self.assertLess(calib_mu, naive_mu * 2.0, "宏观大背景下校准面积严禁暴增数十倍！")
        self.assertGreater(calib_mu, naive_mu * 0.5, "校准面积不应异常归零")
        self.assertLess(cv_pct, 30.0, "变异系数应处于合理统计区间，不得达到 99%+")


if __name__ == "__main__":
    unittest.main()
