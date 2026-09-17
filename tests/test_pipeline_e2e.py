"""
端到端农业遥感空间提取流水线集成回归测试。
"""

import os
import shutil
import unittest
import numpy as np

from src.time_series_builder import TimeSeriesBuilder
from src.crop_classifier import CropClassifier
from src.parcel_segmenter import ParcelSegmenter
from src.vector_exporter import VectorExporter
from src.area_unbiased_estimator import AreaUnbiasedEstimator
from src.report_generator import ExecutiveReportGenerator


class TestPipelineE2E(unittest.TestCase):
    """端到端全链路集成回归测试"""

    @classmethod
    def setUpClass(cls):
        cls.test_output_dir = "output/test_e2e_run"
        os.makedirs(cls.test_output_dir, exist_ok=True)
        cls.config = {
            "spatial": {
                "resolution_meters": 10.0,
                "crs": "EPSG:32650",
                "utm_zone": 50,
                "export_crs": "WGS84",
                "origin_x": 500000.0,
                "origin_y": 4200000.0
            },
            "crop_legend": {
                0: "非农田/背景",
                1: "夏玉米",
                2: "冬小麦",
                3: "大豆"
            },
            "segmentation": {
                "apply_boundary_erosion": True,
                "min_parcel_area_m2": 200.0,
                "max_parcel_area_m2": 500000.0,
                "max_export_parcels": 20,
                "connectivity": 8
            },
            "classification": {
                "model_type": "random_forest",
                "n_estimators": 10,
                "max_depth": 6,
                "random_state": 42
            },
            "unbiased_area_inference": {
                "n_bootstrap": 100,
                "confidence_level": 0.95
            },
            "paths": {
                "output_dir": cls.test_output_dir
            }
        }

    @classmethod
    def tearDownClass(cls):
        # 清理临时测试产物
        if os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir, ignore_errors=True)

    def test_full_pipeline_flow(self):
        """验证从模拟数据生成 -> 分类 -> 切分 -> 矢量化 -> 无偏统计 -> 专报生成的全流程"""
        # 1. 生成仿真多时相测试景观 (60x60 极速模式)
        ts_builder = TimeSeriesBuilder(self.config)
        synthetic_data = ts_builder.generate_synthetic_agricultural_landscape(rows=60, cols=60, random_seed=42)
        cube = synthetic_data["raster_cube"]
        gt_mask = synthetic_data["ground_truth_mask"]

        # 2. 提取物候特征
        features_cube = ts_builder.extract_phenological_features(cube)
        self.assertEqual(features_cube.shape[:2], (60, 60))

        # 3. 训练并预测
        classifier = CropClassifier(self.config)
        X_all = features_cube.reshape(-1, features_cube.shape[2])
        y_all = gt_mask.flatten()
        sample_idx = np.random.choice(len(y_all), size=80, replace=False)
        classifier.fit_from_arrays(X_all[sample_idx], y_all[sample_idx])

        pred_mask, conf_map = classifier.predict_raster_cube(features_cube, batch_size=1000)
        self.assertEqual(pred_mask.shape, (60, 60))

        # 4. 零碎地块形态学切分
        segmenter = ParcelSegmenter(self.config)
        parcel_mask, parcel_meta = segmenter.segment_parcels(pred_mask, conf_map)
        self.assertGreater(len(parcel_meta), 0, "应提取到至少 1 个有效规整地块")

        # 5. 导出矢量 GeoJSON 与属性表
        exporter = VectorExporter(self.config)
        geojson_path = os.path.join(self.test_output_dir, "test_parcels.geojson")
        exporter.export_geojson(parcel_mask, parcel_meta, output_path=geojson_path)
        self.assertTrue(os.path.exists(geojson_path))

        # 检查 WebGIS 地图生成
        map_html_path = geojson_path.replace(".geojson", "_map.html")
        self.assertTrue(os.path.exists(map_html_path))

        # 6. 无偏面积推断
        estimator = AreaUnbiasedEstimator(self.config)
        df_area, _ = estimator.estimate_unbiased_areas(pred_mask, ground_truth_csv="data/ground_truth_area_sample.csv")
        area_csv_path = os.path.join(self.test_output_dir, "acreage_report.csv")
        df_area.to_csv(area_csv_path, index=False, encoding="utf-8-sig")
        self.assertTrue(os.path.exists(area_csv_path))

        # 7. 呈报专报生成
        report_gen = ExecutiveReportGenerator(self.config)
        report_html_path = os.path.join(self.test_output_dir, "test_report.html")
        csv_parcels = geojson_path.replace(".geojson", "_attribute_table.csv")
        out_path = report_gen.generate_report(area_csv_path, csv_parcels, report_html_path)
        self.assertTrue(os.path.exists(out_path))
        self.assertGreater(os.path.getsize(out_path), 5000, "生成的决策专报大小应大于 5KB")


if __name__ == "__main__":
    unittest.main()
