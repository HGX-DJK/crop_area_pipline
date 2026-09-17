"""
大规模遥感影像吞吐与算力性能（工业级落地）专项测试套件。
验证：
1. 零拷贝切片差分与边缘检测等价性；
2. 多核多进程并行矢量化与单线程结果严格一致性；
3. 滑动窗口分块流式推断与全局推断数值一致性；
4. 底层 C 级 argsort 斑块排序正确性。
"""

import os
import unittest
import numpy as np
import tempfile
import json

from src.time_series_builder import TimeSeriesBuilder
from src.crop_classifier import CropClassifier
from src.parcel_segmenter import ParcelSegmenter
from src.vector_exporter import VectorExporter


class TestPerformanceScaling(unittest.TestCase):
    """工业级性能扩展性与流式运算测试"""

    def setUp(self):
        self.config = {
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
                "max_export_parcels": 50,
                "smooth_boundaries": True,
                "connectivity": 8
            },
            "classification": {
                "model_type": "random_forest",
                "n_estimators": 10,
                "max_depth": 6,
                "n_jobs": 1,
                "random_state": 42
            },
            "performance": {
                "streaming_block_size": 32,
                "vectorization_n_jobs": 2
            }
        }

    def test_zero_copy_gradient_slicing(self):
        """验证零拷贝切片差分检测相邻不同作物边界的准确性"""
        mask = np.zeros((10, 10), dtype=np.int32)
        mask[2:5, 2:5] = 1  # 作物 1
        mask[2:5, 5:8] = 2  # 作物 2 (与作物 1 在水平列 4 和列 5 紧密相邻)

        rows, cols = mask.shape
        gradient_edges = np.zeros((rows, cols), dtype=bool)

        if rows > 1:
            diff_v = (mask[:-1, :] > 0) & (mask[1:, :] > 0) & (mask[:-1, :] != mask[1:, :])
            gradient_edges[:-1, :][diff_v] = True
            gradient_edges[1:, :][diff_v] = True
        if cols > 1:
            diff_h = (mask[:, :-1] > 0) & (mask[:, 1:] > 0) & (mask[:, :-1] != mask[:, 1:])
            gradient_edges[:, :-1][diff_h] = True
            gradient_edges[:, 1:][diff_h] = True

        # 列 4 和列 5 的相邻行必须全部被标记为切开边缘
        for r in range(2, 5):
            self.assertTrue(gradient_edges[r, 4], f"像元 ({r}, 4) 应被标记为分界线")
            self.assertTrue(gradient_edges[r, 5], f"像元 ({r}, 5) 应被标记为分界线")

        # 同作物内部像元不得被误标记
        self.assertFalse(gradient_edges[3, 3], "同种作物内部像元不应被误切")
        self.assertFalse(gradient_edges[3, 6], "同种作物内部像元不应被误切")

    def test_c_level_argsort_ranking(self):
        """验证底层 C 级 argsort 降序排序与原生 Python 排序逻辑严格一致"""
        sizes = np.array([120, 50, 430, 12, 900, 310, 80], dtype=np.int64)
        num_features = len(sizes)

        # 优化后：底层 argsort
        sorted_order = np.argsort(-sizes)
        comp_indices_fast = (sorted_order + 1).tolist()

        # 基准：Python 原生 sorted
        comp_indices_py = sorted(list(range(1, num_features + 1)), key=lambda cid: sizes[cid - 1], reverse=True)

        self.assertEqual(comp_indices_fast, comp_indices_py, "快速 argsort 排序结果必须与原生排序 100% 一致")
        # 验证降序特性
        ranked_sizes = [sizes[i - 1] for i in comp_indices_fast]
        self.assertEqual(ranked_sizes, sorted(sizes.tolist(), reverse=True))

    def test_window_streaming_prediction_equivalence(self):
        """验证滑动窗口流式推断结果与全局一次性推断结果完全一致"""
        ts_builder = TimeSeriesBuilder(self.config)
        # 生成小型多时相立方体 (64, 64, 8)
        landscape = ts_builder.generate_synthetic_agricultural_landscape(rows=64, cols=64)
        raster_cube = landscape["raster_cube"]

        classifier = CropClassifier(self.config)
        # 训练模型
        classifier.train_with_samples(
            "data/sample_training_points.csv",
            ts_builder=ts_builder,
            target_t=8
        )

        # 1. 全局一次性推断
        feature_cube = ts_builder.extract_phenological_features(raster_cube)
        mask_global, conf_global = classifier.predict_raster_cube(feature_cube)

        # 2. 分块滑动窗口流式推断 (窗口 32x32)
        mask_stream, conf_stream = classifier.predict_cube_stream(raster_cube, ts_builder, block_size=32)

        # 验证两者分类标签 100% 等价
        self.assertTrue(np.array_equal(mask_global, mask_stream), "流式分块预测分类掩膜必须与全局预测完全一致")
        # 验证置信度浮点数值在数值精度范围内一致
        self.assertTrue(np.allclose(conf_global, conf_stream, atol=1e-5), "流式分块预测置信度矩阵必须与全局预测完全一致")

    def test_parallel_vectorization_equivalence(self):
        """验证多核多进程并行矢量化与单线程串行矢量化结果完全等价"""
        # 构建具备多个地块的掩膜
        mask = np.zeros((80, 80), dtype=np.int32)
        metadata = []
        pid = 1
        for r_idx in range(5):
            for c_idx in range(5):
                r0, r1 = r_idx * 15 + 2, r_idx * 15 + 12
                c0, c1 = c_idx * 15 + 2, c_idx * 15 + 12
                mask[r0:r1, c0:c1] = pid
                crop_code = (pid % 3) + 1
                metadata.append({
                    "parcel_id": f"P{pid:04d}",
                    "internal_id": pid,
                    "crop_code": crop_code,
                    "pixel_count": 100,
                    "area_m2": 10000.0,
                    "area_mu": 15.0,
                    "area_ha": 1.0,
                    "dominant_purity": 1.0,
                    "mean_confidence": 0.95,
                    "centroid_row": float((r0 + r1) / 2.0),
                    "centroid_col": float((c0 + c1) / 2.0)
                })
                pid += 1

        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 单线程导出
            cfg_serial = dict(self.config)
            cfg_serial["performance"] = {"vectorization_n_jobs": 1}
            exporter_serial = VectorExporter(cfg_serial)
            out_serial = os.path.join(tmpdir, "serial.geojson")
            exporter_serial.export_geojson(mask, metadata, out_serial)

            # 2. 多核多进程导出
            cfg_parallel = dict(self.config)
            cfg_parallel["performance"] = {"vectorization_n_jobs": 2}
            exporter_parallel = VectorExporter(cfg_parallel)
            out_parallel = os.path.join(tmpdir, "parallel.geojson")
            exporter_parallel.export_geojson(mask, metadata, out_parallel)

            # 读取对比两份 GeoJSON
            with open(out_serial, "r", encoding="utf-8") as f:
                data_serial = json.load(f)
            with open(out_parallel, "r", encoding="utf-8") as f:
                data_parallel = json.load(f)

            self.assertEqual(len(data_serial["features"]), len(data_parallel["features"]))
            
            # 对比前 5 个要素的坐标与属性
            for feat_s, feat_p in zip(data_serial["features"], data_parallel["features"]):
                self.assertEqual(feat_s["properties"]["parcel_id"], feat_p["properties"]["parcel_id"])
                self.assertEqual(feat_s["properties"]["crop_code"], feat_p["properties"]["crop_code"])
                self.assertAlmostEqual(feat_s["properties"]["perimeter_m"], feat_p["properties"]["perimeter_m"], places=1)
                self.assertAlmostEqual(feat_s["properties"]["center_lon"], feat_p["properties"]["center_lon"], places=5)
                self.assertAlmostEqual(feat_s["properties"]["center_lat"], feat_p["properties"]["center_lat"], places=5)
                # 几何坐标环必须严格相等
                self.assertEqual(feat_s["geometry"]["coordinates"], feat_p["geometry"]["coordinates"])


if __name__ == "__main__":
    unittest.main()
