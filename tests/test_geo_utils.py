"""
地理空间坐标转换与辅助函数单元测试。
"""

import unittest
import numpy as np

from src.utils.geo_utils import (
    utm_to_wgs84,
    wgs84_to_utm,
    is_geographic_system,
    parse_temporal_doy,
    estimate_resolution_meters,
)


class TestGeoUtils(unittest.TestCase):
    """测试地理空间坐标变换的高精度与严密性"""

    def test_utm_wgs84_roundtrip_precision(self):
        """测试 WGS84 与 UTM 投影米制坐标正反算闭合精度 (< 1 毫米误差)"""
        # 选取华北平原粮仓核心测试点 (UTM Zone 50)
        test_points = [
            (500000.0, 4200000.0),  # 中央经线附近
            (450000.0, 4150000.0),  # 西侧
            (550000.0, 4250000.0),  # 东侧
            (480000.0, 4000000.0),  # 南侧
        ]

        for orig_e, orig_n in test_points:
            lon, lat = utm_to_wgs84(orig_e, orig_n, zone=50, northern=True)
            # 验证经纬度在合理中国华北范围内
            self.assertTrue(114.0 <= lon <= 120.0, f"经度异常: {lon}")
            self.assertTrue(35.0 <= lat <= 40.0, f"纬度异常: {lat}")

            # 反算回 UTM
            calc_e, calc_n = wgs84_to_utm(lon, lat, zone=50)

            # 闭合误差应小于 0.001 米 (1 毫米)
            delta_e = abs(calc_e - orig_e)
            delta_n = abs(calc_n - orig_n)
            self.assertLess(delta_e, 0.001, f"Easting 闭合误差过大: {delta_e} 米")
            self.assertLess(delta_n, 0.001, f"Northing 闭合误差过大: {delta_n} 米")

    def test_is_geographic_system(self):
        """测试坐标系类型自适应判别"""
        self.assertTrue(is_geographic_system("EPSG:4326"))
        self.assertTrue(is_geographic_system("WGS84"))
        self.assertTrue(is_geographic_system("CGCS2000"))
        self.assertFalse(is_geographic_system("EPSG:32650"))
        self.assertFalse(is_geographic_system("UTM"))

    def test_parse_temporal_doy(self):
        """测试遥感时相日历日 (DOY) 解析正则"""
        self.assertEqual(parse_temporal_doy("doy_110.tif"), 110)
        self.assertEqual(parse_temporal_doy("Wheat_DOY_200_NDVI.tif"), 200)
        # 2024年4月15日 -> DOY 106 (平年 31+29+31+15 = 106, 闰年 106)
        doy = parse_temporal_doy("GF2_20240415_L2A.tif")
        self.assertTrue(100 <= doy <= 115)
        # 无法匹配时回退默认值
        self.assertEqual(parse_temporal_doy("unknown_raster.tif", default_doy=99), 99)

    def test_estimate_resolution_meters(self):
        """测试空间分辨率估算"""
        # 米制投影直接返回
        self.assertAlmostEqual(estimate_resolution_meters(10.0, is_geographic=False), 10.0)
        # 经纬度度数估算 (0.0001度约 10 米)
        res_m = estimate_resolution_meters(0.0001, is_geographic=True)
        self.assertTrue(8.0 <= res_m <= 12.0)


if __name__ == "__main__":
    unittest.main()
