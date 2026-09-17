"""
地块几何拓扑算法单元测试。
"""

import unittest
import numpy as np

from src.utils.geometry_utils import (
    trace_grid_boundary,
    simplify_polygon,
    chaikin_smooth_ring,
    calculate_isoperimetric_quotient,
)


class TestGeometryUtils(unittest.TestCase):
    """测试多边形边界追踪、抽稀、平滑与紧凑度计算"""

    def test_trace_grid_boundary_simple_box(self):
        """测试对正方形二值掩膜的闭合边界追踪"""
        mask = np.zeros((10, 10), dtype=bool)
        mask[2:6, 2:6] = True  # 4x4 的实心正方形

        pts = trace_grid_boundary(mask)
        self.assertGreaterEqual(len(pts), 4, "边界点数必须不少于 4 个")
        # 首尾闭合检查
        self.assertEqual(pts[0], pts[-1], "几何外环必须首尾重合闭合")

    def test_simplify_polygon_rdp(self):
        """测试 Ramer-Douglas-Peucker 拓扑抽稀"""
        # 带有共线中间点的折线
        line_pts = [(0, 0), (0, 2), (0, 5), (0, 10), (5, 10), (10, 10)]
        simplified = simplify_polygon(line_pts, tolerance=0.1)
        self.assertLess(len(simplified), len(line_pts), "共线冗余点应被成功消除")
        self.assertEqual(simplified[0], (0, 0))
        self.assertEqual(simplified[-1], (10, 10))

    def test_chaikin_smooth_ring_closed(self):
        """测试 Chaikin 平滑算法的闭合性与保形性"""
        square_ring = [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]
        smoothed = chaikin_smooth_ring(square_ring, iterations=1)
        self.assertEqual(smoothed[0], smoothed[-1], "平滑后闭合环必须严格首尾一致")
        self.assertGreater(len(smoothed), len(square_ring), "平滑后顶点数应增加")

    def test_calculate_isoperimetric_quotient(self):
        """测试等周紧凑度指数 (4*pi*Area / P^2)"""
        # 1. 标准圆面积与周长: 紧凑度应趋近于 1.0
        r = 10.0
        circle_area = np.pi * r * r
        circle_perimeter = 2.0 * np.pi * r
        q_circle = calculate_isoperimetric_quotient(circle_area, circle_perimeter)
        self.assertAlmostEqual(q_circle, 1.0, places=3)

        # 2. 标准正方形: 紧凑度 = 4*pi*L^2 / (4L)^2 = pi/4 ≈ 0.785
        side = 10.0
        sq_area = side * side
        sq_perim = 4.0 * side
        q_square = calculate_isoperimetric_quotient(sq_area, sq_perim)
        self.assertAlmostEqual(q_square, np.pi / 4.0, places=3)

        # 3. 狭长条带或边界为0的异常防御
        q_zero = calculate_isoperimetric_quotient(100.0, 0.0)
        self.assertEqual(q_zero, 0.0)


if __name__ == "__main__":
    unittest.main()
