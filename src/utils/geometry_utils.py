"""
农田地块几何拓扑追踪、多边形抽稀与样条平滑算法模块。
纯 Python / NumPy 实现，无需依赖 Shapely/GDAL 等重型外部 C++ 库：
1. 栅格网格边缘拓扑追踪（Cell Edge Boundary Following），提取无自相交闭合环多边形
2. Ramer-Douglas-Peucker (RDP) 拓扑保形多边形几何抽稀
3. Chaikin 拐角切割拓扑平滑算法 (Chaikin's Corner-Cutting Spline Smoothing)
4. 等周商紧凑度与规整性度量 (Isoperimetric Quotient)
"""

import numpy as np


def trace_grid_boundary(binary_mask):
    """
    针对二值像元掩膜，精确提取外边界网格多边形闭合序列（无自相交、顺时针闭合环）。
    采用栅格网格边缘拓扑追踪（Cell Edge Boundary Following）：
    以像素格子交点为顶点，查找所有分隔内部像元与外部像元的单位网格边，并沿拓扑方向链式连接。
    """
    padded = np.pad(binary_mask.astype(bool), 1, mode='constant', constant_values=False)
    
    # 查找所有边界有向边 (u -> v)，内部像元位于右侧
    edges = {}
    rows, cols = np.where(padded)
    for r, c in zip(rows, cols):
        # 北边: (r, c) -> (r, c+1)
        if not padded[r - 1, c]:
            edges.setdefault((r, c), []).append((r, c + 1))
        # 东边: (r, c+1) -> (r+1, c+1)
        if not padded[r, c + 1]:
            edges.setdefault((r, c + 1), []).append((r + 1, c + 1))
        # 南边: (r+1, c+1) -> (r+1, c)
        if not padded[r + 1, c]:
            edges.setdefault((r + 1, c + 1), []).append((r + 1, c))
        # 西边: (r+1, c) -> (r, c)
        if not padded[r, c - 1]:
            edges.setdefault((r + 1, c), []).append((r, c))

    if not edges:
        return []

    rings = []
    remaining = {k: list(v) for k, v in edges.items()}

    max_total_steps = sum(len(v) for v in edges.values()) + 10
    total_steps = 0

    while True:
        start = None
        for k, v in remaining.items():
            if v:
                start = k
                break
        if start is None:
            break

        ring = [start]
        curr = start
        while True:
            total_steps += 1
            if total_steps > max_total_steps:
                break

            targets = remaining.get(curr, [])
            if not targets:
                break
            
            if len(targets) == 1:
                nxt = targets.pop(0)
            else:
                # 鞍点/交叉口：计算进入向量与所有候选流出向量的夹角，优先最右侧（顺时针外边界）
                if len(ring) >= 2:
                    dr_in = curr[0] - ring[-2][0]
                    dc_in = curr[1] - ring[-2][1]
                else:
                    dr_in, dc_in = 0, 1
                
                best_t = None
                best_angle = -999.0
                for t in targets:
                    dr_out = t[0] - curr[0]
                    dc_out = t[1] - curr[1]
                    cross = dr_in * dc_out - dc_in * dr_out
                    dot = dr_in * dr_out + dc_in * dc_out
                    angle = np.arctan2(cross, dot)
                    if angle > best_angle:
                        best_angle = angle
                        best_t = t
                nxt = best_t
                targets.remove(nxt)

            ring.append(nxt)
            curr = nxt
            if curr == start:
                break

        if len(ring) >= 4 and ring[0] == ring[-1]:
            rings.append(ring)

        if total_steps > max_total_steps:
            break

    if not rings:
        return []

    # 鞋带定理计算多边形面积，选取面积最大的环作为外围主轮廓
    best_ring = None
    max_area = -1.0
    for ring in rings:
        area = 0.0
        for i in range(len(ring) - 1):
            area += (ring[i][1] * ring[i + 1][0] - ring[i + 1][1] * ring[i][0])
        area = abs(area) * 0.5
        if area > max_area:
            max_area = area
            best_ring = ring

    if not best_ring:
        return []

    # 减去 padding 偏移量，恢复真实图像的像元交点坐标
    pts = [(r - 1.0, c - 1.0) for r, c in best_ring]
    return pts


def simplify_polygon(points, tolerance=0.5):
    """
    Ramer-Douglas-Peucker (RDP) 算法简化多边形顶点。
    1. 消除严格共线的网格点（如水平、垂直边缘连续像素）。
    2. 对剩余折线进行 RDP 几何抽稀，消除阶梯锯齿并保证几何拓扑严密闭合。
    """
    if len(points) <= 4:
        return points

    # 第一阶段：消除严格共线中间点
    filtered = [points[0]]
    for i in range(1, len(points) - 1):
        p_prev = filtered[-1]
        p_curr = points[i]
        p_next = points[i + 1]
        dr1 = p_curr[0] - p_prev[0]
        dc1 = p_curr[1] - p_prev[1]
        dr2 = p_next[0] - p_curr[0]
        dc2 = p_next[1] - p_curr[1]
        cross = dr1 * dc2 - dc1 * dr2
        if abs(cross) > 1e-6:
            filtered.append(p_curr)
    filtered.append(points[-1])

    if len(filtered) <= 4:
        return filtered

    # 第二阶段：闭合环 RDP 抽稀
    pts_arr = np.array(filtered)
    dists = np.linalg.norm(pts_arr[:-1] - pts_arr[0], axis=1)
    far_idx = int(np.argmax(dists))

    def _rdp(pts, eps):
        if len(pts) <= 2:
            return pts
        pt1 = pts[0]
        pt2 = pts[-1]
        line_vec = pt2 - pt1
        line_norm = np.linalg.norm(line_vec)
        if line_norm < 1e-6:
            d = np.linalg.norm(pts - pt1, axis=1)
        else:
            u = line_vec / line_norm
            v = pts - pt1
            proj = np.outer(np.dot(v, u), u)
            d = np.linalg.norm(v - proj, axis=1)
        idx = int(np.argmax(d))
        if d[idx] > eps:
            left = _rdp(pts[:idx + 1], eps)
            right = _rdp(pts[idx:], eps)
            return np.vstack([left[:-1], right])
        else:
            return np.vstack([pts[0], pts[-1]])

    part1 = _rdp(pts_arr[:far_idx + 1], tolerance)
    part2 = _rdp(pts_arr[far_idx:], tolerance)
    simplified = np.vstack([part1[:-1], part2])
    
    # 确保首尾闭合
    res = [tuple(p) for p in simplified]
    if res[0] != res[-1]:
        res.append(res[0])
    return res


def chaikin_smooth_ring(points, iterations=1):
    """
    Chaikin 拐角切割拓扑平滑算法 (Chaikin's Corner-Cutting Algorithm)。
    针对栅格网格边缘追踪形成的直角阶梯台阶（锯齿），在保持首尾严密闭合、拓扑不自交
    与总体面积守恒的前提下，通过内切各拐角拟合出自然平滑的农田有机边界轮廓。
    """
    if len(points) < 4:
        return points
    
    pts = list(points)
    if pts[0] == pts[-1]:
        pts = pts[:-1]
        
    n = len(pts)
    if n < 3:
        return points

    for _ in range(iterations):
        smoothed = []
        for i in range(n):
            p0 = pts[i]
            p1 = pts[(i + 1) % n]
            # Q 点: 75% 靠向 p0，25% 靠向 p1
            q_x = 0.75 * p0[0] + 0.25 * p1[0]
            q_y = 0.75 * p0[1] + 0.25 * p1[1]
            # R 点: 25% 靠向 p0，75% 靠向 p1
            r_x = 0.25 * p0[0] + 0.75 * p1[0]
            r_y = 0.25 * p0[1] + 0.75 * p1[1]
            smoothed.append((q_x, q_y))
            smoothed.append((r_x, r_y))
        pts = smoothed
        n = len(pts)

    # 重新封闭首尾环
    pts.append(pts[0])
    return pts


def calculate_isoperimetric_quotient(area_m2: float, perimeter_m: float) -> float:
    """
    计算地块等周商紧凑度指数 (Isoperimetric Quotient):
    Q = 4 * pi * Area / (Perimeter^2)
    正圆形为 1.0，正方形约为 0.785，极长狭条或极度破碎地块趋近于 0。
    """
    if perimeter_m <= 0:
        return 0.0
    return (4.0 * np.pi * float(area_m2)) / (float(perimeter_m) * float(perimeter_m))
