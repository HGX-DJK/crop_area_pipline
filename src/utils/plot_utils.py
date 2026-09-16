"""
跨平台中文字体配置与遥感可视化辅助工具模块。
1. 跨平台中文字体自适应探测与负号正常渲染配置（Windows / Linux / macOS）
2. 农业作物离散分类自适应配色方案与 ListedColormap 构建
3. 大幅宽遥感影像自适应快速预览降采样
"""

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np


def setup_chinese_fonts():
    """
    自动配置跨平台中文字体与负号正常渲染。
    优先检索操作系统内置高品质无衬线中文字体：
    - Windows: Microsoft YaHei (微软雅黑), SimHei (黑体), SimSun (宋体)
    - Linux/Docker: WenQuanYi Micro Hei (文泉驿微米黑), Noto Sans CJK SC
    - macOS: PingFang SC (苹方)
    """
    plt.rcParams['font.sans-serif'] = [
        'Microsoft YaHei',    # Windows 首选微软雅黑
        'SimHei',             # Windows 经典黑体
        'SimSun',             # Windows 宋体
        'WenQuanYi Micro Hei',# Linux/Docker 文泉驿
        'Noto Sans CJK SC',   # Linux 常用无衬线中文
        'PingFang SC',        # macOS
        'sans-serif'
    ]
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号 '-' 正常显示而非方块乱码


def get_crop_colormap(num_classes: int, base_background="#ecf0f1"):
    """
    构建符合联合国粮农统计标准的农作物分类专题色带 (ListedColormap)。
    
    默认色标体系：
    0: 背景/非农田 (灰白)
    1: 玉米 (金黄)
    2: 小麦 (草绿)
    3: 大豆 (天蓝)
    4: 水稻 (青碧)
    5: 油菜 (鲜红)
    6: 棉花 (紫罗兰)
    ... 支持自适应扩展到 20+ 种复杂农作体系
    """
    base_palette = [
        base_background, "#f39c12", "#2ecc71", "#3498db",  # 0:背景, 1:玉米, 2:小麦, 3:大豆
        "#1abc9c", "#e74c3c", "#9b59b6", "#f1c40f",        # 4:水稻, 5:油菜, 6:棉花, 7:花生
        "#16a085", "#d35400", "#8e44ad", "#27ae60",        # 8:马铃薯, 9:向日葵, 10:甜菜, 11:甘蔗
        "#7f8c8d", "#c0392b", "#2980b9", "#f368e0"         # 12:设施大棚, 13:蔬菜, 14:果茶园, 15:杂粮
    ]
    if num_classes <= len(base_palette):
        palette = base_palette[:num_classes]
    else:
        cmap_obj = plt.get_cmap("tab20")
        palette = [cmap_obj(i % 20) for i in range(num_classes)]
        palette[0] = base_background
    return ListedColormap(palette)


def downsample_raster_preview(array: np.ndarray, max_dim: int = 1200) -> np.ndarray:
    """
    对超大幅宽高分辨率遥感栅格矩阵进行自适应步长降采样，
    在秒级内生成高清预览图，避免 Matplotlib 内存溢出或渲染卡顿。
    """
    if array.ndim < 2:
        return array
    h, w = array.shape[:2]
    step = max(1, max(h, w) // max_dim)
    if step > 1:
        return array[::step, ::step]
    return array
