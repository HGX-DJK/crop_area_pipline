"""
农作物种植分布与地块矢量化可视化制图模块。
用于生成出版级高清科研/业务图表：
1. 作物多时相物候指纹曲线图（展示冬小麦、夏玉米、大豆的光谱时序分异）
2. 全域农作物空间分类分布图（像素级专题图）
3. 零碎地块矢量边界与田埂切分叠加图（展示田埂分离效果）
4. 联合国无偏校准面积对比柱状图（朴素像元面积 vs 校准无偏面积与 95% 置信区间）
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.utils.plot_utils import (
    setup_chinese_fonts,
    get_crop_colormap,
    downsample_raster_preview,
)

# 自动配置跨平台中文字体支持（Windows 微软雅黑、Linux Noto/文泉驿、macOS 苹方）
setup_chinese_fonts()



class Visualizer:
    def __init__(self, config=None):
        self.config = config or {}
        self.output_dir = self.config.get("paths", {}).get("output_dir", "output")
        self.dpi = self.config.get("visualization", {}).get("plot_dpi", 300)
        self.crop_legend = self.config.get("crop_legend", {
            0: "非农田/背景",
            1: "夏玉米",
            2: "冬小麦",
            3: "大豆"
        })
        os.makedirs(self.output_dir, exist_ok=True)

    def plot_phenology_curves(self, pheno_curves_csv="data/sample_phenology_curves.csv", filename="phenology_signatures.png"):
        """绘制各作物全生长周期的物候时序特征指纹曲线。"""
        import pandas as pd
        df = pd.read_csv(pheno_curves_csv, comment="#")

        doy_cols = [c for c in df.columns if c.startswith("doy_")]
        doys = [int(c.split("_")[1]) for c in doy_cols]

        fig, ax = plt.subplots(figsize=(8, 5))
        colors = ["#95a5a6", "#e67e22", "#27ae60", "#2980b9"]

        for idx, row in df.iterrows():
            c_name = row["crop_name"]
            values = row[doy_cols].values
            color = colors[idx % len(colors)]
            ax.plot(doys, values, marker="o", linewidth=2.4, label=c_name, color=color)

        ax.set_xlabel("日历日 (DOY, Day of Year)", fontsize=11, fontweight="bold")
        ax.set_ylabel("植被指数 (NDVI)", fontsize=11, fontweight="bold")
        ax.set_title("典型作物多时相遥感物候指纹分异曲线", fontsize=13, fontweight="bold", pad=12)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(loc="upper left")
        plt.tight_layout()

        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return out_path

    def plot_crop_classification_map(self, crop_classified_mask, filename="crop_classification_map.png"):
        """绘制全域农作物像素级种植分类专题图。"""
        fig, ax = plt.subplots(figsize=(8, 7))
        
        # 动态自适应类别调色盘与超大幅宽自适应快速降采样
        unique_classes = sorted(list(self.crop_legend.keys()))
        num_classes = max(len(unique_classes), int(np.max(crop_classified_mask)) + 1)
        cmap = get_crop_colormap(num_classes)
        disp_mask = downsample_raster_preview(crop_classified_mask, max_dim=1200)

        im = ax.imshow(disp_mask, cmap=cmap, vmin=-0.5, vmax=num_classes - 0.5, interpolation="nearest")

        ax.set_title("遥感全域农作物种植分布分类图", fontsize=13, fontweight="bold", pad=12)
        ax.set_xlabel("列坐标 (像元)", fontsize=10)
        ax.set_ylabel("行坐标 (像元)", fontsize=10)

        # 自适应图例刻度
        ticks = [k for k in unique_classes if k < num_classes]
        tick_labels = [self.crop_legend.get(k, f"作物_{k}") for k in ticks]
        cbar = plt.colorbar(im, ax=ax, ticks=ticks, fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(tick_labels, fontsize=9)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return out_path

    def plot_parcel_delineation(self, parcel_id_mask, raster_cube_peak, filename="parcel_delineation_boundaries.png"):
        """
        绘制零碎地块矢量边界与狭窄田埂切分叠加图。
        底图采用夏季作物峰值 NDVI，叠加高亮的地块边界。
        """
        fig, ax = plt.subplots(figsize=(8, 7))
        # 动态自适应获取全时段最大植被指数作为农田底图（兼容任意时相数量、2D低内存缩略图或无底图场景）
        if raster_cube_peak is None:
            disp_parcels = downsample_raster_preview(parcel_id_mask, max_dim=1200)
            disp_ndvi = (disp_parcels > 0).astype(np.float32) * 0.5 + 0.2
        elif raster_cube_peak.ndim == 3:
            peak_ndvi = np.max(raster_cube_peak, axis=2)
            disp_ndvi = downsample_raster_preview(peak_ndvi, max_dim=1200)
            disp_parcels = downsample_raster_preview(parcel_id_mask, max_dim=1200)
        else:
            disp_ndvi = downsample_raster_preview(raster_cube_peak, max_dim=1200)
            disp_parcels = downsample_raster_preview(parcel_id_mask, max_dim=1200)

        # 确保底图与地块尺寸严格对齐（应对极少数降采样四舍五入微小像素差异）
        if disp_ndvi.shape != disp_parcels.shape:
            from scipy.ndimage import zoom
            z_r = disp_parcels.shape[0] / max(disp_ndvi.shape[0], 1)
            z_c = disp_parcels.shape[1] / max(disp_ndvi.shape[1], 1)
            disp_ndvi = zoom(disp_ndvi, (z_r, z_c), order=1)

        ax.imshow(disp_ndvi, cmap="YlGn", vmin=0.1, vmax=0.9)

        # 提取地块边界线并高亮叠加（红色线条）
        boundaries = np.zeros_like(disp_parcels, dtype=bool)
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            shifted = np.roll(np.roll(disp_parcels, dr, axis=0), dc, axis=1)
            boundaries |= (disp_parcels > 0) & (shifted != disp_parcels)

        # 绘制地块外轮廓边界
        overlay = np.zeros((*disp_parcels.shape, 4), dtype=np.float32)
        overlay[boundaries] = [0.85, 0.1, 0.1, 0.9]  # 红色半透明边界线
        ax.imshow(overlay)

        ax.set_title("零碎农田地块自动勾勒与田埂分离效果图 (联合国手册第8章)", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("像元 X 坐标", fontsize=10)
        ax.set_ylabel("像元 Y 坐标", fontsize=10)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return out_path

    def plot_area_comparison(self, df_area_report, filename="area_unbiased_comparison.png"):
        """绘制朴素像元面积与联合国无偏校准面积对比柱状图。"""
        fig, ax = plt.subplots(figsize=(8, 5))
        crops = df_area_report["crop_name"].tolist()
        naive_areas = df_area_report["naive_area_mu"].tolist()
        calib_areas = df_area_report["unbiased_calibrated_mu"].tolist()
        ci_lows = df_area_report["ci_95_lower_mu"].tolist()
        ci_highs = df_area_report["ci_95_upper_mu"].tolist()

        x = np.arange(len(crops))
        width = 0.35

        errors = [
            [calib - low for calib, low in zip(calib_areas, ci_lows)],
            [high - calib for calib, high in zip(calib_areas, ci_highs)]
        ]

        ax.bar(x - width/2, naive_areas, width, label="传统朴素像元面积 (含边缘偏差)", color="#e74c3c", alpha=0.85, edgecolor="black")
        ax.bar(x + width/2, calib_areas, width, yerr=errors, capsize=6, label="联合国手册无偏校准面积 (95%置信区间)", color="#27ae60", alpha=0.85, edgecolor="black")

        ax.set_ylabel("种植面积 (亩)", fontsize=11, fontweight="bold")
        ax.set_title("零碎农田种植面积：传统像元统计 vs 联合国无偏校准对比", fontsize=12, fontweight="bold", pad=12)
        ax.set_xticks(x)
        ax.set_xticklabels(crops, fontsize=10, fontweight="bold")
        ax.legend()
        ax.grid(True, axis="y", linestyle=":", alpha=0.6)

        plt.tight_layout()
        out_path = os.path.join(self.output_dir, filename)
        fig.savefig(out_path, dpi=self.dpi)
        plt.close(fig)
        return out_path
