"""
真实遥感影像栅格读取与时序堆叠模块。
支持读取本地多时相 GeoTIFF (.tif/.tiff) 影像，
解析其地理空间投影参考系（CRS）与仿射变换参数（Affine Transform），
将各时相影像堆叠为多时相特征立方体，使提取的地块具备真实地理坐标。
"""

import os
import re
import glob
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from src.utils.geo_utils import (
    parse_temporal_doy,
    is_geographic_system,
    estimate_resolution_meters,
)
from src.utils.logger import get_logger, log_success

# 优雅导入地理空间处理库，若未安装则提供友好提示
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


class RasterLoader:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = get_logger("栅格加载")
        self.spatial_cfg = self.config.get("spatial", {})

    def resolve_tif_files(self, tif_dir_or_list):
        """解析并返回有效的 GeoTIFF 文件路径列表。"""
        if isinstance(tif_dir_or_list, str):
            if os.path.isdir(tif_dir_or_list):
                tif_files = sorted(
                    glob.glob(os.path.join(tif_dir_or_list, "*.tif")) +
                    glob.glob(os.path.join(tif_dir_or_list, "*.tiff"))
                )
            else:
                tif_files = [tif_dir_or_list]
        else:
            tif_files = sorted(list(tif_dir_or_list))

        if not tif_files:
            raise FileNotFoundError(f"未在指定路径检索到任何 GeoTIFF (.tif) 影像文件: {tif_dir_or_list}")
        return tif_files

    def get_multitemporal_metadata(self, tif_dir_or_list):
        """
        轻量解析多时相 GeoTIFF 的时相元数据与空间参考，无需载入任何像元矩阵。
        返回: (sorted_tif_files, geo_info, sorted_doy_list, is_multiband)
        """
        if not HAS_RASTERIO:
            self.logger.warning("未检测到 rasterio 地理空间库。如需直接解析真实 GeoTIFF 的坐标与投影，请执行: pip install rasterio")
            raise ImportError("缺少 rasterio 库，无法解析带有地理坐标的真实 GeoTIFF 影像。")

        tif_files = self.resolve_tif_files(tif_dir_or_list)
        geo_info = {}
        doy_list = []
        is_multiband = False

        # 检验第一景获取核心地理空间元数据
        with rasterio.open(tif_files[0]) as src:
            is_geo = src.crs.is_geographic if src.crs else is_geographic_system(self.spatial_cfg.get("crs", ""))
            res_x = abs(src.transform[0])
            res_y = abs(src.transform[4])
            res_meters = estimate_resolution_meters(res_x, is_geographic=is_geo)

            geo_info = {
                "crs": str(src.crs) if src.crs else self.spatial_cfg.get("crs", "EPSG:32650"),
                "transform": src.transform,
                "bounds": src.bounds,
                "width": src.width,
                "height": src.height,
                "resolution_x": res_x,
                "resolution_y": res_y,
                "resolution_meters": res_meters,
                "is_geographic": is_geo,
                "nodata": src.nodata,
                "count": src.count
            }

            if len(tif_files) == 1 and src.count > 1:
                is_multiband = True
                doy_list = [b * 30 for b in range(1, src.count + 1)]
                return tif_files, geo_info, doy_list, is_multiband

        # 多景文件解析各自时相 DOY
        for idx, tif_path in enumerate(tif_files):
            fname = os.path.basename(tif_path)
            doy = parse_temporal_doy(fname, default_doy=(idx + 1) * 30)
            doy_list.append(doy)

        sorted_order = np.argsort(doy_list)
        sorted_tif_files = [tif_files[i] for i in sorted_order]
        sorted_doy_list = [doy_list[i] for i in sorted_order]

        return sorted_tif_files, geo_info, sorted_doy_list, is_multiband

    def iter_raster_windows(self, tif_dir_or_list, block_size: int = 1024, is_multiband: Optional[bool] = None):
        """
        分块滑动窗口生成器 (Tiling Window Generator)。
        按需流式读取指定窗口的像元时序切片，彻底避免 GB 级影像导致内存崩溃。
        
        生成器依次产出: ((r_start, r_stop), (c_start, c_stop), window_cube)
        """
        from rasterio.windows import Window

        if isinstance(tif_dir_or_list, list) and len(tif_dir_or_list) > 0 and not isinstance(tif_dir_or_list[0], str):
            # 兼容已排序文件列表
            sorted_files = tif_dir_or_list
            geo_info = self.spatial_cfg
            h = geo_info.get("height", 1024)
            w = geo_info.get("width", 1024)
        else:
            sorted_files, geo_info, _, auto_multiband = self.get_multitemporal_metadata(tif_dir_or_list)
            if is_multiband is None:
                is_multiband = auto_multiband
            h, w = geo_info["height"], geo_info["width"]

        # 打开所有文件句柄（仅保留指针，不读取像元）
        src_handles = [rasterio.open(f) for f in sorted_files]
        try:
            for r in range(0, h, block_size):
                bh = min(block_size, h - r)
                for c in range(0, w, block_size):
                    bw = min(block_size, w - c)
                    win = Window(col_off=c, row_off=r, width=bw, height=bh)

                    band_slices = []
                    if is_multiband and len(src_handles) == 1:
                        src = src_handles[0]
                        for b in range(1, src.count + 1):
                            arr = src.read(b, window=win).astype(np.float32)
                            if src.nodata is not None:
                                arr[arr == src.nodata] = np.nan
                            band_slices.append(arr)
                    else:
                        for src in src_handles:
                            arr = src.read(1, window=win).astype(np.float32)
                            if src.nodata is not None:
                                arr[arr == src.nodata] = np.nan
                            band_slices.append(arr)

                    win_cube = np.stack(band_slices, axis=-1)
                    yield (slice(r, r + bh), slice(c, c + bw)), win_cube
        finally:
            for s in src_handles:
                try:
                    s.close()
                except Exception:
                    pass

    def load_preview_thumbnail(self, tif_dir_or_list, max_dim: int = 1200) -> np.ndarray:
        """
        以极低内存极速读取一景降采样缩略底图 (约 1200×1200，仅数兆内存)，
        供成果专题图 (Matplotlib) 作为可视化农田背景底图，彻底避免大图 OOM。
        """
        if not HAS_RASTERIO:
            return np.zeros((100, 100), dtype=np.float32)

        sorted_files, _, _, is_multiband = self.get_multitemporal_metadata(tif_dir_or_list)
        target_file = sorted_files[0]
        with rasterio.open(target_file) as src:
            step = max(1, max(src.height, src.width) // max_dim)
            out_h = max(1, src.height // step)
            out_w = max(1, src.width // step)
            band_idx = (src.count // 2 + 1) if (is_multiband and src.count > 1) else 1
            try:
                from rasterio.enums import Resampling
                thumbnail = src.read(band_idx, out_shape=(out_h, out_w), resampling=Resampling.bilinear).astype(np.float32)
            except Exception:
                thumbnail = src.read(band_idx, out_shape=(out_h, out_w)).astype(np.float32)

            if src.nodata is not None:
                thumbnail[thumbnail == src.nodata] = np.nan

            valid = np.isfinite(thumbnail)
            if np.any(valid):
                vmin = float(np.percentile(thumbnail[valid], 2))
                vmax = float(np.percentile(thumbnail[valid], 98))
                if vmax > vmin:
                    thumbnail = np.clip((thumbnail - vmin) / (vmax - vmin), 0.0, 1.0)
            return thumbnail

    def load_multitemporal_tifs(self, tif_dir_or_list):
        """
        从指定目录中读取多时相 GeoTIFF 影像，并按日历日（DOY）自动排序堆叠。
        对于中小型影像直接返回全图三维矩阵；对于大图推荐配合 iter_raster_windows 流式处理。
        """
        sorted_files, geo_info, doy_list, is_multiband = self.get_multitemporal_metadata(tif_dir_or_list)

        h, w = geo_info["height"], geo_info["width"]
        total_t = len(doy_list)
        estimated_mem_gb = (h * w * total_t * 4) / (1024 ** 3)
        if estimated_mem_gb > 2.0:
            self.logger.warning(f"⚠️ 当前影像全量展开需要约 {estimated_mem_gb:.1f} GB 连续内存。对于超大影像，建议优先启用滑动窗口外核流式推断！")

        self.logger.info(f"检索到 {len(sorted_files)} 景多时相遥感影像，正在解析时相与空间信息...")

        # 就地预分配三维立方体，避免通过列表累积再使用 np.stack 导致的双倍内存峰值
        raster_cube = np.empty((h, w, total_t), dtype=np.float32)

        if is_multiband and len(sorted_files) == 1:
            with rasterio.open(sorted_files[0]) as src:
                self.logger.info(f"  -> 检测到单景多波段影像，包含 {src.count} 个波段，按时序多波段提取...")
                for b in range(1, src.count + 1):
                    arr = src.read(b).astype(np.float32)
                    if src.nodata is not None:
                        arr[arr == src.nodata] = np.nan
                    raster_cube[:, :, b - 1] = arr
        else:
            for idx, tif_path in enumerate(sorted_files):
                with rasterio.open(tif_path) as src:
                    arr = src.read(1).astype(np.float32)
                    if src.nodata is not None:
                        arr[arr == src.nodata] = np.nan
                    raster_cube[:, :, idx] = arr

        res_desc = f"{geo_info['resolution_x']:.5f} 度 (约 {geo_info['resolution_meters']:.1f} 米)" if geo_info.get("is_geographic") else f"{geo_info['resolution_x']:.2f} 米"
        log_success(self.logger, f"影像堆叠完成，空间尺寸: {geo_info['height']} 行 × {geo_info['width']} 列，覆盖 {len(doy_list)} 个生长时相 (CRS: {geo_info['crs']}，空间分辨率: {res_desc})")

        return raster_cube, geo_info, doy_list
