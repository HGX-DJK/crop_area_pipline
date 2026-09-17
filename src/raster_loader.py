"""
真实遥感影像栅格读取与时序堆叠模块。
支持读取本地多时相 GeoTIFF (.tif/.tiff) 影像，
解析其地理空间投影参考系（CRS）与仿射变换参数（Affine Transform），
将各时相影像堆叠为多时相特征立方体，使提取的地块具备真实地理坐标。
"""

import os
import re
import glob
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

    def load_multitemporal_tifs(self, tif_dir_or_list):
        """
        从指定目录中读取多时相 GeoTIFF 影像，并按日历日（DOY）自动排序堆叠。
        
        参数：
            tif_dir_or_list: 包含多时相 .tif 文件的文件夹路径，或 .tif 文件路径列表。
                            文件名通常包含日期或DOY，例如：'20240415_NDVI.tif' 或 'doy_110.tif'。
        返回：
            raster_cube: 形状为 (Rows, Cols, T) 的多时相矩阵
            geo_info: 包含 crs, transform, bounds, resolution, width, height 的空间参考字典
            doy_list: 对应的日历日列表
        """
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

        if not HAS_RASTERIO:
            self.logger.warning("未检测到 rasterio 地理空间库。如需直接解析真实 GeoTIFF 的坐标与投影，请执行: pip install rasterio")
            raise ImportError("缺少 rasterio 库，无法解析带有地理坐标的真实 GeoTIFF 影像。")

        self.logger.info(f"检索到 {len(tif_files)} 景多时相遥感影像，正在解析时相与空间信息...")

        band_arrays = []
        doy_list = []
        geo_info = {}

        for idx, tif_path in enumerate(tif_files):
            fname = os.path.basename(tif_path)
            
            # 解析日期或 DOY (基于 utils.geo_utils)
            doy = parse_temporal_doy(fname, default_doy=(idx + 1) * 30)

            with rasterio.open(tif_path) as src:
                # 记录第一景影像的地理空间元数据
                if idx == 0:
                    is_geo = src.crs.is_geographic if src.crs else is_geographic_system(self.spatial_cfg.get("crs", ""))
                    res_x = abs(src.transform[0])
                    res_y = abs(src.transform[4])
                    # 若为地理坐标系(度)，估算赤道/中纬度每度对应米数 (1度 ≈ 111320米)
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
                        "nodata": src.nodata
                    }

                # 检查是否为单景多波段时序立方体
                if len(tif_files) == 1 and src.count > 1:
                    self.logger.info(f"  -> 检测到单景多波段影像，包含 {src.count} 个波段，按时序多波段提取...")
                    for b in range(1, src.count + 1):
                        arr = src.read(b).astype(np.float32)
                        if src.nodata is not None:
                            arr[arr == src.nodata] = np.nan
                        band_arrays.append(arr)
                        doy_list.append((b) * 30)
                    break
                else:
                    arr = src.read(1).astype(np.float32)
                    if src.nodata is not None:
                        arr[arr == src.nodata] = np.nan
                    band_arrays.append(arr)
                    doy_list.append(doy)

        # 按 DOY 排序各时相
        sorted_order = np.argsort(doy_list)
        band_arrays = [band_arrays[i] for i in sorted_order]
        doy_list = [doy_list[i] for i in sorted_order]

        # 堆叠为三维立方体 (Rows, Cols, T)
        raster_cube = np.stack(band_arrays, axis=-1)
        res_desc = f"{geo_info['resolution_x']:.5f} 度 (约 {geo_info['resolution_meters']:.1f} 米)" if geo_info.get("is_geographic") else f"{geo_info['resolution_x']:.2f} 米"
        log_success(self.logger, f"影像堆叠完成，空间尺寸: {geo_info['height']} 行 × {geo_info['width']} 列，覆盖 {len(doy_list)} 个生长时相 (CRS: {geo_info['crs']}，空间分辨率: {res_desc})")

        return raster_cube, geo_info, doy_list
