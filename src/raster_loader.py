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
    parse_utm_zone,
)
from src.utils.logger import get_logger, log_success

# 优雅导入地理空间处理库，若未安装则提供友好提示
try:
    import rasterio
    import rasterio.env
    proj_cand = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
    if os.path.exists(proj_cand):
        try:
            rasterio.env.set_proj_data_search_path(proj_cand)
            os.environ["PROJ_LIB"] = proj_cand
            os.environ["PROJ_DATA"] = proj_cand
        except Exception:
            pass
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


def _compute_sdc6_physical_indices(b1, b2, b3, b4, b5, global_rows, global_cols):
    """
    针对 SDC30 6波段反射率数据统一计算 NDVI、LSWI 及物理非耕地掩膜（水体湿地、城镇包络、高山野生植被）。
    
    参数：
        b1, b2, b3, b4, b5: 空间浮点反射率矩阵 (Blue, Green, Red, NIR, SWIR1)
        global_rows, global_cols: 全局像素坐标网格 (支持分块局部偏移)
    返回：
        ndvi: 经物理压制后的 NDVI (非农田压低至 <= 0.10)
        lswi: 经物理压制后的 LSWI
    """
    from scipy import ndimage

    # --- NDVI 计算 ---
    denom_ndvi = b4 + b3
    ndvi = np.zeros_like(b3)
    valid_ndvi = denom_ndvi > 0
    ndvi[valid_ndvi] = (b4[valid_ndvi] - b3[valid_ndvi]) / denom_ndvi[valid_ndvi]

    # --- LSWI 计算 ---
    denom_lswi = b4 + b5
    lswi = np.zeros_like(b4)
    valid_lswi = denom_lswi > 0
    lswi[valid_lswi] = (b4[valid_lswi] - b5[valid_lswi]) / denom_lswi[valid_lswi]

    # 1. 水体与湿地沼泽：MNDWI > -0.08，或低近红外水体吸收
    mndwi = (b2 - b5) / np.maximum(b2 + b5, 1e-4)
    is_water_wetland = (mndwi > -0.08) | ((mndwi > -0.15) & (ndvi < 0.20)) | ((b4 < 600.0) & (b2 > b4))

    # 2. 城镇建筑与不透水硬化面空间包络 (Urban Settlement Envelope):
    # 结合高亮金属/混凝土/商业屋顶 (B1 > 1000)、典型沥青/路网、NDBI 与 SWIR1/Red 平坦特征
    b5_b3_ratio = b5 / np.maximum(b3, 1.0)
    ndbi = (b5 - b4) / np.maximum(b5 + b4, 1e-4)
    impervious_core = (
        (b1 > 1000.0) |
        ((b1 > 480.0) & (b5_b3_ratio < 1.85) & (ndvi < 0.40)) |
        ((ndbi > -0.02) & (b5_b3_ratio < 1.75)) |
        ((b5_b3_ratio < 1.55) & (ndvi < 0.35))
    )

    # 街区尺度空间集聚滤波 (21x21 像元窗口 ≈ 630m x 630m)
    dens = ndimage.uniform_filter(impervious_core.astype(np.float32), size=21)
    urban_candidate = dens >= 0.12
    urban_closed = ndimage.binary_closing(urban_candidate, structure=np.ones((7, 7)))
    lbl_u, num_u = ndimage.label(urban_closed, structure=ndimage.generate_binary_structure(2, 2))
    counts_u = np.bincount(lbl_u.ravel())
    large_urban = np.zeros_like(urban_closed, dtype=bool)
    for i in range(1, num_u + 1):
        if counts_u[i] >= 200:
            large_urban[lbl_u == i] = True
    final_urban_mask = ndimage.binary_dilation(large_urban, structure=np.ones((7, 7)))

    # 3. 基于泛化纹理特征的地形粗糙度压制（消除野生林木/山地，保护平原农田）：
    # 摒弃写死的区域空间坐标 (已移除 foothill_col / global_cols 限制)，改用纯粹的数据驱动方法。
    # 山坡林地通常在近红外 (Band 4) 有更强的阴阳坡起伏纹理，而农田平整均一。
    mean_b4 = ndimage.uniform_filter(b4, size=11)
    sq_b4 = ndimage.uniform_filter(b4**2, size=11)
    cv_b4 = np.sqrt(np.maximum(sq_b4 - mean_b4**2, 0.0)) / np.maximum(mean_b4, 1.0)

    # 仅使用较高的纹理变异系数 (CV > 0.05) 和光谱组合（暗红光）识别山地野生植被，跨区域通用
    # 南方（如湖北）林地极度茂密且平缓，CV可能仅在 0.05-0.10，同时冠层阴影导致红光 (b3) 较低
    # 1. 常规山区纹理过滤 (CV > 0.05 + 阴影红光较低)
    is_rough_mountain = (ndvi > 0.30) & (cv_b4 > 0.05) & (b3 < 900.0) & (b5 < 2600.0)
    
    # 2. 茂密暗红光森林过滤 (Dark Dense Vegetation - DDV)
    # 南方平缓山坡上的茂密森林，冠层极度平滑导致 CV < 0.05，但对红光和短波红外吸收极强
    is_dense_forest = (ndvi > 0.55) & (b3 < 550.0) & (b5 < 1600.0)
    
    is_shrub_or_tea = (ndvi > 0.45) & (cv_b4 > 0.035) & (b3 < 1000.0) & (b5 < 2400.0)
    is_mountain_veg = is_rough_mountain | is_dense_forest | is_shrub_or_tea

    # 执行非耕地物理压制：
    ndvi[is_water_wetland] = np.minimum(ndvi[is_water_wetland], -0.05)
    lswi[is_water_wetland] = np.minimum(lswi[is_water_wetland], -0.10)

    # 城镇及其内部社区草坪、高尔夫球场彻底压制
    ndvi[final_urban_mask] = np.minimum(ndvi[final_urban_mask], 0.10)
    lswi[final_urban_mask] = np.minimum(lswi[final_urban_mask], -0.05)

    ndvi[is_mountain_veg] = np.minimum(ndvi[is_mountain_veg], 0.10)
    lswi[is_mountain_veg] = np.minimum(lswi[is_mountain_veg], -0.05)

    return ndvi, lswi


class RasterLoader:
    def __init__(self, config=None):
        self.config = config or {}
        self.logger = get_logger("栅格加载")
        self.spatial_cfg = self.config.get("spatial", {})

    def resolve_tif_files(self, tif_dir_or_list):
        """解析并返回有效的 GeoTIFF 文件路径列表。"""
        if isinstance(tif_dir_or_list, str):
            if not os.path.exists(tif_dir_or_list) and not os.path.isabs(tif_dir_or_list):
                project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                cand = os.path.join(project_root, tif_dir_or_list)
                if os.path.exists(cand):
                    tif_dir_or_list = cand
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

            crs_str = str(src.crs) if src.crs else self.spatial_cfg.get("crs", "EPSG:32650")
            parsed_zone, parsed_northern = parse_utm_zone(crs_str, default_zone=self.spatial_cfg.get("utm_zone", 50))

            # 智能检测是否为 SDC30 (Dataset 26) 6波段地表反射率无缝立方体数据 (1:Blue, 2:Green, 3:Red, 4:NIR, 5:SWIR1, 6:SWIR2)
            first_fname = os.path.basename(tif_files[0])
            is_sdc6 = (src.count == 6) or (src.count >= 4 and ("CSDC" in first_fname or "SDC" in first_fname))

            geo_info = {
                "crs": crs_str,
                "transform": src.transform,
                "bounds": src.bounds,
                "width": src.width,
                "height": src.height,
                "resolution_x": res_x,
                "resolution_y": res_y,
                "resolution_meters": res_meters,
                "is_geographic": is_geo,
                "utm_zone": parsed_zone,
                "northern": parsed_northern,
                "nodata": src.nodata,
                "count": src.count,
                "is_sdc6": is_sdc6
            }

            if is_sdc6:
                self.logger.info("  -> 🛰️ 成功识别为 SDC30 (Dataset 26) 多光谱无缝数据立方体！自动激活近红外/红光物理 NDVI 提取器 (Band 4: NIR, Band 3: Red)。")

            if len(tif_files) == 1:
                if is_sdc6:
                    is_multiband = False
                    doy_list = [parse_temporal_doy(first_fname, default_doy=1)]
                    return tif_files, geo_info, doy_list, is_multiband
                elif src.count > 1:
                    is_multiband = True
                    doy_list = [b * 30 for b in range(1, src.count + 1)]
                    return tif_files, geo_info, doy_list, is_multiband

        # 多景文件检验地理空间范围一致性与解析各自时相 DOY
        base_bounds = geo_info.get("bounds")
        spatial_mismatch_files = []
        for idx, tif_path in enumerate(tif_files):
            fname = os.path.basename(tif_path)
            doy = parse_temporal_doy(fname, default_doy=(idx + 1) * 30)
            doy_list.append(doy)

            if idx > 0 and base_bounds is not None:
                try:
                    with rasterio.open(tif_path) as s:
                        cur_b = s.bounds
                        # 检查空间范围是否明显错位 (经纬度差 > 0.1度 或 投影米制差 > 1000米)
                        tol = 0.1 if geo_info.get("is_geographic") else 1000.0
                        if (abs(cur_b.left - base_bounds.left) > tol or 
                            abs(cur_b.bottom - base_bounds.bottom) > tol):
                            spatial_mismatch_files.append(fname)
                except Exception:
                    pass

        if spatial_mismatch_files:
            self.logger.warning(
                f"⚠️ [遥感数据严重提示] 检测到输入目录中的多个 GeoTIFF 地理空间范围不一致！\n"
                f"  首景参照基准: {os.path.basename(tif_files[0])} (Bounds: {base_bounds})\n"
                f"  错位瓦片示例: {spatial_mismatch_files[:3]}\n"
                f"  【根因分析】这些文件是不同地理区域的空间切片瓦片 (Spatial Tiles)，而非同一区域的多时相时间序列！\n"
                f"  将不同区域的空间大图强行按时序上下堆叠，会导致严重的物候错乱与错误分类。"
            )

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

        is_sdc6 = geo_info.get("is_sdc6", False)

        # 打开所有文件句柄（仅保留指针，不读取像元）
        src_handles = [rasterio.open(f) for f in sorted_files]
        try:
            for r in range(0, h, block_size):
                bh = min(block_size, h - r)
                for c in range(0, w, block_size):
                    bw = min(block_size, w - c)
                    win = Window(col_off=c, row_off=r, width=bw, height=bh)

                    band_slices = []
                    if is_sdc6:
                        # SDC30 (Dataset 26) 6波段反射率数据：提取多光谱联合物理指数 (NDVI + LSWI)
                        # NDVI=(NIR-Red)/(NIR+Red)  物候绿度判别
                        # LSWI=(NIR-SWIR1)/(NIR+SWIR1)  土壤水分/作物冠层含水量判别
                        # 利用 MNDWI/NDBI 辅助抑制非农田伪影 (水体湿地、城镇建筑与裸沙荒漠)
                        for src in src_handles:
                            b1 = src.read(1, window=win).astype(np.float32)
                            b2 = src.read(2, window=win).astype(np.float32)
                            b3 = src.read(3, window=win).astype(np.float32)
                            b4 = src.read(4, window=win).astype(np.float32)
                            b5 = src.read(5, window=win).astype(np.float32)

                            global_rows = r + np.arange(bh, dtype=np.float32)[:, np.newaxis]
                            global_cols = c + np.arange(bw, dtype=np.float32)[np.newaxis, :]
                            ndvi, lswi = _compute_sdc6_physical_indices(b1, b2, b3, b4, b5, global_rows, global_cols)

                            if src.nodata is not None:
                                nodata_mask = (b3 == src.nodata) | (b4 == src.nodata)
                                ndvi[nodata_mask] = np.nan
                                lswi[nodata_mask] = np.nan

                            # 双通道输出：时间步内同时记录 NDVI 和 LSWI
                            band_slices.append(ndvi)
                            band_slices.append(lswi)
                    elif is_multiband and len(src_handles) == 1:
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

        sorted_files, geo_info, _, is_multiband = self.get_multitemporal_metadata(tif_dir_or_list)
        is_sdc6 = geo_info.get("is_sdc6", False)
        target_file = sorted_files[0]
        with rasterio.open(target_file) as src:
            step = max(1, max(src.height, src.width) // max_dim)
            out_h = max(1, src.height // step)
            out_w = max(1, src.width // step)
            if is_sdc6:
                # SDC30 物理反射率数据：提取多光谱联合物理指数生成纯净缩略图
                b1 = src.read(1, out_shape=(out_h, out_w)).astype(np.float32)
                b2 = src.read(2, out_shape=(out_h, out_w)).astype(np.float32)
                b3 = src.read(3, out_shape=(out_h, out_w)).astype(np.float32)
                b4 = src.read(4, out_shape=(out_h, out_w)).astype(np.float32)
                b5 = src.read(5, out_shape=(out_h, out_w)).astype(np.float32)
                thumb_rows = np.arange(out_h, dtype=np.float32)[:, np.newaxis] * step
                thumb_cols = np.arange(out_w, dtype=np.float32)[np.newaxis, :] * step
                thumbnail, _ = _compute_sdc6_physical_indices(b1, b2, b3, b4, b5, thumb_rows, thumb_cols)
            else:
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
        is_sdc6 = geo_info.get("is_sdc6", False)

        h, w = geo_info["height"], geo_info["width"]
        total_t = len(doy_list)
        estimated_mem_gb = (h * w * total_t * 4) / (1024 ** 3)
        if estimated_mem_gb > 2.0:
            self.logger.warning(f"⚠️ 当前影像全量展开需要约 {estimated_mem_gb:.1f} GB 连续内存。对于超大影像，建议优先启用滑动窗口外核流式推断！")

        self.logger.info(f"检索到 {len(sorted_files)} 景多时相遥感影像，正在解析时相与空间信息...")

        # 就地预分配三维立方体，避免通过列表累积再使用 np.stack 导致的双倍内存峰值
        raster_cube = np.empty((h, w, total_t), dtype=np.float32)

        if is_sdc6:
            # SDC30 双通道：为每景影像同时提取 NDVI + LSWI，形状 (H, W, N_files*2)
            raster_cube = np.empty((h, w, total_t * 2), dtype=np.float32)
            global_rows = np.arange(h, dtype=np.float32)[:, np.newaxis]
            global_cols = np.arange(w, dtype=np.float32)[np.newaxis, :]
            for idx, tif_path in enumerate(sorted_files):
                with rasterio.open(tif_path) as src:
                    b1 = src.read(1).astype(np.float32)
                    b2 = src.read(2).astype(np.float32)
                    b3 = src.read(3).astype(np.float32)
                    b4 = src.read(4).astype(np.float32)
                    b5 = src.read(5).astype(np.float32)

                    ndvi, lswi = _compute_sdc6_physical_indices(b1, b2, b3, b4, b5, global_rows, global_cols)

                    if src.nodata is not None:
                        nodata_mask = (b3 == src.nodata) | (b4 == src.nodata)
                        ndvi[nodata_mask] = np.nan
                        lswi[nodata_mask] = np.nan

                    raster_cube[:, :, idx * 2] = ndvi
                    raster_cube[:, :, idx * 2 + 1] = lswi
        elif is_multiband and len(sorted_files) == 1:
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

    def compute_spectral_edge_mask(self, tif_path: str, cropland_mask: np.ndarray, percentile_thresh: float = 80.0) -> np.ndarray:
        """
        基于真实遥感多光谱物理反射率（近红外 Band 4 与红光 Band 3 / NDVI）提取田间道路、灌溉渠与作物交界边缘。
        遵循《联合国农业统计遥感手册》第 8 章技术规范，用于将宏观连片农田沿真实物理分界切断。
        """
        if not HAS_RASTERIO or not os.path.exists(tif_path):
            return np.zeros_like(cropland_mask, dtype=bool)

        try:
            import cv2
            from scipy import ndimage
            with rasterio.open(tif_path) as src:
                if src.count >= 4:
                    b3 = src.read(3).astype(np.float32)
                    b4 = src.read(4).astype(np.float32)
                else:
                    b3 = src.read(1).astype(np.float32)
                    b4 = b3

            denom = b4 + b3
            ndvi = np.zeros_like(b3)
            v = denom > 0
            ndvi[v] = (b4[v] - b3[v]) / denom[v]

            gx_b4 = cv2.Sobel(b4, cv2.CV_32F, 1, 0, ksize=3)
            gy_b4 = cv2.Sobel(b4, cv2.CV_32F, 0, 1, ksize=3)
            mag_b4 = np.sqrt(gx_b4**2 + gy_b4**2)

            gx_ndvi = cv2.Sobel(ndvi, cv2.CV_32F, 1, 0, ksize=3)
            gy_ndvi = cv2.Sobel(ndvi, cv2.CV_32F, 0, 1, ksize=3)
            mag_ndvi = np.sqrt(gx_ndvi**2 + gy_ndvi**2)

            mag_b4_norm = mag_b4 / np.maximum(b4, 1.0)
            combined_edge = mag_b4_norm * 0.4 + mag_ndvi * 0.6

            crop_pixels = combined_edge[cropland_mask > 0]
            if len(crop_pixels) == 0:
                return np.zeros_like(cropland_mask, dtype=bool)

            thresh = float(np.percentile(crop_pixels, percentile_thresh))
            is_edge = (combined_edge > thresh) & (cropland_mask > 0)

            struct_cross = ndimage.generate_binary_structure(2, 1)
            edge_mask = ndimage.binary_dilation(is_edge, structure=struct_cross)
            self.logger.info(f"成功从遥感影像中提取 {np.sum(edge_mask):,} 个真实机耕路/水渠/田埂物理边缘像元。")
            return edge_mask
        except Exception as e:
            self.logger.warning(f"遥感物理边缘解算异常 ({e})，跳过物理边缘阻隔。")
            return np.zeros_like(cropland_mask, dtype=bool)

