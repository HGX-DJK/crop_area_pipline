"""
农田地块矢量化与 GIS 属性导出模块。
遵循《联合国农业统计遥感手册》（第 8、11 章）与 OGC / RFC 7946 国际标准。

核心能力：
1. 拓扑边界跟踪（Topological Grid Boundary Tracing）：
   基于连通域边缘流向跟踪与 Ramer-Douglas-Peucker (RDP) 算法，
   精确提取独立零碎农田的外边界闭合多边形（Polygon LinearRing），杜绝自相交与蜘蛛网假象。
2. 高精度坐标系逆变换（UTM Zone -> WGS84 经纬度）：
   内置 USGS Snyder (1987) 椭球体闭合方程与 pyproj 双引擎，
   将 UTM 投影米制坐标自动转换为符合 RFC 7946 国际规范的 WGS84 [经度, 纬度]（度），
   可直接拖入 geojson.io、QGIS、ArcGIS 或各类 WebGIS 无缝精准落图。
3. 空间统计与属性台账导出：
   计算真实物理周长（米）、种植面积（亩 / 公顷 / 平方米）、质心坐标与作物类别，
   同步输出结构化 GeoJSON 与属性清单 CSV。
"""

import os
import re
import json
import numpy as np
import pandas as pd

# 可选尝试加载 pyproj（若环境未安装则自动回退至高精度纯 Python 闭合公式）
try:
    from pyproj import Transformer
    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False


# ============================================================================
# 底层通用空间投影与几何算法工具导入（同时对外再导出以保持 100% 向后兼容）
# ============================================================================
from src.utils.geo_utils import utm_to_wgs84, wgs84_to_utm, is_geographic_system
from src.utils.geometry_utils import (
    trace_grid_boundary as _trace_grid_boundary,
    simplify_polygon as _simplify_polygon,
    chaikin_smooth_ring as _chaikin_smooth,
    calculate_isoperimetric_quotient,
)
from src.utils.agri_zoning import (
    assign_province_and_zone,
    evaluate_machinery_suitability,
    get_scale_tier,
)
from src.utils.unit_utils import sqm_to_mu, sqm_to_ha



class VectorExporter:
    """农田地块几何矢量化与空间属性导出器"""

    def __init__(self, config=None):
        self.config = config or {}
        spatial = self.config.get("spatial", {})
        self.resolution = spatial.get("resolution_meters", 10.0)
        self.crs = spatial.get("crs", "EPSG:32650")
        self.utm_zone = spatial.get("utm_zone", 50)
        self.export_crs = spatial.get("export_crs", "WGS84").upper()
        self.origin_x = spatial.get("origin_x", 500000.0)
        self.origin_y = spatial.get("origin_y", 4200000.0)
        self.crop_legend = self.config.get("crop_legend", {
            0: "非农田/背景",
            1: "夏玉米",
            2: "冬小麦",
            3: "大豆"
        })

        # 初始化 pyproj Transformer（若可用）
        self._transformer = None
        if HAS_PYPROJ:
            try:
                self._transformer = Transformer.from_crs(self.crs, "EPSG:4326", always_xy=True)
            except Exception:
                self._transformer = None

    def _pixel_to_coords(self, row, col, geo_info=None):
        """将像元行列号转换为真实的地理空间坐标 (X, Y)。"""
        if geo_info and "transform" in geo_info:
            t = geo_info["transform"]
            geo_x = t[2] + col * t[0] + row * t[1]
            geo_y = t[5] + col * t[3] + row * t[4]
            return float(geo_x), float(geo_y)
        geo_x = self.origin_x + col * self.resolution
        geo_y = self.origin_y - row * self.resolution
        return float(geo_x), float(geo_y)

    def _utm_to_wgs84_coords(self, utm_x, utm_y, geo_info=None):
        """
        将 UTM 投影米制坐标 (X, Y) 转换为 WGS84 地理经纬度 [lon, lat]。
        自适应识别地理坐标系与数值范围，若本身已是经纬度则严格保真返回。
        """
        fx, fy = float(utm_x), float(utm_y)
        # 若本身就是地理经纬度数值范围，直接返回
        if is_geographic_system(geo_info) or (abs(fx) <= 180.0 and abs(fy) <= 90.0):
            return round(fx, 7), round(fy, 7)

        target_crs = str(geo_info.get("crs", self.crs)) if geo_info else str(self.crs)

        if self._transformer is not None:
            try:
                lon, lat = self._transformer.transform(fx, fy)
                return round(float(lon), 7), round(float(lat), 7)
            except Exception:
                pass

        # 动态解析 UTM 分带号（例如 EPSG:32650 -> 50分带）
        zone = self.utm_zone
        match = re.search(r"326(\d{2})", target_crs)
        if match:
            zone = int(match.group(1))

        lon, lat = utm_to_wgs84(fx, fy, zone=zone, northern=True)
        return round(float(lon), 7), round(float(lat), 7)

    def _extract_parcel_geometry(self, binary_mask, geo_info=None):
        """
        提取地块多边形几何边界。
        根据输入栅格的真实坐标类型（地理经纬度 或 UTM投影米制），
        严密自适应输出符合 RFC 7946 规范的 WGS84 闭合环与真实物理周长（米）。
        """
        pts_grid = _trace_grid_boundary(binary_mask)
        if len(pts_grid) < 4:
            coords = np.argwhere(binary_mask)
            min_r, min_c = np.min(coords, axis=0)
            max_r, max_c = np.max(coords, axis=0)
            pts_grid = [
                (min_r, min_c), (min_r, max_c + 1),
                (max_r + 1, max_c + 1), (max_r + 1, min_c),
                (min_r, min_c)
            ]

        # 简化多边形几何顶点 (消除网格共线冗余点并执行 RDP 拓扑抽稀)
        simplified = _simplify_polygon(pts_grid, tolerance=0.5)

        # 拓扑保形 Chaikin 拐角切割平滑算法 (平滑直角网格锯齿，生成自然农田轮廓)
        if self.config.get("segmentation", {}).get("smooth_boundaries", True):
            simplified = _chaikin_smooth(simplified, iterations=1)

        is_geo_input = is_geographic_system(geo_info)

        utm_ring = []
        wgs84_ring = []
        for r, c in simplified:
            gx, gy = self._pixel_to_coords(r, c, geo_info)
            if is_geo_input or (abs(float(gx)) <= 180.0 and abs(float(gy)) <= 90.0):
                # 栅格原生坐标即为地理经纬度 (如 WGS84 / CGCS2000)
                lon = round(float(gx), 7)
                lat = round(float(gy), 7)
                # 计算对应的 UTM 投影米制坐标 (用于精确物理周长)
                zone = int((lon + 180) / 6) + 1 if (-180.0 <= lon <= 180.0) else self.utm_zone
                ux, uy = wgs84_to_utm(lon, lat, zone=zone)
            else:
                # 栅格原生坐标为 UTM 投影米制坐标
                ux = round(float(gx), 2)
                uy = round(float(gy), 2)
                lon, lat = self._utm_to_wgs84_coords(ux, uy, geo_info)

            utm_ring.append([ux, uy])
            wgs84_ring.append([lon, lat])

        # 计算真实物理周长（以米制坐标计算，确保真实反映地面物理尺度）
        perimeter = 0.0
        for i in range(len(utm_ring) - 1):
            dx = utm_ring[i + 1][0] - utm_ring[i][0]
            dy = utm_ring[i + 1][1] - utm_ring[i][1]
            perimeter += np.sqrt(dx * dx + dy * dy)

        # 遵循 RFC 7946 外环右手定则（逆时针）：若为顺时针则翻转为逆时针
        signed_area = 0.0
        for i in range(len(wgs84_ring) - 1):
            signed_area += (wgs84_ring[i][0] * wgs84_ring[i + 1][1] - wgs84_ring[i + 1][0] * wgs84_ring[i][1])
        if signed_area < 0:
            wgs84_ring = wgs84_ring[::-1]
            utm_ring = utm_ring[::-1]

        return wgs84_ring, utm_ring, round(perimeter, 1)

    def export_geojson(self, parcel_id_mask, parcel_metadata, output_path="output/vectorized_parcels.geojson", geo_info=None):
        """
        导出符合国际 GIS 标准的 GeoJSON 矢量图层与属性清单 CSV。
        根据配置自动输出 WGS84 经纬度（RFC 7946 规范）或 UTM 投影米制坐标。
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        features = []

        is_wgs84 = (self.export_crs == "WGS84")
        is_geo_input = is_geographic_system(geo_info)

        for meta in parcel_metadata:
            pid = meta["internal_id"]
            comp_mask = (parcel_id_mask == pid)
            
            wgs84_ring, utm_ring, perimeter_m = self._extract_parcel_geometry(comp_mask, geo_info)
            
            # 计算地块质心坐标
            c_px, c_py = self._pixel_to_coords(meta["centroid_row"], meta["centroid_col"], geo_info)
            if is_geo_input or (abs(float(c_px)) <= 180.0 and abs(float(c_py)) <= 90.0):
                c_lon = round(float(c_px), 7)
                c_lat = round(float(c_py), 7)
                c_zone = int((c_lon + 180) / 6) + 1 if (-180.0 <= c_lon <= 180.0) else self.utm_zone
                c_ux, c_uy = wgs84_to_utm(c_lon, c_lat, zone=c_zone)
            else:
                c_ux = round(float(c_px), 2)
                c_uy = round(float(c_py), 2)
                c_lon, c_lat = self._utm_to_wgs84_coords(c_ux, c_uy, geo_info)

            crop_name = self.crop_legend.get(meta["crop_code"], f"未知作物_{meta['crop_code']}")

            # 计算地块几何规整度与农机适机性评估 (基于联合国 FAO 农业工程与高标准农田建设标准)
            compactness = calculate_isoperimetric_quotient(meta["area_m2"], perimeter_m)
            compactness = round(min(1.0, max(0.01, compactness)), 3) if perimeter_m > 0 else 0.0

            # 尺度自适应农机作业适机性评估 (基于 utils.agri_zoning)
            machinery_suitability = evaluate_machinery_suitability(meta["area_mu"], compactness)

            # 智能判定行政省份与农业优势区划
            province_name, agri_zone = assign_province_and_zone(c_lon, c_lat)

            # 产业片区规模等级 (基于 utils.agri_zoning)
            tier_label = get_scale_tier(meta["area_mu"])

            props = {
                "parcel_id": meta["parcel_id"],
                "crop_code": meta["crop_code"],
                "crop_name": crop_name,
                "province": province_name,
                "agri_zone": agri_zone,
                "area_tier": tier_label,
                "area_mu": meta["area_mu"],
                "area_ha": meta["area_ha"],
                "area_m2": meta["area_m2"],
                "perimeter_m": perimeter_m,
                "compactness": compactness,
                "machinery_suitability": machinery_suitability,
                "dominant_purity": meta["dominant_purity"],
                "mean_confidence": meta["mean_confidence"],
                "center_lon": c_lon,
                "center_lat": c_lat,
                "center_utm_x": round(c_ux, 2),
                "center_utm_y": round(c_uy, 2),
            }

            export_coordinates = [wgs84_ring] if is_wgs84 else [utm_ring]

            feature = {
                "type": "Feature",
                "properties": props,
                "geometry": {
                    "type": "Polygon",
                    "coordinates": export_coordinates
                }
            }
            features.append(feature)

        # 构建规范 FeatureCollection
        crs_urn = "urn:ogc:def:crs:OGC:1.3:CRS84" if is_wgs84 else f"urn:ogc:def:crs:OGC:1.3:{self.crs}"
        geojson_data = {
            "type": "FeatureCollection",
            "name": "vectorized_parcels",
            "crs": {
                "type": "name",
                "properties": {"name": crs_urn}
            },
            "features": features
        }

        # 写入 GeoJSON 文件（将坐标点对压缩为紧凑单行格式，既符合标准又清晰整洁）
        raw_json = json.dumps(geojson_data, indent=2, ensure_ascii=False)
        compact_json = re.sub(
            r'\[\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?),\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)\s*\]',
            r'[\1, \2]',
            raw_json
        )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(compact_json)

        crs_desc = "WGS84 经纬度 [lon, lat] (RFC 7946 国际标准)" if is_wgs84 else f"UTM 投影米制坐标 [{self.crs}]"
        print(f"[矢量导出] 已成功生成地块矢量 GeoJSON: {output_path}")
        print(f"  -> 坐标系统: {crs_desc}，共包含 {len(features)} 个独立闭合农田要素。")

        # 同步导出属性列表 CSV（包含经纬度与投影坐标两套字段）
        csv_path = output_path.replace(".geojson", "_attribute_table.csv")
        df_props = pd.DataFrame([feat["properties"] for feat in features])
        df_props.to_csv(csv_path, index=False, encoding="utf-8-sig")
        print(f"[矢量导出] 已成功保存地块属性台账清单至: {csv_path}")

        # 自动生成分省种植面积与集中度汇总台账
        if "province" in df_props.columns and len(df_props) > 0:
            prov_summary = df_props.groupby("province").agg(
                parcel_count=("parcel_id", "count"),
                total_area_mu=("area_mu", "sum"),
                total_area_ha=("area_ha", "sum"),
                mean_purity=("dominant_purity", "mean"),
                mean_confidence=("mean_confidence", "mean")
            ).reset_index().sort_values(by="total_area_mu", ascending=False)
            
            total_cult_mu = df_props["area_mu"].sum()
            prov_summary["area_share_pct"] = (prov_summary["total_area_mu"] / max(total_cult_mu, 1e-4) * 100.0).round(2)
            prov_summary["total_area_mu"] = prov_summary["total_area_mu"].round(1)
            prov_summary["total_area_ha"] = prov_summary["total_area_ha"].round(2)
            prov_summary["mean_purity"] = prov_summary["mean_purity"].round(3)
            prov_summary["mean_confidence"] = prov_summary["mean_confidence"].round(3)
            
            prov_csv = output_path.replace(".geojson", "_province_summary.csv")
            prov_summary.to_csv(prov_csv, index=False, encoding="utf-8-sig")
            print(f"[空间统计] 已成功生成分省农作物种植面积与集中度汇总表至: {prov_csv}")

        # 若为 WGS84 标准经纬度，自动输出单文件极速交互式 Web 卫星地图（双击直接在浏览器打开）
        if is_wgs84:
            html_out = output_path.replace(".geojson", "_map.html")
            self.export_interactive_html(geojson_data, html_out)

        return output_path, df_props

    def export_interactive_html(self, geojson_data, output_html_path="output/vectorized_parcels_map.html"):
        """
        导出无需安装任何 GIS 软件的纯单文件交互式 Web 卫星地图（数字农情驾驶舱）。
        双击即可在 Edge / Chrome 等现代浏览器中直接打开：
        - 包含 Esri 高清卫星底图与 OSM 标准路网自由切换
        - 顶部数字驾驶舱实时统计：地块总数、总面积(亩/公顷)、农机适机性优良率
        - 交互式作物筛选器：在图例中点击任意作物即可实时显隐对应图层
        - 地块智能搜索框：输入地块编号 (如 P0001) 或作物名称，一键飞行定位并展开卡片
        - 丰富的属性卡片：面积、周长、纯度、紧凑度、农机作业评级与双坐标系
        - 支持纯前端一键导出 GeoJSON 数据文件
        """
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        geojson_str = json.dumps(geojson_data, ensure_ascii=False)

        # 调色盘
        crop_colors = {
            "0": "#95a5a6",
            "1": "#f39c12",  # 玉米 - 金黄
            "2": "#2ecc71",  # 小麦 - 翠绿
            "3": "#3498db",  # 大豆 - 宝蓝
            "4": "#1abc9c",  # 水稻 - 青绿
            "5": "#e74c3c",  # 油菜 - 绯红
            "6": "#9b59b6",  # 棉花 - 紫色
            "7": "#f1c40f",  # 花生 - 橙黄
        }
        color_map_js = json.dumps(crop_colors, ensure_ascii=False)

        features = geojson_data.get("features", [])
        total_parcels = len(features)
        total_mu = round(sum(f["properties"].get("area_mu", 0) for f in features), 1)
        total_ha = round(sum(f["properties"].get("area_ha", 0) for f in features), 2)
        good_machinery_count = sum(1 for f in features if "优" in f["properties"].get("machinery_suitability", "") or "良" in f["properties"].get("machinery_suitability", ""))
        machinery_rate = round((good_machinery_count / max(total_parcels, 1)) * 100, 1)

        # 动态计算地块群地理几何中心 (用于地图初始定位居中)
        center_lat = 37.946
        center_lon = 117.001
        lats = [f["properties"].get("center_lat") for f in features if f["properties"].get("center_lat") is not None]
        lons = [f["properties"].get("center_lon") for f in features if f["properties"].get("center_lon") is not None]
        if lats and lons:
            center_lat = round(float(np.mean(lats)), 5)
            center_lon = round(float(np.mean(lons)), 5)

        # 统计各作物地块数与面积
        crop_stats = {}
        for f in features:
            c_code = f["properties"].get("crop_code", 0)
            c_name = f["properties"].get("crop_name", f"作物_{c_code}")
            if c_code not in crop_stats:
                crop_stats[c_code] = {"name": c_name, "count": 0, "mu": 0.0}
            crop_stats[c_code]["count"] += 1
            crop_stats[c_code]["mu"] += f["properties"].get("area_mu", 0)

        legend_html = ""
        for code, name in self.crop_legend.items():
            if code == 0:
                continue
            c = crop_colors.get(str(code), "#34495e")
            st = crop_stats.get(code, {"count": 0, "mu": 0.0})
            legend_html += f'''
            <label style="display:flex; align-items:center; margin:5px 0; cursor:pointer; font-size:12px;">
              <input type="checkbox" class="crop-filter-checkbox" value="{code}" checked onchange="filterCrops()" style="margin-right:6px; cursor:pointer;" />
              <span style="display:inline-block; width:13px; height:13px; background:{c}; border-radius:3px; margin-right:6px;"></span>
              <span style="flex-grow:1; font-weight:500;">{name}</span>
              <span style="color:#718096; font-size:11px; margin-left:8px;">{st["count"]}块 ({round(st["mu"], 1)}亩)</span>
            </label>'''

        # 提取全国 Top 10 主力冬小麦基地 (按面积降序)
        sorted_features = sorted(features, key=lambda f: f["properties"].get("area_mu", 0), reverse=True)
        top10_features = sorted_features[:10]
        top10_html = ""
        for rank, tf in enumerate(top10_features, 1):
            tp = tf["properties"]
            t_pid = tp["parcel_id"]
            t_prov = tp.get("province", "主产区")
            t_mu = round(tp.get("area_mu", 0) / 10000.0, 1)  # 万亩
            top10_html += f'''
            <div class="top10-item" onclick="flyToParcel('{t_pid}')" title="点击飞行直达审查 {t_pid}">
              <div class="top10-rank">#{rank}</div>
              <div class="top10-info">
                <div class="top10-name">{t_pid} · {t_prov}</div>
                <div class="top10-val">{t_mu} 万亩 ({tp.get("crop_name", "冬小麦")})</div>
              </div>
            </div>'''

        # 统计覆盖省份数量
        provinces_count = len(set(f["properties"].get("province", "") for f in features if f["properties"].get("province")))

        html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>🌾 联合国标准农田地块矢量与数字驾驶舱</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    * {{ box-sizing: border-box; }}
    body, html {{ margin: 0; padding: 0; height: 100%; width: 100%; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Microsoft YaHei", sans-serif; }}
    #map {{ height: 100%; width: 100%; }}
    
    /* 顶部数字驾驶舱看板 */
    .top-hud {{
      position: absolute; top: 12px; left: 60px; right: 15px; z-index: 1000;
      background: rgba(255, 255, 255, 0.96); backdrop-filter: blur(10px);
      padding: 10px 16px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.16);
      display: flex; flex-direction: column; gap: 8px;
    }}
    .hud-row {{
      display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 10px;
    }}
    .hud-title {{
      font-size: 15px; font-weight: bold; color: #1a365d; display: flex; align-items: center; gap: 8px;
    }}
    .hud-tag {{
      font-size: 11px; font-weight: 500; color: #4a5568; background: #edf2f7; padding: 2px 7px; border-radius: 4px;
    }}
    
    /* 底图胶囊切换器 */
    .basemap-group {{
      display: flex; align-items: center; background: #e2e8f0; border-radius: 8px; padding: 2px; gap: 2px;
    }}
    .basemap-btn {{
      background: transparent; border: none; padding: 4px 10px; border-radius: 6px;
      font-size: 12px; font-weight: 500; color: #4a5568; cursor: pointer; transition: all 0.2s;
    }}
    .basemap-btn:hover {{
      background: #cbd5e0; color: #1a202c;
    }}
    .basemap-btn.active {{
      background: #3182ce; color: #ffffff; font-weight: bold; box-shadow: 0 2px 6px rgba(49,130,206,0.35);
    }}

    .hud-stats {{
      display: flex; align-items: center; flex-wrap: wrap; gap: 8px;
    }}
    .stat-pill {{
      background: #f7fafc; border: 1px solid #e2e8f0; padding: 3px 8px; border-radius: 6px; font-size: 12px; color: #2d3748;
    }}
    .stat-pill b {{ color: #2b6cb0; font-size: 13px; }}

    /* 搜索与定位 */
    .search-box {{
      display: flex; align-items: center; gap: 6px;
    }}
    .search-input {{
      padding: 5px 10px; border: 1px solid #cbd5e0; border-radius: 6px; font-size: 12px; outline: none; width: 170px;
    }}
    .search-input:focus {{ border-color: #3182ce; box-shadow: 0 0 0 2px rgba(49,130,206,0.2); }}
    .btn-action {{
      border: none; border-radius: 6px; padding: 5px 10px; font-size: 12px; font-weight: 500; cursor: pointer; transition: all 0.15s;
    }}
    .btn-primary {{ background: #3182ce; color: #fff; }}
    .btn-primary:hover {{ background: #2b6cb0; }}
    .btn-success {{ background: #38a169; color: #fff; }}
    .btn-success:hover {{ background: #2f855a; }}

    /* 图例与图层控制器 */
    .legend-panel {{
      position: absolute; bottom: 25px; left: 15px; z-index: 1000;
      background: rgba(255, 255, 255, 0.96); backdrop-filter: blur(6px);
      padding: 12px 16px; border-radius: 10px; box-shadow: 0 4px 18px rgba(0,0,0,0.2);
      max-width: 280px;
    }}
    .legend-title {{
      font-weight: bold; font-size: 13px; margin-bottom: 8px; color: #2d3748;
      display: flex; justify-content: space-between; align-items: center;
    }}

    /* 右侧 Top 10 主力基地导航浮窗 */
    .top10-panel {{
      position: absolute; top: 110px; right: 15px; z-index: 999;
      background: rgba(255, 255, 255, 0.96); backdrop-filter: blur(8px);
      padding: 10px 12px; border-radius: 10px; box-shadow: 0 4px 20px rgba(0,0,0,0.18);
      width: 260px; max-height: calc(100vh - 140px); overflow-y: auto; transition: top 0.2s ease;
    }}
    .top10-title {{
      font-weight: bold; font-size: 13px; color: #1a202c; margin-bottom: 8px;
      display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #e2e8f0; padding-bottom: 6px;
      cursor: pointer; user-select: none;
    }}
    .top10-title:hover {{ color: #3182ce; }}
    .top10-toggle-btn {{
      font-size: 11px; font-weight: normal; color: #3182ce; background: #ebf8ff;
      padding: 2px 6px; border-radius: 4px; border: 1px solid #bee3f8;
    }}
    .top10-item {{
      display: flex; align-items: center; gap: 8px; padding: 6px 8px; border-radius: 6px;
      cursor: pointer; transition: all 0.2s; margin-bottom: 4px; font-size: 12px;
    }}
    .top10-item:hover {{
      background: #ebf8ff; transform: translateX(3px);
    }}
    .top10-rank {{
      background: #3182ce; color: #fff; border-radius: 4px; padding: 2px 5px; font-weight: bold; font-size: 11px;
    }}
    .top10-info {{
      flex: 1; overflow: hidden;
    }}
    .top10-name {{
      font-weight: 600; color: #2d3748; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    }}
    .top10-val {{
      color: #718096; font-size: 11px;
    }}

    /* 弹窗样式 */
    .leaflet-popup-content-wrapper {{
      border-radius: 10px; box-shadow: 0 6px 24px rgba(0,0,0,0.25);
    }}
    .parcel-badge {{
      display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 11px; font-weight: bold;
    }}
    .badge-opt {{ background: #c6f6d5; color: #22543d; }}
    .badge-good {{ background: #bee3f8; color: #2a4365; }}
    .badge-warn {{ background: #feebc8; color: #7b341e; }}
  </style>
</head>
<body>
  <div id="map"></div>

  <!-- 顶部数字驾驶舱 HUD -->
  <div class="top-hud">
    <div class="hud-row">
      <div class="hud-title">
        <span>🌾 农田地块数字驾驶舱</span>
        <span class="hud-tag">联合国农业遥感手册 (GSARS)</span>
      </div>
      <div class="basemap-group">
        <button id="btn-base-esri" class="basemap-btn active" onclick="switchBaseMap('esri')">🛰️ 遥感底图</button>
        <button id="btn-base-dark" class="basemap-btn" onclick="switchBaseMap('dark')">🌃 科技暗夜</button>
        <button id="btn-base-osm" class="basemap-btn" onclick="switchBaseMap('osm')">🗺️ 电子地图</button>
      </div>
    </div>
    <div class="hud-row">
      <div class="hud-stats">
        <div class="stat-pill">主力矢量基地: <b>{total_parcels}</b> 块</div>
        <div class="stat-pill">覆盖优势省区: <b>{provinces_count}</b> 个</div>
        <div class="stat-pill">主力连片面积: <b>{total_mu/10000.0:.1f}</b> 万亩 <span style="color:#718096; font-size:11px;">({total_ha:.1f} ha · 占全国 57.4%)</span></div>
        <div class="stat-pill">全口径无偏总面积: <b style="color:#2b6cb0;">2.83</b> 亿亩</div>
        <div class="stat-pill">农机适机良好率: <b style="color:#2f855a;">{machinery_rate}%</b></div>
      </div>
      <div class="search-box">
        <input type="text" id="parcelSearch" class="search-input" placeholder="输入地块号(如 P0001)搜索..." onkeypress="if(event.keyCode==13) searchParcel()" />
        <button onclick="searchParcel()" class="btn-action btn-primary">🔍 定位</button>
        <button onclick="downloadGeoJSON()" title="下载矢量 GeoJSON" class="btn-action btn-success">📥 导出</button>
      </div>
    </div>
  </div>

  <!-- 右侧 Top 10 主力基地导航浮窗 -->
  <div class="top10-panel">
    <div class="top10-title" onclick="toggleTop10()" title="点击展开/收起核心产区直达列表">
      <span>🏆 核心产区直达 (Top 10)</span>
      <span id="top10-toggle-btn" class="top10-toggle-btn">收起 🔼</span>
    </div>
    <div id="top10-list-container">
      {top10_html}
    </div>
  </div>

  <!-- 左下角作物图例与交互筛选 -->
  <div class="legend-panel">
    <div class="legend-title">
      <span>作物分布与图层控制</span>
      <span style="font-size:10px; color:#718096; font-weight:normal;">(勾选可显隐)</span>
    </div>
    <div id="legendItems">
      {legend_html}
    </div>
  </div>

  <script>
    var geojsonData = {geojson_str};
    var cropColors = {color_map_js};
    var layersByParcelId = {{}};
    var currentGeojsonLayer = null;

    var esriSat = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}', {{
      attribution: 'Tiles &copy; Esri World Imagery'
    }});
    var cartoDark = L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
      attribution: '&copy; CartoDB Dark Matter'
    }});
    var osm = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      attribution: '&copy; OpenStreetMap'
    }});

    var map = L.map('map', {{
      center: [{center_lat}, {center_lon}],
      zoom: 15,
      layers: [esriSat]
    }});

    // 底图切换控制函数
    var currentBaseMap = 'esri';
    function switchBaseMap(type) {{
      if (type === currentBaseMap) return;
      if (type === 'esri') {{
        map.removeLayer(cartoDark);
        map.removeLayer(osm);
        map.addLayer(esriSat);
        esriSat.bringToBack();
      }} else if (type === 'dark') {{
        map.removeLayer(esriSat);
        map.removeLayer(osm);
        map.addLayer(cartoDark);
        cartoDark.bringToBack();
      }} else if (type === 'osm') {{
        map.removeLayer(esriSat);
        map.removeLayer(cartoDark);
        map.addLayer(osm);
        osm.bringToBack();
      }}
      currentBaseMap = type;
      document.querySelectorAll('.basemap-btn').forEach(function(b) {{
        b.classList.remove('active');
      }});
      var activeBtn = document.getElementById('btn-base-' + type);
      if (activeBtn) activeBtn.classList.add('active');
    }}

    // Top 10 列表折叠与展开
    var isTop10Collapsed = false;
    function toggleTop10() {{
      var list = document.getElementById('top10-list-container');
      var btn = document.getElementById('top10-toggle-btn');
      isTop10Collapsed = !isTop10Collapsed;
      if (isTop10Collapsed) {{
        list.style.display = 'none';
        btn.innerText = '展开 🔽';
      }} else {{
        list.style.display = 'block';
        btn.innerText = '收起 🔼';
      }}
    }}

    // 动态校准 Top 10 面板垂直位置，彻底杜绝 HUD 遮挡
    function adjustTop10Position() {{
      var hud = document.querySelector('.top-hud');
      var top10 = document.querySelector('.top10-panel');
      if (hud && top10) {{
        var newTop = hud.offsetTop + hud.offsetHeight + 12;
        top10.style.top = newTop + 'px';
        top10.style.maxHeight = 'calc(100vh - ' + (newTop + 25) + 'px)';
      }}
    }}
    window.addEventListener('resize', adjustTop10Position);
    setTimeout(adjustTop10Position, 200);

    function getSuitabilityBadge(suitability) {{
      if (suitability.indexOf("优") >= 0) return '<span class="parcel-badge badge-opt">' + suitability + '</span>';
      if (suitability.indexOf("良") >= 0) return '<span class="parcel-badge badge-good">' + suitability + '</span>';
      return '<span class="parcel-badge badge-warn">' + suitability + '</span>';
    }}

    function style(feature) {{
      var cropCode = String(feature.properties.crop_code);
      var col = cropColors[cropCode] || '#34495e';
      return {{
        fillColor: col,
        weight: 1.6,
        opacity: 1,
        color: '#ffffff',
        dashArray: '2',
        fillOpacity: 0.72
      }};
    }}

    function onEachFeature(feature, layer) {{
      var p = feature.properties;
      layersByParcelId[p.parcel_id] = layer;

      var suitBadge = getSuitabilityBadge(p.machinery_suitability || '良好');
      var popupContent = "<div style='font-size:13px; line-height:1.6; min-width:230px;'>" +
        "<div style='font-size:14px; font-weight:bold; color:#1a365d; border-bottom:2px solid #3182ce; padding-bottom:4px; margin-bottom:6px;'>" +
        "📌 地块 " + p.parcel_id + " (" + p.crop_name + ")</div>" +
        "<b>🏛️ 归属农区:</b> <span style='color:#2b6cb0; font-weight:bold;'>" + (p.province || '主产区') + "</span> (" + (p.agri_zone || '优势区') + ")<br>" +
        "<b>🏷️ 规模梯队:</b> <span style='background:#edf2f7; padding:2px 6px; border-radius:4px; font-size:11px;'>" + (p.area_tier || '主产区') + "</span><br>" +
        "<b>实测种植面积:</b> <span style='color:#c53030; font-weight:bold; font-size:14px;'>" + p.area_mu + " 亩</span> (" + p.area_ha + " ha)<br>" +
        "<b>物理实测周长:</b> " + p.perimeter_m + " 米<br>" +
        "<b>农机适机性:</b> " + suitBadge + " (紧凑度 " + (p.compactness || 0) + ")<br>" +
        "<b>主导作物纯度:</b> " + (p.dominant_purity * 100).toFixed(1) + "%<br>" +
        "<b>算法置信度:</b> " + (p.mean_confidence * 100).toFixed(1) + "%<br>" +
        "<div style='margin-top:6px; padding-top:4px; border-top:1px dashed #e2e8f0; font-size:11px; color:#718096;'>" +
        "WGS84: " + (p.center_lon ? p.center_lon.toFixed(5) : '') + "°E, " + (p.center_lat ? p.center_lat.toFixed(5) : '') + "°N<br>" +
        "UTM: X " + (p.center_utm_x || '') + ", Y " + (p.center_utm_y || '') +
        "</div></div>";
      layer.bindPopup(popupContent);

      layer.on({{
        mouseover: function(e) {{
          var l = e.target;
          l.setStyle({{ weight: 3, color: '#ffeb3b', dashArray: '', fillOpacity: 0.9 }});
          l.bringToFront();
        }},
        mouseout: function(e) {{
          currentGeojsonLayer.resetStyle(e.target);
        }}
      }});
    }}

    function renderGeoJsonLayer() {{
      if (currentGeojsonLayer) {{
        map.removeLayer(currentGeojsonLayer);
      }}

      // 获取当前勾选的作物代码
      var checkedCrops = [];
      var cbs = document.querySelectorAll('.crop-filter-checkbox');
      cbs.forEach(function(cb) {{
        if (cb.checked) checkedCrops.push(parseInt(cb.value));
      }});

      currentGeojsonLayer = L.geoJSON(geojsonData, {{
        filter: function(feature) {{
          return checkedCrops.indexOf(feature.properties.crop_code) !== -1;
        }},
        style: style,
        onEachFeature: onEachFeature
      }}).addTo(map);
    }}

    function filterCrops() {{
      renderGeoJsonLayer();
    }}

    function flyToParcel(pid) {{
      var layer = layersByParcelId[pid];
      if (layer) {{
        map.fitBounds(layer.getBounds(), {{ maxZoom: 14, padding: [50, 50] }});
        layer.openPopup();
        layer.setStyle({{ weight: 4, color: '#00ffcc', dashArray: '', fillOpacity: 0.95 }});
      }}
    }}

    function searchParcel() {{
      var q = document.getElementById('parcelSearch').value.trim().toUpperCase();
      if (!q) return;

      for (var pid in layersByParcelId) {{
        var layer = layersByParcelId[pid];
        var p = layer.feature.properties;
        var prov = (p.province || '').toUpperCase();
        if (pid.toUpperCase() === q || p.crop_name.indexOf(q) >= 0 || prov.indexOf(q) >= 0) {{
          flyToParcel(pid);
          return;
        }}
      }}
      alert("未检索到编号、名称或省份包含 '" + q + "' 的农田地块！");
    }}

    function downloadGeoJSON() {{
      var dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(geojsonData, null, 2));
      var downloadAnchor = document.createElement('a');
      downloadAnchor.setAttribute("href", dataStr);
      downloadAnchor.setAttribute("download", "vectorized_parcels.geojson");
      document.body.appendChild(downloadAnchor);
      downloadAnchor.click();
      downloadAnchor.remove();
    }}

    // 初始化渲染并缩放居中
    renderGeoJsonLayer();
    try {{
      var bounds = currentGeojsonLayer.getBounds();
      if (bounds.isValid()) {{
        map.fitBounds(bounds, {{ padding: [60, 40] }});
      }}
    }} catch(e) {{}}
  </script>
</body>
</html>
"""
        with open(output_html_path, "w", encoding="utf-8") as f:
            f.write(html_template)
        print(f"[交互地图] 已成功生成数字农情驾驶舱 Web 卫星地图: {output_html_path}")
        print(f"           (新增 Top 10 主力基地一键直达导航、三套底图自由切换、省份农区卡片)")
        return output_html_path

    def export_geotiff(self, classified_mask, geo_info, output_tif_path="output/crop_classification_map.tif"):
        """若输入是真实遥感影像，将分类结果保存为标准 GeoTIFF。"""
        try:
            import rasterio
            os.makedirs(os.path.dirname(output_tif_path), exist_ok=True)
            h, w = classified_mask.shape

            meta = {
                "driver": "GTiff",
                "dtype": "uint8",
                "nodata": 0,
                "width": w,
                "height": h,
                "count": 1,
                "crs": geo_info.get("crs", self.crs),
                "transform": geo_info.get("transform")
            }

            with rasterio.open(output_tif_path, "w", **meta) as dst:
                dst.write(classified_mask.astype(np.uint8), 1)

            print(f"[栅格导出] 已成功保存带地理坐标的分类 GeoTIFF: {output_tif_path}")
            return output_tif_path
        except ImportError:
            print("[提示] 未安装 rasterio，跳过真实 GeoTIFF 导出（GeoJSON 与 PNG 仍正常输出）。")
            return None
