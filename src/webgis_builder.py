"""
数字农情空间驾驶舱与纯前端 WebGIS 交互式地图构建模块。
遵循 OGC / RFC 7946 规范，生成无需任何后端服务器即可单文件离线运行的数字农情驾驶舱：
1. 集成 Esri 高清遥感底图、CartoDB 科技暗夜图与 OSM 标准电子地图平滑切换；
2. 顶部综合农情 HUD 看板：地块总数、分省覆盖、总面积统计与农机作业适机性优良率；
3. 作物多维动态筛选图例 + 行政省份智能过滤联动；
4. 右侧 Top 10 主力战略生产基地一键飞行定位直达；
5. 地块智能检索定位与前瞻性 GeoJSON 数据本地一键导出。
"""

import os
import json
import numpy as np
from typing import Dict, Any, List, Optional


class WebGISDashboardBuilder:
    """数字农情 WebGIS 空间驾驶舱生成器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.crop_legend = self.config.get("crop_legend", {
            0: "非农田/背景",
            1: "夏玉米",
            2: "冬小麦",
            3: "大豆"
        })
        # 调色盘配置 (支持动态扩展作物)
        self.crop_colors = {
            "0": "#95a5a6",
            "1": "#f39c12",  # 玉米 - 金黄
            "2": "#2ecc71",  # 小麦 - 翠绿
            "3": "#3498db",  # 大豆 - 宝蓝
            "4": "#1abc9c",  # 水稻 - 青绿
            "5": "#e74c3c",  # 油菜 - 绯红
            "6": "#9b59b6",  # 棉花 - 紫色
            "7": "#f1c40f",  # 花生 - 橙黄
            "8": "#e67e22",  # 马铃薯 - 棕褐
        }

    def build_dashboard(
        self,
        geojson_data: Dict[str, Any],
        output_html_path: str = "output/vectorized_parcels_map.html"
    ) -> str:
        """
        根据 GeoJSON 数据生成自包含单文件 WebGIS 数字农情交互式地图。
        
        参数：
            geojson_data: 符合 RFC 7946 规范的 FeatureCollection 字典
            output_html_path: 目标 HTML 报告输出路径
        返回：
            生成的文件绝对路径
        """
        os.makedirs(os.path.dirname(output_html_path), exist_ok=True)
        geojson_str = json.dumps(geojson_data, ensure_ascii=False)
        color_map_js = json.dumps(self.crop_colors, ensure_ascii=False)

        features = geojson_data.get("features", [])
        total_parcels = len(features)
        total_mu = round(sum(f["properties"].get("area_mu", 0) for f in features), 1)
        total_ha = round(sum(f["properties"].get("area_ha", 0) for f in features), 2)
        good_machinery_count = sum(
            1 for f in features
            if "优" in str(f["properties"].get("machinery_suitability", ""))
            or "良" in str(f["properties"].get("machinery_suitability", ""))
        )
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
        provinces_set = set()
        for f in features:
            props = f.get("properties", {})
            c_code = props.get("crop_code", 0)
            c_name = props.get("crop_name", f"作物_{c_code}")
            prov = props.get("province", "")
            if prov:
                provinces_set.add(prov)

            if c_code not in crop_stats:
                crop_stats[c_code] = {"name": c_name, "count": 0, "mu": 0.0}
            crop_stats[c_code]["count"] += 1
            crop_stats[c_code]["mu"] += props.get("area_mu", 0)

        provinces_list = sorted(list(provinces_set))
        provinces_count = len(provinces_list)

        # 生成作物图例 HTML
        legend_html = ""
        for code, name in self.crop_legend.items():
            if code == 0:
                continue
            c = self.crop_colors.get(str(code), "#34495e")
            st = crop_stats.get(code, {"count": 0, "mu": 0.0})
            legend_html += f'''
            <label style="display:flex; align-items:center; margin:5px 0; cursor:pointer; font-size:12px;">
              <input type="checkbox" class="crop-filter-checkbox" value="{code}" checked onchange="filterCrops()" style="margin-right:6px; cursor:pointer;" />
              <span style="display:inline-block; width:13px; height:13px; background:{c}; border-radius:3px; margin-right:6px;"></span>
              <span style="flex-grow:1; font-weight:500;">{name}</span>
              <span style="color:#718096; font-size:11px; margin-left:8px;">{st["count"]}块 ({round(st["mu"], 1)}亩)</span>
            </label>'''

        # 生成分省过滤选项
        prov_options_html = '<option value="ALL">全部省份 (全域)</option>'
        for prov in provinces_list:
            prov_options_html += f'<option value="{prov}">{prov}</option>'

        # 提取全国 Top 10 主力战略生产基地 (按面积降序)
        sorted_features = sorted(features, key=lambda f: f["properties"].get("area_mu", 0), reverse=True)
        top10_features = sorted_features[:10]
        top10_html = ""
        for rank, tf in enumerate(top10_features, 1):
            tp = tf["properties"]
            t_pid = tp["parcel_id"]
            t_prov = tp.get("province", "主产区")
            t_mu = round(tp.get("area_mu", 0) / 10000.0, 2)  # 万亩
            top10_html += f'''
            <div class="top10-item" onclick="flyToParcel('{t_pid}')" title="点击飞行直达审查 {t_pid}">
              <div class="top10-rank">#{rank}</div>
              <div class="top10-info">
                <div class="top10-name">{t_pid} · {t_prov}</div>
                <div class="top10-val">{t_mu} 万亩 ({tp.get("crop_name", "冬小麦")})</div>
              </div>
            </div>'''

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
      padding: 5px 10px; border: 1px solid #cbd5e0; border-radius: 6px; font-size: 12px; outline: none; width: 150px;
    }}
    .search-input:focus {{ border-color: #3182ce; box-shadow: 0 0 0 2px rgba(49,130,206,0.2); }}
    .prov-select {{
      padding: 5px 8px; border: 1px solid #cbd5e0; border-radius: 6px; font-size: 12px; outline: none; background: #fff; color: #2d3748;
    }}
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
        <span class="hud-tag">联合国农业遥感手册 (FAO / UNSD)</span>
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
        <div class="stat-pill">主力连片面积: <b>{total_mu/10000.0:.2f}</b> 万亩 <span style="color:#718096; font-size:11px;">({total_ha:.1f} ha)</span></div>
        <div class="stat-pill">农机适机良好率: <b style="color:#2f855a;">{machinery_rate}%</b></div>
      </div>
      <div class="search-box">
        <select id="provFilter" class="prov-select" onchange="filterCrops()" title="按行政省份筛选">
          {prov_options_html}
        </select>
        <input type="text" id="parcelSearch" class="search-input" placeholder="地块号(如 P0001)..." onkeypress="if(event.keyCode==13) searchParcel()" />
        <button onclick="searchParcel()" class="btn-action btn-primary">🔍 定位</button>
        <button onclick="downloadGeoJSON()" title="一键导出标准 GeoJSON 矢量图层" class="btn-action btn-success">📥 导出</button>
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
      <span style="font-size:10px; color:#718096; font-weight:normal;">(勾选显隐)</span>
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
      preferCanvas: true,
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

    // 动态校准 Top 10 面板垂直位置，杜绝 HUD 遮挡
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

      // 获取当前选中的省份
      var selectedProv = "ALL";
      var provSelect = document.getElementById('provFilter');
      if (provSelect) {{
        selectedProv = provSelect.value;
      }}

      currentGeojsonLayer = L.geoJSON(geojsonData, {{
        filter: function(feature) {{
          var matchCrop = checkedCrops.indexOf(feature.properties.crop_code) !== -1;
          var matchProv = (selectedProv === "ALL" || feature.properties.province === selectedProv);
          return matchCrop && matchProv;
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
        return output_html_path
