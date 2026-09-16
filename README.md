# 🌾 农作物种植区域提取与零碎地块矢量化系统 (Crop Area Pipeline)

本项目基于**联合国粮农组织 (FAO) 与联合国统计司 (UNSD)** 联合发布的《农业统计遥感手册》（*UN Handbook on Remote Sensing for Agricultural Statistics*）构建，是一个专门针对**“种植地块零碎、小农田埂密集”**及**“20~30 年长时序历史回溯与持续监测”**打造的自动化、端到端农作物种植区域提取与地块矢量化工程。

---

## 📖 理论渊源与核心技术

### 1. 多时相卫星物候指纹分类（手册第 3 章与第 11 章 sits / SITS）
针对农田作物“同物异谱、异物同谱”的难题，系统不依赖单景静态遥感影像，而是基于作物全生长周期的多光谱时序动态（NDVI、EVI、红边 NDRE）：
- **冬小麦**：春季（DOY 110-140）达到抽穗灌浆峰值，6月上旬急速下降收割；
- **夏玉米**：6月出苗，8月盛夏（DOY 230）达到绿度巅峰，9月下旬收割；
- **大豆**：物候峰期稍晚于玉米，时序斜率与冠层结构具有独特反射特征。
结合时序差分、积分与变化斜率，实现对全域作物种类的精确空间识别。

### 2. 零碎小农田块形态学边缘分离（手册理论篇第 8 章 IBGE & FAO）
针对小农经济下田块破碎、形状不规则、田埂狭窄（仅 1~2 个像元宽）的痛点：
- 采用**作物交界梯度检测**与**形态学边缘腐蚀（Erosion）**，精准切断相邻不同地块的误粘连；
- 运用连通域标记（Connected Component Labeling）与多边形轮廓跟踪，将栅格自动转化为标准的**封闭矢量地块（Polygon）**；
- 过滤细碎杂斑噪声，输出每块田的独立编码（`parcel_id`）、实测面积（**亩** / 公顷）、周长与主导作物类别。

### 3. 联合国官方无偏种植面积推断（手册统计篇第 24 章与第 26 章 Weighted Area / PPI）
针对零碎农田边缘混合像元严重、直接“数像元面积”（Pixel-Counting）会导致 15%~35% 系统性偏差的问题：
- 引入联合国手册推荐的**加权转移矩阵（Weighted Confusion Calibration）**与 **Percentile Bootstrap（2,000 次重抽样）**；
- 在数学上严格消除遥感分类误差，输出具备法律与统计合规性的各作物**真实无偏种植面积**及 **95% 置信区间**。

---

## 📁 目录结构

```
crop_area_pipeline/
├── README.md                          # 本说明文档
├── requirements.txt                   # 运行依赖包清单
├── config.yaml                        # 集中参数配置（像元分辨率、最小地块面积、模型超参数等）
├── demo_quickstart.py                 # 极简独立单文件演示脚本（零门槛极速体验）
├── main.py                            # 端到端主运行流水线
├── data/                              # 数据目录（内置真实模拟的基准测试集）
│   ├── sample_phenology_curves.csv    # 各作物多时相物候基准曲线
│   ├── sample_training_points.csv     # 训练集样本点（多时相特征与标签）
│   └── ground_truth_area_sample.csv   # 联合国手册抽样校准样方数据
├── src/                               # 核心算法源码库
│   ├── __init__.py
│   ├── time_series_builder.py         # 多时相卫星时序立方体构建与物候特征提取
│   ├── crop_classifier.py             # 多时相作物分类器（Random Forest / Gradient Boosting）
│   ├── parcel_segmenter.py            # 零碎地块形态学切分与田埂分离（手册第8章）
│   ├── vector_exporter.py             # 地块几何边界矢量化（导出标准 GeoJSON 与属性表）
│   ├── area_unbiased_estimator.py     # 联合国第24/26章加权无偏面积校准引擎
│   └── visualizer.py                  # 出版级专题制图（物候图、分类图、边界图、面积柱状图）
└── output/                            # 成果输出目录
    ├── crop_classification_map.png    # 农作物种植分布分类专题图
    ├── parcel_delineation_boundaries.png # 零碎地块矢量边界与田埂切分图
    ├── vectorized_parcels.geojson      # 提取的标准 OGC GeoJSON 地块矢量多边形
    ├── vectorized_parcels_attribute_table.csv # 每一个地块的属性台账（亩数、作物、周长）
    ├── acreage_statistics_report.csv   # 官方无偏种植面积统计汇总台账（含95%置信区间）
    └── area_unbiased_comparison.png    # 传统像元面积 vs 联合国无偏校准面积对比图
```

---

## 🚀 快速开始

### 1. 环境准备

推荐在 Python 3.8+（已完美适配 Python 3.13 与 NumPy 2.x）环境下运行：

```bash
cd crop_area_pipeline
pip install -r requirements.txt
```

### 2. 方式一：极速单文件试跑 (Demo Quickstart)

无需配置任何路径，直接执行单文件极速演示：

```bash
python demo_quickstart.py
```

终端将立即展示：
* 80×80 像元零碎农田多时相场景生成；
* 多时相物候特征提取与随机森林模型训练；
* 零碎田块自动勾勒与田埂切分结果；
* 联合国加权无偏种植面积对比台账。

### 3. 方式二：运行完整生产流水线 (Full Pipeline)

```bash
python main.py --config config.yaml
```

执行后，系统将自动完成五大步骤并在 `output/` 目录下生成完整成果：
1. **`output/vectorized_parcels.geojson`**：包含所有独立田块外边界多边形的标准矢量图层，可直接拖入 **QGIS** 或 **ArcGIS** 进行三维漫游与空间分析。
2. **`output/vectorized_parcels_attribute_table.csv`**：全量地块属性清单（地块编号、种植作物、净种植面积(亩)、周长(米)、中心经纬度）。
3. **`output/acreage_statistics_report.csv`**：联合国统计标准的无偏面积台账，记录了每种作物的朴素像元面积、校准无偏面积、95% 置信区间以及偏差修正量。
4. **4 幅出版级高清科研图表**（时序物候曲线、分类图、地块边界图、面积校准对比图）。

---

## 🛰️ 覆盖 20~30 年长时序需求的技术指引

针对您提出的跨越 20~30 年历史数据的需求，结合本手册规范推荐以下配置方案：

1. **近 10 年（2015 至今）**：
   * **数据源**：免费公开的 **Copernicus Sentinel-2**（10米分辨率，5天重访）；
   * **处理模式**：在 `config.yaml` 中设置 `resolution_meters: 10.0`，可直接精确提取每个 1~2 亩以上的零碎田块矢量多边形。
2. **过去 10~30 年（1995 ~ 2015）**：
   * **数据源**：免费公开的 **USGS Landsat 5 / 7 / 8**（30米分辨率，16天重访）；
   * **处理模式**：在 `config.yaml` 中设置 `resolution_meters: 30.0`。针对 30 米像元下零碎田块难以勾勒单个微型田埂的特点，以现代提取的高精度地块矢量为空间底图，运行本工程的 `area_unbiased_estimator.py` 反演历史年份的作物占比与无偏种植总面积，完美满足 30 年连续历史演变分析！
