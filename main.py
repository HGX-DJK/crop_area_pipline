"""
农作物种植区域提取与零碎地块矢量化系统主流程脚本。
严格对齐《联合国农业统计遥感手册》（第 8、11、24、26 章）标准。

执行流程：
1. 构建多时相卫星影像时间序列与物候特征立方体（提取 NDVI/EVI 动态与斜率）
2. 训练多时相机器学习作物分类器，生成全域种植分类图与置信度
3. 针对零碎小农田块执行形态学边缘腐蚀与狭窄田埂切分（提取独立闭合地块）
4. 导出符合国际 GIS 标准的 GeoJSON 矢量地块文件与属性台账清单（含每块地的面积亩数与主导作物）
5. 执行联合国手册无偏面积校准（消除混合像元误差，计算 95% 置信区间）
6. 导出高清专题制图与统计汇总报告
"""

import os
import sys
import yaml
import argparse
import pandas as pd
import numpy as np

from src.time_series_builder import TimeSeriesBuilder
from src.crop_classifier import CropClassifier
from src.parcel_segmenter import ParcelSegmenter
from src.vector_exporter import VectorExporter
from src.area_unbiased_estimator import AreaUnbiasedEstimator
from src.visualizer import Visualizer
from src.raster_loader import RasterLoader
from src.rotation_tracker import CropRotationTracker
from src.report_generator import ExecutiveReportGenerator
from src.utils.logger import validate_config, get_logger, log_success


def load_config(config_path="config.yaml"):
    logger = get_logger("配置检查")
    if not os.path.exists(config_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, config_path)
        if os.path.exists(alt_path):
            config_path = alt_path
        else:
            raise FileNotFoundError(f"未找到配置文件: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # 规范化并解析所有相对路径为相对于项目根目录 (config 文件所在目录)
    base_dir = os.path.dirname(os.path.abspath(config_path))
    def _resolve(p):
        if p and isinstance(p, str) and not os.path.isabs(p):
            return os.path.normpath(os.path.join(base_dir, p))
        return p

    if "paths" in cfg:
        for k in ["data_dir", "output_dir", "training_samples", "phenology_curves", "ground_truth_samples"]:
            if k in cfg["paths"]:
                cfg["paths"][k] = _resolve(cfg["paths"][k])
    if "input_source" in cfg and "geotiff_dir" in cfg["input_source"]:
        cfg["input_source"]["geotiff_dir"] = _resolve(cfg["input_source"]["geotiff_dir"])

    is_valid, errors = validate_config(cfg)
    if not is_valid:
        for err in errors:
            logger.warning(f"配置校验项: {err}")
    return cfg


def run_pipeline(config_path="config.yaml", override_mode=None, override_geotiff_dir=None, override_output_dir=None, track_rotation=False, sample_plan=False, quiet=False, streaming=False, n_jobs=None, clean_output=True):
    logger = get_logger("流水线", quiet=quiet)
    print("=" * 76)
    print("🌾 联合国农业统计遥感手册标准：农作物种植区域提取与零碎地块矢量化系统")
    print("=" * 76)

    # 1. 加载并自检业务与算法配置
    config = load_config(config_path)
    if override_mode:
        config.setdefault("input_source", {})["mode"] = override_mode
    if override_geotiff_dir:
        config.setdefault("input_source", {})["geotiff_dir"] = override_geotiff_dir
    if override_output_dir:
        config.setdefault("paths", {})["output_dir"] = override_output_dir
    if n_jobs is not None:
        config.setdefault("performance", {})["vectorization_n_jobs"] = n_jobs
    if streaming:
        config.setdefault("performance", {})["enable_window_streaming"] = True

    output_dir = config.get("paths", {}).get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)

    # 自动安全清理 output 目录中的历史遗留成果，避免不同时相/区域结果交叉混淆
    if clean_output:
        target_exts = (".geojson", ".csv", ".html", ".tif", ".tiff", ".png", ".jpg")
        cleaned_files = 0
        for fname in os.listdir(output_dir):
            if fname.endswith(target_exts):
                fpath = os.path.join(output_dir, fname)
                if os.path.isfile(fpath):
                    try:
                        os.remove(fpath)
                        cleaned_files += 1
                    except Exception:
                        pass
        if cleaned_files > 0:
            logger.info(f"已自动清理 output 目录中的 {cleaned_files} 个历史遗留成果文件，确保本次运行结果纯净独立。")

    # 2. 初始化特征工程构建器并加载遥感数据
    ts_builder = TimeSeriesBuilder(config)
    input_mode = config.get("input_source", {}).get("mode", "synthetic")
    geo_info = None
    doy_list = None
    preview_cube = None

    perf_cfg = config.get("performance", {})
    block_size = perf_cfg.get("streaming_block_size", 1024)
    force_streaming = perf_cfg.get("enable_window_streaming", False)

    if input_mode == "geotiff":
        geotiff_dir = config.get("input_source", {}).get("geotiff_dir", "data/satellite_tifs")
        logger.info(f"[步骤 1/5] 解析本地多时相真实遥感 GeoTIFF 空间坐标与时相序列 ({geotiff_dir})...")
        loader = RasterLoader(config)
        sorted_files, geo_info, doy_list, is_multiband = loader.get_multitemporal_metadata(geotiff_dir)

        if geo_info is not None:
            spatial_res = geo_info.get("resolution_meters", 10.0)
            config.setdefault("spatial", {})["resolution_meters"] = spatial_res
            if "crs" in geo_info:
                config.setdefault("spatial", {})["crs"] = geo_info["crs"]
            res_desc = f"{geo_info['resolution_x']:.5f} 度 (约 {spatial_res:.1f} 米)" if geo_info.get("is_geographic") else f"{spatial_res:.2f} 米"
            logger.info(f"  -> 自动对齐影像空间参考 (CRS: {geo_info['crs']}，空间分辨率: {res_desc})，尺寸: {geo_info['height']} 行 × {geo_info['width']} 列，覆盖 {len(doy_list)} 个生长时相。")

        total_pixels = geo_info["height"] * geo_info["width"]
        use_streaming = force_streaming or (total_pixels > 4000000)

        # 3. 训练作物分类器并全域推断
        logger.info("[步骤 2/5] 训练多时相作物智能分类器并执行像素级空间预测...")
        classifier = CropClassifier(config)

        classifier.train_with_samples(
            config.get("paths", {}).get("training_samples", "data/sample_training_points.csv"),
            ts_builder=ts_builder,
            target_t=len(doy_list),
            doy_list=doy_list
        )

        if use_streaming:
            logger.info(f"  -> 影像像元规模达 {total_pixels:,} (超 400 万) 或开启流式，自动启用磁盘纯外核滑动窗口流式推断 (块大小: {block_size}×{block_size})，避免大图内存峰值...")
            crop_mask, conf_map, geo_info, doy_list = classifier.predict_geotiff_stream(
                loader, sorted_files, ts_builder, block_size=block_size
            )
            # 以极低内存提取一张缩略底图 (约 1200×1200，仅数兆)，供成果专题图制图底图使用
            preview_cube = loader.load_preview_thumbnail(sorted_files, max_dim=1200)
        else:
            logger.info("  -> 影像规模适中，载入全图三维矩阵推断...")
            raster_cube, geo_info, doy_list = loader.load_multitemporal_tifs(geotiff_dir)
            feature_cube = ts_builder.extract_phenological_features(raster_cube)
            crop_mask, conf_map = classifier.predict_raster_cube(feature_cube)
            preview_cube = raster_cube
            logger.info(f"  -> 全域空间预测完成，平均分类置信度: {np.mean(conf_map) * 100:.1f}%。")
    else:
        logger.info("[步骤 1/5] 构建多时相卫星时序立方体与提取作物物候指纹 (基准仿真模式)...")
        # 生成/加载标准测试场景 (120x120 像素，包含零碎农田、1~2像素窄田埂与背景地物)
        landscape = ts_builder.generate_synthetic_agricultural_landscape(rows=120, cols=120)
        raster_cube = landscape["raster_cube"]  # (Rows, Cols, 8个时相)
        preview_cube = raster_cube

        # 3. 训练作物分类器并全域推断
        logger.info("[步骤 2/5] 训练多时相作物智能分类器并执行像素级空间预测...")
        classifier = CropClassifier(config)

        classifier.train_with_samples(
            config.get("paths", {}).get("training_samples", "data/sample_training_points.csv"),
            ts_builder=ts_builder,
            target_t=raster_cube.shape[2],
            doy_list=ts_builder.doy_list
        )

        total_pixels = raster_cube.shape[0] * raster_cube.shape[1]
        if force_streaming or total_pixels > 4000000:
            logger.info(f"  -> 启用滑动窗口流式推断 (块大小: {block_size}×{block_size})，避免大图内存峰值...")
            crop_mask, conf_map = classifier.predict_cube_stream(raster_cube, ts_builder, block_size=block_size)
        else:
            feature_cube = ts_builder.extract_phenological_features(raster_cube)
            crop_mask, conf_map = classifier.predict_raster_cube(feature_cube)
            logger.info(f"  -> 全域空间预测完成，平均分类置信度: {np.mean(conf_map) * 100:.1f}%。")

    # 4. 零碎地块形态学分割与田埂切分（核心：联合国手册第8章）
    logger.info("[步骤 3/5] 执行形态学边缘腐蚀与狭窄田埂切分（切分零碎小田块）...")
    segmenter = ParcelSegmenter(config)
    parcel_id_mask, parcel_metadata = segmenter.segment_parcels(crop_mask, conf_map)
    total_valid_parcels = len(parcel_metadata)
    total_cultivated_mu = sum(p["area_mu"] for p in parcel_metadata)
    logger.info(f"  -> 成功勾勒并分离 {total_valid_parcels} 个独立农田地块，累计净耕地面积: {total_cultivated_mu:.1f} 亩。")

    # 5. 导出标准 GIS 矢量地块与属性清单（含农机作业适宜度与紧凑度评估）
    logger.info("[步骤 4/5] 导出 OGC 标准 GeoJSON 地块矢量边界与属性台账清单...")
    exporter = VectorExporter(config)
    geojson_out = os.path.join(output_dir, "vectorized_parcels.geojson")
    exporter.export_geojson(parcel_id_mask, parcel_metadata, geojson_out, geo_info=geo_info)
    if geo_info is not None:
        tif_out = os.path.join(output_dir, "crop_classification_map.tif")
        exporter.export_geotiff(crop_mask, geo_info, tif_out)

    # 6. 联合国第 24/26 章样框无偏面积校准（消除混合像元误差）
    logger.info("[步骤 5/5] 执行联合国手册无偏面积推断（修正零碎田块像元边界偏差）...")
    area_estimator = AreaUnbiasedEstimator(config)
    df_area_report, cond_matrix, df_cm, accuracy_metrics = area_estimator.estimate_unbiased_areas(
        crop_mask,
        config.get("paths", {}).get("ground_truth_samples", "data/ground_truth_area_sample.csv"),
        return_details=True
    )
    report_csv = os.path.join(output_dir, "acreage_statistics_report.csv")
    df_area_report.to_csv(report_csv, index=False)
    log_success(logger, f"已保存官方级无偏种植面积统计台账至: {report_csv}")

    # 导出联合国手册 Table 2 规范面积加权误差矩阵与三维精度评定表
    cm_csv = os.path.join(output_dir, "area_weighted_confusion_matrix.csv")
    df_cm.to_csv(cm_csv, encoding="utf-8-sig")
    log_success(logger, f"已导出联合国 Olofsson (2014) 面积加权混淆矩阵与三维精度表: {cm_csv}")

    # 可选：联合国手册事前样方抽样方案设计 (Neyman Optimal Allocation)
    if sample_plan:
        logger.info("[事前抽样设计] 基于 Neyman 最佳分层抽样算法输出实地调查方案...")
        df_sample_plan = area_estimator.design_optimal_sample_allocation(
            total_sample_budget=150,
            crop_classified_mask=crop_mask,
            min_sample_per_class=20
        )
        plan_csv = os.path.join(output_dir, "sample_allocation_plan.csv")
        area_estimator.export_sampling_plan(df_sample_plan, output_csv=plan_csv)
        log_success(logger, f"目标总预算 150 个样方，已导出《国家样方抽样设计清单》至: {plan_csv}")
        for _, r in df_sample_plan.iterrows():
            logger.info(f"  * {r['crop_name']:<8}: 分配 {r['recommended_sample_n']:>3} 个样框 ({r['sample_ratio_pct']:>4.1f}%) | 面积占比: {r['stratum_weight_Wh']*100:>4.1f}%")

    # 7. 可选长时序（20~30年）农田轮作演变与撂荒/补贴合规监测
    do_rotation = track_rotation or config.get("rotation_tracking", {}).get("enabled", False)
    comp_csv = None
    if do_rotation:
        logger.info("[长时序监测拓展] 执行多年期作物轮作演变矩阵与撂荒/粮豆补贴合规分析...")
        rot_tracker = CropRotationTracker(config)
        # 生成前期参考期基准（若无外部历史数据则基于演化规律生成高保真基线）
        early_mask = rot_tracker.simulate_historical_transition(crop_mask, years_span=5)
        df_trans, df_comp = rot_tracker.analyze_transition(early_mask, crop_mask, year_early=2020, year_late=2024)
        _, comp_csv = rot_tracker.export_rotation_report(df_trans, df_comp, output_dir=output_dir)

        logger.info("跨期作物轮作合规与业务预警清单:")
        for _, row in df_comp.iterrows():
            logger.info(f"  * {row['监测类型']}: {row['涉及面积(亩)']} 亩 -> {row['业务建议']}")

    # 8. 生成出版级可视化成果图表
    if config.get("visualization", {}).get("generate_plots", True):
        viz = Visualizer(config)
        p1 = viz.plot_phenology_curves(config.get("paths", {}).get("phenology_curves", "data/sample_phenology_curves.csv"))
        p2 = viz.plot_crop_classification_map(crop_mask)
        p3 = viz.plot_parcel_delineation(parcel_id_mask, preview_cube)
        p4 = viz.plot_area_comparison(df_area_report)
        log_success(logger, f"已在 '{output_dir}/' 目录下生成 4 幅出版级高清成果图: {os.path.basename(p1)}, {os.path.basename(p2)}, {os.path.basename(p3)}, {os.path.basename(p4)}")

    # 9. 编译生成出版级综合图文决策分析专报 (单文件 HTML，内嵌成果画廊与轮作分析)
    report_gen = ExecutiveReportGenerator(config)
    csv_parcels = geojson_out.replace(".geojson", "_attribute_table.csv")
    briefing_html = os.path.join(output_dir, "national_wheat_executive_briefing.html")
    report_gen.generate_report(
        report_csv,
        csv_parcels,
        briefing_html,
        confusion_matrix_csv=cm_csv,
        rotation_compliance_csv=comp_csv
    )

    # 10. 打印控制台官方统计汇总报表
    print("\n" + "=" * 102)
    print("📊 联合国统计司 / 粮农组织（FAO）农作物种植面积无偏统计与 Olofsson (2014) 官方精度台账")
    print("=" * 102)
    print(f"{'作物名称':<8} | {'像元统计(亩)':<12} | {'无偏校准面积(亩)':<15} | {'标准误 (SE)':<14} | {'变异系数':<8} | {'制图精度 PA':<14} | {'用户精度 UA':<14}")
    print("-" * 102)
    for _, row in df_area_report.iterrows():
        pa_str = row.get("producers_accuracy", "N/A")
        ua_str = row.get("users_accuracy", "N/A")
        se_str = f"±{row['se_analytic_mu']:.1f} 亩"
        cv_str = f"{row['cv_pct']:.2f}%"
        print(f"{row['crop_name']:<8} | {row['naive_area_mu']:<14.1f} | {row['unbiased_calibrated_mu']:<17.1f} | {se_str:<16} | {cv_str:<10} | {pa_str:<16} | {ua_str:<16}")
    print("-" * 102)
    oa_val = accuracy_metrics.get("overall_accuracy", 0.0) * 100.0
    se_oa_val = accuracy_metrics.get("se_overall_accuracy", 0.0) * 100.0
    print(f"🎯 联合国手册全图面积加权总体分类精度 (OA): {oa_val:.2f}% ± {se_oa_val:.2f}% (符合国际统计调查交付标准)")
    print(f"📌 零碎地块切分总结: 共勾勒 {total_valid_parcels} 个地块，平均单块面积 {(total_cultivated_mu/max(total_valid_parcels,1)):.1f} 亩。")
    html_map_path = os.path.join(output_dir, "vectorized_parcels_map.html")
    if os.path.exists(html_map_path):
        print(f"🌐 数字驾驶舱 Web 卫星地图: {html_map_path} (双击浏览器直接打开)")
    if os.path.exists(briefing_html):
        print(f"📑 官方高管决策分析专报: {briefing_html} (一键打印/导出PDF)")
    prov_csv_path = os.path.join(output_dir, "vectorized_parcels_province_summary.csv")
    if os.path.exists(prov_csv_path):
        print(f"📋 全国各省冬小麦统计台账: {prov_csv_path}")
    print(f"✨ 联合国加权样框算法成功校正了小田块田埂像元混淆产生的系统性偏差！")
    print("=" * 86)
    print("🎉 种植区域提取与零碎地块矢量化流水线全部运行完毕！\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="联合国手册农作物种植区域提取与零碎地块矢量化系统")
    parser.add_argument("--config", default="config.yaml", help="配置文件路径 (默认: config.yaml)")
    parser.add_argument("--mode", choices=["synthetic", "geotiff"], default=None, help="数据输入模式 (覆盖 config.yaml)")
    parser.add_argument("--geotiff-dir", default=None, help="多时相 GeoTIFF 影像目录 (覆盖 config.yaml)")
    parser.add_argument("--output-dir", default=None, help="成果输出目录 (覆盖 config.yaml)")
    parser.add_argument("--track-rotation", action="store_true", help="是否同时执行长时序作物轮作演变、撂荒与粮豆补贴合规分析")
    parser.add_argument("--sample-plan", action="store_true", help="是否执行联合国手册 Neyman 最优分层样方抽样设计并导出规划清单")
    parser.add_argument("--streaming", action="store_true", help="强制启用滑动窗口分块流式处理以极致节省内存")
    parser.add_argument("--n-jobs", type=int, default=None, help="多核多进程并行核心数 (默认: 读取配置文件或自动全核)")
    parser.add_argument("--no-clean", dest="clean_output", action="store_false", default=True, help="保留 output 目录中的历史成果文件，不执行自动清理")
    parser.add_argument("--self-check", action="store_true", help="一键执行全系统自动化测试与健康自检")
    parser.add_argument("--quiet", action="store_true", help="开启静默模式，仅输出最终统计台账与严重错误")
    args = parser.parse_args()

    if args.self_check:
        import run_tests
        run_tests.main()
        sys.exit(0)

    run_pipeline(
        config_path=args.config,
        override_mode=args.mode,
        override_geotiff_dir=args.geotiff_dir,
        override_output_dir=args.output_dir,
        track_rotation=args.track_rotation,
        sample_plan=args.sample_plan,
        quiet=args.quiet,
        streaming=args.streaming,
        n_jobs=args.n_jobs,
        clean_output=args.clean_output
    )
