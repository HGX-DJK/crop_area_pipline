"""
农业遥感与空间分析公共工具库 (Utils)。
整合空间投影、几何拓扑、农业区划、可视化制图与面积度量衡等底层通用算法。
"""

# 1. 空间与投影工具
from src.utils.geo_utils import (
    utm_to_wgs84,
    wgs84_to_utm,
    is_geographic_system,
    estimate_resolution_meters,
    parse_temporal_doy,
)

# 2. 几何拓扑与平滑工具
from src.utils.geometry_utils import (
    trace_grid_boundary,
    simplify_polygon,
    chaikin_smooth_ring,
    calculate_isoperimetric_quotient,
)

# 3. 农业区划与适机性工具
from src.utils.agri_zoning import (
    assign_province_and_zone,
    evaluate_machinery_suitability,
    get_scale_tier,
)

# 4. 可视化辅助工具
from src.utils.plot_utils import (
    setup_chinese_fonts,
    get_crop_colormap,
    downsample_raster_preview,
)

# 5. 面积与度量衡换算
from src.utils.unit_utils import (
    sqm_to_mu,
    mu_to_sqm,
    sqm_to_ha,
    ha_to_sqm,
    mu_to_ha,
    ha_to_mu,
    format_area,
    SQM_PER_HA,
    SQM_TO_MU_RATIO,
    MU_PER_SQM,
    SQM_PER_MU,
    MU_PER_HA,
    WAN_MU,
)

__all__ = [
    # Geo
    "utm_to_wgs84",
    "wgs84_to_utm",
    "is_geographic_system",
    "estimate_resolution_meters",
    "parse_temporal_doy",
    # Geometry
    "trace_grid_boundary",
    "simplify_polygon",
    "chaikin_smooth_ring",
    "calculate_isoperimetric_quotient",
    # Agri Zoning
    "assign_province_and_zone",
    "evaluate_machinery_suitability",
    "get_scale_tier",
    # Plotting
    "setup_chinese_fonts",
    "get_crop_colormap",
    "downsample_raster_preview",
    # Units
    "sqm_to_mu",
    "mu_to_sqm",
    "sqm_to_ha",
    "ha_to_sqm",
    "mu_to_ha",
    "ha_to_mu",
    "format_area",
    "SQM_PER_HA",
    "SQM_TO_MU_RATIO",
    "MU_PER_SQM",
    "SQM_PER_MU",
    "MU_PER_HA",
    "WAN_MU",
]
