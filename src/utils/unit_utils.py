"""
农业面积与统计度量衡换算通用工具模块。
提供平方米 (m²)、市亩 (mu)、公顷 (ha)、万亩等多向高精度换算。
标准换算基准：
1 市亩 = 666.6666... 平方米 (精确换算系数 1 m² = 0.0015 亩)
1 公顷 (ha) = 10,000 平方米 = 15 市亩
"""

# 精确换算比例常量
SQM_PER_HA = 10000.0
SQM_TO_MU_RATIO = 0.0015
MU_PER_SQM = 0.0015
SQM_PER_MU = 1.0 / 0.0015  # 666.6666...
MU_PER_HA = 15.0
WAN_MU = 10000.0


def sqm_to_mu(sqm: float, decimals: int = None) -> float:
    """平方米转换为市亩。支持可选精度保留位数。"""
    val = float(sqm) * SQM_TO_MU_RATIO
    return round(val, decimals) if decimals is not None else val


def mu_to_sqm(mu: float, decimals: int = None) -> float:
    """市亩转换为平方米。支持可选精度保留位数。"""
    val = float(mu) / SQM_TO_MU_RATIO
    return round(val, decimals) if decimals is not None else val


def sqm_to_ha(sqm: float, decimals: int = None) -> float:
    """平方米转换为公顷。支持可选精度保留位数。"""
    val = float(sqm) / SQM_PER_HA
    return round(val, decimals) if decimals is not None else val


def ha_to_sqm(ha: float, decimals: int = None) -> float:
    """公顷转换为平方米。支持可选精度保留位数。"""
    val = float(ha) * SQM_PER_HA
    return round(val, decimals) if decimals is not None else val


def mu_to_ha(mu: float, decimals: int = None) -> float:
    """市亩转换为公顷。支持可选精度保留位数。"""
    val = float(mu) / MU_PER_HA
    return round(val, decimals) if decimals is not None else val


def ha_to_mu(ha: float, decimals: int = None) -> float:
    """公顷转换为市亩。支持可选精度保留位数。"""
    val = float(ha) * MU_PER_HA
    return round(val, decimals) if decimals is not None else val


def format_area(area_m2: float, precision: int = 2) -> dict:
    """
    输入平方米，返回标准面积多维规格化字典。
    """
    mu_val = sqm_to_mu(area_m2)
    ha_val = sqm_to_ha(area_m2)
    wan_mu_val = mu_val / WAN_MU
    return {
        "area_m2": round(float(area_m2), precision),
        "area_mu": round(mu_val, precision),
        "area_ha": round(ha_val, precision),
        "area_wan_mu": round(wan_mu_val, precision + 2)
    }
