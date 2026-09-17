"""
统一的农情遥感与统计系统日志管理与配置自校验工具。
提供：
1. 统一格式的彩色/符号终端日志输出（INFO, SUCCESS, WARN, ERROR, DEBUG）；
2. 静默模式（Silent Mode）支持（便于作为无干扰 SDK 库导入）；
3. 系统配置防御性自校验（Validate Config）。
"""

import sys
import logging
from typing import Dict, Any, Tuple, List


class AgriColorFormatter(logging.Formatter):
    """带农业遥感业务符号的彩色日志格式化器"""

    # ANSI 颜色码
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    DIM = "\033[2m"

    PREFIX_MAP = {
        logging.DEBUG: ("🔍 [DEBUG]", DIM),
        logging.INFO: ("ℹ️  [INFO] ", ""),
        logging.WARNING: ("⚠️  [WARN] ", YELLOW),
        logging.ERROR: ("🚨 [ERROR]", RED),
        logging.CRITICAL: ("💥 [FATAL]", RED + BOLD),
    }

    def format(self, record: logging.LogRecord) -> str:
        prefix, color = self.PREFIX_MAP.get(record.levelno, ("•", ""))
        msg = record.getMessage()

        # 特殊成功标签支持
        if getattr(record, "is_success", False):
            prefix = "✨ [DONE] "
            color = self.GREEN

        # Windows 终端若不支持 ANSI 转义序列则回退纯文本
        if sys.platform == "win32" and not sys.stdout.isatty():
            return f"{prefix} {msg}"

        return f"{color}{prefix} {msg}{self.RESET}" if color else f"{prefix} {msg}"


def get_logger(name: str = "AgriPipeline", level: int = logging.INFO, quiet: bool = False) -> logging.Logger:
    """获取或初始化统一系统日志器。"""
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(AgriColorFormatter())
        logger.addHandler(handler)

    logger.setLevel(logging.ERROR if quiet else level)
    return logger


def log_success(logger: logging.Logger, msg: str) -> None:
    """记录一条成功完成级别的日志 (绿色完成标记)"""
    logger.info(msg, extra={"is_success": True})


def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """
    针对输入配置执行严密的防御性自校验。
    返回: (is_valid, error_messages)
    """
    errors = []

    # 1. 检查必要段落
    if not isinstance(config, dict):
        return False, ["配置文件根节点必须是有效的字典对象。"]

    # 2. 空间分辨率校验
    spatial = config.get("spatial", {})
    res = spatial.get("resolution_meters", 10.0)
    if not isinstance(res, (int, float)) or res <= 0:
        errors.append(f"spatial.resolution_meters 必须为大于 0 的正数，当前为: {res}")

    # 3. 分割参数校验
    seg = config.get("segmentation", {})
    min_area = seg.get("min_parcel_area_m2", 200.0)
    max_area = seg.get("max_parcel_area_m2", 500000.0)
    if min_area >= max_area:
        errors.append(f"最小地块阈值 ({min_area}㎡) 不能大于或等于最大阈值 ({max_area}㎡)。")

    # 4. 作物图例校验
    legend = config.get("crop_legend", {})
    if not legend:
        errors.append("crop_legend 必须至少配置一个作物图例类别。")
    elif 0 not in legend and "0" not in legend:
        errors.append("crop_legend 建议必须包含类别 0 (背景/非农田)。")

    # 5. 无偏推断参数校验
    unbiased = config.get("unbiased_area_inference", {})
    n_boot = unbiased.get("n_bootstrap", 2000)
    if not isinstance(n_boot, int) or n_boot < 50:
        errors.append(f"unbiased_area_inference.n_bootstrap 抽样次数过少 ({n_boot})，建议不少于 50 次。")

    return len(errors) == 0, errors
