"""
统一的农情遥感与统计系统日志管理与配置自校验工具。
提供：
1. 统一格式的彩色/符号终端日志输出（INFO, SUCCESS, WARN, ERROR, DEBUG）；
2. 模块名标签自动化（如 [地块分割]、[矢量导出]、[作物分类]）；
3. 静默模式（Silent Mode）支持（便于作为无干扰 SDK 库导入）；
4. 系统配置防御性自校验（Validate Config）。
"""

import os
import sys
import logging
from typing import Dict, Any, Tuple, List

# Windows 控制台尝试激活 VT100 ANSI 颜色模式
if sys.platform == "win32":
    try:
        os.system('')
    except Exception:
        pass


class AgriColorFormatter(logging.Formatter):
    """带农业遥感业务符号的统一彩色日志格式化器"""

    # ANSI 颜色码
    RESET = "\033[0m"
    BOLD = "\033[1m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    RED = "\033[31m"
    CYAN = "\033[36m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
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

        # 模块标签构建：若 logger 名称不是 AgriPipeline 且信息开头无中括号，则追加 [模块名]
        tag = ""
        if record.name and record.name not in ("AgriPipeline", "root"):
            if not msg.startswith("[") and not msg.startswith("  ->"):
                tag = f"[{record.name}] "

        # 非交互终端 (如重定向至文件) 则回退无颜色纯文本
        if not sys.stdout.isatty():
            return f"{prefix} {tag}{msg}"

        return f"{color}{prefix} {tag}{msg}{self.RESET}" if color else f"{prefix} {tag}{msg}"


_SHARED_HANDLER = None

def get_shared_handler() -> logging.Handler:
    """获取单例格式化输出 Handler，防止多次添加产生重复打印"""
    global _SHARED_HANDLER
    if _SHARED_HANDLER is None:
        _SHARED_HANDLER = logging.StreamHandler(sys.stdout)
        _SHARED_HANDLER.setFormatter(AgriColorFormatter())
    return _SHARED_HANDLER


def get_logger(name: str = "AgriPipeline", level: int = logging.INFO, quiet: bool = False) -> logging.Logger:
    """
    获取或初始化统一格式的农情系统日志器。
    
    参数:
        name: 模块名称 (如 '地块分割', '矢量导出', '作物分类')
        level: 最低打印级别 (默认 logging.INFO)
        quiet: 是否开启静默模式 (仅严重错误才输出)
    """
    logger = logging.getLogger(name)
    logger.propagate = False

    handler = get_shared_handler()
    if handler not in logger.handlers:
        logger.handlers.clear()
        logger.addHandler(handler)

    logger.setLevel(logging.ERROR if quiet else level)
    return logger


def log_success(logger: logging.Logger, msg: str) -> None:
    """记录一条成功完成级别的日志 (绿色 ✨ [DONE] 标记)"""
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

    # 6. 性能配置校验
    perf = config.get("performance", {})
    if perf:
        blk_sz = perf.get("streaming_block_size", 1024)
        if not isinstance(blk_sz, int) or blk_sz < 16:
            errors.append(f"performance.streaming_block_size 分块尺寸 ({blk_sz}) 必须为大于等于 16 的整数。")

    return len(errors) == 0, errors
