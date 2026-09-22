"""
🌾 联合国农业遥感统计系统自动化测试与全系统自检执行器
使用方法:
    python run_tests.py
"""

import sys
import os
import unittest
import time

# 确保主路径在 Python 模块检索列表中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows 跨平台编码防御：重置控制台标准输出编码为 utf-8 (兼容无法打印 Emoji 的 GBK 终端)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Windows 环境下自动探测并配置 PROJ 数据路径
try:
    import rasterio
    import rasterio.env
    proj_cand = os.path.join(os.path.dirname(rasterio.__file__), "proj_data")
    if os.path.exists(proj_cand):
        rasterio.env.set_proj_data_search_path(proj_cand)
        os.environ["PROJ_LIB"] = proj_cand
        os.environ["PROJ_DATA"] = proj_cand
except Exception:
    pass


def main():
    print("=" * 80)
    print("🌾 联合国农业统计遥感手册系统：自动化全量测试与健康自检")
    print("=" * 80)
    start_time = time.time()

    # 发现 tests 目录下的全部测试
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    elapsed = time.time() - start_time
    print("-" * 80)
    print(f"⏱️  测试执行耗时: {elapsed:.2f} 秒")
    print(f"📊 总运行用例数: {result.testsRun}")
    print(f"✅ 成功通过数量: {result.testsRun - len(result.failures) - len(result.errors)}")
    if result.failures:
        print(f"❌ 失败数量 (Failures): {len(result.failures)}")
    if result.errors:
        print(f"🚨 异常数量 (Errors): {len(result.errors)}")

    if result.wasSuccessful():
        print("✨ [ALL PASS] 全系统所有算法与功能模块自检全部通过！系统处于健康就绪状态。")
        print("=" * 80)
        sys.exit(0)
    else:
        print("⚠️ [WARNING] 部分测试未通过，请检查上方堆栈信息进行排查。")
        print("=" * 80)
        sys.exit(1)


if __name__ == "__main__":
    main()
