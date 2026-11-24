#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""宏观/大盘数据 一键更新编排脚本 (fetch_macro.py)
   不复制任何拉取代码, 直接复用现有脚本:
     [1] 个股数据增量刷新 -> import fetch_data/fetch_baostock.py, 复用其 main() (--refresh)
     [2] 重算全市场换手率 -> subprocess 调 _calc_market_turn.py
     [3] 指数点位补全     -> subprocess 调 _fetch_index_sohu.py
   用法(Windows python):
     powershell -Command "cd 'C:\Users\Vision Works\Documents\GitHub\data_predict'; C:\Python311\python.exe fetch_data\fetch_macro.py"
"""
import sys, io, subprocess, time
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent   # 项目根目录
PY = sys.executable                              # 当前解释器(Windows python)


def banner(t):
    print("\n" + "=" * 62)
    print(f"  {t}")
    print("=" * 62)


def step1_refresh_stocks():
    """[1/3] 个股增量刷新: import 复用 fetch_baostock.main()"""
    banner("[1/3] 个股数据增量刷新 (复用 fetch_baostock.main, --refresh)")
    sys.path.insert(0, str(ROOT / 'fetch_data'))
    import fetch_baostock as fb
    sys.argv = ['fetch_baostock', '--refresh']
    fb.main()


def step2_market_turn():
    """[2/3] 重算全市场换手率 (_calc_market_turn.py)"""
    banner("[2/3] 重算全市场换手率 (_calc_market_turn.py)")
    r = subprocess.run([PY, str(ROOT / '_calc_market_turn.py')], cwd=str(ROOT))
    print(f"  exit={r.returncode}")


def step3_index_points():
    """[3/3] 指数点位补全 (_fetch_index_sohu.py)"""
    banner("[3/3] 指数点位补全 (_fetch_index_sohu.py)")
    r = subprocess.run([PY, str(ROOT / '_fetch_index_sohu.py')], cwd=str(ROOT))
    print(f"  exit={r.returncode}")


def main():
    t0 = time.time()
    print(f"fetch_macro 开始, 解释器: {PY}")
    try:
        step1_refresh_stocks()
    except Exception as e:
        print(f"[1/3] 异常: {e}")
    step2_market_turn()
    step3_index_points()
    print(f"\n=== fetch_macro 完成, 总耗时 {time.time() - t0:.0f}s ===")


if __name__ == "__main__":
    main()
