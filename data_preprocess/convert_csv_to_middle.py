#!/usr/bin/env python3
"""
将stock_data/*.csv转换为middle_data/
生成: mean_adj.npy, change_rate.npy

用法:
  python data_preprocess/convert_csv_to_middle.py
"""
import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
STOCK_DATA_DIR = PROJECT_ROOT / "stock_data"
MIDDLE_DATA_DIR = PROJECT_ROOT / "middle_data_baostock"

# 配置
START_DATE = "2016-01-01"
END_DATE = "2026-05-17"

def load_all_stocks():
    """加载所有股票CSV，按日期对齐"""
    csv_files = sorted([
        f for f in os.listdir(STOCK_DATA_DIR)
        if f.endswith('.csv') and f != 'all_a_stocks_clean.csv'
    ])

    print(f"找到 {len(csv_files)} 个股票文件")

    # 收集所有日期
    all_dates = set()
    stock_data = {}

    for csv_file in tqdm(csv_files, desc="读取股票"):
        code = csv_file.replace('.csv', '').replace('_', '.')
        df = pd.read_csv(STOCK_DATA_DIR / csv_file)

        # 转换日期
        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)]

        if len(df) == 0:
            continue

        # 前复权价 = (开+高+低+收) / 4
        # df['mean_adj'] = df['amount'] / df['volume']
        # 注意: 不能用 amount/volume 算均价, 因为 baostock 的 amount 和 volume 不复权,
        # 算出来的是真实历史成交价, 与 adjustflag="2" 的前复权 OHLC 不一致, 会产生除权除息断层
        df['mean_adj'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
        # 换手率: 百分比转小数
        df['change_rate'] = df['turn'] / 100.0
        # 日内波动率: (最高-最低) / 均价, 天然归一化
        df['high_low_ratio'] = (df['high'] - df['low']) / df['mean_adj']
        # 换手率变化量: (今日-昨日)/昨日, 反映换手率加速/减速, 第0日为0
        change_vals = df['change_rate'].values
        delta = np.zeros_like(change_vals)
        if len(change_vals) > 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                delta[1:] = (change_vals[1:] - change_vals[:-1]) / change_vals[:-1]
                delta = np.where(np.isfinite(delta), delta, 0)
        df['change_delta'] = delta

        stock_data[code] = df.set_index('date')[['mean_adj', 'change_rate', 'high_low_ratio', 'change_delta']]
        all_dates.update(df['date'].tolist())

    # 按日期排序
    all_dates = sorted(all_dates)
    print(f"日期范围: {all_dates[0]} ~ {all_dates[-1]}, 共 {len(all_dates)} 天")

    return stock_data, all_dates

def build_matrices(stock_data, all_dates):
    """构建 (days, stocks) 矩阵"""
    codes = list(stock_data.keys())
    days = len(all_dates)
    stocks = len(codes)

    mean_adj = np.full((days, stocks), np.nan, dtype=np.float64)
    change_rate = np.full((days, stocks), np.nan, dtype=np.float64)
    high_low_ratio = np.full((days, stocks), np.nan, dtype=np.float64)
    change_delta = np.full((days, stocks), np.nan, dtype=np.float64)

    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for j, code in enumerate(tqdm(codes, desc="构建矩阵")):
        df = stock_data[code]
        for _, row in df.iterrows():
            date = row.name
            if date in date_to_idx:
                i = date_to_idx[date]
                mean_adj[i, j] = row['mean_adj']
                change_rate[i, j] = row['change_rate']
                high_low_ratio[i, j] = row['high_low_ratio']
                change_delta[i, j] = row['change_delta']

    return mean_adj, change_rate, high_low_ratio, change_delta, codes

def main():
    print("=" * 60)
    print("转换 stock_data/*.csv -> middle_data/*.npy")
    print("=" * 60)

    # 创建输出目录
    MIDDLE_DATA_DIR.mkdir(exist_ok=True)

    # 加载数据
    stock_data, all_dates = load_all_stocks()

    # 构建矩阵
    mean_adj, change_rate, high_low_ratio, change_delta, codes = build_matrices(stock_data, all_dates)

    print(f"\n矩阵形状: {mean_adj.shape}")
    print(f"mean_adj 非NaN比例: {np.sum(~np.isnan(mean_adj)) / mean_adj.size * 100:.2f}%")
    print(f"change_rate 非NaN比例: {np.sum(~np.isnan(change_rate)) / change_rate.size * 100:.2f}%")
    print(f"high_low_ratio 非NaN比例: {np.sum(~np.isnan(high_low_ratio)) / high_low_ratio.size * 100:.2f}%")
    print(f"change_delta 非NaN比例: {np.sum(~np.isnan(change_delta)) / change_delta.size * 100:.2f}%")

    # 保存
    np.save(MIDDLE_DATA_DIR / "mean_adj.npy", mean_adj)
    np.save(MIDDLE_DATA_DIR / "change_rate.npy", change_rate)
    np.save(MIDDLE_DATA_DIR / "high_low_ratio.npy", high_low_ratio)
    np.save(MIDDLE_DATA_DIR / "change_delta.npy", change_delta)

    # 保存日期和股票代码列表
    dates_df = pd.DataFrame({'date': all_dates})
    dates_df.to_csv(MIDDLE_DATA_DIR / "dates.csv", index=False)

    codes_df = pd.DataFrame({'code': codes})
    codes_df.to_csv(MIDDLE_DATA_DIR / "codes.csv", index=False)

    print(f"\n已保存到 {MIDDLE_DATA_DIR}/")
    print(f"  - mean_adj.npy: {mean_adj.shape}")
    print(f"  - change_rate.npy: {change_rate.shape}")
    print(f"  - high_low_ratio.npy: {high_low_ratio.shape}")
    print(f"  - change_delta.npy: {change_delta.shape}")
    print(f"  - dates.csv: {len(all_dates)} 天")
    print(f"  - codes.csv: {len(codes)} 只股票")

if __name__ == "__main__":
    main()
