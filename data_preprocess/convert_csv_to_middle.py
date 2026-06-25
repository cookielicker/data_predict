#!/usr/bin/env python3
"""
将stock_data/*.csv转换为middle_data/
生成: open_pct, high_pct, low_pct, close_pct, change_rate, change_delta
每个OHLC特征 = value[t]/close[t-1] - 1 (相对前收盘的涨跌幅)

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
END_DATE = "2026-12-31"


def load_all_stocks():
    """加载所有股票CSV，按日期对齐"""
    csv_files = sorted([
        f for f in os.listdir(STOCK_DATA_DIR)
        if f.endswith('.csv') and f != 'all_a_stocks_clean.csv'
    ])

    print(f"找到 {len(csv_files)} 个股票文件")

    all_dates = set()
    stock_data = {}

    bad_files = []
    for csv_file in tqdm(csv_files, desc="读取股票"):
        code = csv_file.replace('.csv', '').replace('_', '.')
        try:
            df = pd.read_csv(STOCK_DATA_DIR / csv_file)
        except Exception as e:
            print(f"\n  跳过损坏文件: {csv_file} ({e})")
            continue

        # 检查脏行: 日期格式不是 YYYY-MM-DD
        date_str = df['date'].astype(str)
        import re
        date_pat = re.compile(r'^\d{4}-\d{2}-\d{2}$')
        bad_mask = ~date_str.str.match(date_pat)
        if bad_mask.any():
            bad_files.append(csv_file)
            bad_rows = df[bad_mask]
            print(f"\n  脏文件: {csv_file} ({bad_mask.sum()} 行)")
            for _, row in bad_rows.head(3).iterrows():
                vals = [str(v)[:30] for v in row.values]
                print(f"    脏行: {vals[:5]}...")
            # 删掉脏行
            df = df[~bad_mask]

        df['date'] = pd.to_datetime(df['date'])
        df = df[(df['date'] >= START_DATE) & (df['date'] <= END_DATE)]
        if len(df) == 0:
            continue

        # 过滤停牌日
        df = df[df['tradestatus'].astype(str) != '0']
        if len(df) == 0:
            continue

        # 数值化
        for col in ['open', 'high', 'low', 'close', 'turn']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # 均价 = (H+L+C)/3, 前一日均价
        df['avg_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['prev_avg'] = df['avg_price'].shift(1)

        with np.errstate(divide='ignore', invalid='ignore'):
            # 均价涨跌: 替代 mean_adj
            df['price_pct'] = np.where(df['prev_avg'] > 0,
                                       df['avg_price'] / df['prev_avg'] - 1, 0)
            # 日内振幅: 替代 high_low_ratio
            df['amplitude'] = np.where(df['avg_price'] > 0,
                                       (df['high'] - df['low']) / df['avg_price'], 0)
            # 日内方向: 替代 OHLC 四个独立特征
            df['day_chg'] = np.where(df['avg_price'] > 0,
                                     (df['close'] - df['open']) / df['avg_price'], 0)

        # RSI (14日)
        close = df['close'].values.astype(np.float64)
        diff = np.diff(close, prepend=close[0])
        gain = np.where(diff > 0, diff, 0)
        loss = np.where(diff < 0, -diff, 0)
        avg_gain = pd.Series(gain).rolling(14, min_periods=14).mean().values
        avg_loss = pd.Series(loss).rolling(14, min_periods=14).mean().values
        with np.errstate(divide='ignore', invalid='ignore'):
            df['rsi'] = np.where(avg_loss > 0, 1 - 1/(1 + avg_gain/avg_loss), 0.5)

        # 布林带 (N=20, k=2)
        roll_mean = df['close'].rolling(20, min_periods=20).mean()
        roll_std  = df['close'].rolling(20, min_periods=20).std()
        upper = roll_mean + 2 * roll_std
        lower = roll_mean - 2 * roll_std
        with np.errstate(divide='ignore', invalid='ignore'):
            df['bb_position'] = np.where(upper > lower, (df['close'] - lower) / (upper - lower), 0.5)
            df['bb_width'] = np.where(roll_mean > 0, (upper - lower) / roll_mean, 0)
            df['bb_position'] = np.where(np.isfinite(df['bb_position']), df['bb_position'], 0.5)
            df['bb_width'] = np.where(np.isfinite(df['bb_width']), df['bb_width'], 0)

        # 换手率: 百分比转小数
        df['change_rate'] = df['turn'].fillna(0) / 100.0

        # 换手率变化量
        change_vals = df['change_rate'].values
        delta = np.zeros_like(change_vals)
        if len(change_vals) > 1:
            with np.errstate(divide='ignore', invalid='ignore'):
                delta[1:] = np.where(change_vals[:-1] != 0,
                                     (change_vals[1:] - change_vals[:-1]) / change_vals[:-1], 0)
        df['change_delta'] = delta

        stock_data[code] = df.set_index('date')[
            ['price_pct', 'amplitude', 'day_chg', 'change_rate', 'change_delta',
             'bb_position', 'bb_width', 'rsi']
        ]
        all_dates.update(df['date'].tolist())

    all_dates = sorted(all_dates)
    print(f"日期范围: {all_dates[0]} ~ {all_dates[-1]}, 共 {len(all_dates)} 天")
    return stock_data, all_dates


def build_matrices(stock_data, all_dates):
    """构建 (days, stocks) 矩阵"""
    codes = list(stock_data.keys())
    days = len(all_dates)
    stocks = len(codes)

    feature_names = ['price_pct', 'amplitude', 'day_chg', 'change_rate', 'change_delta',
                     'bb_position', 'bb_width', 'rsi']
    matrices = {name: np.full((days, stocks), np.nan, dtype=np.float32) for name in feature_names}
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for j, code in enumerate(tqdm(codes, desc="构建矩阵")):
        df = stock_data[code]
        for _, row in df.iterrows():
            date = row.name
            if date in date_to_idx:
                i = date_to_idx[date]
                for name in feature_names:
                    matrices[name][i, j] = row[name]

    result = [matrices[name] for name in feature_names]
    return (*result, codes)


def main():
    print("=" * 60)
    print("转换 stock_data/*.csv -> middle_data/*.npy (8特征)")
    print("=" * 60)

    MIDDLE_DATA_DIR.mkdir(exist_ok=True)

    stock_data, all_dates = load_all_stocks()
    price_pct, amplitude, day_chg, change_rate, change_delta, bb_position, bb_width, rsi, codes = \
        build_matrices(stock_data, all_dates)

    feature_names = ['price_pct', 'amplitude', 'day_chg', 'change_rate', 'change_delta',
                     'bb_position', 'bb_width', 'rsi']
    matrices = [price_pct, amplitude, day_chg, change_rate, change_delta, bb_position, bb_width, rsi]

    print(f"\n矩阵形状: {price_pct.shape}")
    for name, mat in zip(feature_names, matrices):
        print(f"  {name}: 非NaN={np.sum(~np.isnan(mat))/mat.size*100:.1f}%")

    for name, mat in zip(feature_names, matrices):
        np.save(MIDDLE_DATA_DIR / f"{name}.npy", mat)

    dates_df = pd.DataFrame({'date': all_dates})
    dates_df.to_csv(MIDDLE_DATA_DIR / "dates.csv", index=False)

    codes_df = pd.DataFrame({'code': codes})
    codes_df.to_csv(MIDDLE_DATA_DIR / "codes.csv", index=False)

    print(f"\n已保存到 {MIDDLE_DATA_DIR}/")
    for name in feature_names:
        print(f"  - {name}.npy")
    print(f"  - dates.csv: {len(all_dates)} 天")
    print(f"  - codes.csv: {len(codes)} 只股票")


if __name__ == "__main__":
    main()
