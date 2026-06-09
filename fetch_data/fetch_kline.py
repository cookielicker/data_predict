#!/usr/bin/env python3
"""
获取股票历史K线数据
使用 baostock 包
"""
import os
import sys
from pathlib import Path

import baostock as bs
import pandas as pd

# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "stock_data"
DATA_DIR.mkdir(exist_ok=True)


def login():
    """登录BaoStock系统"""
    lg = bs.login()
    print('登录返回代码：', lg.error_code)
    print('登录返回信息：', lg.error_msg)
    return lg


def logout():
    """登出BaoStock系统"""
    bs.logout()


def fetch_kline_data(code, start_date, end_date, save_dir=None):
    """
    获取历史K线数据

    Args:
        code: 股票代码，如 'sh.600000'
        start_date: 开始日期，如 '2023-01-01'
        end_date: 结束日期，如 '2023-12-31'
        save_dir: 保存目录，不传则不保存

    Returns:
        DataFrame: K线数据
    """
    rs = bs.query_history_k_data_plus(
        code,
        fields="date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"  # 不复权
    )

    print(f'K线数据查询返回代码：{rs.error_code}, 股票：{code}')

    if rs.error_code != '0':
        print(f'查询失败：{rs.error_msg}')
        return None

    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())

    df = pd.DataFrame(data_list, columns=rs.fields)
    print(f'获取数据 {len(df)} 条')

    if save_dir:
        stock_code = code.replace('.', '_')
        filename = f'{stock_code}_{start_date}_{end_date}.csv'
        filepath = save_dir / filename
        df.to_csv(filepath, index=False)
        print(f'已保存至：{filepath}')

    return df


def get_all_stocks():
    """获取所有股票列表"""
    rs = bs.query_all_stock(day='2024-01-01')
    stocks = []
    while rs.next():
        stocks.append(rs.get_row_data())
    return pd.DataFrame(stocks, columns=rs.fields)


if __name__ == "__main__":
    print("=" * 60)
    print("BaoStock 数据获取测试")
    print("=" * 60)

    # 1. 登录
    login()

    # 2. 测试获取单只股票数据
    print("\n--- 测试：获取浦发银行2023年数据 ---")
    df = fetch_kline_data(
        code='sh.600000',
        start_date='2023-01-01',
        end_date='2024-12-31',
        save_dir=DATA_DIR
    )
    if df is not None:
        print("\n数据预览：")
        print(df.head())

    # 3. 登出
    logout()
    print("\n完成！")