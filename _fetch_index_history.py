#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 BaoStock 拉指数日线到 1990/1991 起点, 覆盖/新建本地CSV
   目标: 上证指数 sh.000001(1990起), 深证成指 sz.399001(1991起)
   字段统一: date,open,high,low,close,volume,amount"""
import sys, io, time, socket
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import baostock as bs
import pandas as pd

socket.setdefaulttimeout(60)

TARGETS = [
    ('sh.000001', 'index_sh000001.csv', '上证指数'),
    ('sz.399001', 'index_399001.csv', '深证成指'),
]

bs.login()

for code, fn, name in TARGETS:
    fields = "date,open,high,low,close,volume,amount"
    rs = bs.query_history_k_data_plus(code, fields, start_date='1990-01-01',
                                      end_date='2026-12-31', frequency='d')
    if rs.error_code != '0':
        print(f"{name} ({code}) 拉取失败: {rs.error_msg}")
        continue
    rows = []
    while rs.next():
        rows.append(rs.get_row_data())
    df = pd.DataFrame(rows, columns=rs.fields)
    df = df.rename(columns={})
    # 转数值
    for c in ['open','high','low','close','volume','amount']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] >= '1990-01-01'].sort_values('date').reset_index(drop=True)
    df = df.dropna(subset=['close'])
    df.to_csv(fn, index=False)
    print(f"{name} ({code}): {len(df)} 天, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}, "
          f"收盘范围 {df['close'].min():.2f} ~ {df['close'].max():.2f}")
    time.sleep(1)

bs.logout()
print("done")
