#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 2010 年前上市的在市股票, 往前补到 2000-01-01 (或IPO日, 取较晚)
   目的: 让全市场换手率覆盖到 2000 年, 500日滚动打分在 2002 年起完整
   断点续传 + 网络重试(每只最多3次)"""
import sys, io, time, socket, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
import baostock as bs
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler('fetch_pre2000.log', mode='a'),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger()

socket.setdefaulttimeout(60)
DATA_DIR = Path('stock_data')
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
PRE = '2000-01-01'          # 补到的最早起点
CUTOFF = '2010-01-01'       # 只补 2010 年前上市的在市股

def login():
    for _ in range(3):
        try:
            if bs.login().error_code == '0':
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def fetch_one(code, start, end, max_retry=3):
    for attempt in range(max_retry):
        try:
            rs = bs.query_history_k_data_plus(code, FIELDS, start_date=start,
                                              end_date=end, frequency='d', adjustflag='2')
            if rs is not None and rs.error_code == '0':
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rows:
                    return pd.DataFrame(rows, columns=rs.fields)
            time.sleep(1.5)
        except Exception:
            time.sleep(1.5)
    return None

# 1. 拿在市股票清单, 筛 2010 年前上市
if not login():
    print("登录失败"); sys.exit(1)
rs = bs.query_stock_basic()
rows = []
while rs.next():
    rows.append(rs.get_row_data())
basic = pd.DataFrame(rows, columns=rs.fields)
bs.logout()

listed = basic[(basic['type'] == '1') & (basic['status'] == '1')].copy()
listed['ipoDate'] = pd.to_datetime(listed['ipoDate'], errors='coerce')
targets = listed[listed['ipoDate'] < CUTOFF]
print(f"2010 年前上市的在市股票: {len(targets)} 只")

# 2. 找出需要补的(现有文件最早日期 > 补的目标起点)
todo = []
for _, r in targets.iterrows():
    code = r['code']
    fn = DATA_DIR / (code.replace('.', '_') + '.csv')
    if not fn.exists():
        continue
    try:
        df = pd.read_csv(fn, usecols=['date'], dtype=str)
        first = df['date'].min()  # 现有最早日期, 形如 2010-01-04
        ipo = r['ipoDate'].strftime('%Y-%m-%d')
        start = max(ipo, PRE)      # 补的起点
        if first > start:
            todo.append((code, start, first))
    except Exception:
        continue

print(f"待往前补: {len(todo)} 只")

if not todo:
    print("全部完成")
    sys.exit(0)

if not login():
    print("登录失败"); sys.exit(1)

ok = 0; fail = 0; empty = 0
for i, (code, start, first) in enumerate(todo):
    fn = DATA_DIR / (code.replace('.', '_') + '.csv')
    df_new = fetch_one(code, start, first)
    if df_new is None:
        fail += 1
        logger.warning(f"[{code}] 拉取失败")
    elif len(df_new) == 0:
        empty += 1
    else:
        try:
            olddf = pd.read_csv(fn, dtype=str)
            merged = pd.concat([df_new, olddf], ignore_index=True)
            merged = merged.drop_duplicates(subset=['date'], keep='last')
            merged = merged.sort_values('date').reset_index(drop=True)
            merged.to_csv(fn, index=False)
            ok += 1
        except Exception as e:
            fail += 1
            logger.warning(f"[{code}] 合并失败: {e}")

    if (i+1) % 20 == 0:
        try: bs.logout()
        except Exception: pass
        login()
        print(f"[{i+1}/{len(todo)}] ok={ok} fail={fail} empty={empty}", flush=True)
        time.sleep(0.5)

try: bs.logout()
except Exception: pass
print(f"\n老股票往前补到2000完成: ok={ok} fail={fail} empty={empty}")
