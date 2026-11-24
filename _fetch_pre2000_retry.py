#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重试补2000年数据失败的36只股票"""
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
PRE = '2000-01-01'
CUTOFF = '2010-01-01'

FAILED = ['sh.600776','sh.600790','sz.000691','sz.000692','sz.000695','sz.000697',
'sz.000698','sz.000700','sz.000701','sz.000702','sz.000703','sz.000705',
'sz.000707','sz.000708','sz.000709','sz.000710','sz.000711','sz.000712',
'sz.000713','sz.000715','sz.000716','sz.000717','sz.000718','sz.000719',
'sz.000720','sz.000721','sz.000722','sz.000723','sz.000725','sz.000726',
'sz.000727','sz.000735','sz.000883','sz.000952','sz.000953','sz.000988']

def login():
    for _ in range(3):
        try:
            if bs.login().error_code == '0':
                return True
        except Exception:
            pass
        time.sleep(2)
    return False

def fetch_one(code, start, end, max_retry=5):
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
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return None

# 拿 IPO 日期, 确定每只的补的起点
if not login():
    print("登录失败"); sys.exit(1)
rs = bs.query_stock_basic()
rows = []
while rs.next():
    rows.append(rs.get_row_data())
basic = pd.DataFrame(rows, columns=rs.fields)
bs.logout()
ipo_map = dict(zip(basic['code'], basic['ipoDate']))

if not login():
    print("登录失败"); sys.exit(1)

ok = 0; fail = 0
for i, code in enumerate(FAILED):
    fn = DATA_DIR / (code.replace('.', '_') + '.csv')
    ipo = ipo_map.get(code, PRE)
    start = max(ipo, PRE)
    # 现有文件最早日期
    try:
        existing = pd.read_csv(fn, usecols=['date'], dtype=str)
        first = existing['date'].min()
    except Exception:
        first = CUTOFF
    df_new = fetch_one(code, start, first)
    if df_new is None:
        fail += 1
        print(f"[{i+1}/{len(FAILED)}] {code} 仍失败", flush=True)
    else:
        try:
            olddf = pd.read_csv(fn, dtype=str)
            merged = pd.concat([df_new, olddf], ignore_index=True)
            merged = merged.drop_duplicates(subset=['date'], keep='last')
            merged = merged.sort_values('date').reset_index(drop=True)
            merged.to_csv(fn, index=False)
            ok += 1
            print(f"[{i+1}/{len(FAILED)}] {code} ok ({len(df_new)}行补充)", flush=True)
        except Exception as e:
            fail += 1
            print(f"[{i+1}/{len(FAILED)}] {code} 合并失败: {e}", flush=True)
    if (i+1) % 5 == 0:
        try: bs.logout()
        except Exception: pass
        login()
        time.sleep(0.5)

try: bs.logout()
except Exception: pass
print(f"\n重试完成: ok={ok} fail={fail}")
