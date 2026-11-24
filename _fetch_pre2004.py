#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 2004 年前上市的在市老股票, 从现有起点(2010/2012)往前补到 2004-01-01
   覆盖 04-07 大牛市的换手率计算
   断点续传: 跳过已有 2004 数据的股票"""
import sys, io, time, socket, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
import baostock as bs
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler('fetch_pre2004.log', mode='a'),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger()

socket.setdefaulttimeout(40)
DATA_DIR = Path('stock_data')
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
PRE = '2004-01-01'

def login():
    try:
        return bs.login().error_code == '0'
    except Exception:
        return False

# 1. 拿在市股票清单, 过滤 2004 年前上市
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
old = listed[listed['ipoDate'] < PRE]
print(f"2004 年前上市的在市股票: {len(old)} 只")

# 2. 找出需要补的(现有文件最早日期 > 2004-01-01)
todo = []
for _, r in old.iterrows():
    code = r['code']
    fn = DATA_DIR / (code.replace('.', '_') + '.csv')
    if not fn.exists():
        continue  # 不应该发生, 在市股票应该有文件
    try:
        df = pd.read_csv(fn, usecols=['date'], dtype=str)
        first = df['date'].min()
        if first > PRE:
            todo.append((code, first))
    except Exception:
        continue

print(f"待往前补: {len(todo)} 只 (其余已有 2004 数据)")

if not todo:
    print("全部完成")
    sys.exit(0)

if not login():
    print("登录失败"); sys.exit(1)

ok = 0; fail = 0; empty = 0
for i, (code, first) in enumerate(todo):
    fn = DATA_DIR / (code.replace('.', '_') + '.csv')
    try:
        rs = bs.query_history_k_data_plus(code, FIELDS, start_date=PRE,
                                          end_date=first[:10],
                                          frequency='d', adjustflag='2')
        if rs.error_code != '0':
            fail += 1
            logger.warning(f"[{code}] error: {rs.error_msg[:50]}")
        else:
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                empty += 1
            else:
                newdf = pd.DataFrame(rows, columns=rs.fields)
                olddf = pd.read_csv(fn, dtype=str)
                merged = pd.concat([newdf, olddf], ignore_index=True)
                merged = merged.drop_duplicates(subset=['date'], keep='last')
                merged = merged.sort_values('date').reset_index(drop=True)
                merged.to_csv(fn, index=False)
                ok += 1
    except Exception as e:
        fail += 1
        logger.warning(f"[{code}] 异常: {e}")

    if (i+1) % 30 == 0:
        try: bs.logout()
        except Exception: pass
        login()
        print(f"[{i+1}/{len(todo)}] ok={ok} fail={fail} empty={empty}", flush=True)
        time.sleep(0.5)

try: bs.logout()
except Exception: pass
print(f"\n老股票往前补完成: ok={ok} fail={fail} empty={empty}")
