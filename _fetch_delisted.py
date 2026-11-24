#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""拉取退市股票历史日线(前复权), 补进 stock_data/
   退市股票此前完全没在换手率计算里, 补入后历史换手率更准确
   断点续传: 跳过已有文件的退市股"""
import sys, io, time, socket, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
import baostock as bs
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler('fetch_delisted.log', mode='a'),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger()

socket.setdefaulttimeout(40)
DATA_DIR = Path('stock_data')
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"

def login():
    try:
        return bs.login().error_code == '0'
    except Exception:
        return False

# 1. 拿退市股清单
if not login():
    print("登录失败"); sys.exit(1)
rs = bs.query_stock_basic()
rows = []
while rs.next():
    rows.append(rs.get_row_data())
basic = pd.DataFrame(rows, columns=rs.fields)
bs.logout()

delisted = basic[(basic['type'] == '1') & (basic['status'] == '0')]
print(f"退市股票共 {len(delisted)} 只")

# 2. 过滤已存在的
todo = []
for _, r in delisted.iterrows():
    code = r['code']  # sh.600625
    fn = DATA_DIR / (code.replace('.', '_') + '.csv')
    if fn.exists():
        continue
    todo.append(r)

print(f"待拉取: {len(todo)} 只 (已存在 {len(delisted)-len(todo)} 只跳过)")

if not todo:
    print("全部完成")
    sys.exit(0)

if not login():
    print("登录失败"); sys.exit(1)

ok = 0; fail = 0; empty = 0
for i, r in enumerate(todo):
    code = r['code']
    fn = DATA_DIR / (code.replace('.', '_') + '.csv')
    try:
        rs = bs.query_history_k_data_plus(code, FIELDS, start_date='1990-01-01',
                                          end_date='2026-12-31', frequency='d', adjustflag='2')
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
                df = pd.DataFrame(rows, columns=rs.fields)
                df.to_csv(fn, index=False)
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
print(f"\n退市股拉取完成: ok={ok} fail={fail} empty={empty}")
