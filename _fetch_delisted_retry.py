#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重试拉取失败的退市股(33只), 加大超时与重试"""
import sys, io, time, socket, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
import baostock as bs
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler('fetch_delisted.log', mode='a'),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger()

socket.setdefaulttimeout(60)
DATA_DIR = Path('stock_data')
FIELDS = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"

FAILED = ["sh.600253","sh.600260","sh.600290","sh.600291","sh.600296","sh.600297",
"sh.600306","sh.600311","sh.600317","sh.600321","sh.600355","sh.600357",
"sh.600385","sh.600387","sh.600393","sh.600401","sh.600421","sh.600432",
"sh.600466","sh.600472","sh.600485","sh.600532","sh.600553","sh.600766",
"sh.600767","sz.000569","sz.000578","sz.000621","sz.000622","sz.000627",
"sz.000638","sz.000653","sz.000658"]

def login():
    try:
        return bs.login().error_code == '0'
    except Exception:
        return False

def fetch_with_retry(code, max_retry=3):
    for attempt in range(max_retry):
        try:
            rs = bs.query_history_k_data_plus(code, FIELDS, start_date='1990-01-01',
                                              end_date='2026-12-31', frequency='d', adjustflag='2')
            if rs.error_code == '0':
                rows = []
                while rs.next():
                    rows.append(rs.get_row_data())
                if rows:
                    return pd.DataFrame(rows, columns=rs.fields)
            time.sleep(2)
        except Exception as e:
            logger.warning(f"[{code}] 第{attempt+1}次异常: {e}")
            time.sleep(2)
    return None

if not login():
    print("登录失败"); sys.exit(1)

ok = 0; fail = 0
for i, code in enumerate(FAILED):
    fn = DATA_DIR / (code.replace('.', '_') + '.csv')
    df = fetch_with_retry(code)
    if df is not None:
        df.to_csv(fn, index=False)
        ok += 1
        print(f"[{i+1}/{len(FAILED)}] {code} ok ({len(df)}行)", flush=True)
    else:
        fail += 1
        print(f"[{i+1}/{len(FAILED)}] {code} 仍失败", flush=True)
    if (i+1) % 5 == 0:
        try: bs.logout()
        except Exception: pass
        login()
        time.sleep(0.5)

try: bs.logout()
except Exception: pass
print(f"\n重试完成: ok={ok} fail={fail}")
