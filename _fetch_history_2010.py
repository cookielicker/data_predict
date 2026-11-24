#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补历史数据(断点续传版): 把个股K线补到 2010-01-01
   改进: ①跳过已有2010数据的股票 ②更频繁重登录(每50只) ③异常更稳健"""
import sys, io, time, socket, logging
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from pathlib import Path
import glob
import baostock as bs
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    handlers=[logging.FileHandler('fetch_history_2010.log', mode='a'),
                              logging.StreamHandler(sys.stdout)])
logger = logging.getLogger()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / 'stock_data'
START = '2010-01-01'
END = '2012-01-03'
socket.setdefaulttimeout(40)

def has_history(f):
    """判断该CSV是否已含2010年数据"""
    try:
        with open(f, 'r', encoding='utf-8-sig') as fh:
            first = fh.readline().strip()          # 表头
            second = fh.readline().strip()          # 第一行数据
        return second.startswith('2010-')
    except Exception:
        return False

def login():
    try:
        return bs.login().error_code == '0'
    except Exception:
        return False

def fetch_one(code):
    fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"
    rs = bs.query_history_k_data_plus(code, fields=fields, start_date=START, end_date=END,
                                      frequency="d", adjustflag="2")
    if rs is None or rs.error_code != '0':
        return None
    rows = []
    try:
        while rs.next():
            rows.append(rs.get_row_data())
    except Exception:
        pass
    if not rows:
        return None
    try:
        return pd.DataFrame(rows, columns=rs.fields)
    except Exception:
        return None

files = sorted(glob.glob(str(DATA_DIR / 'sh_*.csv')) + glob.glob(str(DATA_DIR / 'sz_*.csv')))
todo = [f for f in files if not has_history(f)]
print(f"总 {len(files)} 只, 已完成 {len(files)-len(todo)} 只, 待补 {len(todo)} 只", flush=True)

if not todo:
    print("全部完成，无需重跑")
    sys.exit(0)

if not login():
    print("登录失败")
    sys.exit(1)

ok = 0; skip = 0; fail = 0
for i, f in enumerate(todo):
    code = Path(f).stem.replace('_', '.')
    df = None
    try:
        df = fetch_one(code)
    except Exception as e:
        logger.warning(f"[{code}] 拉取异常: {e}")
    if df is None or len(df) == 0:
        skip += 1
    else:
        try:
            old = pd.read_csv(f, dtype=str)
            merged = pd.concat([df, old], ignore_index=True)
            merged = merged.drop_duplicates(subset=['date'], keep='last')
            merged = merged.sort_values('date').reset_index(drop=True)
            merged.to_csv(f, index=False)
            ok += 1
        except Exception as e:
            logger.warning(f"[{code}] 合并失败: {e}")
            fail += 1
    if (i+1) % 50 == 0:
        try: bs.logout()
        except Exception: pass
        try: login()
        except Exception: pass
        print(f"[{i+1}/{len(todo)}] ok={ok} skip={skip} fail={fail}", flush=True)
        time.sleep(0.5)

try: bs.logout()
except Exception: pass
print(f"\n补历史完成: ok={ok} skip={skip} fail={fail}")
