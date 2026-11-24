#!/usr/bin/env python3
"""增量补历史: 只给 2016-06 前已上市的 2407 只股票补 2012-01-01 ~ 2016-05-31 的前复权历史
   关键处理:
   1. 前复权基准: 用 adjustflag=2 (与现有CSV一致, 已验证价格能对齐)
   2. 交界日期重复: 拉到 2016-05-31 为止, 现有CSV从2016-06-01起, 无重叠
   3. 合并: 新历史段 + 现有CSV, 按 date 去重(keep=last), 排序
   只处理 _need_history.txt 里列出且首行==2016-06-01 的股票"""
import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import pandas as pd
import baostock as bs
from pathlib import Path

DATA_DIR = Path("stock_data")
HIST_START = "2012-01-01"
HIST_END = "2016-05-31"  # 拉到5月底, 避免和现有6-01重叠

def login():
    for _ in range(5):
        lg = bs.login()
        if lg.error_code == '0':
            return True
        time.sleep(2)
    return False

def main():
    # 读需补清单 (只补首行==2016-06-01的, 即"2016-06前已上市且未覆盖")
    targets = []
    with open('_need_history.txt') as fh:
        for line in fh:
            code_file = line.strip()
            if not code_file:
                continue
            csv_path = DATA_DIR / code_file
            if not csv_path.exists():
                continue
            # 确认首行日期是 2016-06-01 (未覆盖)
            try:
                lines = open(csv_path, 'r', encoding='utf-8-sig', errors='replace').readlines()
                if len(lines) < 2:
                    continue
                first = lines[1].split(',')[0].strip()
                if first != '2016-06-01':
                    continue  # 已覆盖 或 后上市, 跳过
            except Exception:
                continue
            code = code_file.replace('.csv', '').replace('_', '.')
            targets.append((code, csv_path))

    print(f"需补历史股票: {len(targets)} 只")

    if not login():
        print("登录失败")
        return

    success = 0; fail = 0; failed = []
    for i, (code, csv_path) in enumerate(targets):
        # 拉 2012-01-01 ~ 2016-05-31 前复权
        for attempt in range(3):
            rs = bs.query_history_k_data_plus(
                code,
                fields="date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
                start_date=HIST_START, end_date=HIST_END,
                frequency="d", adjustflag="2")
            if rs.error_code == '0':
                break
            time.sleep(2)
        hist = rs.get_data()
        if hist is None or len(hist) == 0:
            fail += 1; failed.append(code)
            if (i+1) % 50 == 0:
                print(f"  [{i+1}/{len(targets)}] fail={fail}")
            continue

        # 读现有 CSV (2016-06-01 起)
        try:
            existing = pd.read_csv(csv_path)
        except Exception:
            fail += 1; failed.append(code); continue

        # 合并: 历史 + 现有, 按 date 去重(keep=last 保留现有的), 排序
        combined = pd.concat([hist, existing], ignore_index=True)
        combined = combined.drop_duplicates(subset=['date'], keep='last')
        combined = combined.sort_values('date').reset_index(drop=True)
        combined.to_csv(csv_path, index=False)
        success += 1

        if (i+1) % 50 == 0:
            print(f"  [{i+1}/{len(targets)}] success={success}, fail={fail}")
            try: bs.logout()
            except: pass
            if not login():
                print("  重登录失败, 继续...")

    print(f"\n完成! success={success}, fail={fail}")
    if failed:
        with open('_need_history_failed.txt', 'w') as fh:
            fh.write('\n'.join(failed))
        print(f"失败列表: _need_history_failed.txt ({len(failed)} 只)")

    try: bs.logout()
    except: pass

if __name__ == '__main__':
    main()
