#!/usr/bin/env python3
"""
批量获取大A股票近10年K线数据 (前复权)
保存为CSV格式到 stock_data/ 目录

用法:
  python fetch_data/fetch_baostock.py
  python fetch_data/fetch_baostock.py --start 2020-01-01 --end 2026-06-01
"""
import os
import sys
import time
import socket
import logging
import threading
import argparse
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('fetch_baostock.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).parent.parent))
import baostock as bs
import pandas as pd

# ============ 配置参数 ============
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "stock_data"
STOCK_LIST_FILE = DATA_DIR / "all_a_stocks_clean.csv"
SOCKET_TIMEOUT = 60       # 单次网络请求超时(秒)
PER_STOCK_TIMEOUT = 180   # 单只股票总超时(秒)
# ==================================


def login():
    lg = bs.login()
    return lg.error_code == '0'


def logout():
    try:
        bs.logout()
    except Exception:
        pass


def _fetch_one_query(code, start_date, end_date):
    """执行一次查询 (非sz股票: 单次; sz股票: 分段)"""
    fields = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST"

    if not code.startswith('sz.'):
        rs = bs.query_history_k_data_plus(
            code, fields=fields,
            start_date=start_date, end_date=end_date,
            frequency="d", adjustflag="2"
        )
        if rs.error_code != '0':
            raise ConnectionError(f"query failed: {rs.error_msg}")

        rows = []
        while rs.next():
            rows.append(rs.get_row_data())
        if not rows:
            return None
        return pd.DataFrame(rows, columns=rs.fields)

    # sz 股票分段查询
    chunk_days = 365
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')

    all_rows = []
    chunk_start = start_dt
    while chunk_start < end_dt:
        chunk_end = min(chunk_start + timedelta(days=chunk_days), end_dt)

        rs = bs.query_history_k_data_plus(
            code, fields=fields,
            start_date=chunk_start.strftime('%Y-%m-%d'),
            end_date=chunk_end.strftime('%Y-%m-%d'),
            frequency="d", adjustflag="2"
        )
        if rs.error_code != '0':
            raise ConnectionError(f"sz chunk query failed: {rs.error_msg}")

        while rs.next():
            all_rows.append(rs.get_row_data())

        chunk_start = chunk_end
        time.sleep(0.3)

    if not all_rows:
        return None
    return pd.DataFrame(all_rows, columns=rs.fields)


def fetch_kline_data(code, start_date, end_date):
    """获取单只股票K线数据，带重试"""
    max_retries = 3
    last_error = None

    for attempt in range(max_retries):
        try:
            return _fetch_one_query(code, start_date, end_date)
        except (socket.timeout, TimeoutError, ConnectionError, OSError) as e:
            last_error = e
            logger.warning(f"[{code}] 超时/连接错误 (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(3)
            logout()
            login()
        except Exception as e:
            last_error = e
            logger.warning(f"[{code}] 异常 (attempt {attempt+1}/{max_retries}): {e}")
            time.sleep(2)

    logger.error(f"[{code}] 重试{max_retries}次后仍失败: {last_error}")
    return None


def _fetch_with_timeout(code, start_date, end_date, timeout):
    """在线程中执行fetch，超时则返回None (线程无法杀死但会被socket超时释放)"""
    result = [None]

    def target():
        try:
            result[0] = fetch_kline_data(code, start_date, end_date)
        except Exception as e:
            logger.error(f"[{code}] fetch异常: {e}")

    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout)

    if t.is_alive():
        logger.warning(f"[{code}] 整体超时 ({timeout}s)，跳过 (后台线程将由socket超时释放)")
        return None
    return result[0]


EXPECTED_COLS = ["date", "code", "open", "high", "low", "close", "preclose",
                 "volume", "amount", "adjustflag", "turn", "tradestatus", "pctChg", "isST"]

def _validate_and_clean(df):
    """过滤 baostock 返回中的无效行 (列数不对/日期格式错/缺失关键字段)"""
    if df is None or len(df) == 0:
        return None
    # 1. 只保留有正确列数的行 (逗号数不对的滤掉)
    ncols = len(EXPECTED_COLS)
    if df.shape[1] != ncols:
        # 尝试读取时可能列数不匹配, 强制用 EXPECTED_COLS 做列名
        if df.shape[1] > ncols:
            df = df.iloc[:, :ncols]
        df.columns = EXPECTED_COLS[:df.shape[1]]
        if df.shape[1] < ncols:
            return None
    # 2. 日期格式校验 YYYY-MM-DD
    import re
    date_pattern = re.compile(r'^\d{4}-\d{2}-\d{2}$')
    valid_dates = df['date'].astype(str).str.match(date_pattern)
    df = df[valid_dates].copy()
    if len(df) == 0:
        return None
    # 3. 数值列检查 (open/high/low/close 不能全是0)
    for col in ['open', 'high', 'low', 'close']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['open', 'high', 'low', 'close'])
    df = df[~((df['open'] == 0) & (df['high'] == 0) & (df['low'] == 0) & (df['close'] == 0))]
    if len(df) == 0:
        return None
    return df.reset_index(drop=True)

def main():
    parser = argparse.ArgumentParser(description="批量获取大A股票K线数据 (前复权)")
    parser.add_argument('--start', type=str, default='2016-06-01',
                        help='起始日期 YYYY-MM-DD (默认: 2016-06-01)')
    parser.add_argument('--end', type=str, default='2026-05-29',
                        help='结束日期 YYYY-MM-DD (默认: 2026-05-29)')
    parser.add_argument('--force', action='store_true',
                        help='强制重新获取 (即使CSV已存在)')
    parser.add_argument('--refresh', action='store_true',
                        help='增量刷新: 只更新数据滞后于今天的股票')
    args = parser.parse_args()

    start_date = args.start
    end_date = args.end
    force = args.force
    refresh = args.refresh

    # 设置全局socket超时
    socket.setdefaulttimeout(SOCKET_TIMEOUT)
    logger.info(f"Socket timeout: {SOCKET_TIMEOUT}s, Per-stock timeout: {PER_STOCK_TIMEOUT}s")

    df_stocks = pd.read_csv(STOCK_LIST_FILE)
    # 过滤退市股
    if 'tradeStatus' in df_stocks.columns:
        df_stocks = df_stocks[df_stocks['tradeStatus'] == 1]
    if 'code_name' in df_stocks.columns:
        df_stocks = df_stocks[~df_stocks['code_name'].str.contains('退', na=False)]
    stocks = df_stocks['code'].tolist()
    total_stocks = len(stocks)
    today = datetime.now().strftime('%Y-%m-%d')

    if refresh:
        print("=" * 60)
        print("BaoStock 增量刷新模式")
        print("=" * 60)
        print(f"目标日期: {today}")
        print(f"股票数量: {total_stocks}")
    else:
        print("=" * 60)
        print("BaoStock 批量获取K线数据 (前复权)")
        print("=" * 60)
        print(f"时间范围: {start_date} ~ {end_date}")
        if force:
            print("强制模式: 覆盖已存在的CSV")
        print(f"股票数量: {total_stocks}")

    DATA_DIR.mkdir(exist_ok=True)

    if not login():
        logger.error("登录失败!")
        return

    try:
        success = 0
        skip = 0
        fail = 0
        updated = 0
        failed_stocks = []

        print(f"\n开始获取...")
        print("-" * 60)

        for i, code in enumerate(stocks):
            stock_file = DATA_DIR / f"{code.replace('.', '_')}.csv"

            if refresh:
                # 增量模式: 只拉缺失部分
                need_fetch = True
                last_date = start_date
                if stock_file.exists():
                    try:
                        # 只读最后一行检查日期 (快)
                        last_line = ''
                        with open(stock_file, 'r') as fh:
                            for line in fh:
                                if line.strip():
                                    last_line = line
                        if last_line:
                            last = last_line.split(',')[0].strip()
                            # 2天内算新鲜 (非交易日/未开盘时 today 无数据)
                            fresh_date = (datetime.now() - timedelta(days=2)).strftime('%Y-%m-%d')
                            if last >= fresh_date:
                                need_fetch = False
                            else:
                                last_date = (datetime.strptime(last, '%Y-%m-%d') - timedelta(days=2)).strftime('%Y-%m-%d')
                    except Exception:
                        pass

                if need_fetch:
                    df = _fetch_with_timeout(code, last_date, today, PER_STOCK_TIMEOUT)
                    df = _validate_and_clean(df)
                    if df is not None and len(df) > 0:
                        # 检测除权除息: 新旧重叠日 close 价不同 → 前复权调整, 需全量重拉
                        need_full = False
                        if stock_file.exists():
                            try:
                                old = pd.read_csv(stock_file)
                                overlap_date = df['date'].iloc[0]
                                old_row = old[old['date'] == overlap_date]
                                if len(old_row) > 0:
                                    old_close = float(old_row['close'].iloc[0])
                                    new_close = float(df[df['date'] == overlap_date]['close'].iloc[0])
                                    if abs(old_close - new_close) > 0.01:
                                        logger.info(f"[{code}] 除权除息检测到, 全量重拉")
                                        need_full = True
                                if not need_full:
                                    combined = pd.concat([old, df], ignore_index=True)
                                    combined = combined.drop_duplicates(subset=['date'], keep='last')
                                    combined = combined.sort_values('date').reset_index(drop=True)
                                    combined.to_csv(stock_file, index=False)
                            except Exception:
                                df.to_csv(stock_file, index=False)
                        else:
                            df.to_csv(stock_file, index=False)

                        if need_full:
                            # 全量重拉替换旧文件
                            full_df = _fetch_with_timeout(code, start_date, today, PER_STOCK_TIMEOUT)
                            full_df = _validate_and_clean(full_df)
                            if full_df is not None and len(full_df) > 0:
                                full_df.to_csv(stock_file, index=False)
                            else:
                                fail += 1
                                failed_stocks.append(code)
                        updated += 1
                    else:
                        fail += 1
                        failed_stocks.append(code)
                else:
                    skip += 1

            else:
                # 全量模式
                if stock_file.exists() and not force:
                    skip += 1
                else:
                    df = _fetch_with_timeout(code, start_date, end_date, PER_STOCK_TIMEOUT)
                    df = _validate_and_clean(df)
                    if df is not None and len(df) > 0:
                        df.to_csv(stock_file, index=False)
                        success += 1
                    else:
                        fail += 1
                        failed_stocks.append(code)
                        logger.warning(f"[{code}] 失败，已记录 (累计失败: {fail})")

            # 每50次重新登录 (skip 也会走到这里)
            if (i + 1) % 50 == 0:
                logout()
                if not login():
                    logger.error("重新登录失败!")
                    break
                if refresh:
                    logger.info(f"[{i+1}/{total_stocks}] re-login, updated={updated}, skip={skip}, fail={fail}")
                else:
                    logger.info(f"[{i+1}/{total_stocks}] re-login, success={success}, skip={skip}, fail={fail}")

            # 进度 & 限速
            if (i + 1) % 200 == 0:
                if refresh:
                    logger.info(f"[{i+1}/{total_stocks}] updated={updated}, skip={skip}, fail={fail}")
                else:
                    logger.info(f"[{i+1}/{total_stocks}] success={success}, skip={skip}, fail={fail}")
                time.sleep(0.3)

        if refresh:
            logger.info(f"第一轮完成! updated={updated}, skip={skip}, fail={fail}")
        else:
            logger.info(f"第一轮完成! success={success}, skip={skip}, fail={fail}")

        # ─── 自动重试失败股票 ───
        if failed_stocks:
            retry_count = len(failed_stocks)
            logger.info(f"开始重试 {retry_count} 只失败股票...")
            print(f"\n重试失败股票 ({retry_count} 只)...")

            logout()
            if not login():
                logger.error("重试登录失败!")
            else:
                still_failed = []
                retry_success = 0
                retry_end = today if refresh else end_date
                for i, code in enumerate(failed_stocks):
                    stock_file = DATA_DIR / f"{code.replace('.', '_')}.csv"
                    if stock_file.exists() and not force:
                        skip += 1
                        retry_success += 1
                    elif refresh:
                        last = start_date
                        if stock_file.exists():
                            try:
                                with open(stock_file, 'r') as fh:
                                    last_line = None
                                    for line in fh:
                                        if line.strip(): last_line = line
                                if last_line:
                                    last = (datetime.strptime(last_line.split(',')[0].strip(), '%Y-%m-%d') - timedelta(days=2)).strftime('%Y-%m-%d')
                            except: pass
                        df = _fetch_with_timeout(code, last, today, PER_STOCK_TIMEOUT)
                        df = _validate_and_clean(df)
                        if df is not None and len(df) > 0:
                            if stock_file.exists():
                                try:
                                    old = pd.read_csv(stock_file)
                                    combined = pd.concat([old, df], ignore_index=True)
                                    combined = combined.drop_duplicates(subset=['date'], keep='last')
                                    combined = combined.sort_values('date').reset_index(drop=True)
                                    combined.to_csv(stock_file, index=False)
                                except: df.to_csv(stock_file, index=False)
                            else: df.to_csv(stock_file, index=False)
                            updated += 1; retry_success += 1
                        else: still_failed.append(code)
                    else:
                        df = _fetch_with_timeout(code, start_date, retry_end, PER_STOCK_TIMEOUT)
                        df = _validate_and_clean(df)
                        if df is not None and len(df) > 0:
                            df.to_csv(stock_file, index=False)
                            success += 1
                            retry_success += 1
                        else:
                            still_failed.append(code)

                    if (i + 1) % 50 == 0:
                        logout()
                        if not login():
                            logger.error("重试重新登录失败!")
                            break
                    if (i + 1) % 10 == 0:
                        logger.info(f"重试 [{i+1}/{retry_count}] retry_ok={retry_success}")
                        time.sleep(0.3)

                fail = len(still_failed)
                failed_stocks = still_failed

                if still_failed:
                    failed_file = DATA_DIR / "failed_stocks.txt"
                    with open(failed_file, "w") as f:
                        f.write("\n".join(still_failed))
                    logger.info(f"重试完成: {retry_success} 恢复, {fail} 仍然失败 → {failed_file}")

        print("-" * 60)
        print(f"\n获取完成!")
        if refresh:
            print(f"  增量更新: {updated} 只")
            print(f"  已是最新: {skip} 只")
        else:
            print(f"  成功: {success} 只")
            print(f"  跳过(已存在): {skip} 只")
        print(f"  失败: {fail} 只")
        if failed_stocks:
            print(f"  失败列表已保存至: {DATA_DIR / 'failed_stocks.txt'}")
        print(f"数据目录: {DATA_DIR}")

    finally:
        logout()


if __name__ == "__main__":
    main()
