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

def _merge_and_save(stock_file, df, is_refresh):
    """CSV 合并保存 (线程安全, 不含 baostock 调用).
    返回 (success, need_full_refetch)"""
    if is_refresh and stock_file.exists():
        try:
            old = pd.read_csv(stock_file)
            overlap_date = df['date'].iloc[0]
            old_row = old[old['date'] == overlap_date]
            if len(old_row) > 0:
                old_close = float(old_row['close'].iloc[0])
                new_close = float(df[df['date'] == overlap_date]['close'].iloc[0])
                if abs(old_close - new_close) > 0.01:
                    return (False, True)  # 除权除息, 需全量重拉
            combined = pd.concat([old, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=['date'], keep='last')
            combined = combined.sort_values('date').reset_index(drop=True)
            combined.to_csv(stock_file, index=False)
            return (True, False)
        except Exception:
            df.to_csv(stock_file, index=False)
            return (True, False)
    else:
        df.to_csv(stock_file, index=False)
        return (True, False)


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

    # 股票列表不存在时自动重建
    if not STOCK_LIST_FILE.exists():
        print("股票列表文件不存在, 自动从 Baostock 重建...")
        if not login():
            logger.error("登录失败!")
            return
        # 用 query_all_stock 一次获取全市场股票
        day = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        rs = bs.query_all_stock(day=day)
        if rs.error_code != '0':
            logger.error(f"获取股票列表失败: {rs.error_msg}")
            return
        df_all = rs.get_data()  # 用 get_data() 不是 next()
        # 过滤: 只保留正常交易的A股个股, 排除指数
        # 只保留A股个股 (排除指数: sh.000xxx, 排除ETF: sz.159xxx)
        df_all = df_all[(df_all['code'] >= 'sh.600000') & (df_all['code'] < 'sz.399000')]
        df_all = df_all[~df_all['code'].str.match(r'sz\.159\d{3}')]
        df_all = df_all[df_all['tradeStatus'] == '1']
        df_all = df_all[~df_all['code_name'].str.contains('退', na=False)]
        df_all = df_all[['code', 'tradeStatus', 'code_name']]
        df_all.to_csv(STOCK_LIST_FILE, index=False)
        logout()
        print(f'已重建股票列表: {len(df_all)} 只')
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

    # Probe: 拉一只大盘股确认 baostock 最新可用日期 (处理非交易日)
    probe_code = 'sh.600000'
    if refresh:
        print(f"探针: 查询 {probe_code} 确定最新可用日期...")
        probe_df = _fetch_with_timeout(probe_code, (datetime.now() - timedelta(days=10)).strftime('%Y-%m-%d'), today, 30)
        if probe_df is not None and len(probe_df) > 0:
            latest_available = probe_df['date'].iloc[-1]
            print(f"  baostock 最新日期: {latest_available}")
        else:
            latest_available = today
            print(f"  探针失败, 使用 today={today}")
    else:
        latest_available = today

    try:
        success = 0; skip = 0; fail = 0; updated = 0
        failed_stocks = []

        if refresh:
            # ── Phase 1: 多线程扫描, 确定哪些需要更新 (只读CSV, 无API) ──
            from concurrent.futures import ThreadPoolExecutor, as_completed

            def _check_one(task):
                code, stock_file_str = task
                stock_file = Path(stock_file_str)
                if not stock_file.exists():
                    return (code, True, start_date, stock_file_str)
                try:
                    last_line = ''
                    with open(stock_file, 'r') as fh:
                        for line in fh:
                            if line.strip(): last_line = line
                    if last_line:
                        last = last_line.split(',')[0].strip()
                        if last >= latest_available:
                            return (code, False, None, None)  # skip: 已最新, 前复权数据天然已调整
                        last_date = (datetime.strptime(last, '%Y-%m-%d') - timedelta(days=2)).strftime('%Y-%m-%d')
                        return (code, True, last_date, stock_file_str)  # fetch
                except Exception:
                    pass
                return (code, True, start_date, stock_file_str)  # fetch (fallback)

            scan_workers = min(16, len(stocks))
            print(f"Phase 1: {scan_workers} 线程扫描 {total_stocks} 只股票...")
            tasks = [(code, str(DATA_DIR / f"{code.replace('.', '_')}.csv")) for code in stocks]
            to_fetch = []
            with ThreadPoolExecutor(max_workers=scan_workers) as scanner:
                for code, need, last_d, sf in scanner.map(_check_one, tasks):
                    if need:
                        to_fetch.append((code, last_d, sf))
                    else:
                        skip += 1
            print(f"  需更新: {len(to_fetch)} 只, 已跳过: {skip} 只")

            # ── Phase 2: 主线程串行 API 拉取 + 多线程 CSV 合并 ──
            csv_workers = min(8, max(1, len(to_fetch)))
            csv_executor = ThreadPoolExecutor(max_workers=csv_workers)
            csv_futures = []

            from tqdm import tqdm
            pbar = tqdm(to_fetch, desc="Phase 2: API拉取", unit="只")
            for code, last_d, stock_file_str in pbar:
                df = _fetch_with_timeout(code, last_d, today, PER_STOCK_TIMEOUT)
                df = _validate_and_clean(df)
                stock_file = Path(stock_file_str)

                if df is not None and len(df) > 0:
                    csv_futures.append((
                        code,
                        csv_executor.submit(_merge_and_save, stock_file, df, True),
                        stock_file
                    ))
                    updated += 1
                else:
                    fail += 1
                    failed_stocks.append(code)

                pbar.set_postfix(updated=updated, fail=fail)
                if (updated + fail) % 50 == 0:
                    try: bs.logout()
                    except: pass
                    if not login():
                        logger.error("重新登录失败, 继续尝试...")

            # 等 CSV 写入, 处理除权除息
            print(f"\n等待 CSV 合并完成...")
            for code, fut, stock_file in tqdm(csv_futures, desc="Phase 3: CSV合并", unit="只"):
                try:
                    ok, need_full_refetch = fut.result(timeout=30)
                    if need_full_refetch:
                        logger.info(f"[{code}] 除权除息, 全量重拉")
                        full_df = _fetch_with_timeout(code, start_date, today, PER_STOCK_TIMEOUT)
                        full_df = _validate_and_clean(full_df)
                        if full_df is not None and len(full_df) > 0:
                            full_df.to_csv(stock_file, index=False)
                except Exception as e:
                    logger.warning(f"[{code}] CSV合并失败: {e}")

            csv_executor.shutdown(wait=True)

        else:
            # 全量模式 (单进程)
            for i, code in enumerate(stocks):
                stock_file = DATA_DIR / f"{code.replace('.', '_')}.csv"
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

                if (i + 1) % 50 == 0:
                    try: logout()
                    except: pass
                    for retry in range(3):
                        if login(): break
                        time.sleep(2)
                    logger.info(f"[{i+1}/{total_stocks}] re-login, success={success}, skip={skip}, fail={fail}")
                if (i + 1) % 200 == 0:
                    logger.info(f"[{i+1}/{total_stocks}] success={success}, skip={skip}, fail={fail}")
                    time.sleep(0.3)

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
