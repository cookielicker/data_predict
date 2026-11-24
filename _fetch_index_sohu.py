#!/usr/bin/env python3
"""用搜狐接口拉创业板指/科创50 历史日线, 存本地CSV (兜底BaoStock断连)
搜狐接口: https://q.stock.sohu.com/hisHq?code=zs_399006&start=20100101&end=20261102&stat=1&order=D&period=d
返回 JSON, 字段: date, open, close, high, low, volume, amount, ...
"""
import sys, io, json, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import urllib.request
import pandas as pd

def fetch_sohu(code, name, start='20100101', end='20261102'):
    # code: 搜狐用 zs_ 前缀 (指数)
    url = f"https://q.stock.sohu.com/hisHq?code=zs_{code}&start={start}&end={end}&stat=1&order=D&period=d"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    try:
        resp = urllib.request.urlopen(req, timeout=30)
        raw = resp.read().decode('utf-8')
    except Exception as e:
        print(f"  {name} 拉取失败: {e}")
        return None
    # 解析 JSON: 形如 [{"status":0,"hq":[[date,open,close,...,amount],...]}]
    data = json.loads(raw)
    if not data or data[0].get('status') != 0:
        print(f"  {name} 返回异常: {str(data)[:100]}")
        return None
    hq = data[0]['hq']
    # hq 每行: [date, open, close, change?, ..., high, low, volume, amount, ...]
    # 搜狐格式: [日期,开盘,收盘,涨跌额,涨跌幅,最低,最高,成交量(手),成交额(万),换手率]
    rows = []
    for r in hq:
        d = r[0]
        open_ = float(r[1]); close = float(r[2])
        low = float(r[5]); high = float(r[6])
        vol = float(r[7]); amt = float(r[8])
        rows.append([d, open_, high, low, close, vol, amt])
    df = pd.DataFrame(rows, columns=['date','open','high','low','close','volume','amount'])
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    print(f"  {name}: {len(df)} 天, {df['date'].iloc[0].date()} ~ {df['date'].iloc[-1].date()}")
    return df

targets = [('399006', '创业板指'), ('000688', '科创50'), ('000001', '上证指数')]
for code, name in targets:
    df = fetch_sohu(code, name)
    if df is not None:
        fn = f'index_{code}.csv'
        df.to_csv(fn, index=False)
        print(f"  已保存 {fn}")
    time.sleep(1)
print("done")
