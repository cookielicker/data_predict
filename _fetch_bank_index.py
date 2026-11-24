import urllib.request, json
import pandas as pd
import socket
socket.setdefaulttimeout(30)

def em_kline(secid, beg='20150101', end='20261106'):
    url = (f"https://push2his.eastmoney.com/api/qt/stock/kline/get"
           f"?secid={secid}&fields1=f1,f2,f3,f4,f5,f6"
           f"&fields2=f51,f52,f53,f54,f55,f56,f57"
           f"&klt=101&fqt=1&beg={beg}&end={end}")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    resp = urllib.request.urlopen(req, timeout=30)
    return json.loads(resp.read().decode('utf-8', errors='replace'))

# 拉银行板块指数完整历史
kd = em_kline('90.BK0475', beg='20150101', end='20261106')
data = kd.get('data') or {}
klines = data.get('klines', [])
rows = []
for k in klines:
    p = k.split(',')
    rows.append({'date': p[0], 'close': float(p[2])})
bank = pd.DataFrame(rows)
bank['date'] = pd.to_datetime(bank['date'])
bank.to_csv('bank_index_em.csv', index=False)
print(f'银行板块指数: {len(bank)} 行, {bank["date"].iloc[0].date()} ~ {bank["date"].iloc[-1].date()}')

# 读宏观大表取上证大级别确认日
mt = pd.read_csv('macro_analysis/data/macro_table.csv', encoding='utf-8-sig')
mt['date'] = pd.to_datetime(mt['date'])
sig = mt[mt['sh_major_signal'].notna()][['date', 'sh_major_signal']]
print('\n上证大级别确认日:')
for _, r in sig.iterrows():
    print(f'  {r["date"].date()}  {r["sh_major_signal"]}')
