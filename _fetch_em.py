import sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import requests
from bs4 import BeautifulSoup
from pathlib import Path

OUT = Path(r"C:\Users\Vision Works\Documents\GitHub\data_predict\macro_analysis\docs\题材连板\原文")
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36'}

# 东财系正文选择器 + 通用选择器
SELS = ['xeditor_content', 'g_content', 'zwaddcontent', 'article-body', 'txtinfos',
        'news_content', 'art_content', 'article', 'content', 'rich_media_content']

targets = [
    ("02_ST股涨跌幅_东方财富", "https://finance.eastmoney.com/a/202607063794924487.html"),
    ("15_高低板换手分歧买点_东财财富号", "https://caifuhao.eastmoney.com/news/20260913204435966157880"),
    ("16_三板组识别_东财财富号", "https://caifuhao.eastmoney.com/news/20260319050148166167340"),
    ("17_陈小群龙头战法_东财财富号", "https://caifuhao.eastmoney.com/news/20261028195506438610120"),
    ("13_短线术语_东方财富", "https://emcreative.eastmoney.com/app_fortune/article?artcode=20240211075416187777350&postid=1399529934"),
]

def extract(html):
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style']):
        tag.decompose()
    for sel in SELS:
        for el in soup.select(f'.{sel}, #{sel}, .{sel.replace("_"," ")}'):
            t = el.get_text('\n', strip=True)
            if t and len(t) > 100:
                return t
    # 兜底：所有 p
    ps = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 1]
    return '\n\n'.join(ps) if ps else ''

for name, url in targets:
    try:
        r = requests.get(url, headers=UA, timeout=25)
        r.encoding = r.apparent_encoding or 'utf-8'
        if r.status_code != 200:
            print(f'[失败] {name}: HTTP {r.status_code}')
            continue
        text = extract(r.text)
        if text and len(text) > 100:
            (OUT / f'{name}.md').write_text(f'# {name}\n\n> 来源：{url}\n\n---\n\n{text}\n', encoding='utf-8')
            print(f'[成功] {name}: {len(text)}字')
        else:
            print(f'[失败] {name}: 正文过短({len(text)}字)')
    except Exception as e:
        print(f'[失败] {name}: {e}')
    time.sleep(1)
