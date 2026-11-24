import subprocess, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from bs4 import BeautifulSoup
from pathlib import Path

OUT = Path(r"C:\Users\Vision Works\Documents\GitHub\data_predict\macro_analysis\docs\题材连板\原文")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

targets = [
    ("15_高低板换手分歧买点_东财财富号", "https://caifuhao.eastmoney.com/news/20260913204435966157880"),
    ("16_三板组识别_东财财富号", "https://caifuhao.eastmoney.com/news/20260319050148166167340"),
    ("17_陈小群龙头战法_东财财富号", "https://caifuhao.eastmoney.com/news/20261028195506438610120"),
]

def extract(html):
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style']):
        tag.decompose()
    for cls in ['xeditor_content', 'g_content', 'zwaddcontent', 'article-body', 'txtinfos']:
        el = soup.select_one(f'.{cls}') or soup.select_one(f'#{cls}')
        if el:
            t = el.get_text('\n', strip=True)
            if t and len(t) > 100:
                return t
    ps = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 1]
    return '\n\n'.join(ps) if ps else ''

for name, url in targets:
    # 用 curl 下载（requests 被东财反爬返回空）
    r = subprocess.run(['curl', '-s', '-m', '25', '-A', UA, '-H', 'Referer: https://caifuhao.eastmoney.com/', url],
                       capture_output=True)
    html = r.stdout.decode('utf-8', errors='replace')
    if len(html) < 1000:
        print(f'[失败] {name}: curl 下载过短({len(html)}字节)')
        continue
    text = extract(html)
    if text and len(text) > 100:
        (OUT / f'{name}.md').write_text(f'# {name}\n\n> 来源：{url}\n\n---\n\n{text}\n', encoding='utf-8')
        print(f'[成功] {name}: {len(text)}字')
    else:
        print(f'[失败] {name}: 正文过短({len(text)}字)')
    time.sleep(1)
