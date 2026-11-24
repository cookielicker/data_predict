import subprocess, sys, io, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
from bs4 import BeautifulSoup
from pathlib import Path

OUT = Path(r"C:\Users\Vision Works\Documents\GitHub\data_predict\macro_analysis\docs\题材连板\原文")
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"

targets = [
    ("14_龙头战法全解_CSDN", "https://blog.csdn.net/zhangyunchou2015/article/details/156942937"),
    ("09_龙头情绪周期_淘股吧", "https://m.tgb.cn/a/2vd5rCv6MkU"),
    ("13_短线术语_东方财富", "https://emcreative.eastmoney.com/app_fortune/article?artcode=20240211075416187777350&postid=1399529934"),
]

def extract(html):
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style']):
        tag.decompose()
    for cls in ['article_content', 'content', 'article', 'blog-content-box', 'xeditor_content', 'g_content',
                'txtinfos', 'rich_media_content', 'markdown_views', 'htmledit_views']:
        el = soup.select_one(f'.{cls}') or soup.select_one(f'#{cls}')
        if el:
            t = el.get_text('\n', strip=True)
            if t and len(t) > 100:
                return t
    ps = [p.get_text(strip=True) for p in soup.find_all('p') if len(p.get_text(strip=True)) > 1]
    return '\n\n'.join(ps) if ps else ''

for name, url in targets:
    r = subprocess.run(['curl', '-s', '-m', '30', '-L', '-A', UA, url], capture_output=True)
    html = r.stdout.decode('utf-8', errors='replace')
    if len(html) < 1000:
        print(f'[失败] {name}: curl 下载过短({len(html)}字节)')
        continue
    text = extract(html)
    if text and len(text) > 100:
        (OUT / f'{name}.md').write_text(f'# {name}\n\n> 来源：{url}\n\n---\n\n{text}\n', encoding='utf-8')
        print(f'[成功] {name}: {len(text)}字')
    else:
        print(f'[失败] {name}: 正文过短({len(text)}字), HTML {len(html)}字节')
    time.sleep(1)
