import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import requests
from pathlib import Path

OUT = Path(r"C:\Users\Vision Works\Documents\GitHub\data_predict\macro_analysis\docs\题材连板\原文")
PROXY = {'http': 'http://127.0.0.1:7897', 'https': 'http://127.0.0.1:7897'}
UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36'}

targets = [
    ("19_龙行刺情绪周期体系_md", "https://cdn.jsdelivr.net/gh/zhouqinglong520/trading-mastery@main/%E4%BA%A4%E6%98%93%E4%BD%93%E7%B3%BB/01_%E9%BE%99%E8%A1%8C%E5%88%BA%E6%83%85%E7%BB%AA%E5%91%A8%E6%9C%9F%E4%BA%A4%E6%98%93%E4%BD%93%E7%B3%BB.md"),
    ("20_龙头首阴战法_Scribd", "https://www.scribd.com/document/798031249/%E9%BE%99%E5%A4%B4%E8%82%A1%E9%A6%96%E9%98%B4%E6%88%98%E6%B3%95"),
    ("14_龙头战法全解_CSDN", "https://blog.csdn.net/zhangyunchou2015/article/details/156942937"),
]

for name, url in targets:
    try:
        r = requests.get(url, headers=UA, proxies=PROXY, timeout=40, allow_redirects=True)
        print(f'{name}: HTTP {r.status_code}, {len(r.content)} bytes')
        if r.status_code == 200:
            # 判断是不是 markdown 或 html
            ct = r.headers.get('Content-Type', '')
            if 'text/plain' in ct or name.endswith('_md') or url.endswith('.md'):
                text = r.text
            else:
                text = r.text  # 先存原文，稍后处理
            fname = OUT / f'{name}.md'
            fname.write_text(f'# {name}\n\n> 来源：{url}\n\n---\n\n{text}\n', encoding='utf-8')
            print(f'  [成功] 已存 {name}.md ({len(text)}字)')
    except Exception as e:
        print(f'{name}: 失败 {e}')
