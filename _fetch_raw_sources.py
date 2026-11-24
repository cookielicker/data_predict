#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量抓取题材连板文档的原始来源网页，存成 markdown 原文。

来源列表来自 docs/题材连板/连板题材炒作入门.md 末尾的「来源」章节。
存到 docs/题材连板/原文/ 目录，每个来源一个 .md（文件名 = 简短标识）。
"""
import sys, io, time, os, re
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import requests
from bs4 import BeautifulSoup
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'macro_analysis' / 'docs' / '题材连板' / '原文'
OUT.mkdir(parents=True, exist_ok=True)

UA = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36'}

# (文件名, URL) —— 文件名用简短中文标识
SOURCES = [
    ("01_ST股涨跌幅调整_上海金融监管局", "https://jrj.sh.gov.cn/ZXYW178/20260702/a8fb63ac6e9e49f9a47a4024a98fd7fb.html"),
    ("02_ST股涨跌幅_东方财富", "https://finance.eastmoney.com/a/202607063794924487.html"),
    ("03_主板ST股创业板改革_21财经", "https://www.21jingji.com/article/20260410/herald/15660b9ceeb28f76e283a3c1f8021f27.html"),
    ("04_科创板交易制度_天风证券", "https://edu.tfzq.com/index/article/?id=18603"),
    ("05_创业板交易规则_南方基金", "https://www.nffund.com/main/servicecenter/rdwtfl/tzzjy/tjtx/95174.shtml"),
    ("06_情绪周期与战法_闽发论坛", "https://www.xiarj.com/24984.html"),
    ("07_主力操盘情绪周期_太平洋证券", "https://edu.tpyzq.com/investor/mobile/Details.jsp?id=394963"),
    ("08_情绪周期七阶段_新浪", "https://www.sina.cn/news/detail/5347063085926950.html"),
    ("09_龙头情绪周期_淘股吧", "https://m.tgb.cn/a/2vd5rCv6MkU"),
    ("12_北交所交易规则", "https://www.bseinfo.net/jygl_list/200028217.html"),
    ("13_短线术语_东方财富", "https://emcreative.eastmoney.com/app_fortune/article?artcode=20240211075416187777350&postid=1399529934"),
    ("14_龙头战法全解_CSDN", "https://blog.csdn.net/zhangyunchou2015/article/details/156942937"),
    ("15_高低板换手分歧买点_东财财富号", "https://caifuhao.eastmoney.com/news/20260913204435966157880"),
    ("16_三板组识别_东财财富号", "https://caifuhao.eastmoney.com/news/20260319050148166167340"),
    ("17_陈小群龙头战法_东财财富号", "https://caifuhao.eastmoney.com/news/20261028195506438610120"),
    ("18_炒股养家语录100条_淘股吧", "https://tgb.cn/a/2wlfN5KqiQx"),
    ("19_龙行刺情绪周期体系_md", "https://cdn.jsdelivr.net/gh/zhouqinglong520/trading-mastery@main/%E4%BA%A4%E6%98%93%E4%BD%93%E7%B3%BB/01_%E9%BE%99%E8%A1%8C%E5%88%BA%E6%83%85%E7%BB%AA%E5%91%A8%E6%9C%9F%E4%BA%A4%E6%98%93%E4%BD%93%E7%B3%BB.md"),
    ("20_龙头首阴战法_Scribd", "https://www.scribd.com/document/798031249/%E9%BE%99%E5%A4%B4%E8%82%A1%E9%A6%96%E9%98%B4%E6%88%98%E6%B3%95"),
]

PDFS = [
    ("10_深交所交易规则2026", "http://docs.static.szse.cn/www/lawrules/rule/trade/current/W020260424690713155663.pdf"),
    ("11_上交所交易规则2026", "https://www.lczq.com/upload/20260703/20260703152140244008.pdf"),
]


def extract_text(html):
    """从 HTML 提取正文文本。优先 article/content，否则全部 p 标签。"""
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style', 'noscript']):
        tag.decompose()
    # 找正文容器
    main = None
    for sel in ['article', '.article', '.content', '.art_content', '.news_content', '.article-content', '#content', '.rich_media_content']:
        main = soup.select_one(sel)
        if main:
            break
    if main is None:
        main = soup.body or soup
    # 提取段落文本
    paras = []
    for p in main.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'li']):
        t = p.get_text(strip=True)
        if t and len(t) > 1:
            paras.append(t)
    if not paras:
        # 兜底：全文文本
        t = main.get_text('\n', strip=True)
        paras = [t]
    return '\n\n'.join(paras)


def main():
    results = []
    for name, url in SOURCES:
        try:
            r = requests.get(url, headers=UA, timeout=25, allow_redirects=True)
            r.encoding = r.apparent_encoding or 'utf-8'
            if r.status_code != 200:
                results.append((name, f'HTTP {r.status_code}', ''))
                print(f'[失败] {name}: HTTP {r.status_code}')
                continue
            text = extract_text(r.text)
            if not text or len(text) < 50:
                results.append((name, '正文过短/提取失败', ''))
                print(f'[失败] {name}: 正文过短({len(text)}字)')
                continue
            fname = OUT / f'{name}.md'
            content = f'# {name}\n\n> 来源：{url}\n\n---\n\n{text}\n'
            fname.write_text(content, encoding='utf-8')
            results.append((name, 'OK', f'{len(text)}字'))
            print(f'[成功] {name}: {len(text)}字')
        except Exception as e:
            results.append((name, f'异常: {e}', ''))
            print(f'[失败] {name}: {e}')
        time.sleep(1)

    # PDF 直接下载
    for name, url in PDFS:
        try:
            r = requests.get(url, headers=UA, timeout=60, allow_redirects=True)
            if r.status_code == 200 and len(r.content) > 10000:
                fpath = OUT / f'{name}.pdf'
                fpath.write_bytes(r.content)
                results.append((name, 'PDF下载OK', f'{len(r.content)//1024}KB'))
                print(f'[成功] {name}: PDF {len(r.content)//1024}KB')
            else:
                results.append((name, f'PDF下载失败 HTTP{r.status_code}', ''))
                print(f'[失败] {name}: PDF HTTP {r.status_code}')
        except Exception as e:
            results.append((name, f'异常: {e}', ''))
            print(f'[失败] {name}: {e}')
        time.sleep(1)

    print('\n' + '=' * 50)
    ok = sum(1 for _, s, _ in results if 'OK' in s)
    print(f'共 {len(results)} 个来源，成功 {ok} 个，失败 {len(results)-ok} 个')
    print('\n失败清单:')
    for name, status, _ in results:
        if 'OK' not in status:
            print(f'  {name}: {status}')


if __name__ == '__main__':
    main()
