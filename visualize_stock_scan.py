#!/usr/bin/env python3
"""
Stock Scan Viewer v2 — 基于 stock_data/ 滑动窗口预测, 按股票+日期浏览
用法:
  python visualize_stock_scan.py
"""
import os, sys, subprocess
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.gridspec import GridSpec

import pandas as pd

# ============ 常量 ============
PREVIOUS_NUM = 30
PREDICT_NUM = 3
PP = PREVIOUS_NUM + PREDICT_NUM  # 33
INPUT_DIM = PREVIOUS_NUM * 4     # 120

LABEL_NAMES = {0: "L0 (跌>10%)", 1: "L1 (-10%~0)", 2: "L2 (0~10%)", 3: "L3 (涨>10%)"}
LABEL_COLORS = {0: "#e74c3c", 1: "#e67e22", 2: "#3498db", 3: "#2ecc71"}
NUM_CLASSES = 4

DEFAULT_MODEL = "best_fc_twostage_30d_h256_l4_20260601_222325.pt"
STOCK_DATA_DIR = "stock_data"


def compute_features_and_label(df, end_idx):
    """取 end_idx 前30天做特征, 后面0-3天做标签.
    end_idx: 30天历史最后一天的索引 (0-based in df)"""
    start_idx = end_idx - PREVIOUS_NUM + 1
    if start_idx < 0 or end_idx >= len(df):
        return None

    hist = df.iloc[start_idx:end_idx + 1]
    if len(hist) < PREVIOUS_NUM:
        return None

    mean_adj = ((hist['open'] + hist['high'] + hist['low'] + hist['close']) / 4).values.astype(np.float32)
    change_rate = (hist['turn'].fillna(0) / 100.0).values.astype(np.float32)
    high_low_ratio = ((hist['high'] - hist['low']) / mean_adj).values.astype(np.float32)
    high_low_ratio = np.where(np.isfinite(high_low_ratio), high_low_ratio, 0)

    change_delta = np.zeros(PREVIOUS_NUM, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        d = np.zeros(PREVIOUS_NUM)
        d[1:] = np.where(change_rate[:-1] != 0,
                        (change_rate[1:] - change_rate[:-1]) / change_rate[:-1], 0)
        change_delta[:] = d

    price_pct = np.zeros(PREVIOUS_NUM, dtype=np.float32)
    with np.errstate(divide='ignore', invalid='ignore'):
        r = np.zeros(PREVIOUS_NUM)
        r[1:] = np.where(mean_adj[:PREVIOUS_NUM-1] != 0,
                        (mean_adj[1:PREVIOUS_NUM] - mean_adj[:PREVIOUS_NUM-1])
                        / mean_adj[:PREVIOUS_NUM-1], 0)
        price_pct[:] = r

    features = np.stack([
        price_pct, change_rate[:PREVIOUS_NUM],
        high_low_ratio[:PREVIOUS_NUM], change_delta[:PREVIOUS_NUM],
    ], axis=-1).astype(np.float32)

    # True label: end_idx 后最多3天
    future_start = end_idx + 1
    future_end = min(end_idx + PREDICT_NUM + 1, len(df))
    future_days = future_end - future_start
    true_label = -1
    ret_pct = 0.0
    true_days_info = ""

    fut_mean = np.array([], dtype=np.float32)
    fut_dates = []
    if future_days > 0:
        fut = df.iloc[future_start:future_end]
        fut_mean = ((fut['open'] + fut['high'] + fut['low'] + fut['close']) / 4).values.astype(np.float32)
        fut_dates = fut['date'].apply(lambda x: x[5:] if len(str(x)) >= 10 else str(x)).tolist()
        if len(fut_mean) > 0 and mean_adj[-1] != 0:
            ret_pct = (fut_mean[-1] - mean_adj[-1]) / mean_adj[-1]
            if np.isfinite(ret_pct):
                # 4-class: L0<-10%, L1 -10%~0, L2 0~10%, L3 >10%
                if ret_pct < -0.1: true_label = 0
                elif ret_pct < 0: true_label = 1
                elif ret_pct <= 0.1: true_label = 2
                else: true_label = 3
                if future_days < PREDICT_NUM:
                    true_days_info = f" (仅{future_days}天)"
            else:
                ret_pct = 0.0

    code = df['code'].iloc[end_idx] if 'code' in df.columns else '?'
    date = df['date'].iloc[end_idx] if 'date' in df.columns else '?'
    hist_dates = hist['date'].apply(lambda x: x[5:] if len(str(x)) >= 10 else str(x)).tolist()

    raw_data = {
        'mean_adj': mean_adj, 'change_rate': change_rate,
        'high_low_ratio': high_low_ratio, 'change_delta': change_delta,
        'price_pct': price_pct, 'date': date, 'hist_dates': hist_dates,
        'stock_code': code, 'true_label': true_label, 'return_pct': ret_pct,
        'future_days': future_days, 'true_days_info': true_days_info,
        'fut_mean': fut_mean, 'fut_dates': fut_dates,
    }
    return features, raw_data


class StockScanViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Scan Viewer v2 — 滑动窗口预测")
        self.root.geometry("1400x950")
        self.root.configure(bg="#2b2b2b")

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._stocks = []
        self._stock_index = 0
        self._df_cache = None
        self._cache_code = None
        self._day_index = 0

        self._latest_results = []
        self._filter_mode = "all"
        self._filtered = []
        self._filter_pos = 0

        self._build_ui()

    # ─── UI Build ────────────────────────────────────────────────────
    def _build_ui(self):
        top = tk.Frame(self.root, bg="#2b2b2b")
        top.pack(fill=tk.X, padx=8, pady=(8, 0))

        tk.Button(top, text="Load Model (.pt)", command=self._load_model,
                  bg="#3a3a3a", fg="#ecf0f1", bd=1, padx=10, pady=2,
                  activebackground="#555555").pack(side=tk.LEFT, padx=4)

        self._scan_btn = tk.Button(top, text="▶ Refresh + Scan", command=self._start_scan,
                  bg="#1a6e3e", fg="#ecf0f1", bd=1, padx=10, pady=2,
                  activebackground="#2a8e5e", state=tk.DISABLED)
        self._scan_btn.pack(side=tk.LEFT, padx=4)

        tk.Label(top, text="  Search:", bg="#2b2b2b", fg="#7f8c8d", font=("Arial", 9)).pack(side=tk.LEFT)
        self._search_entry = tk.Entry(top, width=14, bg="#3a3a3a", fg="#ecf0f1",
                                       insertbackground="#ecf0f1", bd=1, font=("Consolas", 10))
        self._search_entry.pack(side=tk.LEFT, padx=4)
        self._search_entry.bind("<Return>", self._search_stock)
        self._search_entry.config(state=tk.DISABLED)

        self._status_label = tk.Label(top, text="请加载模型, 然后点 Refresh+Scan", bg="#2b2b2b",
                                       fg="#7f8c8d", font=("Arial", 10))
        self._status_label.pack(side=tk.LEFT, padx=16)

        # ── Filter bar ──
        filt = tk.Frame(self.root, bg="#2b2b2b")
        filt.pack(fill=tk.X, padx=8, pady=4)

        self._filter_btns = []
        def _add_btn(text, mode, bg_c, fg_c):
            btn = tk.Button(filt, text=text, command=lambda m=mode: self._set_filter(m),
                          bg=bg_c, fg=fg_c, bd=1, padx=8, pady=2,
                          activebackground="#555555", activeforeground="#ffffff",
                          state=tk.DISABLED)
            btn.pack(side=tk.LEFT, padx=2)
            self._filter_btns.append(btn)

        _add_btn("All", "all", "#3a3a3a", "#ecf0f1")
        _add_btn("Pred L0", "pred_l0", "#5a2020", "#e74c3c")
        _add_btn("Pred L1", "pred_l1", "#4a2000", "#e67e22")
        _add_btn("Pred L2", "pred_l2", "#002040", "#3498db")
        _add_btn("Pred L3", "pred_l3", "#204020", "#2ecc71")

        self._filter_breakdown = tk.Label(filt, text="", bg="#2b2b2b", fg="#7f8c8d",
                                           font=("Consolas", 10))
        self._filter_breakdown.pack(side=tk.LEFT, padx=10)

        # ── Navigation ──
        nav = tk.Frame(self.root, bg="#2b2b2b")
        nav.pack(fill=tk.X, padx=8, pady=4)
        b_opts = dict(bg="#3a3a3a", fg="#ecf0f1", bd=1, padx=6, pady=2,
                      activebackground="#555555", activeforeground="#ffffff")

        tk.Button(nav, text="◀◀ Stock", command=self._prev_stock, **b_opts).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text="◀ Day", command=self._prev_day, **b_opts).pack(side=tk.LEFT, padx=2)

        self._date_label = tk.Label(nav, text="", font=("Consolas", 12, "bold"),
                                     bg="#2b2b2b", fg="#f39c12", width=14)
        self._date_label.pack(side=tk.LEFT, padx=6)

        tk.Button(nav, text="Day ▶", command=self._next_day, **b_opts).pack(side=tk.LEFT, padx=2)
        tk.Button(nav, text="Stock ▶▶", command=self._next_stock, **b_opts).pack(side=tk.LEFT, padx=2)

        self._code_display = tk.Label(nav, text="", font=("Consolas", 16, "bold"),
                                       bg="#2b2b2b", fg="#f39c12")
        self._code_display.pack(side=tk.LEFT, padx=16)

        self._pred_display = tk.Label(nav, text="", font=("Arial", 14, "bold"),
                                       bg="#2b2b2b", fg="#ecf0f1")
        self._pred_display.pack(side=tk.LEFT, padx=10)

        prob_frame = tk.Frame(nav, bg="#2b2b2b")
        prob_frame.pack(side=tk.LEFT, padx=16)
        self._prob_labels = {}
        for i, (name, color) in enumerate([("L0", "#e74c3c"), ("L1", "#e67e22"), ("L2", "#3498db"), ("L3", "#2ecc71")]):
            lbl = tk.Label(prob_frame, text=f"{name}: --%", font=("Consolas", 12, "bold"),
                          bg="#2b2b2b", fg=color)
            lbl.pack(side=tk.LEFT, padx=6)
            self._prob_labels[i] = lbl

        tk.Label(nav, text="| Stk:", bg="#2b2b2b", fg="#7f8c8d", font=("Arial", 9)).pack(side=tk.LEFT, padx=(10,2))
        self._stock_count_label = tk.Label(nav, text="0/0", bg="#2b2b2b", fg="#bdc3c7",
                                            font=("Consolas", 10))
        self._stock_count_label.pack(side=tk.LEFT)

        # ── Matplotlib (3 子图, 无 sharex) ──
        self.fig = Figure(figsize=(13, 8), facecolor="#2b2b2b")
        gs = GridSpec(3, 1, figure=self.fig, height_ratios=[3, 2, 2],
                      top=0.97, bottom=0.05, left=0.08, right=0.93, hspace=0.12)
        self.ax_price = self.fig.add_subplot(gs[0])
        self.ax_turnover = self.fig.add_subplot(gs[1])
        self.ax_delta = self.fig.add_subplot(gs[2])

        for ax in [self.ax_price, self.ax_turnover, self.ax_delta]:
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#aaaaaa")
            for sp_name in ["bottom", "left"]:
                ax.spines[sp_name].set_color("#555555")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Bottom ──
        bottom = tk.Frame(self.root, bg="#1e1e1e")
        bottom.pack(fill=tk.X, padx=8, pady=(0, 4))
        self._progress = ttk.Progressbar(bottom, orient=tk.HORIZONTAL, length=500, mode='determinate')
        self._progress.pack(side=tk.RIGHT, padx=8)
        tk.Label(bottom, text="◀◀/▶▶ 切换股票  |  ◀/▶ 切换日期  |  Filter 分类  |  Search 搜索",
                 bg="#1e1e1e", fg="#555555", font=("Arial", 8)).pack(side=tk.LEFT)

        self.root.bind("<Left>", lambda e: self._prev_day())
        self.root.bind("<Right>", lambda e: self._next_day())
        self.root.bind("<Prior>", lambda e: self._prev_stock())
        self.root.bind("<Next>", lambda e: self._next_stock())

    # ─── Helpers ─────────────────────────────────────────────────────
    def _set_progress(self, val, max_val, text):
        self._progress['maximum'] = max_val
        self._progress['value'] = val
        self._status_label.config(text=text)
        try: self.root.update()
        except: pass

    # ─── Model ────────────────────────────────────────────────────────
    def _load_model(self):
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")],
            initialfile=os.path.basename(DEFAULT_MODEL),
        )
        if not path: return
        from models.FCmodel import FCmodel
        self.model = FCmodel(feature_dim=4, num_class=NUM_CLASSES, hidden_size=256, num_layers=4, seq_len=PREVIOUS_NUM)
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self._status_label.config(
            text=f"模型: {os.path.basename(path)} | {sum(p.numel() for p in self.model.parameters()):,} params ({self.device})"
        )
        self._scan_btn.config(state=tk.NORMAL)

    # ─── Scan ────────────────────────────────────────────────────────
    def _start_scan(self):
        if self.model is None:
            self._status_label.config(text="请先加载模型!")
            return

        self._scan_btn.config(state=tk.DISABLED)
        self._set_progress(0, 100, "加载股票列表...")
        self.root.update()

        # 构建股票列表 (过滤退市股 tradeStatus=0)
        stock_list_file = Path(STOCK_DATA_DIR) / "all_a_stocks_clean.csv"
        df_stocks = pd.read_csv(stock_list_file)
        df_stocks = df_stocks[df_stocks['tradeStatus'] == 1]
        df_stocks = df_stocks[~df_stocks['code_name'].str.contains('退', na=False)]
        all_codes = df_stocks['code'].tolist()
        all_names = df_stocks.get('code_name', ['?'] * len(df_stocks)).tolist()
        self._stocks = []
        for code, name in zip(all_codes, all_names):
            csv_path = Path(STOCK_DATA_DIR) / f"{code.replace('.', '_')}.csv"
            if csv_path.exists():
                self._stocks.append((code, str(name), str(csv_path)))

        # Phase 1: 增量刷新
        self._set_progress(0, 100, "增量刷新 stock_data (见终端)...")
        self.root.update()
        print("\n" + "="*50)
        print("[visual_scan] 开始增量刷新...")
        print("="*50)
        try:
            result = subprocess.run(
                [sys.executable, 'fetch_data/fetch_baostock.py', '--refresh'],
                timeout=1800, cwd=Path(__file__).parent,
            )
            self._status_label.config(text="stock_data 已刷新")
        except subprocess.TimeoutExpired:
            print("[visual_scan] 刷新超时, 继续使用现有数据")
            self._status_label.config(text="刷新超时, 使用现有数据")
        except Exception as e:
            print(f"[visual_scan] 刷新出错: {e}")
            self._status_label.config(text=f"刷新出错: {e}")
        print("="*50 + "\n")

        # Phase 2: 扫描最新预测
        total = len(self._stocks)
        self._set_progress(0, total, f"扫描 {total} 只股票最新预测...")
        self.root.update()

        self._latest_results = []
        bs = 256
        batch_features = []
        batch_meta = []

        for i, (code, name, csv_path) in enumerate(self._stocks):
            try:
                df = pd.read_csv(csv_path)
                for col in ['open','high','low','close','turn']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['open','high','low','close'])
                if len(df) < PREVIOUS_NUM:
                    continue
                df = df.reset_index(drop=True)
            except Exception:
                continue

            # 最新有效位置
            best_idx = len(df) - 1  # 默认最新一天
            if best_idx < PREVIOUS_NUM - 1:
                continue

            result = compute_features_and_label(df, best_idx)
            if result is None:
                continue

            features, _ = result
            batch_features.append(features)
            batch_meta.append({
                'code': code, 'name': name, 'csv_path': csv_path,
                'day_idx': best_idx, 'pred': -1, 'probs': None,
            })

            if len(batch_features) >= bs:
                self._infer_batch(batch_features, batch_meta)
                batch_features = []; batch_meta = []

            if i % 200 == 0:
                self._set_progress(i + 1, total, f"扫描 {i+1}/{total} (已识别{len(self._latest_results)})")

        if batch_features:
            self._infer_batch(batch_features, batch_meta)

        valid_results = [r for r in self._latest_results if r['pred'] >= 0]
        n_l0 = sum(1 for r in valid_results if r['pred'] == 0)
        n_l1 = sum(1 for r in valid_results if r['pred'] == 1)
        n_l2 = sum(1 for r in valid_results if r['pred'] == 2)
        n_l3 = sum(1 for r in valid_results if r['pred'] == 3)
        summary = f"完成! {len(valid_results)}/{total} 有效 | L0={n_l0} L1={n_l1} L2={n_l2} L3={n_l3}"
        print(f"\n[visual_scan] {summary}")
        self._set_progress(total, total, summary)

        for btn in self._filter_btns:
            btn.config(state=tk.NORMAL)
        self._scan_btn.config(state=tk.NORMAL)
        self._search_entry.config(state=tk.NORMAL)

        # 搜索索引
        self._lookup = {}
        for idx, r in enumerate(self._latest_results):
            key = r['code'].lower()
            self._lookup[key] = idx
            if '.' in r['code']:
                self._lookup[r['code'].split('.')[1].lower()] = idx
            self._lookup[r['name'].lower()] = idx

        self._set_filter("pred_l2")

    def _infer_batch(self, batch_features, batch_meta):
        feats = np.stack(batch_features, axis=0)
        with torch.no_grad():
            inputs = torch.from_numpy(feats).float().to(self.device)
            logits = self.model(inputs)
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        for meta, pred, prob in zip(batch_meta, preds, probs):
            meta['pred'] = int(pred)
            meta['probs'] = prob.astype(np.float32)
            self._latest_results.append(meta)

    # ─── Filter ──────────────────────────────────────────────────────
    def _set_filter(self, mode):
        self._filter_mode = mode
        self._filtered = []
        for i, r in enumerate(self._latest_results):
            p = r['pred']
            if mode == "all" or \
               (mode == "pred_l0" and p == 0) or \
               (mode == "pred_l1" and p == 1) or \
               (mode == "pred_l2" and p == 2) or \
               (mode == "pred_l3" and p == 3):
                self._filtered.append(i)

        self._filter_pos = 0
        total = len(self._filtered)
        if total > 0:
            fps = [self._latest_results[i]['pred'] for i in self._filtered]
            n_l0, n_l1, n_l2, n_l3 = fps.count(0), fps.count(1), fps.count(2), fps.count(3)
            self._filter_breakdown.config(
                text=f"{mode}: {total} stocks | L0={n_l0} L1={n_l1} L2={n_l2} L3={n_l3}")
            self._load_stock_at_pos()
        else:
            self._filter_breakdown.config(text=f"{mode}: 0 stocks")

    # ─── Search ──────────────────────────────────────────────────────
    def _search_stock(self, event):
        query = self._search_entry.get().strip().lower()
        if not query or not self._latest_results:
            return
        if query in self._lookup:
            target = self._lookup[query]
            try: self._filter_pos = self._filtered.index(target)
            except ValueError:
                self._set_filter("all")
                self._filter_pos = self._filtered.index(target)
            self._load_stock_at_pos()
        else:
            matches = [i for i, r in enumerate(self._latest_results)
                       if query in f"{r['code']} {r['name']}".lower()]
            if matches:
                self._set_filter("all")
                self._filter_pos = self._filtered.index(matches[0])
                self._load_stock_at_pos()

    # ─── Navigation ──────────────────────────────────────────────────
    def _load_stock_at_pos(self):
        if not self._filtered: return
        self._stock_index = self._filtered[self._filter_pos]
        r = self._latest_results[self._stock_index]
        self._day_index = r['day_idx']
        self._load_and_draw(r['csv_path'], r['code'], r['name'])

    def _prev_stock(self):
        if not self._filtered or self._filter_pos <= 0: return
        self._filter_pos -= 1
        self._load_stock_at_pos()

    def _next_stock(self):
        if not self._filtered or self._filter_pos >= len(self._filtered) - 1: return
        self._filter_pos += 1
        self._load_stock_at_pos()

    def _prev_day(self):
        r = self._latest_results[self._stock_index]
        if self._day_index > PREVIOUS_NUM - 1:
            self._day_index -= 1
            self._load_and_draw(r['csv_path'], r['code'], r['name'])

    def _next_day(self):
        r = self._latest_results[self._stock_index]
        try:
            df = pd.read_csv(r['csv_path'])
            max_day = len(df) - 1  # 允许到最后一天 (0天未来)
        except Exception:
            max_day = self._day_index
        if self._day_index < max_day:
            self._day_index += 1
            self._load_and_draw(r['csv_path'], r['code'], r['name'])

    def _load_and_draw(self, csv_path, code, name):
        try:
            if csv_path != self._cache_code:
                df = pd.read_csv(csv_path)
                for col in ['open','high','low','close','turn']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                df = df.dropna(subset=['open','high','low','close'])
                df = df.reset_index(drop=True)
                self._df_cache = df
                self._cache_code = csv_path
            else:
                df = self._df_cache

            result = compute_features_and_label(df, self._day_index)
            if result is None:
                self._status_label.config(text=f"数据不足 @ day {self._day_index}")
                return

            features, raw_data = result

            with torch.no_grad():
                inputs = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
                logits = self.model(inputs)
                probs = F.softmax(logits, dim=-1)[0].cpu().numpy()
                pred = int(np.argmax(probs))

            self._draw_impl(raw_data, pred, probs, code, name)

        except Exception as e:
            print(f"ERROR: {e}")
            import traceback; traceback.print_exc()

    # ─── Drawing ─────────────────────────────────────────────────────
    def _draw_impl(self, raw, pred, probs, code, name):
        mean = raw['mean_adj']
        change = raw['change_rate']
        hl = raw['high_low_ratio']
        change_delta = raw['change_delta']
        date = raw['date']
        true_label = raw['true_label']
        ret_pct = raw['return_pct']
        future_days = raw['future_days']
        true_days_info = raw['true_days_info']
        hist_dates = raw['hist_dates']

        STYLE = dict(facecolor="#1e1e1e", tick_params=dict(colors="#aaaaaa"),
                     grid=dict(alpha=0.15, color="#888888"))

        # ── Top: Price ──
        ax = self.ax_price
        ax.clear()
        ax.set_facecolor(STYLE["facecolor"])

        # 拼接历史+未来数据
        fut_mean = raw.get('fut_mean', np.array([], dtype=np.float32))
        fut_dates = raw.get('fut_dates', [])
        all_price = np.concatenate([mean, fut_mean]) if len(fut_mean) > 0 else mean
        all_dates = hist_dates + fut_dates
        n_hist = len(mean)
        n_fut = len(fut_mean)
        n_all = len(all_price)

        days = np.arange(n_all)
        # 历史部分蓝色, 未来(target)部分橙色
        ax.plot(days[:n_hist], mean, color="#3498db", linewidth=1.3, marker="o", markersize=3)
        if n_fut > 0:
            ax.plot(days[n_hist-1:], all_price[n_hist-1:], color="#f39c12", linewidth=1.6, marker="D", markersize=5)
            ax.axvspan(n_hist - 1, n_all - 1, alpha=0.10, color="#f39c12")
            ax.scatter([n_all - 1], [all_price[-1]],
                       color="#2ecc71" if ret_pct >= 0 else "#e74c3c", s=55, zorder=5, marker="D",
                       label="target")

        ax.axvline(x=PREVIOUS_NUM - 1, color="#f39c12", linestyle="--", linewidth=1.2)
        ax.scatter([PREVIOUS_NUM - 1], [mean[PREVIOUS_NUM - 1]], color="#e74c3c", s=40, zorder=5,
                   label="history end")

        pred_str = LABEL_NAMES.get(pred, f"Pred_{pred}")
        pred_color = LABEL_COLORS.get(pred, "#ecf0f1")

        if true_label >= 0:
            true_str = LABEL_NAMES[true_label]
            match = "CORRECT" if pred == true_label else f"WRONG (True={true_str})"
            match_color = "#2ecc71" if pred == true_label else "#e74c3c"
        else:
            match = "Forward: T+3 预测"
            match_color = "#f39c12"

        ax.set_title(
            f"{code} ({name})  |  Date: {date}  |  Ret: {ret_pct*100:+.2f}%  |  "
            f"Pred: {pred_str}  |  {match}",
            color=match_color, fontsize=11, fontweight="bold",
        )
        ax.set_ylabel("Price (OHLC均价)", color="#bbbbbb")
        ax.legend(loc="upper left", fontsize=8, facecolor="#333333", edgecolor="#555555",
                  labelcolor="#cccccc")
        ax.grid(True, **STYLE["grid"])
        ax.tick_params(**STYLE["tick_params"])
        for sp in ax.spines.values():
            sp.set_color("#555555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # x轴每天标注
        tick_pos = list(range(0, n_all))
        tick_labels = [all_dates[i] if i < len(all_dates) else '' for i in tick_pos]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_labels, rotation=90, ha='center', fontsize=6, color='#aaaaaa')

        # ── Middle: Turnover + Volatility ──
        ax = self.ax_turnover
        ax.clear()
        if hasattr(self, '_twinx') and self._twinx in self.fig.axes:
            self.fig.delaxes(self._twinx)
        ax.set_facecolor(STYLE["facecolor"])
        x = np.arange(PREVIOUS_NUM)
        ax.bar(x - 0.15, change[:PREVIOUS_NUM], width=0.3,
               color="#9b59b6", alpha=0.8, label="change_rate")
        ax2_ = ax.twinx()
        self._twinx = ax2_
        ax2_.bar(x + 0.15, hl[:PREVIOUS_NUM], width=0.3,
                 color="#1abc9c", alpha=0.8, label="high_low_ratio")
        ax.set_ylabel("Turnover", color="#9b59b6")
        ax2_.set_ylabel("Volatility", color="#1abc9c")
        ax.set_title("Turnover & Volatility", color="#cccccc", fontsize=10)
        ax.grid(True, axis="y", **STYLE["grid"])
        ax.tick_params(**STYLE["tick_params"])
        ax2_.tick_params(**STYLE["tick_params"])
        # 日期
        h_tick_pos = list(range(0, n_hist))
        h_tick_labels = hist_dates[:n_hist]
        ax.set_xticks(h_tick_pos)
        ax.set_xticklabels(h_tick_labels, rotation=90, ha='center', fontsize=6, color='#aaaaaa')
        for sp in list(ax.spines.values()) + list(ax2_.spines.values()):
            sp.set_color("#555555")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2_.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8,
                  facecolor="#333333", edgecolor="#555555", labelcolor="#cccccc")

        # ── Bottom: Change delta + 日期标签 ──
        ax = self.ax_delta
        ax.clear()
        ax.set_facecolor(STYLE["facecolor"])
        x = np.arange(PREVIOUS_NUM)
        delta_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in change_delta[:PREVIOUS_NUM]]
        ax.bar(x, change_delta[:PREVIOUS_NUM], color=delta_colors, width=0.6, alpha=0.85)
        ax.axhline(y=0, color="#666666", linewidth=0.8)
        ax.set_ylabel("Δ Turnover", color="#bbbbbb")
        ax.set_title("Turnover Change Rate", color="#cccccc", fontsize=10)
        ax.grid(True, axis="y", **STYLE["grid"])
        ax.tick_params(**STYLE["tick_params"])
        for sp in ax.spines.values():
            sp.set_color("#555555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # 日期 (每天)
        ax.set_xticks(h_tick_pos)
        ax.set_xticklabels(h_tick_labels, rotation=90, ha='center', fontsize=6, color='#aaaaaa')

        self.canvas.draw()

        # Nav
        display_name = f"{code} ({name})" if name and name != '?' else code
        self._code_display.config(text=f"  {display_name}  ")
        self._date_label.config(text=f"  {date}  ")

        true_display = f"True: {LABEL_NAMES.get(true_label, '?')}{true_days_info}" if true_label >= 0 else "True: 无"
        self._pred_display.config(text=f"Pred: {pred_str}  |  {true_display}", fg=pred_color)

        self._stock_count_label.config(text=f"{self._filter_pos+1}/{len(self._filtered)}")

        for i in range(NUM_CLASSES):
            marker = "◀" if pred == i else "  "
            self._prob_labels[i].config(
                text=f"{marker} {LABEL_NAMES[i].split()[0]}: {probs[i]*100:.1f}%",
                fg=LABEL_COLORS[i],
            )


def main():
    root = tk.Tk()
    _app = StockScanViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
