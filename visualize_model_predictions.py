#!/usr/bin/env python3
"""
模型推理可视化 — 加载训练好的模型, 在测试集上推理, 按预测类别浏览样本

用法:
  python visualize_model_predictions.py
"""
import os, sys
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

import torch

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec

PREVIOUS_NUM = 30
PREDICT_NUM = 3
PP = PREVIOUS_NUM + PREDICT_NUM  # 33
INPUT_DIM = PREVIOUS_NUM * 4     # 120

LABEL_NAMES = {0: "L0 (跌 >10%)", 1: "L1 (震荡)", 2: "L2 (涨 >10%)"}
LABEL_COLORS = {0: "#e74c3c", 1: "#95a5a6", 2: "#2ecc71"}


def _cal_pct(d):
    r = np.zeros_like(d)
    with np.errstate(divide="ignore", invalid="ignore"):
        r[1:] = (d[1:] - d[:-1]) / d[:-1]
        r = np.where(np.isfinite(r), r, 0)
    return r


def compute_label(mean_slice):
    pct = (mean_slice[-1] - mean_slice[PREVIOUS_NUM - 1]) / mean_slice[PREVIOUS_NUM - 1]
    if pct < -0.1:
        return 0, pct
    elif pct > 0.1:
        return 2, pct
    return 1, pct


class ModelPredictionViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Model Prediction Viewer - 模型推理可视化")
        self.root.geometry("1400x950")
        self.root.configure(bg="#2b2b2b")

        self._test_raw = None       # (test_n, 132) only test set
        self._test_codes = None     # (test_n,) stock codes
        self._predictions = None
        self._true_labels = None
        self._all_features = None   # precomputed (test_n, 120)

        self._filtered_indices = []
        self._current_filtered_pos = 0
        self._filter_mode = "all"
        self._correctness_mode = "all_results"

        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self._build_ui()

    def _build_ui(self):
        # ── Top bar ──
        top = tk.Frame(self.root, bg="#2b2b2b")
        top.pack(fill=tk.X, padx=8, pady=(8, 0))

        tk.Button(top, text="Load Dataset Dir", command=self._load_dataset,
                  bg="#3a3a3a", fg="#ecf0f1", bd=1, padx=10, pady=2,
                  activebackground="#555555").pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Load Model (.pt)", command=self._load_model,
                  bg="#3a3a3a", fg="#ecf0f1", bd=1, padx=10, pady=2,
                  activebackground="#555555").pack(side=tk.LEFT, padx=4)
        self._infer_btn = tk.Button(top, text="Run Inference", command=self._run_inference,
                  bg="#1a6e3e", fg="#ecf0f1", bd=1, padx=10, pady=2,
                  activebackground="#2a8e5e", state=tk.DISABLED)
        self._infer_btn.pack(side=tk.LEFT, padx=4)

        self._status_label = tk.Label(top, text="请加载数据集和模型", bg="#2b2b2b",
                                       fg="#7f8c8d", font=("Arial", 10))
        self._status_label.pack(side=tk.LEFT, padx=16)

        # ── Filter bar ──
        filt = tk.Frame(self.root, bg="#2b2b2b")
        filt.pack(fill=tk.X, padx=8, pady=4)

        b_opts = dict(bg="#3a3a3a", fg="#ecf0f1", bd=1, padx=8, pady=2,
                      activebackground="#555555", activeforeground="#ffffff")

        self._filter_btns = []
        def _add_filter_btn(text, mode, bg_color, fg_color):
            btn = tk.Button(filt, text=text, command=lambda m=mode: self._set_filter(m),
                          bg=bg_color, fg=fg_color, bd=1, padx=8, pady=2,
                          activebackground="#555555", activeforeground="#ffffff",
                          state=tk.DISABLED)
            btn.pack(side=tk.LEFT, padx=2)
            self._filter_btns.append(btn)

        _add_filter_btn("All", "all", "#3a3a3a", "#ecf0f1")
        _add_filter_btn("Pred L0", "pred_l0", "#5a2020", "#e74c3c")
        _add_filter_btn("Pred L1", "pred_l1", "#3a3a3a", "#95a5a6")
        _add_filter_btn("Pred L2", "pred_l2", "#204020", "#2ecc71")

        tk.Label(filt, text="  |  ", bg="#2b2b2b", fg="#555555").pack(side=tk.LEFT, padx=4)

        self._correctness_btns = []
        def _add_correctness_btn(text, mode, bg_color, fg_color):
            btn = tk.Button(filt, text=text, command=lambda m=mode: self._set_correctness(m),
                          bg=bg_color, fg=fg_color, bd=1, padx=8, pady=2,
                          activebackground="#555555", activeforeground="#ffffff",
                          state=tk.DISABLED)
            btn.pack(side=tk.LEFT, padx=2)
            self._correctness_btns.append(btn)

        _add_correctness_btn("All Results", "all_results", "#3a3a3a", "#ecf0f1")
        _add_correctness_btn("Correct Only", "correct", "#204020", "#2ecc71")
        _add_correctness_btn("Wrong Only", "wrong", "#5a2020", "#e74c3c")

        self._filter_breakdown = tk.Label(filt, text="", bg="#2b2b2b", fg="#7f8c8d",
                                           font=("Consolas", 10))
        self._filter_breakdown.pack(side=tk.LEFT, padx=20)

        # ── Navigation ──
        nav = tk.Frame(self.root, bg="#2b2b2b")
        nav.pack(fill=tk.X, padx=8, pady=4)

        tk.Button(nav, text="◀  Prev (←)", command=self._prev_sample, **b_opts).pack(side=tk.LEFT, padx=3)

        self._idx_entry = tk.Entry(nav, width=8, justify="center", bg="#3a3a3a", fg="#ecf0f1",
                                   insertbackground="#ecf0f1", bd=1, font=("Consolas", 11))
        self._idx_entry.pack(side=tk.LEFT, padx=6)
        self._idx_entry.bind("<Return>", self._jump_to)

        self._count_label = tk.Label(nav, text="/ 0", bg="#2b2b2b", fg="#bdc3c7",
                                      font=("Consolas", 11))
        self._count_label.pack(side=tk.LEFT)

        tk.Button(nav, text="Next (→)  ▶", command=self._next_sample, **b_opts).pack(side=tk.LEFT, padx=3)

        self._pred_display = tk.Label(nav, text="", font=("Arial", 16, "bold"),
                                       bg="#2b2b2b", fg="#ecf0f1")
        self._pred_display.pack(side=tk.LEFT, padx=30)

        self._code_display = tk.Label(nav, text="", font=("Consolas", 12, "bold"),
                                       bg="#2b2b2b", fg="#f39c12")
        self._code_display.pack(side=tk.LEFT, padx=10)

        self._return_display = tk.Label(nav, text="", font=("Consolas", 12),
                                         bg="#2b2b2b", fg="#bdc3c7")
        self._return_display.pack(side=tk.LEFT, padx=10)

        # ── Matplotlib ──
        self.fig = Figure(figsize=(13, 9), facecolor="#2b2b2b")
        gs = GridSpec(4, 1, figure=self.fig, height_ratios=[2.5, 1, 1.5, 1.5],
                      top=0.97, bottom=0.05, left=0.08, right=0.93, hspace=0.08)
        self.ax1 = self.fig.add_subplot(gs[0])
        self.ax2 = self.fig.add_subplot(gs[1], sharex=self.ax1)
        self.ax3 = self.fig.add_subplot(gs[2])
        self.ax4 = self.fig.add_subplot(gs[3])

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#aaaaaa")
            ax.spines["bottom"].set_color("#555555")
            ax.spines["left"].set_color("#555555")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Bottom: progress bar + status ──
        bottom = tk.Frame(self.root, bg="#1e1e1e")
        bottom.pack(fill=tk.X, padx=8, pady=(0, 4))

        self._progress = ttk.Progressbar(bottom, orient=tk.HORIZONTAL, length=400, mode='determinate')
        self._progress.pack(side=tk.RIGHT, padx=8)

        tk.Label(bottom,
                 text="← → 切换  |  Filter: All/PredL0/PredL1/PredL2  |  Correct Only / Wrong Only  |  输入序号回车跳转",
                 bg="#1e1e1e", fg="#555555", font=("Arial", 8)).pack(side=tk.LEFT)

        # Keyboard
        self.root.bind("<Left>", lambda e: self._prev_sample())
        self.root.bind("<Right>", lambda e: self._next_sample())
        self.root.bind("<Prior>", lambda e: self._prev_sample())
        self.root.bind("<Next>", lambda e: self._next_sample())

    # ─── Progress helper ───────────────────────────────────────────

    def _check_ready(self):
        """当数据和模型都加载后, 启用推理按钮"""
        if self._test_raw is not None and self.model is not None:
            self._infer_btn.config(state=tk.NORMAL, text="Run Inference")
            self._status_label.config(text="数据和模型已就绪, 点击 Run Inference")

    def _set_progress(self, val, max_val, text):
        self._progress['maximum'] = max_val
        self._progress['value'] = val
        self._status_label.config(text=text)
        try:
            self.root.update()
        except Exception:
            pass

    # ─── Data Loading (只加载测试集) ────────────────────────────────

    def _load_dataset(self):
        d = filedialog.askdirectory(title="选择数据集目录 (baostock_dataset_30)")
        if not d:
            return

        files = [os.path.join(d, f) for f in os.listdir(d) if f.endswith('.npy')]
        if not files:
            self._status_label.config(text="目录中没有 .npy 文件")
            return

        self._set_progress(0, len(files), "扫描数据文件...")

        # Step 1: mmap files to get shapes, build cumulative row counts
        file_sizes = []
        code_lists = []  # per-file code list
        for i, f in enumerate(files):
            data = np.load(f, mmap_mode='r')
            file_sizes.append(data.shape[0])
            codes_file = f.replace('.npy', '_codes.txt')
            if os.path.exists(codes_file):
                with open(codes_file) as cf:
                    code_lists.append(np.array([line.strip() for line in cf]))
            else:
                code_lists.append(np.array(['?'] * data.shape[0]))
            self._set_progress(i + 1, len(files), f"扫描 {i+1}/{len(files)}: {os.path.basename(f)}")

        total = sum(file_sizes)
        cumsum = np.cumsum([0] + file_sizes)  # cumsum[i] = start of file i

        # Step 2: 复现训练的 shuffle (seed=42), 用 np.random.shuffle 而非 permutation
        # 因为训练脚本用的是 np.random.shuffle(all_data), 必须用同样的操作才能得到相同的测试集
        self._set_progress(0, 1, "生成 shuffle 索引...")
        perm = np.arange(total)
        np.random.seed(42)
        np.random.shuffle(perm)
        split = int(0.8 * total)
        test_perm = perm[split:]  # 训练时用 perm[:split] 做训练集, perm[split:] 做测试集
        test_n = len(test_perm)

        # Step 3: for each test position, map to file + row
        self._set_progress(0, 1, f"映射测试集索引 ({test_n:,} samples)...")
        file_idx = np.searchsorted(cumsum, test_perm, side='right') - 1
        file_idx = np.clip(file_idx, 0, len(files) - 1)
        local_row = test_perm - cumsum[file_idx]

        # Step 4: group by file, load only needed rows
        self._test_raw = np.zeros((test_n, 132), dtype=np.float32)
        self._test_codes = np.empty(test_n, dtype=object)

        unique_files = np.unique(file_idx)
        self._set_progress(0, len(unique_files), "加载测试集数据...")

        for prog_i, fi in enumerate(unique_files):
            mask = file_idx == fi
            rows = local_row[mask]
            test_positions = np.where(mask)[0]

            data = np.load(files[fi], mmap_mode='r')
            self._test_raw[test_positions] = data[rows]
            self._test_codes[test_positions] = code_lists[fi][rows]

            self._set_progress(prog_i + 1, len(unique_files),
                              f"加载文件 {fi+1}/{len(files)} ({len(rows):,} rows)")

        self._set_progress(0, 0,
            f"数据集: {os.path.basename(d)} | 测试集: {test_n:,} 样本 (~{self._test_raw.nbytes/1024**3:.1f}GB)"
        )
        self._check_ready()

    # ─── Model Loading ────────────────────────────────────────────

    def _load_model(self):
        path = filedialog.askopenfilename(
            title="选择模型文件",
            filetypes=[("PyTorch files", "*.pt"), ("All files", "*.*")],
        )
        if not path:
            return

        from models.fcmodel import FCmodel
        self.model = FCmodel(INPUT_DIM, num_class=3, hidden_size=256, num_layers=4)
        state = torch.load(path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        self._status_label.config(
            text=f"模型: {os.path.basename(path)} | {sum(p.numel() for p in self.model.parameters()):,} params ({self.device})"
        )
        self._check_ready()

    # ─── Inference ────────────────────────────────────────────────

    def _run_inference(self):
        if self.model is None or self._test_raw is None:
            self._status_label.config(text="请先加载数据集和模型!")
            return

        test_n = len(self._test_raw)
        self._all_features = np.zeros((test_n, INPUT_DIM), dtype=np.float32)
        self._true_labels = np.zeros(test_n, dtype=np.int64)
        self._predictions = np.zeros(test_n, dtype=np.int64)

        bs = 4096
        total_batches = (test_n + bs - 1) // bs

        for batch_idx, start in enumerate(range(0, test_n, bs)):
            end = min(start + bs, test_n)
            self._set_progress(batch_idx, total_batches,
                              f"推理 {batch_idx+1}/{total_batches} ({start:,}/{test_n:,})")
            batch_raw = self._test_raw[start:end]
            batch_n = end - start

            # Extract features (vectorized)
            mean = batch_raw[:, :PP]                     # (B, 33)
            change = batch_raw[:, PP:2*PP]               # (B, 33)
            hl = batch_raw[:, 2*PP:3*PP]                 # (B, 33)
            change_delta = batch_raw[:, 3*PP:4*PP]       # (B, 33)

            price_pct = np.zeros((batch_n, PREVIOUS_NUM), dtype=np.float32)
            with np.errstate(divide='ignore', invalid='ignore'):
                diff = mean[:, 1:PREVIOUS_NUM] - mean[:, :PREVIOUS_NUM-1]
                pct = diff / mean[:, :PREVIOUS_NUM-1]
                pct = np.where(np.isfinite(pct), pct, 0)
                price_pct[:, 1:] = pct

            # Label
            pct_chg = (mean[:, -1] - mean[:, PREVIOUS_NUM - 1]) / mean[:, PREVIOUS_NUM - 1]
            labels = np.where(pct_chg < -0.1, 0, np.where(pct_chg > 0.1, 2, 1))

            features = np.concatenate([
                price_pct,
                change[:, :PREVIOUS_NUM],
                hl[:, :PREVIOUS_NUM],
                change_delta[:, :PREVIOUS_NUM],
            ], axis=1)

            self._all_features[start:end] = features
            self._true_labels[start:end] = labels

            with torch.no_grad():
                inputs = torch.from_numpy(features).float().to(self.device)
                outputs = self.model(inputs)
                _, pred = torch.max(outputs, 1)
                self._predictions[start:end] = pred.cpu().numpy()

            if batch_idx % 5 == 0:
                try:
                    self.root.update()
                except Exception:
                    pass

        # Confusion summary
        cm = np.zeros((3, 3), dtype=np.int64)
        for p, t in zip(self._predictions, self._true_labels):
            cm[p][t] += 1

        summary = (f"推理完成! Acc={100*cm.trace()/cm.sum():.1f}% | "
                   f"Pred_L0={cm[0].sum():,} Pred_L1={cm[1].sum():,} Pred_L2={cm[2].sum():,}")
        self._status_label.config(text=summary)
        self._set_progress(0, 0, summary)

        # Enable filter + correctness buttons
        for btn in self._filter_btns:
            btn.config(state=tk.NORMAL)
        for btn in self._correctness_btns:
            btn.config(state=tk.NORMAL)

        self._set_filter("pred_l2")

    # ─── Filtering ─────────────────────────────────────────────────

    def _set_filter(self, mode):
        if self._predictions is None:
            print("DEBUG: _predictions is None, returning")
            self._filter_breakdown.config(text="请先运行推理!")
            return

        self._filter_mode = mode
        self._apply_masks()

    def _set_correctness(self, mode):
        self._correctness_mode = mode
        self._apply_masks()

    def _apply_masks(self):
        p = self._predictions
        t = self._true_labels

        # Pred-class filter
        mode = self._filter_mode
        if mode == "all":
            mask = np.ones(len(p), dtype=bool)
        elif mode == "pred_l0":
            mask = p == 0
        elif mode == "pred_l1":
            mask = p == 1
        elif mode == "pred_l2":
            mask = p == 2
        else:
            mask = np.ones(len(p), dtype=bool)

        # Correctness sub-filter
        cm = self._correctness_mode
        if cm == "correct":
            mask = mask & (p == t)
        elif cm == "wrong":
            mask = mask & (p != t)

        self._filtered_indices = np.where(mask)[0].tolist()
        self._current_filtered_pos = 0

        if len(self._filtered_indices) > 0:
            filt_t = t[self._filtered_indices]
            filt_p = p[self._filtered_indices]
            n_l0 = int((filt_t == 0).sum())
            n_l1 = int((filt_t == 1).sum())
            n_l2 = int((filt_t == 2).sum())
            n_correct = int((filt_p == filt_t).sum())
            total = len(self._filtered_indices)
            self._filter_breakdown.config(
                text=f"Filter={mode} | {cm}: {total} samples | "
                     f"True L0={n_l0} ({100*n_l0/total:.0f}%), "
                     f"True L1={n_l1} ({100*n_l1/total:.0f}%), "
                     f"True L2={n_l2} ({100*n_l2/total:.0f}%) | "
                     f"Correct={n_correct}/{total} ({100*n_correct/total:.0f}%)"
            )
        else:
            self._filter_breakdown.config(text=f"Filter={mode} | {cm}: 0 samples")

        self._draw()

    # ─── Navigation ────────────────────────────────────────────────

    def _prev_sample(self):
        if len(self._filtered_indices) == 0:
            return
        if self._current_filtered_pos > 0:
            self._current_filtered_pos -= 1
        self._draw()

    def _next_sample(self):
        if len(self._filtered_indices) == 0:
            return
        if self._current_filtered_pos < len(self._filtered_indices) - 1:
            self._current_filtered_pos += 1
        self._draw()

    def _jump_to(self, event):
        try:
            idx = int(self._idx_entry.get())
            if 0 <= idx < len(self._filtered_indices):
                self._current_filtered_pos = idx
                self._draw()
        except ValueError:
            pass

    # ─── Drawing ───────────────────────────────────────────────────

    def _draw(self):
        if len(self._filtered_indices) == 0:
            self._count_label.config(text="/ 0")
            return

        try:
            self._draw_impl()
        except Exception as e:
            print(f"ERROR in _draw: {e}")
            import traceback
            traceback.print_exc()

    def _draw_impl(self):
        test_idx = self._filtered_indices[self._current_filtered_pos]
        raw = self._test_raw[test_idx]
        stock_code = self._test_codes[test_idx] if self._test_codes is not None else "?"

        mean = raw[:PP]
        change = raw[PP:2 * PP]
        hl = raw[2 * PP:3 * PP]
        change_delta = raw[3 * PP:4 * PP]
        price_pct = _cal_pct(mean[:PREVIOUS_NUM])

        true_label, ret_pct = compute_label(mean)
        pred_label = int(self._predictions[test_idx]) if self._predictions is not None else -1

        STYLE = dict(facecolor="#1e1e1e", tick_params=dict(colors="#aaaaaa"),
                     grid=dict(alpha=0.15, color="#888888"))

        # ── Ax1: Price ──
        ax = self.ax1
        ax.clear()
        ax.set_facecolor(STYLE["facecolor"])
        days = np.arange(PP)
        ax.plot(days, mean, color="#3498db", linewidth=1.3, marker="o", markersize=3)
        ax.axvspan(PREVIOUS_NUM - 1, PP - 1, alpha=0.10, color="#f39c12")
        ax.axvline(x=PREVIOUS_NUM - 1, color="#f39c12", linestyle="--", linewidth=1.2)
        ax.scatter([PREVIOUS_NUM - 1], [mean[PREVIOUS_NUM - 1]], color="#e74c3c", s=40, zorder=5,
                   label="day 29 (history end)")
        ax.scatter([PP - 1], [mean[-1]],
                   color="#2ecc71" if ret_pct >= 0 else "#e74c3c", s=55, zorder=5, marker="D",
                   label="day 32 (target)")

        pred_str = LABEL_NAMES.get(pred_label, f"Pred_{pred_label}")
        true_str = LABEL_NAMES[true_label]
        if pred_label == true_label:
            match_str = "CORRECT"
            match_color = "#2ecc71"
        else:
            match_str = f"WRONG (True={true_str})"
            match_color = "#e74c3c"

        ax.set_title(
            f"[{stock_code}]  Test idx={test_idx}  |  3d return: {ret_pct * 100:+.2f}%  |  "
            f"Pred: {pred_str}  |  {match_str}",
            color=match_color, fontsize=11, fontweight="bold",
        )
        ax.set_ylabel("Price", color="#bbbbbb")
        ax.legend(loc="upper left", fontsize=8, facecolor="#333333", edgecolor="#555555",
                  labelcolor="#cccccc")
        ax.grid(True, **STYLE["grid"])
        ax.tick_params(**STYLE["tick_params"])
        plt.setp(ax.get_xticklabels(), visible=False)
        for sp in ax.spines.values():
            sp.set_color("#555555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ── Ax2: Price pct_change ──
        ax = self.ax2
        ax.clear()
        ax.set_facecolor(STYLE["facecolor"])
        x = np.arange(PREVIOUS_NUM)
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in price_pct]
        ax.bar(x, price_pct, color=colors, width=0.7, alpha=0.85)
        ax.axhline(y=0, color="#666666", linewidth=0.8)
        ax.set_ylabel("Δ%", color="#bbbbbb", fontsize=9)
        ax.set_xlabel("Day (price 折线图每个相邻点的涨跌幅度)", color="#888888", fontsize=8)
        ax.grid(True, axis="y", **STYLE["grid"])
        ax.tick_params(**STYLE["tick_params"])
        for sp in ax.spines.values():
            sp.set_color("#555555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(-0.5, PP - 0.5)

        # ── Ax3: Turnover + Volatility ──
        ax = self.ax3
        ax.clear()
        if hasattr(self, '_ax3_twinx') and self._ax3_twinx in self.fig.axes:
            self.fig.delaxes(self._ax3_twinx)
        ax.set_facecolor(STYLE["facecolor"])
        x = np.arange(PREVIOUS_NUM)
        ax.bar(x - 0.15, change[:PREVIOUS_NUM], width=0.3,
               color="#9b59b6", alpha=0.8, label="change_rate (换手率)")
        ax2_ = ax.twinx()
        self._ax3_twinx = ax2_
        ax2_.bar(x + 0.15, hl[:PREVIOUS_NUM], width=0.3,
                 color="#1abc9c", alpha=0.8, label="high_low_ratio (波动率)")
        ax.set_xlabel("Day (0-29 = 30-day history)", color="#888888", fontsize=8)
        ax.set_ylabel("Turnover", color="#9b59b6")
        ax2_.set_ylabel("Volatility", color="#1abc9c")
        ax.set_title("Turnover & Volatility", color="#cccccc", fontsize=10)
        ax.grid(True, axis="y", **STYLE["grid"])
        ax.tick_params(**STYLE["tick_params"])
        ax2_.tick_params(**STYLE["tick_params"])
        for sp in list(ax.spines.values()) + list(ax2_.spines.values()):
            sp.set_color("#555555")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2_.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8,
                  facecolor="#333333", edgecolor="#555555", labelcolor="#cccccc")

        # ── Ax4: Change rate delta ──
        ax = self.ax4
        ax.clear()
        ax.set_facecolor(STYLE["facecolor"])
        x = np.arange(PREVIOUS_NUM)
        delta_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in change_delta[:PREVIOUS_NUM]]
        ax.bar(x, change_delta[:PREVIOUS_NUM], color=delta_colors, width=0.6, alpha=0.85)
        ax.axhline(y=0, color="#666666", linewidth=0.8)
        ax.set_xlabel("Day (0-29 = 30-day history)", color="#888888", fontsize=8)
        ax.set_ylabel("Δ Turnover", color="#bbbbbb")
        ax.set_title("Turnover Change Rate (换手率变化量)", color="#cccccc", fontsize=10)
        ax.grid(True, axis="y", **STYLE["grid"])
        ax.tick_params(**STYLE["tick_params"])
        for sp in ax.spines.values():
            sp.set_color("#555555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        self.canvas.draw()

        # Update nav
        self._idx_entry.delete(0, tk.END)
        self._idx_entry.insert(0, str(self._current_filtered_pos))
        self._count_label.config(text=f"/ {len(self._filtered_indices) - 1}")

        self._pred_display.config(
            text=f"Pred: {pred_str}  |  True: {true_str}",
            fg=LABEL_COLORS.get(pred_label, "#ecf0f1")
        )
        self._code_display.config(text=f"  {stock_code}  ")
        self._return_display.config(
            text=f"return: {ret_pct * 100:+.2f}%",
            fg="#2ecc71" if ret_pct >= 0 else "#e74c3c",
        )


def main():
    root = tk.Tk()
    _app = ModelPredictionViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
