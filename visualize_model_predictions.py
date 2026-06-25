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
FEATURE_DIM = 8
INPUT_DIM = PREVIOUS_NUM * FEATURE_DIM  # 240

LABEL_NAMES = {0: "L0 (跌>10%)", 1: "L1 (-10%~0)", 2: "L2 (0~10%)", 3: "L3 (涨>10%)"}
LABEL_COLORS = {0: "#e74c3c", 1: "#e67e22", 2: "#3498db", 3: "#2ecc71"}
NUM_CLASSES = 4


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
        _add_filter_btn("Pred L1", "pred_l1", "#4a2000", "#e67e22")
        _add_filter_btn("Pred L2", "pred_l2", "#002040", "#3498db")
        _add_filter_btn("Pred L3", "pred_l3", "#204020", "#2ecc71")

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
        gs = GridSpec(3, 1, figure=self.fig, height_ratios=[3, 2, 2],
                      top=0.97, bottom=0.05, left=0.08, right=0.93, hspace=0.08)
        self.ax_price = self.fig.add_subplot(gs[0])
        self.ax_turnover = self.fig.add_subplot(gs[1])
        self.ax_delta = self.fig.add_subplot(gs[2])

        for ax in [self.ax_price, self.ax_turnover, self.ax_delta]:
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
        self.model = FCmodel(feature_dim=FEATURE_DIM, num_class=NUM_CLASSES, hidden_size=256, num_layers=4, seq_len=PREVIOUS_NUM)
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

            # Extract 7 features
            pct     = batch_raw[:, 0*PP:1*PP]; ampl   = batch_raw[:, 1*PP:2*PP]
            daychg  = batch_raw[:, 2*PP:3*PP]
            chg     = batch_raw[:, 3*PP:4*PP]; cd_    = batch_raw[:, 4*PP:5*PP]
            bb_pos  = batch_raw[:, 5*PP:6*PP]; bb_w   = batch_raw[:, 6*PP:7*PP]
            rsi     = batch_raw[:, 7*PP:8*PP]

            pct_chg = (1+pct[:, PREVIOUS_NUM])*(1+pct[:, PREVIOUS_NUM+1])*(1+pct[:, PREVIOUS_NUM+2]) - 1
            labels = np.zeros(batch_n, dtype=np.int64)
            labels = np.where(pct_chg < -0.1, 0,
                     np.where(pct_chg < 0, 1,
                     np.where(pct_chg <= 0.1, 2, 3)))

            features = np.concatenate([
                pct[:, :PREVIOUS_NUM], ampl[:, :PREVIOUS_NUM], daychg[:, :PREVIOUS_NUM],
                chg[:, :PREVIOUS_NUM], cd_[:, :PREVIOUS_NUM],
                bb_pos[:, :PREVIOUS_NUM], bb_w[:, :PREVIOUS_NUM],
                rsi[:, :PREVIOUS_NUM],
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
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
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
        elif mode == "pred_l3":
            mask = p == 3
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

        pct     = raw[0*PP:1*PP]; ampl   = raw[1*PP:2*PP]
        daychg  = raw[2*PP:3*PP]
        chg     = raw[3*PP:4*PP]; cd_    = raw[4*PP:5*PP]

        pct_chg = (1+pct[PREVIOUS_NUM])*(1+pct[PREVIOUS_NUM+1])*(1+pct[PREVIOUS_NUM+2]) - 1
        if pct_chg < -0.1: true_label = 0
        elif pct_chg < 0: true_label = 1
        elif pct_chg <= 0.1: true_label = 2
        else: true_label = 3
        ret_pct = pct_chg
        pred_label = int(self._predictions[test_idx]) if self._predictions is not None else -1

        STYLE = dict(facecolor="#1e1e1e", tick_params=dict(colors="#aaaaaa"),
                     grid=dict(alpha=0.15, color="#888888"))

        # ── Ax1: Price ──
        ax = self.ax_price
        ax.clear()
        ax.set_facecolor(STYLE["facecolor"])
        days = np.arange(PP)
        ax.plot(days, close_p, color="#3498db", linewidth=1.3, marker="o", markersize=3)
        ax.axvspan(PREVIOUS_NUM - 1, PP - 1, alpha=0.10, color="#f39c12")
        ax.axvline(x=PREVIOUS_NUM - 1, color="#f39c12", linestyle="--", linewidth=1.2)
        ax.scatter([PREVIOUS_NUM - 1], [pct[PREVIOUS_NUM - 1]], color="#e74c3c", s=40, zorder=5,
                   label="day 29 (history end)")
        ax.scatter([PP - 1], [close_p[-1]],
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

        # ── Ax2: Turnover ──
        ax = self.ax_turnover
        ax.clear()
        ax.set_facecolor(STYLE["facecolor"])
        x = np.arange(PREVIOUS_NUM)
        ax.bar(x, chg[:PREVIOUS_NUM], width=0.7, color="#9b59b6", alpha=0.8, label="change_rate")
        ax.set_xlabel("Day (0-29 = 30-day history)", color="#888888", fontsize=8)
        ax.set_ylabel("Turnover", color="#9b59b6")
        ax.set_title("Turnover (change_rate)", color="#cccccc", fontsize=10)
        ax.grid(True, axis="y", **STYLE["grid"])
        ax.tick_params(**STYLE["tick_params"])
        for sp in ax.spines.values():
            sp.set_color("#555555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ── Ax4: Change rate delta ──
        ax = self.ax_delta
        ax.clear()
        ax.set_facecolor(STYLE["facecolor"])
        x = np.arange(PREVIOUS_NUM)
        delta_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in cd_[:PREVIOUS_NUM]]
        ax.bar(x, cd_[:PREVIOUS_NUM], color=delta_colors, width=0.6, alpha=0.85)
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
