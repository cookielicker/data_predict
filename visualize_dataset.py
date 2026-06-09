#!/usr/bin/env python3
"""
Dataset Visualization Tool — 浏览训练数据样本，查看特征和标签

用法:
  python visualize_dataset.py
"""

import os
import sys
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.gridspec import GridSpec

PREVIOUS_NUM = 30
PREDICT_NUM = 3
PP = PREVIOUS_NUM + PREDICT_NUM  # 33


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


LABEL_NAMES = {0: "L0 (跌 >10%)", 1: "L1 (震荡)", 2: "L2 (涨 >10%)"}
LABEL_COLORS = {0: "#e74c3c", 1: "#95a5a6", 2: "#2ecc71"}


class DatasetViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Dataset Viewer")
        self.root.geometry("1300x900")
        self.root.configure(bg="#2b2b2b")

        self._all_samples = None
        self._all_codes = None  # 每个样本对应的股票代码
        self._file_list = []
        self._current_file_idx = -1
        self._total_samples = 0
        self._current_idx = 0

        self._build_ui()

    def _build_ui(self):
        # Top bar
        top = tk.Frame(self.root, bg="#2b2b2b")
        top.pack(fill=tk.X, padx=8, pady=(8, 0))

        tk.Button(top, text="Load .npy File", command=self._load_file,
                  bg="#3a3a3a", fg="#ecf0f1", bd=1, padx=10, pady=2,
                  activebackground="#555555", activeforeground="#ffffff").pack(side=tk.LEFT, padx=4)
        tk.Button(top, text="Load Directory", command=self._load_directory,
                  bg="#3a3a3a", fg="#ecf0f1", bd=1, padx=10, pady=2,
                  activebackground="#555555", activeforeground="#ffffff").pack(side=tk.LEFT, padx=4)

        self._file_label = tk.Label(top, text="No data loaded — 请加载 .npy 文件或目录",
                                    bg="#2b2b2b", fg="#7f8c8d", font=("Arial", 10))
        self._file_label.pack(side=tk.LEFT, padx=16)

        # Navigation bar
        nav = tk.Frame(self.root, bg="#2b2b2b")
        nav.pack(fill=tk.X, padx=8, pady=4)

        b_opts = dict(bg="#3a3a3a", fg="#ecf0f1", bd=1, padx=8, pady=2,
                      activebackground="#555555", activeforeground="#ffffff")

        tk.Button(nav, text="Prev File  ◀◀", command=self._prev_file, **b_opts).pack(side=tk.LEFT, padx=3)
        tk.Button(nav, text="◀  Prev (←)", command=self._prev_sample, **b_opts).pack(side=tk.LEFT, padx=3)

        self._idx_entry = tk.Entry(nav, width=8, justify="center", bg="#3a3a3a", fg="#ecf0f1",
                                   insertbackground="#ecf0f1", bd=1, font=("Consolas", 11))
        self._idx_entry.pack(side=tk.LEFT, padx=6)
        self._idx_entry.bind("<Return>", self._jump_to)

        self._count_label = tk.Label(nav, text="/ 0", bg="#2b2b2b", fg="#bdc3c7",
                                     font=("Consolas", 11))
        self._count_label.pack(side=tk.LEFT)

        tk.Button(nav, text="Next (→)  ▶", command=self._next_sample, **b_opts).pack(side=tk.LEFT, padx=3)
        tk.Button(nav, text="Next File  ▶▶", command=self._next_file, **b_opts).pack(side=tk.LEFT, padx=3)

        # Label display — tk.Label supports fg natively (ttk.Label doesn't on all themes)
        self._label_display = tk.Label(nav, text="", font=("Arial", 18, "bold"),
                                       bg="#2b2b2b", fg="#ecf0f1")
        self._label_display.pack(side=tk.LEFT, padx=30)

        self._code_display = tk.Label(nav, text="", font=("Consolas", 12, "bold"),
                                       bg="#2b2b2b", fg="#f39c12")
        self._code_display.pack(side=tk.LEFT, padx=10)

        self._pct_display = tk.Label(nav, text="", font=("Consolas", 12),
                                     bg="#2b2b2b", fg="#bdc3c7")
        self._pct_display.pack(side=tk.LEFT, padx=10)

        self.fig = Figure(figsize=(12, 8), facecolor="#2b2b2b")

        # Matplotlib — GridSpec: 价格 + 变化率 + 换手率/波动率 + 换手率变化量
        gs = GridSpec(4, 1, figure=self.fig, height_ratios=[2.5, 1, 1.5, 1.5],
                      top=0.97, bottom=0.05, left=0.08, right=0.93, hspace=0.08)
        self.ax1 = self.fig.add_subplot(gs[0])   # 价格
        self.ax2 = self.fig.add_subplot(gs[1], sharex=self.ax1)  # 变化率 (与价格共享X轴)
        self.ax3 = self.fig.add_subplot(gs[2])   # 换手率+波动率
        self.ax4 = self.fig.add_subplot(gs[3])   # 换手率变化量

        for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:
            ax.set_facecolor("#1e1e1e")
            ax.tick_params(colors="#aaaaaa")
            ax.spines["bottom"].set_color("#555555")
            ax.spines["left"].set_color("#555555")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        NavigationToolbar2Tk(self.canvas, self.root)

        # Status bar
        status = tk.Frame(self.root, bg="#1e1e1e")
        status.pack(fill=tk.X, padx=8, pady=(0, 4))
        tk.Label(status, text="← → 切换样本  |  Ctrl+← → 切换文件  |  PgUp/PgDn 翻页  |  输入序号回车跳转",
                 bg="#1e1e1e", fg="#555555", font=("Arial", 8)).pack(side=tk.LEFT)

        # Keyboard bindings
        self.root.bind("<Left>", lambda e: self._prev_sample())
        self.root.bind("<Right>", lambda e: self._next_sample())
        self.root.bind("<Control-Left>", lambda e: self._prev_file())
        self.root.bind("<Control-Right>", lambda e: self._next_file())
        self.root.bind("<Prior>", lambda e: self._prev_sample())
        self.root.bind("<Next>", lambda e: self._next_sample())

    # ─── Data loading ──────────────────────────────────────────────
    def _load_codes(self, npy_path):
        """加载 _codes.txt 文件 (如果存在)"""
        codes_path = npy_path.replace(".npy", "_codes.txt")
        if os.path.exists(codes_path):
            with open(codes_path, "r") as f:
                return [line.strip() for line in f]
        return None

    def _load_file(self):
        path = filedialog.askopenfilename(
            title="选择 .npy 数据文件",
            filetypes=[("NumPy files", "*.npy"), ("All files", "*.*")],
        )
        if not path:
            return
        data = np.load(path)
        if data.ndim != 2 or data.shape[1] != PP * 4:
            self._file_label.config(text=f"格式错误: shape={data.shape}, 需要 (N, {PP * 4})")
            return
        self._all_samples = data
        self._all_codes = self._load_codes(path)
        self._file_list = [path]
        self._current_file_idx = 0
        self._total_samples = len(data)
        self._current_idx = 0
        self._file_label.config(text=f"[1/1] {os.path.basename(path)}  ({self._total_samples:,} samples)")
        self._draw()

    def _load_directory(self):
        d = filedialog.askdirectory(title="选择包含 .npy 文件的目录")
        if not d:
            return
        files = sorted([os.path.join(d, f) for f in os.listdir(d)
                       if f.endswith(".npy") and not f.endswith("_codes.txt")])
        if not files:
            self._file_label.config(text="目录中没有 .npy 文件")
            return
        self._file_list = files
        self._current_file_idx = 0
        self._load_current_file()

    def _load_current_file(self):
        path = self._file_list[self._current_file_idx]
        data = np.load(path)
        self._all_samples = data
        self._all_codes = self._load_codes(path)
        self._total_samples = len(data)
        self._current_idx = 0
        self._file_label.config(
            text=f"[{self._current_file_idx + 1}/{len(self._file_list)}]  "
            f"{os.path.basename(path)}  ({self._total_samples:,} samples in file)"
        )
        self._draw()

    # ─── Navigation ────────────────────────────────────────────────
    def _prev_sample(self):
        if self._all_samples is None:
            return
        if self._current_idx > 0:
            self._current_idx -= 1
        elif self._current_file_idx > 0:
            self._current_file_idx -= 1
            self._load_current_file()
            self._current_idx = self._total_samples - 1
        self._draw()

    def _next_sample(self):
        if self._all_samples is None:
            return
        if self._current_idx < self._total_samples - 1:
            self._current_idx += 1
        elif self._current_file_idx < len(self._file_list) - 1:
            self._current_file_idx += 1
            self._load_current_file()
        else:
            return
        self._draw()

    def _prev_file(self):
        if self._current_file_idx > 0:
            self._current_file_idx -= 1
            self._load_current_file()

    def _next_file(self):
        if self._current_file_idx < len(self._file_list) - 1:
            self._current_file_idx += 1
            self._load_current_file()

    def _jump_to(self, event):
        try:
            idx = int(self._idx_entry.get())
            if 0 <= idx < self._total_samples:
                self._current_idx = idx
                self._draw()
        except ValueError:
            pass

    # ─── Drawing ───────────────────────────────────────────────────
    def _draw(self):
        if self._all_samples is None:
            return

        raw = self._all_samples[self._current_idx]
        mean = raw[:PP]
        change = raw[PP : 2 * PP]
        hl = raw[2 * PP : 3 * PP]
        change_delta = raw[3 * PP : 4 * PP]
        price_pct = _cal_pct(mean[:PREVIOUS_NUM])

        label, ret_pct = compute_label(mean)
        label_color = LABEL_COLORS[label]
        stock_code = self._all_codes[self._current_idx] if self._all_codes else "?"

        STYLE = dict(facecolor="#1e1e1e", tick_params=dict(colors="#aaaaaa"),
                     grid=dict(alpha=0.15, color="#888888"))

        # ── Ax1: Price (mean_adj) — 33天, 与下方的pct_change共享X轴 ──
        ax = self.ax1
        ax.clear()
        ax.set_facecolor(STYLE["facecolor"])
        days = np.arange(PP)
        ax.plot(days, mean, color="#3498db", linewidth=1.3, marker="o", markersize=3)
        ax.axvspan(PREVIOUS_NUM - 1, PP - 1, alpha=0.10, color="#f39c12")
        ax.axvline(x=PREVIOUS_NUM - 1, color="#f39c12", linestyle="--", linewidth=1.2)
        ax.scatter([PREVIOUS_NUM - 1], [mean[PREVIOUS_NUM - 1]], color="#e74c3c", s=40, zorder=5,
                   label=f"day 29 (history end)")
        ax.scatter([PP - 1], [mean[-1]],
                   color="#2ecc71" if ret_pct >= 0 else "#e74c3c", s=55, zorder=5, marker="D",
                   label=f"day 32 (target)")
        ax.set_ylabel("Price", color="#bbbbbb")
        ax.set_title(
            f"[{stock_code}]  Sample {self._current_idx:,}  |  "
            f"3d return: {ret_pct * 100:+.2f}%  |  {LABEL_NAMES[label]}",
            color=label_color, fontsize=12, fontweight="bold",
        )
        ax.legend(loc="upper left", fontsize=8, facecolor="#333333", edgecolor="#555555",
                  labelcolor="#cccccc")
        ax.grid(True, **STYLE["grid"])
        ax.tick_params(**STYLE["tick_params"])
        plt.setp(ax.get_xticklabels(), visible=False)  # X标签隐藏，让ax2显示
        for sp in ax.spines.values():
            sp.set_color("#555555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # ── Ax2: Price pct_change — 30天, 与ax1共享X轴 ──
        ax = self.ax2
        ax.clear()
        ax.set_facecolor(STYLE["facecolor"])
        x = np.arange(PREVIOUS_NUM)
        colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in price_pct]
        ax.bar(x, price_pct, color=colors, width=0.7, alpha=0.85)
        ax.axhline(y=0, color="#666666", linewidth=0.8)
        # 标记每个bar的值范围
        ax.set_ylabel("Δ%", color="#bbbbbb", fontsize=9)
        ax.set_xlabel("Day  (price 折线图每个相邻点的涨跌幅度 ↓ 一一对齐)", color="#888888", fontsize=8)
        ax.grid(True, axis="y", **STYLE["grid"])
        ax.tick_params(**STYLE["tick_params"])
        for sp in ax.spines.values():
            sp.set_color("#555555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # 限制X轴范围对齐
        ax.set_xlim(-0.5, PP - 0.5)

        # ── Ax3: Change rate + High-low ratio ──
        ax = self.ax3
        ax.clear()
        # 清理旧的 twinx，避免切换样本时右轴重叠
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
        ax.set_title("Turnover & Volatility (模型输入特征)", color="#cccccc", fontsize=10)
        ax.grid(True, axis="y", **STYLE["grid"])
        ax.tick_params(**STYLE["tick_params"])
        ax2_.tick_params(**STYLE["tick_params"])
        for sp in list(ax.spines.values()) + list(ax2_.spines.values()):
            sp.set_color("#555555")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2_.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=8,
                  facecolor="#333333", edgecolor="#555555", labelcolor="#cccccc")

        # ── Ax4: Change rate delta (换手率变化量) ──
        ax = self.ax4
        ax.clear()
        ax.set_facecolor(STYLE["facecolor"])
        x = np.arange(PREVIOUS_NUM)
        delta_colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in change_delta[:PREVIOUS_NUM]]
        ax.bar(x, change_delta[:PREVIOUS_NUM], color=delta_colors, width=0.6, alpha=0.85)
        ax.axhline(y=0, color="#666666", linewidth=0.8)
        ax.set_xlabel("Day (0-29 = 30-day history)", color="#888888", fontsize=8)
        ax.set_ylabel("Δ Turnover", color="#bbbbbb")
        ax.set_title("Turnover Change Rate (换手率变化量, 模型输入特征)", color="#cccccc", fontsize=10)
        ax.grid(True, axis="y", **STYLE["grid"])
        ax.tick_params(**STYLE["tick_params"])
        for sp in ax.spines.values():
            sp.set_color("#555555")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        self.canvas.draw()

        # Update nav UI
        self._idx_entry.delete(0, tk.END)
        self._idx_entry.insert(0, str(self._current_idx))
        self._count_label.config(text=f"/ {self._total_samples - 1:,}")
        self._code_display.config(text=f"  {stock_code}  ")
        self._label_display.config(text=f"[{LABEL_NAMES[label]}]", fg=label_color)
        self._pct_display.config(
            text=f"return: {ret_pct * 100:+.2f}%",
            fg="#2ecc71" if ret_pct >= 0 else "#e74c3c",
        )


def main():
    root = tk.Tk()
    _app = DatasetViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
