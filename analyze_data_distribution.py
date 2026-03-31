"""
数据分布分析脚本
分析数据集中各个类别的分布情况，用于诊断是否存在类别不平衡问题
"""
import os
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
from tqdm import tqdm

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 设置路径
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from dataloader.dockdataset import Dockdataset


def calculate_label(data, previous_num=15):
    """
    根据数据计算标签（与Dockdataset中的逻辑一致）
    """
    pct = (data[-1] - data[0]) / data[0]
    if pct < -0.1:
        label = 0
    elif -0.1 <= pct < -0.03:
        label = 1
    elif -0.03 <= pct <= 0.03:
        label = 2
    elif 0.03 < pct <= 0.1:
        label = 3
    else:
        label = 4
    return label


def analyze_label_distribution(dataset_path, total_num, sample_rate=0.1):
    """
    分析整个数据集的标签分布
    
    参数:
        dataset_path: 数据集路径
        total_num: 总数据量
        sample_rate: 采样率（为了加快速度，可以只采样一部分）
    
    返回:
        标签计数器，标签百分比字典
    """
    print(f"开始分析数据分布... (采样率: {sample_rate*100}%)")
    
    label_counter = Counter()
    sample_num = int(total_num * sample_rate)
    
    # 创建dataset来遍历数据
    dataset = Dockdataset(dataset_path, 0, sample_num, total_num)
    
    # 使用进度条
    for i in tqdm(range(len(dataset)), desc="处理数据"):
        try:
            _, label = dataset[i]
            label_counter[int(label)] += 1
        except Exception as e:
            print(f"处理索引 {i} 时出错: {e}")
            continue
    
    # 计算百分比
    total_samples = sum(label_counter.values())
    label_percentages = {
        label: (count / total_samples) * 100 
        for label, count in label_counter.items()
    }
    
    return label_counter, label_percentages, total_samples


def analyze_raw_files(dataset_path, previous_num=15, predict_num=3, limit=10000):
    """
    直接分析原始文件的标签分布（更快更准确）
    """
    print("直接分析原始数据文件...")
    
    label_counter = Counter()
    file_count = 0
    sample_count = 0
    
    # 获取所有.npy文件
    npy_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.npy')])
    
    for filename in tqdm(npy_files, desc="处理文件"):
        filepath = os.path.join(dataset_path, filename)
        try:
            data = np.load(filepath)
            # data shape: (limit, features)
            for i in range(len(data)):
                raw_data = data[i]
                # 提取mean部分（前previous+predict个）
                mean_data = raw_data[:previous_num + predict_num]
                label = calculate_label(mean_data, previous_num)
                label_counter[label] += 1
                sample_count += 1
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")
            continue
        
        file_count += 1
    
    # 计算百分比
    label_percentages = {
        label: (count / sample_count) * 100 
        for label, count in sorted(label_counter.items())
    }
    
    return label_counter, label_percentages, sample_count


def visualize_distribution(label_counter, label_percentages, output_path=None):
    """
    使用matplotlib可视化标签分布
    """
    labels = ['下跌>10%\n(label 0)', 
              '-10%~-3%\n(label 1)', 
              '-3%~3%\n(label 2)',
              '3%~10%\n(label 3)', 
              '上升>10%\n(label 4)']
    
    counts = [label_counter.get(i, 0) for i in range(5)]
    percentages = [label_percentages.get(i, 0) for i in range(5)]
    
    # 创建图表
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('数据集标签分布分析', fontsize=16, fontweight='bold')
    
    # 1. 柱状图 - 样本数
    ax1 = axes[0, 0]
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4', '#9467bd']
    bars = ax1.bar(range(5), counts, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax1.set_xlabel('标签类别', fontsize=11, fontweight='bold')
    ax1.set_ylabel('样本数', fontsize=11, fontweight='bold')
    ax1.set_title('各类别样本数量', fontsize=12, fontweight='bold')
    ax1.set_xticks(range(5))
    ax1.set_xticklabels(labels)
    ax1.grid(axis='y', alpha=0.3)
    
    # 添加数值标签
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(count):,}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 2. 饼图 - 百分比
    ax2 = axes[0, 1]
    wedges, texts, autotexts = ax2.pie(percentages, labels=labels, autopct='%1.2f%%',
                                         colors=colors, startangle=90, 
                                         textprops={'fontsize': 9})
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    ax2.set_title('类别分布比例', fontsize=12, fontweight='bold')
    
    # 3. 水平柱状图 - 更清晰的比较
    ax3 = axes[1, 0]
    y_pos = np.arange(5)
    ax3.barh(y_pos, percentages, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels)
    ax3.set_xlabel('百分比 (%)', fontsize=11, fontweight='bold')
    ax3.set_title('类别分布百分比（升序）', fontsize=12, fontweight='bold')
    ax3.grid(axis='x', alpha=0.3)
    
    # 添加百分比标签
    for i, (pct, count) in enumerate(zip(percentages, counts)):
        ax3.text(pct + 0.5, i, f'{pct:.2f}% ({int(count):,})',
                va='center', fontsize=9, fontweight='bold')
    
    # 4. 统计信息表
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # 计算统计信息
    total_samples = sum(counts)
    max_label = max(range(5), key=lambda i: counts[i])
    min_label = min(range(5), key=lambda i: counts[i])
    max_ratio = max(percentages) / min(percentages) if min(percentages) > 0 else float('inf')
    
    stats_text = f"""
    数据统计信息:
    {'='*50}
    总样本数: {total_samples:,}
    
    最多的类别: label {max_label} - {percentages[max_label]:.2f}%
    最少的类别: label {min_label} - {percentages[min_label]:.2f}%
    
    类别不平衡比例: {max_ratio:.2f}x
    
    各类别详细信息:
    {'-'*50}
    """
    
    for i in range(5):
        stats_text += f"Label {i}: {counts[i]:>10,} ({percentages[i]:>6.2f}%)\n"
    
    stats_text += "\n" + "="*50
    if max_ratio > 2:
        stats_text += "\n⚠️  警告: 存在严重的类别不平衡问题！"
    elif max_ratio > 1.5:
        stats_text += "\n⚠️  注意: 存在中等程度的类别不平衡。"
    else:
        stats_text += "\n✓ 类别分布相对均衡。"
    
    ax4.text(0.1, 0.95, stats_text, transform=ax4.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.tight_layout()
    
    # 保存图表
    if output_path is None:
        output_path = os.path.join(project_root, 'data_distribution_analysis.png')
    
    try:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ 图表已保存到: {output_path}")
    except Exception as e:
        print(f"\n✗ 保存图表时出错: {e}")
    
    # 移除 plt.show() 以避免在无GUI环境中挂起
    # plt.show()
    
    return {
        'total_samples': total_samples,
        'class_distribution': dict(zip(range(5), counts)),
        'class_percentages': {i: percentages[i] for i in range(5)},
        'imbalance_ratio': max_ratio
    }


if __name__ == "__main__":
    dataset_path = os.path.join(project_root, "dataset")
    total_num = 3805550
    
    print("="*60)
    print("数据分布分析工具")
    print("="*60)
    
    # 分析数据分布
    label_counter, label_percentages, sample_count = analyze_raw_files(
        dataset_path, 
        previous_num=15, 
        predict_num=3, 
        limit=10000
    )
    
    print("\n" + "="*60)
    print("分析结果:")
    print("="*60)
    print(f"总样本数: {sample_count:,}\n")
    
    print("各类别分布:")
    for i in range(5):
        count = label_counter.get(i, 0)
        pct = label_percentages.get(i, 0)
        print(f"  Label {i}: {count:>10,} ({pct:>6.2f}%)")
    
    # 计算不平衡比例
    counts = [label_counter.get(i, 0) for i in range(5)]
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
    
    print(f"\n类别不平衡比例 (最多/最少): {imbalance_ratio:.2f}x")
    
    # 绘制可视化
    stats = visualize_distribution(label_counter, label_percentages)
    
    print("\n" + "="*60)
    print("分析完成！")
    print("="*60)
