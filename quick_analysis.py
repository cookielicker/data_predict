#!/usr/bin/env python3
"""
快速数据分布分析脚本 - 诊断数据不平衡问题
运行方式: python quick_analysis.py
"""
import os
import sys
from pathlib import Path
from collections import Counter
import numpy as np

# 设置路径
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

def calculate_label(data, previous_num=15):
    """根据数据计算标签"""
    pct = (data[-1] - data[0]) / data[0]
    if pct < -0.1:
        return 0
    elif -0.1 <= pct < -0.03:
        return 1
    elif -0.03 <= pct <= 0.03:
        return 2
    elif 0.03 < pct <= 0.1:
        return 3
    else:
        return 4

def analyze_distribution():
    """分析数据分布"""
    dataset_path = project_root / "dataset"
    total_num = 3805550
    
    print("\n" + "="*70)
    print("数据分布快速分析")
    print("="*70)
    
    # 获取所有npy文件
    npy_files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.npy')])
    
    if not npy_files:
        print("❌ 错误: 未找到数据文件!")
        return
    
    print(f"\n📁 找到 {len(npy_files)} 个数据文件")
    
    label_counter = Counter()
    sample_count = 0
    file_count = 0
    previous_num = 15
    predict_num = 3
    limit = 10000
    
    # 处理数据文件
    print("\n📊 正在分析数据分布...\n")
    
    for i, filename in enumerate(npy_files):
        filepath = dataset_path / filename
        try:
            data = np.load(filepath)
            
            for row_idx in range(len(data)):
                raw_data = data[row_idx]
                mean_data = raw_data[:previous_num + predict_num]
                label = calculate_label(mean_data, previous_num)
                label_counter[label] += 1
                sample_count += 1
        except Exception as e:
            print(f"⚠️  处理文件 {filename} 时出错: {e}")
            continue
        
        file_count += 1
        
        # 显示进度
        if (i + 1) % 10 == 0 or (i + 1) == len(npy_files):
            progress = (i + 1) / len(npy_files) * 100
            print(f"  进度: [{i+1:4d}/{len(npy_files):4d}] ({progress:5.1f}%) - "
                  f"已处理 {sample_count:>10,} 条数据", end='\r')
    
    print()  # 换行
    
    # 计算统计信息
    label_percentages = {
        i: (label_counter.get(i, 0) / sample_count) * 100 
        for i in range(5)
    }
    
    counts = [label_counter.get(i, 0) for i in range(5)]
    max_ratio = max(counts) / min(counts) if min(counts) > 0 else float('inf')
    
    # 显示分析结果
    print("\n" + "="*70)
    print("✅ 分析完成!")
    print("="*70)
    
    print(f"\n📈 数据统计:")
    print(f"  总样本数: {sample_count:,}")
    print(f"  处理文件: {file_count}")
    print(f"  类别数: 5")
    
    print(f"\n📊 各类别分布:")
    print(f"  {'类别':<15} {'样本数':<15} {'百分比':<12} {'分布':<40}")
    print(f"  {'-'*70}")
    
    labels_name = ['下跌>10%', '-10%~-3%', '-3%~3%', '3%~10%', '上升>10%']
    for i in range(5):
        count = counts[i]
        pct = label_percentages[i]
        bar_len = int(pct / 2.5)  # 缩放到40个字符
        bar = '█' * bar_len + '░' * (40 - bar_len)
        print(f"  Label {i:<2}: {labels_name[i]:<13} {count:>13,} ({pct:>6.2f}%) {bar}")
    
    # 诊断
    print(f"\n🔍 不平衡分析:")
    print(f"  最多类别: Label {counts.index(max(counts))} - {max(counts):,} ({max(label_percentages.values()):.2f}%)")
    print(f"  最少类别: Label {counts.index(min(counts))} - {min(counts):,} ({min(label_percentages.values()):.2f}%)")
    print(f"  不平衡比例: {max_ratio:.2f}x")
    
    print(f"\n⚠️  问题诊断:")
    if max_ratio > 5:
        print(f"  🔴 【严重问题】存在极度不平衡（{max_ratio:.2f}x > 5x）")
        print(f"     • 模型可能严重倾向于预测多数类")
        print(f"     • 少数类的识别率极低")
        print(f"     • 总体准确率可能虚高")
        print(f"\n     建议解决方案:")
        print(f"     1. 使用加权交叉熵损失函数")
        print(f"     2. 尝试焦点损失 (Focal Loss)")
        print(f"     3. 进行样本过采样/欠采样")
        print(f"     4. 调整类权重在DataLoader中")
        
    elif max_ratio > 2:
        print(f"  🟠 【中等问题】存在明显不平衡（2x < {max_ratio:.2f}x < 5x）")
        print(f"     • 少数类性能会显著下降")
        print(f"     • 建议使用加权损失函数")
        
    elif max_ratio > 1.3:
        print(f"  🟡 【轻微问题】存在轻微不平衡（1.3x < {max_ratio:.2f}x < 2x）")
        print(f"     • 影响有限但需关注")
        print(f"     • 可尝试微调损失权重")
    else:
        print(f"  ✅ 【良好】类别分布相对均衡（{max_ratio:.2f}x < 1.3x）")
    
    print("\n" + "="*70)
    print("💡 其他可能影响模型性能的因素:")
    print("="*70)
    print("""
  1. 特征缩放问题
     - 检查特征范围是否正确
     - 验证是否需要归一化或标准化

  2. 数据质量问题
     - 检查是否有缺失值或异常值
     - 验证数据预处理逻辑

  3. 模型架构问题
     - FC层数是否合适
     - 是否需要添加正则化（Dropout/BatchNorm）
     - 激活函数选择是否合理

  4. 训练超参数问题
     - 学习率是否合适
     - 批大小是否太小/太大
     - 优化器的选择（Adam/SGD等）

  5. 数据分割问题
     - 训练/测试集分割是否合理
     - 是否需要分层采样确保分布一致
    """)
    
    print("="*70)
    return label_counter, label_percentages, sample_count

if __name__ == "__main__":
    try:
        analyze_distribution()
    except KeyboardInterrupt:
        print("\n\n⚠️  分析被中断")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
