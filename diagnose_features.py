"""
特征诊断脚本 - 检查特征统计分布和数据质量
"""
import os
import sys
from pathlib import Path
import numpy as np
from collections import Counter
from tqdm import tqdm

project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

from dataloader.dockdataset import Dockdataset

def diagnose_features():
    """诊断特征分布"""
    print("\n" + "="*70)
    print("特征诊断分析")
    print("="*70)
    
    dataset_path = project_root / "dataset"
    total_num = 3805550
    split_index = int(0.8 * total_num)
    
    # 采样数据（为了加快速度）
    print("\n📊 采样数据用于特征分析...")
    sample_size = 10000
    sample_indices = np.random.choice(split_index, size=sample_size, replace=False)
    
    dataset = Dockdataset(dataset_path, 0, split_index, total_num)
    
    all_features = []
    all_labels = []
    label_counts = Counter()
    
    for idx in tqdm(sample_indices, desc="加载样本"):
        try:
            data, label = dataset[idx]
            all_features.append(data)
            all_labels.append(label)
            label_counts[int(label)] += 1
        except Exception as e:
            print(f"  ⚠️  加载样本 {idx} 时出错: {e}")
            continue
    
    all_features = np.array(all_features)
    all_labels = np.array(all_labels)
    
    print(f"\n✓ 成功加载 {len(all_features)} 条样本")
    print(f"  特征维度: {all_features.shape}")
    
    # 特征统计
    print("\n" + "="*70)
    print("📈 特征统计信息")
    print("="*70)
    
    feature_means = all_features.mean(axis=0)
    feature_stds = all_features.std(axis=0)
    feature_mins = all_features.min(axis=0)
    feature_maxs = all_features.max(axis=0)
    
    print(f"\n{'特征':<10} {'均值':<12} {'标准差':<12} {'最小值':<12} {'最大值':<12} {'范围':<12}")
    print("-" * 70)
    
    for i in range(all_features.shape[1]):
        mean = feature_means[i]
        std = feature_stds[i]
        fmin = feature_mins[i]
        fmax = feature_maxs[i]
        frange = fmax - fmin
        
        # 标记可能的问题
        issue = ""
        if std < 0.001:
            issue = "⚠️ 方差很小"
        elif std > 10:
            issue = "⚠️ 方差很大"
        elif abs(mean) > 100:
            issue = "⚠️ 均值偏离0"
        
        print(f"F{i:<8d} {mean:>10.4f}  {std:>10.4f}  {fmin:>10.4f}  {fmax:>10.4f}  {frange:>10.4f} {issue}")
    
    # 数据质量检查
    print("\n" + "="*70)
    print("🔍 数据质量检查")
    print("="*70)
    
    nan_count = np.isnan(all_features).sum()
    inf_count = np.isinf(all_features).sum()
    zero_variance_features = np.sum(feature_stds < 1e-6)
    
    print(f"\n  NaN值数量: {nan_count}")
    if nan_count > 0:
        print(f"    ⚠️  存在缺失值 (位置: {np.where(np.isnan(all_features))})")
    
    print(f"  Inf值数量: {inf_count}")
    if inf_count > 0:
        print(f"    ⚠️  存在无穷值")
    
    print(f"  零方差特征数: {zero_variance_features}")
    if zero_variance_features > 0:
        print(f"    ⚠️  这些特征没有变化，信息无用:")
        zero_var_indices = np.where(feature_stds < 1e-6)[0]
        print(f"    {list(zero_var_indices)}")
    
    # 特征相关性
    print("\n" + "="*70)
    print("🔗 特征间相关性分析")
    print("="*70)
    
    corr_matrix = np.corrcoef(all_features.T)
    
    # 找出高度相关的特征对
    high_corr_pairs = []
    for i in range(len(corr_matrix)):
        for j in range(i+1, len(corr_matrix)):
            if abs(corr_matrix[i][j]) > 0.95:  # 相关系数>0.95
                high_corr_pairs.append((i, j, corr_matrix[i][j]))
    
    if high_corr_pairs:
        print(f"\n  发现 {len(high_corr_pairs)} 对高度相关的特征 (相关系数 > 0.95):")
        for i, j, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:10]:
            print(f"    特征{i} - 特征{j}: {corr:.4f}")
    else:
        print("\n  ✓ 没有发现过高相关的特征对")
    
    # 特征与标签的关系
    print("\n" + "="*70)
    print("🎯 特征与标签的关系")
    print("="*70)
    
    print(f"\n标签分布:")
    for label in sorted(label_counts.keys()):
        count = label_counts[label]
        pct = count / len(all_labels) * 100
        print(f"  Label {label}: {count:>5} ({pct:>5.1f}%)")
    
    # 每个类别的特征统计
    print(f"\n各类别的特征均值差异:")
    print("-" * 70)
    
    class_feature_means = {}
    for label in range(5):
        label_mask = all_labels == label
        if label_mask.sum() > 0:
            class_feature_means[label] = all_features[label_mask].mean(axis=0)
    
    # 计算特征在各类别间的变异系数
    if len(class_feature_means) > 1:
        feature_variances = []
        for i in range(all_features.shape[1]):
            means = [class_feature_means[label][i] for label in sorted(class_feature_means.keys())]
            cv = np.std(means) / (np.mean(np.abs(means)) + 1e-6)  # 变异系数
            feature_variances.append((i, cv))
        
        # 排序，找出区分度最好的特征
        feature_variances.sort(key=lambda x: x[1], reverse=True)
        
        print(f"\n区分度最高的特征 (Top 10):")
        for idx, (feat_idx, cv) in enumerate(feature_variances[:10], 1):
            print(f"  {idx}. 特征{feat_idx}: 变异系数 = {cv:.4f}")
        
        print(f"\n区分度最低的特征 (Last 5):")
        for idx, (feat_idx, cv) in enumerate(feature_variances[-5:], 1):
            print(f"  {idx}. 特征{feat_idx}: 变异系数 = {cv:.4f}")
    
    # 诊断建议
    print("\n" + "="*70)
    print("💡 诊断建议")
    print("="*70)
    
    issues = []
    
    if nan_count > 0 or inf_count > 0:
        issues.append("❌ 存在NaN或Inf值，需要数据清理")
    
    if zero_variance_features > 0:
        issues.append(f"⚠️  {zero_variance_features}个特征方差为0，建议删除")
    
    # 检查特征缩放
    max_scale = feature_maxs.max()
    min_scale = feature_mins.min()
    if max_scale > 100 or min_scale < -100:
        issues.append(f"⚠️  特征范围很大 ({min_scale:.2f}~{max_scale:.2f})，建议标准化")
    
    if len(high_corr_pairs) > 3:
        issues.append(f"⚠️  特征间高度相关 ({len(high_corr_pairs)}对)，考虑特征选择")
    
    if issues:
        print("\n发现的问题:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ 特征质量良好，没有明显问题")
    
    print("\n建议行动:")
    print("  1. 如果存在NaN/Inf → 修复数据加载或预处理逻辑")
    print("  2. 如果特征方差太大 → 应用标准化/归一化")
    print("  3. 如果某些特征无用 → 特征选择或删除")
    print("  4. 如果特征间高度相关 → 考虑PCA或特征融合")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    diagnose_features()
