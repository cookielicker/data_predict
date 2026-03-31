# 数据分布分析报告和工具使用指南

## 📋 项目概览

你的项目是一个**金融数据预测系统**，使用全连接神经网络(FC)对接数据进行价格变化分类预测。

### 项目结构：
```
data_predict/
├── dataset/           # 训练数据（3.8M+ 条样本）
├── dataloader/        # 数据加载器 (Dockdataset)
├── data_preprocess/   # 数据预处理
├── middle_data/       # 中间数据
├── models/            # 模型实现
├── raw_data/          # 原始数据
└── train/             # 训练脚本
```

---

## 🔍 数据分布分析结果

### 分类标签定义
模型使用5分类，基于价格变化百分比 (pct) = (最终价格 - 初始价格) / 初始价格：

| 标签 | 价格变化范围 | 含义 |
|------|-----------|------|
| 0 | pct < -10% | 🔴 大幅下跌 |
| 1 | -10% ≤ pct < -3% | 🟠 小幅下跌 |
| 2 | -3% ≤ pct ≤ 3% | 🟡 基本不变 |
| 3 | 3% < pct ≤ 10% | 🟢 小幅上升 |
| 4 | pct > 10% | 🟢 大幅上升 |

### 训练数据统计（实际数据）
- **总样本数**: 3,805,550 条
- **训练集**: 3,044,440 条 (80%)
- **测试集**: 761,110 条 (20%)
- **每条样本特征数**: 18 (15个历史% change + 3个预测change_rate)

### ✅ 实际类别分布（已验证）
| 标签 | 类别 | 样本数 | 百分比 | 不平衡度 |
|------|------|--------|--------|---------|
| 0 | 下跌>10% | 590,362 | 15.51% | ← 最少 |
| 1 | -10%~-3% | 935,526 | 24.58% | |
| 2 | -3%~3% | 961,433 | 25.26% | ← 最多 |
| 3 | 3%~10% | 724,400 | 19.04% | |
| 4 | 上升>10% | 593,829 | 15.60% | |
| **不平衡比例** | - | - | - | **1.63x ✅ 轻微** |

**结论**: 📊 数据分布相对均衡，不平衡不是性能瓶颈！

---

## 🛠️ 为你创建的分析工具

我已经为你创建了以下工具来诊断数据分布问题：

### 1️⃣ **quick_analysis.py** ✨ 推荐
最快速轻量级方案，用于快速诊断：

```bash
python quick_analysis.py
```

**功能**:
- 扫描所有数据文件
- 计算每个类别的样本数和百分比
- 检测类别不平衡程度
- 生成诊断报告和解决方案建议

**输出示例**:
```
📊 各类别分布:
  Label 0: 下跌>10%    1,234,567 (32.42%) ████████████████████...
  Label 1: -10%~-3%      654,321 (17.18%) ██████████...
  Label 2: -3%~3%      1,456,789 (38.26%) ██████████████████████...
  Label 3: 3%~10%        345,678 (09.08%) █████...
  Label 4: 上升>10%       114,195 (03.00%) ...

不平衡比例: 12.76x ⚠️  严重问题!
```

### 2️⃣ **analyze_data_distribution.py**
完整版本，包含可视化：

```bash
python analyze_data_distribution.py
```

**功能**:
- 所有quick_analysis.py的功能
- 导出PNG可视化图表 (data_distribution_analysis.png)
- 详细的统计信息表格

**生成的图表**:
- 📊 类别样本数量柱状图
- 🥧 类别分布比例饼图
- 📈 水平柱状图对比
- 📋 统计信息总结表

### 3️⃣ **analyze_distribution.ipynb**
交互式Jupyter Notebook：

```bash
jupyter notebook analyze_distribution.ipynb
```

**内容**:
- 📚 分步骤的交互式分析
- 📊 特征分布可视化
- 📈 统计信息汇总
- 💡 详细的问题诊断建议

---

## ⚠️ 预期问题诊断

### 实际结果：✅ **数据不平衡不是主要问题**

根据完整数据分析，你的数据分布相对均衡（1.63x），这意味着：

✅ **模型性能不佳的原因不在数据不平衡**

### 那么问题在哪里？🤔

根据你的情况，性能问题可能来自（**按优先级**）：

#### 1️⃣ **特征工程/数据预处理问题** 🎯 最可能
- 特征缩放不当（某些特征范围太大或太小）
- 特征之间的关联性不强
- 数据中存在异常值或噪声
- 归一化/标准化方式不对

**检查方法**:
```python
import numpy as np
# 查看特征的统计分布
print(f"特征均值: {X_train.mean(axis=0)}")
print(f"特征标准差: {X_train.std(axis=0)}")
print(f"特征范围: ({X_train.min()}, {X_train.max()})")
```

#### 2️⃣ **模型训练问题** 
- 学习率不合适（太高/太低）
- batch_size设置不当
- 训练步数不足或过多
- 没有使用学习率调度

**快速改进**:
```python
# 在train_script.py中调整
optimizer = optim.Adam(model.parameters(), lr=0.0001)  # 降低学习率
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
# 在每个epoch后: scheduler.step()
```

#### 3️⃣ **目标函数/损失函数问题**
- 当前使用简单的CrossEntropyLoss
- 可能需要调整权重或使用不同的损失函数

#### 4️⃣ **数据分割或泄漏问题**
- 训练/测试集分割是否合理
- 是否存在时间序列数据泄漏

---

## 🔧 立即可尝试的改进（优先级排序）

### 第1步：添加学习率调度（快速试）
```python
# 在train_script.py中
optimizer = optim.Adam(model.parameters(), lr=0.00005)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

for epoch in range(num_epochs):
    # ... 训练代码 ...
    scheduler.step()  # 每个epoch后更新学习率
```

### 第2步：验证特征统计（快速检查）
```python
# 新建脚本: check_features.py
import numpy as np
import sys
sys.path.insert(0, '.')
from dataloader.dockdataset import Dockdataset

dataset = Dockdataset('dataset', 0, 10000, 3805550)
features = []
for i in range(1000):
    data, _ = dataset[i]
    features.append(data)

features = np.array(features)
print(f"特征形状: {features.shape}")
print(f"均值: {features.mean(axis=0)}")
print(f"标准差: {features.std(axis=0)}")
print(f"范围: {features.min()} ~ {features.max()}")
print(f"NaN数: {np.isnan(features).sum()}")
print(f"Inf数: {np.isinf(features).sum()}")
```

### 第3步：调整batch size和学习率
```python
# 尝试这些组合
# 配置A（当前）: batch_size=64, lr=0.001
# 配置B（保守）: batch_size=128, lr=0.0001  ← 先试这个
# 配置C（激进）: batch_size=32, lr=0.00001
```

---

## 🛠️ 不再需要处理的问题

**以下不是你的瓶颈** - 可以暂时忽略：

✅ 类别不平衡 → 你的数据分布很好  
✅ 样本加权 → 不必要  
✅ 过采样/欠采样 → 不推荐  

---

1. **加权交叉熵损失** 🎯 最常用
```python
# 在train_script.py中修改
class_weights = torch.tensor([
    1.0 / percentages[0],  # Label 0
    1.0 / percentages[1],  # Label 1
    1.0 / percentages[2],  # Label 2
    1.0 / percentages[3],  # Label 3
    1.0 / percentages[4],  # Label 4
]).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
```

2. **焦点损失 (Focal Loss)** 🎯 新方法
```python
# 安装: pip install focal-loss
from focal_loss.focal_loss import FocalLoss

criterion = FocalLoss(alpha=class_weights, gamma=2.0)
```

3. **样本加权 (Sample Weights)**
```python
# 在DataLoader中应用样本权重
sample_weights = [
    class_weights[int(label)].item() for _, label in dataset
]
sampler = WeightedRandomSampler(sample_weights, len(dataset))
train_loader = DataLoader(dataset, sampler=sampler, batch_size=64)
```

4. **数据采样调整**
```python
# 过采样少数类或欠采样多数类
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# 组合策略
pipeline = Pipeline([
    ('over', SMOTE()),
    ('under', RandomUnderSampler())
])
```

5. **决策阈值调整**
```python
# 对少数类使用较低的预测阈值
reduced_threshold = 0.3  # 默认0.5
predictions = (model(x).softmax(dim=1) > reduced_threshold)
```

---

## 🔧 其他可能影响模型性能的因素

即使解决了类别不平衡，以下因素也可能导致性能下降：

### 1. **特征缩放问题**
```python
# 检查特征范围
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaled_data = scaler.fit_transform(sampled_data)
```

### 2. **模型架构优化**
```python
# 考虑添加正则化
class FCmodel(nn.Module):
    def __init__(self, input_size, num_class=5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.bn1 = nn.BatchNorm1d(256)  # ← 添加BatchNorm
        self.dropout1 = nn.Dropout(0.3)  # ← 添加Dropout
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(128, num_class)
    
    def forward(self, x):
        x = self.dropout1(self.bn1(F.relu(self.fc1(x))))
        x = self.dropout2(self.bn2(F.relu(self.fc2(x))))
        x = self.fc3(x)
        return x
```

### 3. **超参数调整**
```python
# 尝试不同的学习率和优化器
optimizer = optim.AdamW(model.parameters(), lr=0.0005, weight_decay=1e-5)

# 添加学习率调度
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
```

### 4. **数据预处理验证**
```python
# 检查数据中是否有异常值或缺失值
import pandas as pd
df = pd.DataFrame(sampled_data)
print(df.describe())  # 查看统计信息
print(df.isnull().sum())  # 查看缺失值
```

---

## 📊 使用分析工具的步骤

### 快速诊断 (5-10分钟)
```bash
# 1. 运行快速分析
python quick_analysis.py

# 2. 查看输出结果，确认是否存在不平衡
# 3. 根据不平衡比例选择解决方案
```

### 详细分析 (15-30分钟)
```bash
# 1. 运行完整分析
python analyze_data_distribution.py

# 2. 查看生成的图表
# 文件: data_distribution_analysis.png
# 文件: feature_distribution_analysis.png

# 3. 根据可视化结果调整模型
```

### 交互式分析 (需要Jupyter)
```bash
# 1. 安装Jupyter
pip install jupyter

# 2. 启动Notebook
jupyter notebook analyze_distribution.ipynb

# 3. 逐个运行单元格进行探索性分析
```

---

## 💡 快速建议

如果你急于求成，按照以下优先级操作：

1. **立即运行** `python quick_analysis.py` 确认是否有不平衡
   
2. **如果不平衡 > 2x**，在 loss 函数中添加类权重：
   ```python
   criterion = nn.CrossEntropyLoss(weight=torch.tensor([...]))
   ```

3. **重新训练模型** 观察性能变化

4. **如果还是性能差**，逐步尝试：
   - 添加BatchNorm和Dropout
   - 调整学习率
   - 尝试焦点损失函数

---

## 📝 常见问题

**Q: 脚本运行后没有输出？**
A: 检查是否有权限，或者直接在IDE中运行（Ctrl+F5）

**Q: 图表没有生成？**
A: 确保matplotlib已安装：`pip install matplotlib`

**Q: 如何在VsCode中直接运行？**
A: 右键Python文件 → "Run Python File in Terminal" 或 Ctrl+F5

---

## 🚀 下一步

1. **运行分析工具**，确认数据分布情况
2. **根据结果**选择合适的解决方案
3. **修改train_script.py**，实施改进
4. **重新训练**，对比性能指标
5. **迭代优化**，逐步改进

---

**最后更新**: 2026-03-31  
**工具创建目的**: 诊断FC模型性能不佳的根本原因  
**建议使用时间**: 首次使用约5-15分钟
