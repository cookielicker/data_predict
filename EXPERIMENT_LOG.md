# 实验记录

## 数据集信息

### dataset（旧，15天特征）
- 总样本: 3,805,550
- 训练集: 3,044,440 (80%)
- 测试集: 761,110 (20%)
- 特征维度: 30 (价格变化率15维 + 换手率15维)
- 三分类: L0(下跌>10%), L1(震荡-10%~10%), L2(上涨>10%)

### 训练集类别分布
| 类别 | 数量 | 占比 |
|------|------|------|
| L0 | 69,321 | 2.3% |
| L1 | 2,867,591 | 94.2% |
| L2 | 107,528 | 3.5% |

### 类别权重 (total / (n_classes * count))
| 类别 | 权重 |
|------|------|
| L0 | 14.64 |
| L1 | 0.35 |
| L2 | 9.44 |

---

### baostock_dataset_30（新，30天特征）
- 总样本: 8,765,911（约876万）
- 特征维度: 66 (价格变化率30维 + 换手率30维 + 预测价格3维 + 预测换手率3维)
- 时间范围: 2016-05-19 ~ 2026-05-15（10年）
- 股票数: 4875只
- 目录: `baostock_dataset_30/`

---

## 模型结构

### FCmodel
```
Sequential(
  Linear(30, 256), ReLU,
  Linear(256, 256), ReLU,
  Linear(256, 256), ReLU,
  Linear(256, 256), ReLU,
  Linear(256, 3)
)
```

### TransformerSequenceModel
- 输入: (batch, 30) → reshape to (batch, 15, 2)
- d_model=256, num_layers=4

---

## 评估指标说明
- **Precision**: 预测为某类的样本中，真正属于该类的比例
- **Recall**: 某类样本中，被正确预测的比例
- **F1**: 2 * Precision * Recall / (Precision + Recall)
- **Macro F1**: 所有类F1的算术平均

---

## 实验结果

### Exp #1: Focal Loss gamma=2.5
- **日期**: (待记录)
- **数据集**: dataset（15天，30特征，380万样本）
- **模型**: FCmodel
- **配置**: gamma=2.5, alpha=weights
- **结果**: 失败，模型几乎不预测L1

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | - | - | - |
| Recall | - | ≈0% | - |
| F1 | - | - | - |

**结论**: gamma太高，模型完全偏向少数类

---

### Exp #2: Focal Loss gamma=1.5
- **日期**: (待记录)
- **数据集**: dataset（15天，30特征，380万样本）
- **模型**: FCmodel
- **配置**: gamma=1.5, alpha=weights
- **最佳结果**: Epoch 3, Macro F1 = 17.9%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.07 | 0.98 | 0.05 |
| Recall | 0.72 | 0.19 | 0.64 |
| F1 | 0.13 | 0.31 | 0.09 |

**结论**: Precision极低，大量误判L1为L0/L2

---

### Exp #3: Focal Loss gamma=1.0
- **日期**: (待记录)
- **数据集**: dataset（15天，30特征，380万样本）
- **模型**: FCmodel
- **配置**: gamma=1.0, alpha=weights
- **结果**: Epoch 12, Acc=32.7%, MacroF1=24.0%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.09 | 0.98 | 0.06 |
| Recall | 0.74 | 0.30 | 0.65 |
| F1 | 0.15 | 0.46 | 0.11 |

---

### Exp #4: Focal Loss gamma=0.5
- **日期**: (待记录)
- **数据集**: dataset（15天，30特征，380万样本）
- **模型**: FCmodel
- **配置**: gamma=0.5, alpha=weights
- **结果**: Epoch 14, Acc=49.0%, MacroF1=31.1%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.08 | 0.98 | 0.08 |
| Recall | 0.74 | 0.48 | 0.55 |
| F1 | 0.15 | 0.64 | 0.14 |

---

### Exp #5: Weighted Cross Entropy (baseline)
- **日期**: (待记录)
- **数据集**: dataset（15天，30特征，380万样本）
- **模型**: FCmodel
- **配置**: weights=[14.64, 0.35, 9.44]
- **结果**: Epoch 12, Acc=57.3%, MacroF1=34.5%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.09 | 0.97 | 0.09 |
| Recall | 0.72 | 0.57 | 0.49 |
| F1 | 0.16 | 0.72 | 0.15 |

**备注**: Weighted CE的L0/L2 Precision最高(9%)

---

### Exp #6: Transformer
- **日期**: (待记录)
- **数据集**: dataset（15天，30特征，380万样本）
- **模型**: TransformerSequenceModel
- **配置**: d_model=256, num_layers=4, lr=0.0001, weighted CE
- **结果**: Epoch 3, Acc=57.9%, MacroF1=32.3%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.06 | 0.96 | 0.09 |
| Recall | 0.62 | 0.59 | 0.26 |
| F1 | 0.11 | 0.73 | 0.13 |

**备注**: Transformer的L0/L2 Precision并未比FCmodel更好

---

### Exp #7: 降低类别权重 - Scheme A [5, 1, 5]
- **日期**: 2026-05-14
- **数据集**: dataset（15天，30特征，380万样本）
- **模型**: FCmodel
- **配置**: weights=[5.0, 1.0, 3.0], lr=0.001, 训练200K样本快速测试
- **结果**: Epoch 19, MacroF1=51.3%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.38 | 0.96 | 0.36 |
| Recall | 0.32 | 0.98 | 0.16 |
| F1 | 0.35 | 0.97 | 0.22 |

**结论**: 降低权重后L0/L2 Precision显著提升！L0 P从0.09→0.38, L2 P从0.09→0.36

---

### Exp #8: 降低类别权重 - Scheme B [3, 1, 3]
- **日期**: 2026-05-14
- **数据集**: dataset（15天，30特征，380万样本）
- **模型**: FCmodel
- **配置**: weights=[3.0, 1.0, 3.0], lr=0.001, 训练200K样本快速测试
- **结果**: Epoch 27, MacroF1=51.3%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.44 | 0.96 | 0.50 |
| Recall | 0.31 | 0.99 | 0.13 |
| F1 | 0.36 | 0.97 | 0.20 |

**结论**: 继续降低权重，L0/L2 Precision进一步提升！L0 P=0.44, L2 P=0.50

---

### Exp #9: 降低类别权重 - Scheme C [2, 1, 2]
- **日期**: 2026-05-14
- **数据集**: dataset（15天，30特征，380万样本）
- **模型**: FCmodel
- **配置**: weights=[2.0, 1.0, 2.0], lr=0.001, 训练200K样本快速测试
- **结果**: Epoch 25, MacroF1=51.3%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.47 | 0.96 | 0.52 |
| Recall | 0.27 | 0.99 | 0.14 |
| F1 | 0.34 | 0.97 | 0.22 |

**结论**: 权重越低，L0/L2 Precision越高！L0 P=0.47, L2 P=0.52，但Recall下降

---

### Exp #10: 全量数据训练 - Scheme C [2, 1, 2]
- **日期**: 2026-05-14
- **数据集**: dataset（15天，30特征，380万样本）
- **模型**: FCmodel
- **配置**: weights=[2.0, 1.0, 2.0], lr=0.001, 全量数据3M, 早停patience=15
- **结果**: Epoch 12, Acc=93.7%, MacroF1=55.6%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.58 | 0.95 | 0.67 |
| Recall | 0.30 | 0.99 | 0.19 |
| F1 | 0.39 | 0.97 | 0.30 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 6,575 | 4,301 | 456 |
| Pred_L1 | 15,015 | 700,415 | 25,153 |
| Pred_L2 | 121 | 2,871 | 6,203 |

**结论**: 全量数据训练后L0/L2 Precision进一步提升！L0 P=0.58, L2 P=0.67，MacroF1=55.6%为目前最佳

---

### Exp #11: 全量数据训练 - Scheme C' [1.5, 1.0, 1.5]
- **日期**: 2026-05-14
- **数据集**: dataset（15天，30特征，380万样本）
- **模型**: FCmodel
- **配置**: weights=[1.5, 1.0, 1.5], lr=0.001, 全量数据3M, 早停patience=15
- **结果**: Epoch 30, Acc=93.6%, MacroF1=55.3%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.59 | 0.95 | 0.63 |
| Recall | 0.29 | 0.99 | 0.20 |
| F1 | 0.39 | 0.97 | 0.31 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 6,348 | 4,043 | 348 |
| Pred_L1 | 15,166 | 700,039 | 25,161 |
| Pred_L2 | 197 | 3,505 | 6,303 |

**结论**: L0 P略提升(0.59)，但L2 P下降(0.63 vs 0.67)。整体MacroF1略低(55.3% vs 55.6%)

---

### Exp #12: 30天特征长度 - dataset_30
- **日期**: 2026-05-14
- **数据集**: dataset_30（30天，60特征，380万样本）
- **模型**: FCmodel
- **配置**: weights=[2.0, 1.0, 2.0], lr=0.001, previous_num=30, input_dim=60
- **结果**: Epoch 9, Acc=93.9%, MacroF1=58.9%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.63 | 0.95 | 0.65 |
| Recall | 0.37 | 0.99 | 0.22 |
| F1 | 0.47 | 0.97 | 0.33 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 7,820 | 4,254 | 397 |
| Pred_L1 | 12,985 | 680,726 | 23,591 |
| Pred_L2 | 115 | 3,513 | 6,775 |

**结论**: 30天特征显著提升！MacroF1=58.9% (vs 15天55.6%), L0 R=0.37 (vs 15天0.30), L2 R=0.22 (vs 15天0.19)

---

### Exp #13: Transformer 30天特征长度
- **日期**: 2026-05-16
- **数据集**: dataset_30（30天，60特征，380万样本）
- **模型**: TransformerSequenceModel
- **配置**: weights=[2.0, 1.0, 2.0], lr=0.0001, previous_num=30, d_model=256, num_layers=4
- **结果**: Epoch 46, Acc=92.9%, MacroF1=59.6%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.48 | 0.95 | 0.54 |
| Recall | 0.46 | 0.98 | 0.26 |
| F1 | 0.47 | 0.96 | 0.35 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 9,689 | 10,063 | 540 |
| Pred_L1 | 11,072 | 671,609 | 22,149 |
| Pred_L2 | 159 | 6,821 | 8,074 |

**结论**: Transformer 30天 MacroF1=59.6%略高于FC 30天(58.9%)，但L0/L2 Precision较低。L0 Recall=0.46高于FC的0.37。

---

### Exp #14: Transformer 30天 d_model=128
- **日期**: 2026-05-16
- **数据集**: dataset_30（30天，60特征，380万样本）
- **模型**: TransformerSequenceModel
- **配置**: weights=[2.0, 1.0, 2.0], lr=0.0001, previous_num=30, d_model=128, num_layers=4
- **结果**: Epoch 59, Acc=92.9%, MacroF1=59.2%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.50 | 0.95 | 0.48 |
| Recall | 0.44 | 0.97 | 0.27 |
| F1 | 0.47 | 0.96 | 0.35 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 9,247 | 8,960 | 468 |
| Pred_L1 | 11,510 | 670,930 | 22,061 |
| Pred_L2 | 163 | 8,603 | 8,234 |

**结论**: d_model=128参数量更少(801K vs 3.2M)，MacroF1=59.2%略低于256的59.6%。L0/L2 Precision较低，可尝试降低权重。

---

### Exp #15: Transformer 30天 d_model=128, layers=8
- **日期**: 2026-05-17
- **数据集**: dataset_30（30天，60特征，380万样本）
- **模型**: TransformerSequenceModel
- **配置**: weights=[2.0, 1.0, 2.0], lr=0.0001, previous_num=30, d_model=128, num_layers=8
- **结果**: Epoch 56, Acc=93.4%, MacroF1=60.2%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.51 | 0.95 | 0.56 |
| Recall | 0.47 | 0.98 | 0.25 |
| F1 | 0.49 | 0.96 | 0.35 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 9,885 | 8,993 | 616 |
| Pred_L1 | 10,878 | 673,629 | 22,338 |
| Pred_L2 | 157 | 5,871 | 7,809 |

**结论**: layers=8后 MacroF1=60.2%是目前最佳！但L0/L2 Precision仍然较低

---

### Exp #16: Transformer baostock_dataset_30 (876万样本)
- **日期**: 2026-05-20
- **数据集**: baostock_dataset_30（30天，66特征，876万样本，全10年数据）
- **模型**: TransformerSequenceModel
- **配置**: weights=[2.0, 1.0, 2.0], lr=0.0001, previous_num=30, d_model=128, num_layers=4
- **结果**: Epoch 84, Acc=91.7%, MacroF1=51.3%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.38 | 0.94 | 0.36 |
| Recall | 0.34 | 0.97 | 0.16 |
| F1 | 0.36 | 0.96 | 0.22 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 17,342 | 26,318 | 1,565 |
| Pred_L1 | 33,463 | 1,583,253 | 58,834 |
| Pred_L2 | 627 | 20,155 | 11,619 |

**结论**: 样本量增到876万后 MacroF1=51.3% 低于旧数据集的60.2%。L0/L2 Precision和Recall都下降了。可能原因：旧数据380万是筛选过的"有效"样本，新数据包含更多噪声。

---

### Exp #17: Transformer baostock_dataset_30 续训 (权重调低)
- **日期**: 2026-05-22
- **数据集**: baostock_dataset_30（30天，66特征，876万样本）
- **模型**: TransformerSequenceModel
- **配置**: weights=[1.5, 1.0, 1.5], lr=0.0001, previous_num=30, d_model=128, num_layers=4
- **训练**: 100 epochs后续训，Early stop at epoch 41
- **结果**: Epoch 26 (best), Acc=92.6%, MacroF1=52.0%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.42 | 0.94 | 0.39 |
| Recall | 0.34 | 0.98 | 0.16 |
| F1 | 0.37 | 0.96 | 0.23 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 17,407 | 22,650 | 1,386 |
| Pred_L1 | 33,539 | 1,589,271 | 59,113 |
| Pred_L2 | 486 | 17,805 | 11,519 |

**结论**: 权重从[2.0,1.0,2.0]降到[1.5,1.0,1.5]后，MacroF1从51.3%提升到52.0%，L0/L2 Precision略升。但Recall依然很低（L0=0.34, L2=0.16），模型仍偏向预测L1。

---

### Exp #18: Transformer baostock_dataset_30 (d_model=256)
- **日期**: 2026-05-23
- **数据集**: baostock_dataset_30（30天，66特征，876万样本）
- **模型**: TransformerSequenceModel
- **配置**: weights=[1.5, 1.0, 1.5], lr=0.0001, previous_num=30, d_model=256, num_layers=4
- **训练**: 100 epochs，Early stop at epoch 85
- **结果**: Epoch 70 (best), Acc=92.6%, MacroF1=52.2%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.46 | 0.94 | 0.42 |
| Recall | 0.33 | 0.98 | 0.15 |
| F1 | 0.38 | 0.96 | 0.23 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 16,852 | 19,148 | 1,026 |
| Pred_L1 | 34,150 | 1,596,121 | 60,131 |
| Pred_L2 | 430 | 14,457 | 10,861 |

**结论**: d_model从128提升到256后，MacroF1从52.0%提升到52.2%，L0 Precision从0.42提升到0.46，L2从0.39提升到0.42。参数量从80万增到319万。Epoch 70 best，训练更稳定无明显震荡。

---

### Exp #19: FC baostock_dataset_30
- **日期**: 2026-05-23
- **数据集**: baostock_dataset_30（30天，66特征，876万样本）
- **模型**: FCmodel
- **配置**: weights=[1.5, 1.0, 1.5], lr=0.0001, previous_num=30, hidden=256, num_layers=4
- **训练**: 100 epochs，Early stop at epoch 27
- **结果**: Epoch 12 (best), Acc=93.4%, MacroF1=50.1%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.58 | 0.94 | 0.60 |
| Recall | 0.25 | 0.99 | 0.11 |
| F1 | 0.35 | 0.97 | 0.19 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 12,845 | 8,598 | 758 |
| Pred_L1 | 38,279 | 1,615,994 | 63,177 |
| Pred_L2 | 308 | 5,134 | 8,083 |

**续训后下降**: 从best_model继续训练后MacroF1降至49.0%，L0 P=0.42, L2 P=0.34。模型出现明显震荡，best_epoch从12降至3。

**结论**: FC模型在baostock_dataset_30上表现不如Transformer。初期L0/L2 Precision较高（0.58/0.60）但Recall极低，续训后性能下降明显。FC模型稳定性差，不适合此数据集。

---

### Exp #20: Decoder baostock_dataset_30
- **日期**: 2026-05-25
- **数据集**: baostock_dataset_30（30天，66特征，876万样本，未shuffle）
- **模型**: DecoderTransformerModel
- **配置**: weights=[1.5, 1.0, 1.5], lr=0.0001, previous_num=30, d_model=128, num_layers=4
- **训练**: 1000 epochs，Early stop at epoch 130
- **结果**: Epoch 100 (best), Acc=92.5%, MacroF1=50.9%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.42 | 0.94 | 0.42 |
| Recall | 0.30 | 0.98 | 0.14 |
| F1 | 0.35 | 0.96 | 0.21 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 15,395 | 20,126 | 1,124 |
| Pred_L1 | 35,332 | 1,596,154 | 60,464 |
| Pred_L2 | 705 | 13,446 | 10,430 |

**结论**: Decoder收敛慢（best epoch 100），但训练非常稳定（epoch 49-130波动在50.2-50.9）。L0/L2 Precision=0.42/0.42与Encoder同期持平。

---

### Exp #21: Transformer encoder baostock_dataset_30 (shuffle)
- **日期**: 2026-05-26
- **数据集**: baostock_dataset_30（30天，66特征，876万样本，shuffle seed=42）
- **模型**: TransformerSequenceModel
- **配置**: weights=[1.5, 1.0, 1.5], lr=0.0001, previous_num=30, d_model=128, num_layers=4
- **训练**: 1000 epochs，Early stop at epoch 185
- **结果**: Epoch 155 (best), Acc=94.1%, MacroF1=51.7%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.41 | 0.96 | 0.37 |
| Recall | 0.33 | 0.98 | 0.15 |
| F1 | 0.37 | 0.97 | 0.21 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 12,312 | 17,233 | 809 |
| Pred_L1 | 24,405 | 1,628,430 | 46,954 |
| Pred_L2 | 609 | 13,811 | 8,613 |

**对比 Exp #17 (同配置，非shuffle)**: MacroF1 52.0% → 51.7%，L0 P 0.42 → 0.41，L2 P 0.39 → 0.37。Best epoch 26 → 155，收敛大幅变慢。

**结论**: Shuffle后所有指标均略有下降，收敛速度严重变慢（epoch 26→155）。原因：时间序列预测中shuffle破坏了数据的时间连续性，不同时间段的数据被混合后模型更难学习pattern。**不建议对时间序列数据使用shuffle**。

---

### Exp #22: FC baostock_dataset_30 (shuffle, L0L2F1)
- **日期**: 2026-05-26
- **数据集**: baostock_dataset_30（30天，60特征，876万样本，shuffle seed=42）
- **模型**: FCmodel
- **配置**: weights=[1.5, 1.0, 1.5], lr=0.0001, previous_num=30, hidden=256, num_layers=4
- **训练**: 1000 epochs，Early stop at epoch 52
- **结果**: Epoch 22 (best), Acc=94.7%, L0L2-F1=25.0%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.46 | 0.96 | 0.52 |
| Recall | 0.26 | 0.99 | 0.10 |
| F1 | ~0.33 | 0.97 | ~0.17 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 9,711 | 10,439 | 1,026 |
| Pred_L1 | 27,214 | 1,644,231 | 49,716 |
| Pred_L2 | 401 | 4,804 | 5,634 |

**对比 Exp #19 (FC 非shuffle)**:
- L0 P: 0.58 → 0.46
- L2 P: 0.60 → 0.52
- L0 R: 0.25 → 0.26
- L2 R: 0.11 → 0.10
- L0L2F1: 26.8% → 25.0%

**结论**: FC在shuffle数据上L0/L2 Precision明显下降（-12p/+8p），与Transformer趋势一致。shuffle对FC负面影响更大（Precision降幅超过Transformer）。FC在shuffle下收敛快（epoch 22）但指标低。

---

### Exp #23: FC baostock_dataset_30 新3特征 (high_low_ratio)
- **日期**: 2026-05-27
- **数据集**: baostock_dataset_30（30天，**99特征**，877万样本，**3特征×33天**）
- **特征**: 价格变化率30维 + 换手率30维 + 波动率(high_low_ratio)30维 = 90维
- **模型**: FCmodel
- **配置**: weights=[1.5, 1.0, 1.5], lr=0.0001, previous_num=30, hidden=256, num_layers=4
- **训练**: 200 epochs，Early stop at epoch 37
- **结果**: Epoch 17 (best), Acc=94.8%, L0L2-F1=27.7%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.50 | 0.96 | 0.52 |
| Recall | 0.29 | 0.99 | 0.11 |
| F1 | 0.37 | 0.97 | 0.18 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 10,810 | 9,850 | 1,000 |
| Pred_L1 | 26,137 | 1,644,151 | 48,938 |
| Pred_L2 | 379 | 5,473 | 6,438 |

**对比旧2特征 Exp #19 (FC, 66特征, 非shuffle)**:
- L0 P: 0.58 → 0.50 (-8p)
- L2 P: 0.60 → 0.52 (-8p)
- L0 R: 0.25 → 0.29 (+4p)
- L2 R: 0.11 → 0.11 (持平)
- L0L2-F1: 26.8% → 27.7% (+0.9p)

**结论**: 加入 high_low_ratio 波动率特征后，L0/L2 Precision 反而下降（0.58→0.50, 0.60→0.52），但 L0 Recall 略升（0.25→0.29）。整体 L0L2-F1 微升 0.9 个点。波动率特征带来了更多噪声，Precision 下降说明模型更难从 3 特征中区分极端行情。**数据特征质量问题而非模型架构问题是核心瓶颈**。

---

### Exp #24: Transformer 新3特征 baostock_dataset_30
- **日期**: 2026-05-28
- **数据集**: baostock_dataset_30（30天，99特征，877万样本）
- **模型**: TransformerSequenceModel (Encoder)
- **配置**: weights=[1.5, 1.0, 1.5], lr=0.0001, previous_num=30, d_model=128, num_layers=4
- **训练**: 1000 epochs，Early stop at epoch 141
- **结果**: Epoch 121 (best), Acc=94.1%, L0L2-F1=31.0%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.42 | 0.96 | 0.38 |
| Recall | 0.38 | 0.98 | 0.16 |
| F1 | 0.40 | 0.97 | 0.23 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 14,039 | 18,456 | 902 |
| Pred_L1 | 22,780 | 1,627,126 | 46,593 |
| Pred_L2 | 507 | 13,892 | 8,881 |

**对比 FC (Exp #23, 同配置)**:

| 指标 | FC | Transformer | 变化 |
|------|-----|-------------|------|
| L0 P | 0.50 | 0.42 | -8p |
| L2 P | 0.52 | 0.38 | -14p |
| L0 R | 0.29 | 0.38 | **+9p** |
| L2 R | 0.11 | 0.16 | **+5p** |
| L0L2-F1 | 27.7% | 31.0% | **+3.3p** |

**结论**: Transformer 在 Recall 上显著优于 FC（L0 +9p, L2 +5p），但 Precision 代价明显（L0 -8p, L2 -14p）。L0L2-F1 从 27.7% 提升到 31.0%，Transformer 的时序注意力机制确实从 3 特征中提取到了更多信号。与前几轮实验一致：**FC 偏 Precision，Transformer 偏 Recall**。收敛慢（best epoch 121 vs FC 的 17），但训练非常稳定。

---

### Exp #25: MoE Decoder 新3特征 baostock_dataset_30
- **日期**: 2026-05-30
- **数据集**: baostock_dataset_30（30天，99特征，877万样本）
- **模型**: DecoderTransformerModelMoE (4 experts, top_k=1)
- **配置**: weights=[1.5, 1.0, 1.5], lr=0.0001, previous_num=30, d_model=128, num_layers=4
- **训练**: 200 epochs，Early stop at epoch 193
- **结果**: Epoch 173 (best), Acc=94.1%, L0L2-F1=29.8%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.40 | 0.96 | 0.41 |
| Recall | 0.37 | 0.98 | 0.15 |
| F1 | 0.38 | 0.97 | 0.22 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|-------------------------|---------|---------|---------|
| Pred_L0 | 13,626 | 19,714 | 929 |
| Pred_L1 | 23,245 | 1,628,289 | 47,173 |
| Pred_L2 | 455 | 11,471 | 8,274 |

**三模型对比 (同配置: 3特征, 30天, weights=[1.5,1.0,1.5])**:

| 指标 | FC (Exp #23) | Transformer (Exp #24) | MoE (Exp #25) |
|------|-------------|----------------------|---------------|
| L0L2-F1 | 27.7% | **31.0%** | 29.8% |
| L0 P/R | 0.50/0.29 | 0.42/0.38 | 0.40/0.37 |
| L2 P/R | 0.52/0.11 | 0.38/0.16 | 0.41/0.15 |
| 参数量 | 1.08M | 0.80M | **0.54M** |
| Best epoch | 17 | 121 | 173 |

**结论**: MoE 夹在 FC 和 Transformer 之间，27.7% < 29.8% < 31.0%。比 FC 好 2.1p，比 Transformer 差 1.2p。Recall 接近 Transformer（L0=0.37 vs 0.38），但 Precision 不如。以最小参数量（0.54M）拿到中间成绩。**架构排序: Transformer Encoder > MoE Decoder > FC**。收敛最慢（epoch 173），MoE 的路由 + 负载均衡损失优化更难。验证了之前的判断：在这个数据质量下，架构差异带来的收益远小于特征工程。

---

### Exp #26: FC 新4特征 (change_delta) baostock_dataset_30
- **日期**: 2026-05-31
- **数据集**: baostock_dataset_30（30天，**132特征**，877万样本，**4特征×33天**）
- **特征**: 价格变化率30维 + 换手率30维 + 波动率30维 + **换手率变化量(change_delta)30维** = 120维
- **模型**: FCmodel
- **配置**: weights=[1.5, 1.0, 1.5], lr=0.0001, previous_num=30, hidden=256, num_layers=4
- **训练**: 200 epochs，Early stop at epoch 39
- **结果**: Epoch 19 (best), Acc=95.3%, L0L2-F1=26.0%

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.53 | 0.96 | 0.56 |
| Recall | 0.25 | 0.99 | 0.11 |
| F1 | 0.34 | 0.97 | 0.18 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|--------------------------|---------|---------|---------|
| Pred_L0 | 8,010 | 6,430 | 551 |
| Pred_L1 | 24,163 | 1,661,456 | 46,704 |
| Pred_L2 | 360 | 4,233 | 5,767 |

**对比 Exp #23 (FC 3特征, 同配置)**:

| 指标 | 3特征 (Exp #23) | 4特征 (Exp #26) | 变化 |
|------|----------------|-----------------|------|
| L0 P | 0.50 | 0.53 | **+3p** |
| L2 P | 0.52 | 0.56 | **+4p** |
| L0 R | 0.29 | 0.25 | -4p |
| L2 R | 0.11 | 0.11 | 持平 |
| L0L2-F1 | 27.7% | 26.0% | -1.7p |
| Acc | 94.8% | 95.3% | **+0.5p** |

**分析**: 换手率变化量(change_delta)带来了Precision的明显提升（L0 +3p, L2 +4p, Acc +0.5p），但Recall下滑（L0 -4p）。Precision早期达到L0=0.62/L2=0.69（epoch 10），之后随着Recall上升而衰减。Precision/Recall权衡曲线比3特征更高（同Recall下Precision更好），但最优F1点略低。作为数学上可从change_rate推导出的特征，换手率变化量有效降低了模型学习时序变化关系的难度。（FC无法有效建模相邻时间步间的除法和减法关系 → Conv1d/Attention可缓解减法但除法仍需特征工程）

---

### Exp #26: FC 新4特征 (change_delta) baostock_dataset_30 ⚠️ 已废弃
- **日期**: 2026-05-31
- **⚠️ 发现 `model.state_dict().copy()` 浅拷贝bug，模型文件被后续epoch的Adam in-place更新污染，保存的参数不等于best epoch。结果数据不可靠，已重跑Exp #28**
- **模型文件**: best_fc_30d_h256_l4_20260531_183759.pt（已损坏）
- 详见 Exp #28

---

### Exp #27: FC Two-Stage 训练 ⚠️ 已废弃
- **日期**: 2026-06-01
- **⚠️ 同上浅拷贝bug，模型文件污染。已重跑Exp #29**
- **模型文件**: best_fc_twostage_30d_h256_l4_20260601_050717.pt（已损坏）

---

### Exp #28: FC 新4特征 (复现, deepcopy修复)
- **日期**: 2026-06-01
- **数据集**: baostock_dataset_30（30天，**132特征**，877万样本，**4特征×33天**）
- **特征**: 价格变化率30维 + 换手率30维 + 波动率30维 + 换手率变化量(change_delta)30维 = 120维
- **模型**: FCmodel
- **配置**: weights=[1.5, 1.0, 1.5], lr=0.0001, previous_num=30, hidden=256, num_layers=4
- **训练**: 200 epochs，Early stop at epoch 49
- **结果**: Epoch 29 (best), Acc=95.0%, L0L2-F1=26.1%
- **模型文件**: `best_fc_30d_h256_l4_20260601_112339.pt`（✅ 已验证独立推理一致）

| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.47 | 0.96 | 0.40 |
| Recall | 0.25 | 0.99 | 0.13 |
| F1 | 0.33 | 0.97 | 0.19 |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|--------------------------|---------|---------|---------|
| Pred_L0 | 8,184 | 8,510 | 682 |
| Pred_L1 | 23,681 | 1,654,073 | 45,569 |
| Pred_L2 | 668 | 9,536 | 6,771 |

**🐛 Bug发现**: `model.state_dict().copy()` 是浅拷贝——OrderedDict里的tensor和模型参数共享内存。Adam的`addcdiv_()`是in-place操作，后续epoch持续覆盖best model state。修复为`copy.deepcopy(model.state_dict())`。

**结论**: 不同随机种子下L0L2-F1=26.1%与旧Exp #26的26.0%一致，但Precision/Recall构成不同（L0 P 0.47 vs 0.53, L2 P 0.40 vs 0.56）。随机种子导致的正常波动。

---

### Exp #29: FC Two-Stage 训练 (deepcopy修复)
- **日期**: 2026-06-01
- **数据集**: baostock_dataset_30（4特征，132维，877万样本）
- **模型**: FCmodel
- **配置**: Stage1 weights=[5.0, 1.0, 5.0], Stage2 weights=[1.0, 1.0, 1.0], lr=0.0001, hidden=256, layers=4
- **训练**: Stage1 34 epochs (best ep14), Stage2 23 epochs (best ep3)
- **模型文件**: `best_fc_twostage_30d_h256_l4_20260601_222325.pt`（✅ deepcopy修复）

**Stage 1 最佳 (Epoch 14)**:
| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | 0.33 | 0.97 | 0.27 |
| Recall | 0.37 | 0.97 | 0.22 |
| L0L2-F1 | 29.4% | | |

**Stage 2 最佳 (Epoch 3)**:
| 指标 | L0 | L1 | L2 |
|------|-----|-----|-----|
| Precision | **0.67** | 0.96 | **0.71** |
| Recall | 0.19 | 1.00 | 0.08 |
| L0L2-AvgP | **69.0%** | | |

| Confusion (pred x true) | True_L0 | True_L1 | True_L2 |
|--------------------------|---------|---------|---------|
| Pred_L0 | 6,228 | 2,865 | 193 |
| Pred_L1 | 26,160 | 1,667,706 | 48,695 |
| Pred_L2 | 145 | 1,548 | 4,134 |

**对比单阶段 Exp #28 ([1.5,1.0,1.5])**:
| 指标 | 单阶段 (Exp #28) | Two-Stage Stage2 | 变化 |
|------|----------|------------------|------|
| L0 Precision | 0.47 | **0.67** | **+20p** |
| L2 Precision | 0.40 | **0.71** | **+31p** |
| L0 Recall | 0.25 | 0.19 | -6p |
| L2 Recall | 0.13 | 0.08 | -5p |
| Pred_L0 | 17,376 | **9,286** | -47% (更精准) |
| Pred_L2 | 16,975 | **5,827** | -66% (更精准) |

**发现**:
1. **Precision 大幅领先单阶段**: Stage2 L0 P=0.67 vs 0.47 (+20p), L2 P=0.71 vs 0.40 (+31p)。等权重微调成功将Stage1的Recall资产转化为高Precision。
2. **假阳性大幅减少**: Pred_L0从17,376降到9,286 (-47%), Pred_L2从16,975降到5,827 (-66%)。模型更「克制」，只在高置信度时预测极端类别。
3. **Stage2 最佳在 Epoch 3**: 之后 Precision 开始下降（L0 0.67→0.46, L2 0.71→0.40），等权重下模型逐渐重新偏向L1。
4. **这次 Two-stage 成功超越单阶段**: 相比旧 Exp #27（浅拷贝bug），修复后 Stage2 Precision 显著提升（0.67/0.71 vs 旧0.51/0.50），说明之前 bug 也严重影响了 two-stage 的结果。

**结论**: **Two-stage 是目前Precision最优方案**。Stage1高权重拉Recall → Stage2等权重微调收Precision。代价是Recall很低（L0=0.19, L2=0.08），但预测信号可信度高。适合「宁可错过不可错判」的投资策略。

---

### Exp #30: FC 1M 新4分类基准 (json配置, 新模型结构)
- **日期**: 2026-06-04
- **数据集**: baostock_dataset_30（4特征，877万样本）
- **模型**: FCmodel (新结构: embed + MLP blocks + classifier)
- **配置**: `configs/FC_1M.json`, hidden=256, layers=4, weights=[1,1,1,1], lr=0.0001
- **训练**: 200 epochs，Early stop at epoch 38
- **结果**: Epoch 18 (best)，Score=66.9
- **模型文件**: `best_fc_30d_h256_l4_20260603_235931.pt`

| 指标 | L0 | L1 | L2 | L3 |
|------|-----|-----|-----|-----|
| Precision | 0.65 | 0.62 | 0.63 | 0.69 |
| Recall | 0.23 | 0.72 | 0.56 | 0.09 |

| Confusion | True_L0 | True_L1 | True_L2 | True_L3 |
|-----------|---------|---------|---------|---------|
| Pred_L0 | 7,467 | 2,828 | 803 | 351 |
| Pred_L1 | 20,801 | 625,526 | 348,845 | 21,657 |
| Pred_L2 | 4,007 | 236,755 | 455,444 | 26,245 |
| Pred_L3 | 258 | 569 | 1,349 | 4,769 |

---

### Exp #31: FC 1M Two-Stage 4分类 (json配置)
- **日期**: 2026-06-04
- **配置**: `configs/FC_1M.json --two-stage`, Stage1 weights=[5,1,1,5], Stage2 weights=[1,1,1,1]
- **训练**: Stage1 32 epochs (best ep12), Stage2 21 epochs (best ep1)
- **模型文件**: `best_fc_twostage_256_20260604_103337.pt`

| 阶段 | L0 P | L1 P | L2 P | L3 P | L0 R | L1 R | L2 R | L3 R | Score |
|------|------|------|------|------|------|------|------|------|-------|
| Stage1 | 0.32 | 0.59 | 0.61 | 0.21 | 0.40 | 0.68 | 0.49 | 0.29 | 26.6 |
| Stage2 | 0.63 | 0.59 | 0.61 | 0.64 | 0.22 | 0.71 | 0.53 | 0.09 | 63.6 |

**对比单阶段 (Exp #30)**:
| 指标 | 单阶段 | Two-Stage | 变化 |
|------|--------|-----------|------|
| L0 P | 0.65 | 0.63 | -2p |
| L3 P | 0.69 | 0.64 | -5p |
| L0 R | 0.23 | 0.22 | -1p |
| L3 R | 0.09 | 0.09 | 持平 |
| Score | 66.9 | 63.6 | -3.3p |

**结论**: 4分类下 two-stage 仍未超越单阶段。Stage1 高权重拉 Recall 的扰动在 Stage2 中无法完全恢复。直接等权重训练效果最好。

---

---

### Exp #32: FC 3M 4分类 (scaling test)
- **日期**: 2026-06-05
- **配置**: `configs/FC_3M.json`, hidden=512, layers=6, weights=[1,1,1,1]
- **训练**: 200 epochs，Early stop at epoch 28
- **结果**: Epoch 8 (best)，Score=67.7
- **模型文件**: `best_fc_30d_h512_l6_20260605_001604.pt`

| 指标 | L0 | L1 | L2 | L3 |
|------|-----|-----|-----|-----|
| Precision | 0.67 | 0.61 | 0.63 | 0.68 |
| Recall | 0.23 | 0.71 | 0.57 | 0.09 |

**FC Scaling 对比**:
| 模型 | 参数量 | Score | L0 P | L3 P | Best Epoch |
|------|--------|-------|------|------|------------|
| FC 1M (Exp #30) | 1.08M | 66.9 | 0.65 | 0.69 | 18 |
| FC 3M (Exp #32) | 6.37M | **67.7** | 0.67 | 0.68 | 8 |

**结论**: 6x 参数只换来 +0.8 Score，且大模型更早过拟合（best epoch 8 vs 18）。**当前数据信噪比已到天花板**，加容量无显著收益。单阶段 FC 1M 是性价比最优。

---

---

### Exp #33: Encoder d128 l4 4分类基准
- **日期**: 2026-06-05
- **配置**: `configs/Encoder_d128_l4.json`, d_model=128, layers=4, weights=[1,1,1,1]
- **训练**: 200 epochs，Early stop at epoch 25
- **结果**: Epoch 5 (best)，Score=54.3
- **模型文件**: `best_encoder_30d_d128_l4_20260605_131208.pt`

| 指标 | L0 | L1 | L2 | L3 |
|------|-----|-----|-----|-----|
| Precision | 0.52 | 0.54 | 0.53 | 0.57 |
| Recall | 0.05 | 0.59 | 0.52 | 0.04 |

**FC vs Encoder 4分类对比**:
| 指标 | FC 1M (Exp #30) | Encoder d128 (Exp #33) |
|------|-----------------|----------------------|
| Score | **66.9** | 54.3 |
| L0 P/R | 0.65/0.23 | 0.52/0.05 |
| L3 P/R | 0.69/0.09 | 0.57/0.04 |
| 参数量 | 1.08M | 0.80M |
| Best Epoch | 18 | 5 |

**结论**: 4分类下 Encoder 显著差于 FC（Score -12.6）。Best epoch 仅 5 即过拟合。与旧 3 特征时期 FC < Encoder 的排序完全反转。可能原因：1) 4 特征 ≤ 旧特征工程信号质量 2) Encoder 可能需要更低 lr 或更大 d_model。

---

---

### Exp #34: Encoder d256 l6 4分类 (scaling test)
- **日期**: 2026-06-06
- **配置**: `configs/Encoder_d256_l6.json`, d_model=256, layers=6, weights=[1,1,1,1]
- **训练**: 200 epochs，Early stop at epoch 46
- **结果**: Epoch 26 (best)，Score=56.9
- **模型文件**: `best_encoder_30d_d256_l6_20260606_195745.pt`

| 指标 | L0 | L1 | L2 | L3 |
|------|-----|-----|-----|-----|
| Precision | 0.55 | 0.65 | 0.65 | 0.59 |
| Recall | 0.36 | 0.70 | 0.64 | 0.12 |

**Encoder Scaling 对比**:
| 模型 | 参数量 | Score | L0 P/R | L3 P/R | Best Epoch |
|------|--------|-------|--------|--------|------------|
| Encoder d128 l4 (Exp #33) | 0.80M | 54.3 | 0.52/0.05 | 0.57/0.04 | 5 |
| Encoder d256 l6 (Exp #34) | 4.77M | **56.9** | 0.55/0.36 | 0.59/0.12 | 26 |

**4分类完整对比**:
| 模型 | 参数量 | Score | L0 P | L3 P | L0 R | L3 R |
|------|--------|-------|------|------|------|------|
| **FC 1M** | 1.08M | **66.9** | **0.65** | **0.69** | 0.23 | 0.09 |
| FC 3M | 6.37M | 67.7 | 0.67 | 0.68 | 0.23 | 0.09 |
| Encoder d128 l4 | 0.80M | 54.3 | 0.52 | 0.57 | 0.05 | 0.04 |
| Encoder d256 l6 | 4.77M | 56.9 | 0.55 | 0.59 | 0.36 | 0.12 |

**结论**: 
- FC 在 Precision 上全面碾压 Encoder（+10p L0, +10p L3），符合你「Precision 优先」的目标
- Encoder 的 Recall 更高但 Precision 代价太大，Score 差 10 个点
- 规律和旧 3 特征时期完全一致：**FC 偏 Precision，Encoder 偏 Recall**
- 4 分类最优解：**FC 1M**（性价比 + 精度双赢）

---

---

### Exp #35: Decoder d128 l4 4分类
- **日期**: 2026-06-07
- **配置**: `configs/Decoder_d128_l4.json`, d_model=128, layers=4, weights=[1,1,1,1]
- **训练**: 200 epochs，Early stop at epoch 28
- **结果**: Epoch 8 (best)，Score=54.1
- **模型文件**: `best_decoder_30d_d128_l4_20260607_043727.pt`

| 指标 | L0 | L1 | L2 | L3 |
|------|-----|-----|-----|-----|
| Precision | 0.53 | 0.56 | 0.55 | 0.56 |
| Recall | 0.13 | 0.63 | 0.52 | 0.05 |

**4分类完整对比**:
| 模型 | 参数量 | Score | L0 P | L3 P | L0 R | L3 R |
|------|--------|-------|------|------|------|------|
| **FC 1M** | 1.08M | **66.9** | **0.65** | **0.69** | 0.23 | 0.09 |
| FC 3M | 6.37M | 67.7 | 0.67 | 0.68 | 0.23 | 0.09 |
| Encoder d128 l4 | 0.80M | 54.3 | 0.52 | 0.57 | 0.05 | 0.04 |
| Encoder d256 l6 | 4.77M | 56.9 | 0.55 | 0.59 | 0.36 | 0.12 |
| Decoder d128 l4 | 1.07M | 54.1 | 0.53 | 0.56 | 0.13 | 0.05 |

---

### Exp #36: Encoder d128 l4 4分类 (新Score重跑)
- **日期**: 2026-06-09
- **配置**: `configs/Encoder_d128_l4.json`, d_model=128, layers=4, weights=[1,1,1,1]
- **训练**: 200 epochs full，新Score=距离加权全量样本
- **结果**: Epoch 197 (best)，Score=88.7
- **模型文件**: `best_encoder_30d_d128_l4_20260609_012534.pt`

| 指标 | L0 | L1 | L2 | L3 |
|------|-----|-----|-----|-----|
| Precision | 0.44 | 0.65 | 0.65 | 0.54 |
| Recall | 0.40 | 0.71 | 0.62 | 0.13 |

**vs 旧Score**: 54.3(旧) → 88.7(新)。新 metric 下 Score 整体偏高，因为 L1/L2 样本占 ~94%，差一档也有 0.7 分。极端类 Precision 反而不如旧 metric 下高（0.44 vs 0.52），因为新 metric 选 best 时更看重全量样本而非极端类。

---

**首次4分类基准**: L0/L3 Precision 0.65/0.69 明显高于3分类时期（~0.53/0.56），L1拆分为L1(-10%~0)和L2(0~10%)后模型能有效区分。Recall 分布合理（L3低是因为极端上涨稀有）。模型结构改为embed+backbone+classifier，特征顺序对齐stack(axis=0)。

---

## 重要Bug修复记录

### `model.state_dict().copy()` 浅拷贝Bug (2026-06-01)
- **影响**: 所有之前保存的模型文件（Exp #19-#27）的.pt文件参数被后续epoch污染
- **原因**: `OrderedDict.copy()`是浅拷贝，tensor对象和模型参数共享内存。Adam optimizer的`addcdiv_()`是in-place操作
- **修复**: 改为`copy.deepcopy(model.state_dict())`（train.py + train_two_stage.py）
- **来源**: Claude Code 对话中通过对比.pt文件推理结果与训练日志发现不一致，追踪到save逻辑

---

## 结论汇总 (updated 2026-06-01)

### 当前最佳方案

| 方案 | 模型 | L0 P | L2 P | L0 R | L2 R | 模型文件 |
|------|------|------|------|------|------|---------|
| **Precision优先** | FC Two-Stage (Exp #29) | **0.67** | **0.71** | 0.19 | 0.08 | best_fc_twostage_30d_h256_l4_20260601_222325.pt |
| 均衡 | FC Single (Exp #28) | 0.47 | 0.40 | 0.25 | 0.13 | best_fc_30d_h256_l4_20260601_112339.pt |

- **Two-Stage**: Stage1 weights=[5,1,5] 拉Recall → Stage2 weights=[1,1,1] 收Precision。Precision极致（0.67/0.71），预测信号可信，但Recall低
- **单阶段**: weights=[1.5,1.0,1.5]，Precision/Recall均衡

### 关键发现
1. **降低类别权重显著提升L0/L2 Precision**：
   - 原权重 [14.64, 0.35, 9.44] → L0 P=0.09, L2 P=0.09
   - Scheme A [5, 1, 5] → L0 P=0.38, L2 P=0.36 (200K样本)
   - Scheme B [3, 1, 3] → L0 P=0.44, L2 P=0.50 (200K样本)
   - **Scheme C [2, 1, 2] → L0 P=0.58, L2 P=0.67 (全量3M数据, 15天)** ← 原最佳
   - **Scheme C' [1.5, 1.0, 1.5] → L0 P=0.59, L2 P=0.63** (L0略高但L2下降)

2. **增加特征长度到30天效果显著** (Exp #12 vs #10)：
   - 15天FC: MacroF1=55.6%, L0 P=0.58, L2 P=0.67
   - **30天FC: MacroF1=58.9%, L0 P=0.63, L2 P=0.65, L0 R=0.37, L2 R=0.22**
   - **30天Transformer: MacroF1=59.6%, L0 R=0.46, L2 R=0.26 (Recall更高)** ← 新最佳

3. **FC vs Transformer（新3特征数据集）**：
   - FC: L0L2-F1=27.7%, L0 P=0.50, L2 P=0.52, L0 R=0.29, L2 R=0.11
   - **Transformer: L0L2-F1=31.0%, L0 P=0.42, L2 P=0.38, L0 R=0.38, L2 R=0.16** ← 新最佳
   - 规律：FC 偏 Precision，Transformer 偏 Recall。L0L2-F1 差 3.3p

4. **全量数据 vs 采样数据**：
   - 相同权重下，全量数据训练的L0/L2 Precision明显更高
   - 全量数据: MacroF1=55.6%, L0 P=0.58, L2 P=0.67
   - 200K采样: MacroF1=51.3%, L0 P=0.47, L2 P=0.52

4. **baostock新数据降权效果** (Exp #17)：
   - weights [2.0,1.0,2.0] → MacroF1=51.3%
   - weights [1.5,1.0,1.5] → MacroF1=52.0%，有提升但有限
   - 核心问题：L0/L2 Recall极低（L0=0.34, L2=0.16），模型偏向L1预测

4. **权衡**：权重越低，L0/L2 Precision越高，但Recall越低

5. **之前 Focal Loss 实验的问题根源**：权重过高导致模型过度预测少数类，牺牲了Precision

### 目标说明
对于"筛选高波动股票"的目标：
- L0 (下跌) 和 L2 (上涨) 的 **Precision** 和 **Recall** 都重要
- **不接受通过大量误判L1来提高L0/L2 Recall的方式**

### 下一步方向
1. [x] 尝试权重 [1.5, 1.0, 1.5] 看能否进一步提升 ✓
2. [x] 增加特征长度到30天（需验证数据是否支持）✓
3. [x] 添加技术指标特征 (high_low_ratio) — Exp #23 结果：L0L2-F1=27.7%，Precision反而下降
4. [ ] Transformer 在3特征数据集上的表现
5. [ ] Decoder/MoE 在3特征数据集上的表现

---

## 待做实验
- [x] 降低类别权重实验 (Exp #7, #8, #9) ✓
- [x] 全量数据训练 Scheme C [2, 1, 2] (Exp #10) ✓
- [x] 全量数据训练 Scheme C' [1.5, 1.0, 1.5] (Exp #11) ✓
- [x] 30天特征长度FC训练 (Exp #12) ✓
- [x] 30天特征长度Transformer d_model=256训练 (Exp #13) ✓
- [x] 30天特征长度Transformer d_model=128训练 (Exp #14) ✓
- [ ] Transformer d_model=128 + 降低权重实验

---

## 数据集配置说明

### dataset (15天)
- previous_num=15, predict_num=3
- 特征: 30维 (价格变化率15维 + 换手率15维)

### dataset_30 (30天, 旧2特征)
- previous_num=30, predict_num=3
- 特征: 60维 (价格变化率30维 + 换手率30维)
- 生成命令: `python data_preprocess/preprocess.py`
- 输出目录: `dataset_30/`

### baostock_dataset_30 (30天, 新3特征) ← 当前使用
- previous_num=30, predict_num=3
- **特征: 99维 (价格变化率30维 + 换手率30维 + 波动率30维 + 预测价格3维 + 预测换手率3维 + 预测波动率3维)**
- 原始middle数据: `middle_data_baostock/` (mean_adj, change_rate, high_low_ratio)
- 生成命令: `python data_preprocess/preprocess.py --middle middle_data_baostock --output baostock_dataset_30`
- 样本量: ~877万，文件数: 877个 × 10000
