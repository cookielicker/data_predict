# Stock Prediction — A股三天涨跌预测

4分类：L0(跌>10%) / L1(-10%~0%) / L2(0%~10%) / L3(涨>10%)

## 数据管线

```bash
# 1. 拉取全A股K线 (前复权, 2016-06-01 ~ 最新)
python fetch_data/fetch_baostock.py

# 增量刷新 (只拉缺失的最近几天, 快)
python fetch_data/fetch_baostock.py --refresh

# 指定日期范围 + 强制覆盖
python fetch_data/fetch_baostock.py --start 2024-01-01 --end 2026-06-03 --force

# 2. CSV → 中间矩阵 (按特征×日期对齐所有股票)
python data_preprocess/convert_csv_to_middle.py

# 3. 中间矩阵 → 最终数据集 (30天滑动窗口, 132维, ~877万样本)
python data_preprocess/preprocess.py --middle middle_data_baostock --output baostock_dataset_30
```

## 训练

```bash
# === 单阶段 (推荐) ===
python train/train_config.py --config configs/FC_1M.json         # FC 1M 基准
python train/train_config.py --config configs/FC_3M.json         # FC 3M
python train/train_config.py --config configs/Encoder_d128_l4.json  # Transformer Encoder 小
python train/train_config.py --config configs/Encoder_d256_l6.json  # Transformer Encoder 大
python train/train_config.py --config configs/Decoder_d128_l4.json  # Decoder-only
python train/train_config.py --config configs/MoE_d128_l4.json      # MoE Decoder

# === Two-Stage (Stage1高权重拉Recall → Stage2等权重收Precision) ===
python train/train_config.py --config configs/FC_1M.json --two-stage

# === 覆盖参数 ===
python train/train_config.py --config configs/FC_1M.json --override training.lr=0.01 model.hidden_size=512

# === 固定Backbone只训新分类头 (换类别数) ===
python train/train_classifier.py --backbone <model.pt> --classes 4 --unfreeze 2
```

## 可视化

```bash
# 全A股实时扫描 (刷新数据 + 滑动窗口预测)
python visualize_stock_scan.py

# 测试集回测浏览 (加载模型, 按预测类别查看样本)
python visualize_model_predictions.py

# 原始数据集浏览 (查看4特征走势)
python visualize_dataset.py
```

## 模型配置

| 配置文件 | 架构 | 参数量 |
|----------|------|--------|
| `configs/FC_1M.json` | FC h256 l4 | 1.08M |
| `configs/FC_3M.json` | FC h512 l6 | 6.37M |
| `configs/Encoder_d128_l4.json` | Encoder d128 l4 | 0.80M |
| `configs/Encoder_d256_l6.json` | Encoder d256 l6 | 4.77M |
| `configs/Decoder_d128_l4.json` | Decoder d128 l4 | 1.07M |
| `configs/MoE_d128_l4.json` | MoE Decoder d128 l4 | 0.54M |

## 目录结构

```
baostock_dataset_30/    ← 最终数据集 (879个.npy, 每文件10000行)
stock_data/             ← baostock原始CSV (每只股票一个文件)
middle_data_baostock/   ← 中间矩阵 (4个特征×(天×股票))
configs/                ← JSON训练配置
models/                 ← 模型定义 (FCmodel, sequence_models)
train/                  ← 训练脚本 (train_config, train_classifier, train, train_two_stage)
fetch_data/             ← 数据获取 (fetch_baostock)
data_preprocess/        ← 数据预处理 (convert_csv_to_middle, preprocess)
EXPERIMENT_LOG.md       ← 实验记录
```

## 评估

Score = 全量样本距离加权得分，按样本数加权 ×100:

| Pred \ True | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| L0 | 1.0 | 0.7 | 0.3 | 0.0 |
| L1 | 0.7 | 1.0 | 0.7 | 0.3 |
| L2 | 0.3 | 0.7 | 1.0 | 0.7 |
| L3 | 0.0 | 0.3 | 0.7 | 1.0 |

当前最佳: **FC 1M** — 详见 `EXPERIMENT_LOG.md`

## 模型架构

所有模型统一输入 `[bs, 30, 4]` (30天×4特征)，Backbone + Classifier 分离，支持冻结Backbone只训分类头。

| 模型 | 结构 | Backbone | Classifier |
|------|------|----------|------------|
| FCmodel | embed → MLP blocks | embed + layers | Linear(hidden, n_class) |
| EncoderModel | embedding → TransformerEncoder | embed + pos + encoder | MLP(d_model→n_class) |
| DecoderTransformerModel | embedding → TransformerDecoder | embed + pos + decoder | MLP(d_model→n_class) |
| DecoderTransformerModelMoE | embedding → MoE Decoder | embed + pos + moe_layers | MLP(d_model→n_class) |
