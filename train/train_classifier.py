#!/usr/bin/env python3
"""
固定 Backbone, 只训练新的分类头 (适应任意类别数)
用法:
  python train/train_classifier.py --backbone best_fc_twostage_30d_h256_l4_20260601_222325.pt --classes 4
"""
import os, sys, time, argparse, copy
from datetime import datetime
sys.path.insert(0, '.')
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm

# ============ 默认参数 ============
PREVIOUS_NUM = 30
PREDICT_NUM = 3
DATASET_PATH = "baostock_dataset_30"
LR = 0.001          # head 学习率可以高一些
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_SIZE = 2048
HIDDEN_SIZE = 256
NUM_LAYERS = 4

# ============ Data Pipeline (同 train.py, 含内存大小判断) ============
_global_data = None

def _cal_pct(d):
    r = np.zeros_like(d)
    with np.errstate(divide='ignore', invalid='ignore'):
        r[1:] = (d[1:] - d[:-1]) / d[:-1]
        r = np.where(np.isfinite(r), r, 0)
    return r

def _get_label(pct_chg):
    if pct_chg < -0.1: return 0
    elif pct_chg < 0: return 1
    elif pct_chg <= 0.1: return 2
    else: return 3

class DS(Dataset):
    def __init__(self, data, start, end, previous=30, predict=3):
        self.data = data[start:end]
        self.previous = previous; self.predict = predict
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        raw = self.data[idx]
        pp = self.previous + self.predict
        mean = raw[:pp]; change = raw[pp:2*pp]; hl = raw[2*pp:3*pp]; cd_ = raw[3*pp:4*pp]
        price_pct = _cal_pct(mean[:self.previous])
        pct_chg = (mean[-1] - mean[self.previous - 1]) / mean[self.previous - 1]
        label = _get_label(pct_chg)
        features = np.stack([price_pct, change[:self.previous], hl[:self.previous], cd_[:self.previous]], axis=-1)
        return features.astype(np.float32), label

class DSShared(Dataset):
    def __init__(self, start, end, previous=30, predict=3):
        self.start = start; self.end = end
        self.previous = previous; self.predict = predict
    def __len__(self):
        return self.end - self.start
    def __getitem__(self, idx):
        raw = _global_data[idx + self.start].numpy()
        pp = self.previous + self.predict
        mean = raw[:pp]; change = raw[pp:2*pp]; hl = raw[2*pp:3*pp]; cd_ = raw[3*pp:4*pp]
        price_pct = _cal_pct(mean[:self.previous])
        pct_chg = (mean[-1] - mean[self.previous - 1]) / mean[self.previous - 1]
        label = _get_label(pct_chg)
        features = np.stack([price_pct, change[:self.previous], hl[:self.previous], cd_[:self.previous]], axis=-1)
        return features.astype(np.float32), label


def main():
    parser = argparse.ArgumentParser(description="固定 Backbone 训练新分类头")
    parser.add_argument('--backbone', type=str, required=True, help='预训练 .pt 文件路径')
    parser.add_argument('--classes', type=int, default=4, help='新分类头类别数 (默认4)')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH)
    parser.add_argument('--lr', type=float, default=LR)
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS)
    parser.add_argument('--patience', type=int, default=PATIENCE)
    parser.add_argument('--weights', type=float, nargs='+', default=None,
                        help=f'类别权重 (默认全1.0)')
    parser.add_argument('--unfreeze', type=int, default=2,
                        help='解冻 backbone 最后 N 层 (默认2, 0=只训classifier)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_class = args.classes

    # 权重: 默认全 1.0
    if args.weights:
        weights = args.weights
    else:
        weights = [1.0] * num_class

    print("=" * 60)
    print(f"固定 Backbone 训练新分类头 ({num_class}类)")
    print("=" * 60)
    print(f"Backbone: {args.backbone}")
    print(f"类别数: {num_class}")
    print(f"类别权重: {weights}")
    print(f"学习率: {args.lr}")
    print(f"解冻最后: {args.unfreeze} 层")

    # ── 加载数据 (同 train.py 的内存判断) ──
    global _global_data
    files = sorted([f for f in os.listdir(args.dataset) if f.endswith('.npy')])
    total_bytes = sum(os.path.getsize(os.path.join(args.dataset, f)) for f in files)
    IN_MEMORY_THRESHOLD = 8 * 1024**3  # 8GB

    if total_bytes < IN_MEMORY_THRESHOLD:
        print(f'数据大小: {total_bytes/1024**3:.2f}GB < 8GB, 加载到共享内存')
        all_data = []
        for f in files:
            all_data.append(np.load(os.path.join(args.dataset, f)))
        all_data = np.concatenate(all_data, axis=0)
        print(f'Data shape: {all_data.shape}')
        np.random.seed(42)
        np.random.shuffle(all_data)
        _global_data = torch.from_numpy(all_data).share_memory_()
        del all_data

        total_num = len(_global_data)
        split = int(0.8 * total_num)
        print(f'训练集: {split:,}, 测试集: {total_num - split:,}')
        train_ds = DSShared(0, split, PREVIOUS_NUM, PREDICT_NUM)
        test_ds = DSShared(split, total_num, PREVIOUS_NUM, PREDICT_NUM)
    else:
        print(f'数据大小: {total_bytes/1024**3:.2f}GB > 8GB, 使用单进程')
        all_data = []
        for f in files:
            all_data.append(np.load(os.path.join(args.dataset, f)))
        all_data = np.concatenate(all_data, axis=0)
        print(f'Data shape: {all_data.shape}')
        np.random.seed(42)
        np.random.shuffle(all_data)

        total_num = len(all_data)
        split = int(0.8 * total_num)
        print(f'训练集: {split:,}, 测试集: {total_num - split:,}')
        train_ds = DS(all_data, 0, split, PREVIOUS_NUM, PREDICT_NUM)
        test_ds = DS(all_data, split, total_num, PREVIOUS_NUM, PREDICT_NUM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # ── 构建模型 ──
    from models.FCmodel import FCmodel
    model = FCmodel(feature_dim=4, num_class=num_class, hidden_size=HIDDEN_SIZE,
                    num_layers=NUM_LAYERS, seq_len=PREVIOUS_NUM).to(device)
    # 加载 backbone
    backbone_state = FCmodel.extract_backbone_from_pt(args.backbone, map_location=device)
    model.load_backbone_state_dict(backbone_state)
    model.freeze_backbone(unfreeze_last=args.unfreeze)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"可训练参数: {trainable_params:,}  冻结参数: {frozen_params:,}")

    # ── 训练 ──
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # 注意: freeze_backbone 后只有 classifier 参数有 require_grad=True

    n_classes = num_class
    best_score = -1  # 加权得分(%), 越大越好
    best_epoch = 0
    best_model_state = None
    best_confusion = None
    no_improve = 0
    best_precision = [0.0] * n_classes
    best_recall = [0.0] * n_classes

    # 打印表头
    p_heads = ' '.join([f'P_L{i}' for i in range(n_classes)])
    r_heads = ' '.join([f'R_L{i}' for i in range(n_classes)])
    print(f"\n{'='*70}")
    print(f"Epoch | Acc(%) | Score | [{p_heads}] | [{r_heads}] | Best")
    print(f"{'='*70}")

    for epoch in range(args.epochs):
        t0 = time.time()
        # 训练: 只更新 classifier
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} Train', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 评估
        model.eval()
        confusion_matrix = [[0] * n_classes for _ in range(n_classes)]
        with torch.no_grad():
            pbar_test = tqdm(test_loader, desc=f'Epoch {epoch+1} Test ', leave=False)
            for inputs, labels in pbar_test:
                inputs = inputs.float().to(device)
                outputs = model(inputs)
                _, pred = torch.max(outputs, 1)
                for p, t in zip(pred.cpu().numpy(), labels.numpy()):
                    confusion_matrix[p][t] += 1

        precision = [0.0] * n_classes
        recall = [0.0] * n_classes
        f1 = [0.0] * n_classes
        for c in range(n_classes):
            tp = confusion_matrix[c][c]
            fp = sum(confusion_matrix[c]) - tp
            fn = sum(row[c] for row in confusion_matrix) - tp
            precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1[c] = 2 * precision[c] * recall[c] / (precision[c] + recall[c]) if (precision[c] + recall[c]) > 0 else 0

        total_correct = sum(confusion_matrix[i][i] for i in range(n_classes))
        accuracy = 100.0 * total_correct / sum(sum(row) for row in confusion_matrix)
        macro_f1 = sum(f1) / n_classes * 100
        avg_extreme_p = (precision[0] + precision[n_classes - 1]) / 2 * 100

        # 主要指标: 距离加权得分 — 正确1分, 差1档0.7, 差2档0.3, 远0
        score_weights = [[0.0]*n_classes for _ in range(n_classes)]
        for p in range(n_classes):
            for t in range(n_classes):
                d = abs(p - t)
                score_weights[p][t] = 1.0 if d == 0 else (0.7 if d == 1 else (0.3 if d == 2 else 0.0))
        total_score = 0
        total_samples = 0
        for t in range(n_classes):
            for p in range(n_classes):
                count = confusion_matrix[p][t]
                total_score += score_weights[p][t] * count
                total_samples += count
        weighted_score = total_score / total_samples * 100 if total_samples > 0 else 0

        is_best = weighted_score > best_score
        if is_best:
            best_score = weighted_score
            best_epoch = epoch + 1
            best_confusion = [row[:] for row in confusion_matrix]
            best_precision = precision[:]
            best_recall = recall[:]
            best_model_state = copy.deepcopy(model.state_dict())
            no_improve = 0
            marker = '*'
        else:
            no_improve += 1
            marker = ''

        elapsed = time.time() - t0
        p_str = ' '.join([f'{precision[i]:.2f}' for i in range(n_classes)])
        r_str = ' '.join([f'{recall[i]:.2f}' for i in range(n_classes)])
        print(f'{epoch+1:>5} | {accuracy:>6.1f} | {weighted_score:>5.1f} | [{p_str}] | [{r_str}] | {best_epoch}{marker}')

        if no_improve >= args.patience:
            print(f'\nEarly stop at epoch {epoch+1}')
            break

    # ── 最终结果 ──
    print(f"\n{'='*70}")
    print(f"Best: Epoch {best_epoch}, Score={best_score:.3f} (L0P={best_precision[0]:.2f} L{n_classes-1}P={best_precision[n_classes-1]:.2f})")
    print(f"Precision: {' '.join([f'L{i}={best_precision[i]:.2f}' for i in range(n_classes)])}")
    print(f"Recall:    {' '.join([f'L{i}={best_recall[i]:.2f}' for i in range(n_classes)])}")
    print(f"\nConfusion Matrix (pred rows x true cols):")
    print(f"         {' '.join([f'True_L{i}  ' for i in range(n_classes)])}")
    for i, row in enumerate(best_confusion):
        print(f"Pred_L{i}  {'  '.join([f'{v:>9}' for v in row])}")

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'best_head_{num_class}class_{PREVIOUS_NUM}d_h{HIDDEN_SIZE}_l{NUM_LAYERS}_{timestamp}.pt'
    torch.save(best_model_state, model_path)
    print(f'\nModel saved: {model_path}')


if __name__ == "__main__":
    main()
