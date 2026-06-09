#!/usr/bin/env python3
"""
两阶段训练: Stage 1 高权重拉Recall → Stage 2 等权重收Precision
用法:
  python train/train_two_stage.py --model fc --previous 30 --dataset baostock_dataset_30
"""
import os, sys, time, argparse
from datetime import datetime
sys.path.insert(0, '.')
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import torch
import torch.nn as nn
import copy
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm

# ============ 默认参数 ============
PREVIOUS_NUM = 30
PREDICT_NUM = 3
DATASET_PATH = "baostock_dataset_30"
STAGE1_WEIGHTS = [5.0, 1.0, 1.0, 5.0]
STAGE2_WEIGHTS = [1.0, 1.0, 1.0, 1.0]
LEARNING_RATE = 0.0001
BATCH_SIZE = 2048
STAGE1_EPOCHS = 300
STAGE2_EPOCHS = 300
PATIENCE = 20
HIDDEN_SIZE = 256
NUM_LAYERS = 4
D_MODEL = 128
MODEL_TYPE = "fc"
# ==================================

_global_data = None


class DS:
    def __init__(self, data, start, end, previous=30, predict=3, transformer_format=False):
        self.data = data[start:end]
        self.previous = previous
        self.predict = predict
        self.transformer_format = transformer_format

    def __len__(self):
        return len(self.data)

    def _cal_pct(self, d):
        r = np.zeros_like(d)
        with np.errstate(divide='ignore', invalid='ignore'):
            r[1:] = (d[1:] - d[:-1]) / d[:-1]
            r = np.where(np.isfinite(r), r, 0)
        return r

    def __getitem__(self, idx):
        raw = self.data[idx]
        pp = self.previous + self.predict
        mean = raw[:pp]
        change = raw[pp:2 * pp]
        high_low = raw[2 * pp:3 * pp]
        change_delta = raw[3 * pp:4 * pp]

        price_pct = self._cal_pct(mean[:self.previous])
        pct_chg = (mean[-1] - mean[self.previous - 1]) / mean[self.previous - 1]
        if pct_chg < -0.1: label = 0
        elif pct_chg < 0: label = 1
        elif pct_chg <= 0.1: label = 2
        else: label = 3

        features = np.stack([price_pct, change[:self.previous], high_low[:self.previous], change_delta[:self.previous]], axis=-1)
        return features.astype(np.float32), label


class DSShared(Dataset):
    def __init__(self, start, end, previous=30, predict=3, transformer_format=False):
        self.start = start
        self.end = end
        self.previous = previous
        self.predict = predict
        self.transformer_format = transformer_format

    def __len__(self):
        return self.end - self.start

    def _cal_pct(self, d):
        r = np.zeros_like(d)
        with np.errstate(divide='ignore', invalid='ignore'):
            r[1:] = (d[1:] - d[:-1]) / d[:-1]
            r = np.where(np.isfinite(r), r, 0)
        return r

    def __getitem__(self, idx):
        real_idx = idx + self.start
        raw = _global_data[real_idx].numpy()
        pp = self.previous + self.predict
        mean = raw[:pp]
        change = raw[pp:2 * pp]
        high_low = raw[2 * pp:3 * pp]
        change_delta = raw[3 * pp:4 * pp]

        price_pct = self._cal_pct(mean[:self.previous])
        pct_chg = (mean[-1] - mean[self.previous - 1]) / mean[self.previous - 1]
        if pct_chg < -0.1: label = 0
        elif pct_chg < 0: label = 1
        elif pct_chg <= 0.1: label = 2
        else: label = 3

        features = np.stack([price_pct, change[:self.previous], high_low[:self.previous], change_delta[:self.previous]], axis=-1)
        return features.astype(np.float32), label


def parse_args():
    p = argparse.ArgumentParser(description='两阶段训练: Stage1 高权重→Recall, Stage2 等权重→Precision')
    p.add_argument('--model', type=str, default=MODEL_TYPE, choices=['fc', 'transformer', 'decoder', 'moe'])
    p.add_argument('--previous', type=int, default=PREVIOUS_NUM)
    p.add_argument('--predict', type=int, default=PREDICT_NUM)
    p.add_argument('--dataset', type=str, default=DATASET_PATH)
    p.add_argument('--stage1-weights', type=float, nargs='+', default=STAGE1_WEIGHTS,
                   help=f'Stage1 权重')
    p.add_argument('--stage2-weights', type=float, nargs='+', default=STAGE2_WEIGHTS,
                   help=f'Stage2 权重')
    p.add_argument('--lr', type=float, default=LEARNING_RATE)
    p.add_argument('--batch', type=int, default=BATCH_SIZE)
    p.add_argument('--stage1-epochs', type=int, default=STAGE1_EPOCHS)
    p.add_argument('--stage2-epochs', type=int, default=STAGE2_EPOCHS)
    p.add_argument('--patience', type=int, default=PATIENCE)
    p.add_argument('--hidden', type=int, default=HIDDEN_SIZE)
    p.add_argument('--layers', type=int, default=NUM_LAYERS)
    p.add_argument('--dmodel', type=int, default=D_MODEL)
    return p.parse_args()


def load_data(dataset_path, previous, predict, model_type):
    global _global_data
    files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.npy')])
    total_bytes = sum(os.path.getsize(os.path.join(dataset_path, f)) for f in files)
    IN_MEMORY_THRESHOLD = 8 * 1024 ** 3

    if total_bytes < IN_MEMORY_THRESHOLD:
        print(f'数据大小: {total_bytes/1024**3:.2f}GB < {IN_MEMORY_THRESHOLD/1024**3:.0f}GB, 共享内存模式')
        all_data = []
        for f in files:
            all_data.append(np.load(os.path.join(dataset_path, f)))
        all_data = np.concatenate(all_data, axis=0)
        print(f'Data shape: {all_data.shape}')
        np.random.seed(42)
        np.random.shuffle(all_data)
        _global_data = torch.from_numpy(all_data).share_memory_()
        del all_data

        total_num = len(_global_data)
        split = int(0.8 * total_num)

        import platform
        nw = 0 if platform.system() == 'Windows' else 4
        pf = None if platform.system() == 'Windows' else 2

        tf = model_type in ("transformer", "decoder", "moe")
        train_ds = DSShared(0, split, previous, predict, transformer_format=tf)
        test_ds = DSShared(split, total_num, previous, predict, transformer_format=tf)
    else:
        print(f'数据大小: {total_bytes/1024**3:.2f}GB > {IN_MEMORY_THRESHOLD/1024**3:.0f}GB, 单进程模式')
        all_data = []
        for f in files:
            all_data.append(np.load(os.path.join(dataset_path, f)))
        all_data = np.concatenate(all_data, axis=0)
        print(f'Data shape: {all_data.shape}')
        np.random.seed(42)
        np.random.shuffle(all_data)

        total_num = len(all_data)
        split = int(0.8 * total_num)

        nw = 0
        pf = None

        tf = model_type in ("transformer", "decoder", "moe")
        train_ds = DS(all_data, 0, split, previous, predict, transformer_format=tf)
        test_ds = DS(all_data, split, total_num, previous, predict, transformer_format=tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=nw, prefetch_factor=pf)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=nw, prefetch_factor=pf)
    return train_loader, test_loader, split, total_num


def create_model(model_type, previous, hidden, num_layers, d_model, device, num_class=4):
    if model_type == "encoder":
        from models.sequence_models import EncoderModel
        return EncoderModel(feature_dim=4, seq_len=previous, num_class=num_class, d_model=d_model, num_layers=num_layers).to(device)
    elif model_type == "decoder":
        from models.sequence_models import DecoderTransformerModel
        return DecoderTransformerModel(feature_dim=4, seq_len=previous, num_class=num_class, d_model=d_model, num_layers=num_layers).to(device)
    elif model_type == "moe":
        from models.sequence_models import DecoderTransformerModelMoE
        return DecoderTransformerModelMoE(feature_dim=4, seq_len=previous, num_class=num_class, d_model=d_model, num_layers=num_layers).to(device)
    else:
        from models.FCmodel import FCmodel
        return FCmodel(feature_dim=4, seq_len=previous, num_class=num_class, hidden_size=hidden, num_layers=num_layers).to(device)


def evaluate(model, test_loader, model_type, device):
    model.eval()
    cm = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating', leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            result = model(inputs.float())
            if model_type == "moe":
                outputs, _ = result
            else:
                outputs = result
            _, predicted = torch.max(outputs, 1)
            for p, t in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                cm[p][t] += 1

    precision = [0.0, 0.0, 0.0]
    recall = [0.0, 0.0, 0.0]
    for c in range(3):
        tp = cm[c][c]
        fp = sum(cm[c]) - tp
        fn = sum(row[c] for row in cm) - tp
        precision[c] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[c] = tp / (tp + fn) if (tp + fn) > 0 else 0

    total_correct = sum(cm[i][i] for i in range(3))
    accuracy = 100.0 * total_correct / sum(sum(row) for row in cm)
    return accuracy, precision, recall, cm


def train_stage(model, train_loader, test_loader, weights, epochs, patience, model_type, device, label, monitor_metric='f1'):
    """单个训练阶段
    monitor_metric: 'f1' (L0L2-F1) 或 'precision' ((L0_P + L2_P)/2)
    """
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights).float().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_score = 0
    best_epoch = 0
    best_state = None
    best_metrics = None
    no_improve = 0

    header = f'{"="*70}\n{label}\n{"="*70}'
    if monitor_metric == 'f1':
        header += '\nEpoch | Acc(%) | L0L2F1(%) | P=[L0,L1,L2] | R=[L0,L1,L2] | Best'
    else:
        header += '\nEpoch | Acc(%) | L0L2AvgP(%) | P=[L0,L1,L2] | R=[L0,L1,L2] | Best'
    print(header)

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        pbar = tqdm(train_loader, desc=f'Ep {epoch+1}/{epochs}', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            result = model(inputs.float())
            if model_type == "moe":
                outputs, aux_loss = result
                loss = criterion(outputs, labels) + 0.01 * aux_loss / NUM_LAYERS
            else:
                outputs = result
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        acc, prec, rec, cm = evaluate(model, test_loader, model_type, device)

        if monitor_metric == 'f1':
            f1_0 = 2 * prec[0] * rec[0] / (prec[0] + rec[0]) if (prec[0] + rec[0]) > 0 else 0
            f1_2 = 2 * prec[2] * rec[2] / (prec[2] + rec[2]) if (prec[2] + rec[2]) > 0 else 0
            score = (f1_0 + f1_2) / 2 * 100
            score_name = 'L0L2F1'
        else:
            score = (prec[0] + prec[2]) / 2 * 100
            score_name = 'L0L2AvgP'

        is_best = score > best_score
        if is_best:
            best_score = score
            best_epoch = epoch + 1
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = (acc, prec, rec, cm)
            no_improve = 0
            marker = '*'
        else:
            no_improve += 1
            marker = ''

        elapsed = time.time() - t0
        print(f'{epoch+1:>5} | {acc:>6.1f} | {score:>9.1f} | '
              f'[{prec[0]:.2f},{prec[1]:.2f},{prec[2]:.2f}] | '
              f'[{rec[0]:.2f},{rec[1]:.2f},{rec[2]:.2f}] | '
              f'{best_epoch}{marker}')

        if no_improve >= patience:
            print(f'\nEarly stop at epoch {epoch+1}')
            break

    # Restore best
    model.load_state_dict(best_state)
    return model, best_epoch, best_score, best_metrics


def main():
    args = parse_args()

    print(f'\n{"="*70}')
    print(f'两阶段训练')
    print(f'{"="*70}')
    print(f'模型: {args.model.upper()}')
    print(f'数据集: {args.dataset}')
    print(f'历史天数: {args.previous}, 预测天数: {args.predict}')
    print(f'Stage 1 权重: L0={args.stage1_weights[0]}, L1={args.stage1_weights[1]}, L2={args.stage1_weights[2]}')
    print(f'Stage 2 权重: L0={args.stage2_weights[0]}, L1={args.stage2_weights[1]}, L2={args.stage2_weights[2]}')
    print(f'学习率: {args.lr}, Batch: {args.batch}')
    print(f'Patience: {args.patience}')
    print(f'{"="*70}')

    train_loader, test_loader, train_n, total_n = load_data(
        args.dataset, args.previous, args.predict, args.model
    )
    print(f'训练集: {train_n:,}, 测试集: {total_n - train_n:,}')

    global NUM_LAYERS
    NUM_LAYERS = args.layers

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model = create_model(args.model, args.previous, args.hidden, args.layers, args.dmodel, device)
    print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')

    # ========================
    # Stage 1: 高权重拉Recall
    # ========================
    model, s1_epoch, s1_score, s1_metrics = train_stage(
        model, train_loader, test_loader,
        weights=args.stage1_weights,
        epochs=args.stage1_epochs,
        patience=args.patience,
        model_type=args.model,
        device=device,
        label=f'Stage 1: 高权重 [{" ".join(map(str, args.stage1_weights))}] 拉Recall',
        monitor_metric='f1'
    )

    s1_acc, s1_prec, s1_rec, s1_cm = s1_metrics
    print(f'\nStage 1 完成: Epoch={s1_epoch}, L0L2-F1={s1_score:.1f}%')
    print(f'  P: L0={s1_prec[0]:.2f}, L1={s1_prec[1]:.2f}, L2={s1_prec[2]:.2f}')
    print(f'  R: L0={s1_rec[0]:.2f}, L1={s1_rec[1]:.2f}, L2={s1_rec[2]:.2f}')

    # ================================
    # Stage 2: 等权重收Precision
    # ================================
    print(f'\n{"="*70}')
    print(f'从 Stage 1 best checkpoint 继续训练...')
    print(f'{"="*70}')

    model, s2_epoch, s2_score, s2_metrics = train_stage(
        model, train_loader, test_loader,
        weights=args.stage2_weights,
        epochs=args.stage2_epochs,
        patience=args.patience,
        model_type=args.model,
        device=device,
        label=f'Stage 2: 等权重 [{" ".join(map(str, args.stage2_weights))}] 收Precision',
        monitor_metric='precision'
    )

    s2_acc, s2_prec, s2_rec, s2_cm = s2_metrics

    # ========================
    # 最终报告
    # ========================
    print(f'\n{"="*70}')
    print(f'最终结果对比')
    print(f'{"="*70}')

    f1_0_s1 = 2 * s1_prec[0] * s1_rec[0] / (s1_prec[0] + s1_rec[0]) if (s1_prec[0] + s1_rec[0]) > 0 else 0
    f1_2_s1 = 2 * s1_prec[2] * s1_rec[2] / (s1_prec[2] + s1_rec[2]) if (s1_prec[2] + s1_rec[2]) > 0 else 0
    f1_0_s2 = 2 * s2_prec[0] * s2_rec[0] / (s2_prec[0] + s2_rec[0]) if (s2_prec[0] + s2_rec[0]) > 0 else 0
    f1_2_s2 = 2 * s2_prec[2] * s2_rec[2] / (s2_prec[2] + s2_rec[2]) if (s2_prec[2] + s2_rec[2]) > 0 else 0

    print(f'{"指标":<20} {"Stage 1 (高权重)":>20} {"Stage 2 (等权重)":>20} {"变化":>10}')
    print(f'{"-"*70}')
    print(f'{"Best Epoch":<20} {s1_epoch:>20} {s2_epoch:>20}')
    print(f'{"Accuracy":<20} {s1_acc:>19.1f}% {s2_acc:>19.1f}%')
    print(f'{"L0 Precision":<20} {s1_prec[0]:>19.2f}  {s2_prec[0]:>19.2f}  {s2_prec[0]-s1_prec[0]:>+9.2f}')
    print(f'{"L2 Precision":<20} {s1_prec[2]:>19.2f}  {s2_prec[2]:>19.2f}  {s2_prec[2]-s1_prec[2]:>+9.2f}')
    print(f'{"L0 Recall":<20} {s1_rec[0]:>19.2f}  {s2_rec[0]:>19.2f}  {s2_rec[0]-s1_rec[0]:>+9.2f}')
    print(f'{"L2 Recall":<20} {s1_rec[2]:>19.2f}  {s2_rec[2]:>19.2f}  {s2_rec[2]-s1_rec[2]:>+9.2f}')
    print(f'{"L0L2-F1":<20} {(f1_0_s1+f1_2_s1)/2*100:>19.1f}% {(f1_0_s2+f1_2_s2)/2*100:>19.1f}%')
    print(f'{"L0L2-AvgP":<20} {(s1_prec[0]+s1_prec[2])/2*100:>19.1f}% {(s2_prec[0]+s2_prec[2])/2*100:>19.1f}%')

    print(f'\nStage 2 Confusion Matrix (pred rows x true cols):')
    print(f'         True_L0    True_L1    True_L2')
    for i, row in enumerate(s2_cm):
        print(f'Pred_L{i}  {row[0]:>9}  {row[1]:>9}  {row[2]:>9}')

    # Save model
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    extra = f'd{args.dmodel}' if args.model in ("transformer", "decoder", "moe") else f'h{args.hidden}'
    model_path = f'best_{args.model}_twostage_{args.previous}d_{extra}_l{args.layers}_{timestamp}.pt'
    torch.save(model.state_dict(), model_path)
    print(f'\nModel saved: {model_path}')


if __name__ == "__main__":
    main()
