#!/usr/bin/env python3
"""
统一训练脚本 - 支持FC和Transformer，可配置数据长度
用法:
  python train/train.py --model fc --previous 30 --dataset dataset_30
  python train/train.py --model transformer --previous 30 --dataset dataset_30
  python train/train.py --model fc --previous 15 --dataset dataset --weights 2.0 1.0 2.0
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
PREVIOUS_NUM = 15
PREDICT_NUM = 3
DATASET_PATH = "dataset"
WEIGHTS = [1.0, 1.0, 1.0, 1.0]
LEARNING_RATE = 0.0001
BATCH_SIZE = 2048
MAX_EPOCHS = 1000
PATIENCE = 20
HIDDEN_SIZE = 256
NUM_LAYERS = 4
D_MODEL = 256
MODEL_TYPE = "fc"  # fc or transformer
# ==================================

def parse_args():
    parser = argparse.ArgumentParser(description='训练股票预测模型')
    parser.add_argument('--model', type=str, default=MODEL_TYPE, choices=['fc', 'encoder', 'decoder', 'moe'],
                        help='模型类型: fc / transformer / decoder / moe')
    parser.add_argument('--previous', type=int, default=PREVIOUS_NUM,
                        help=f'历史数据天数 (默认 {PREVIOUS_NUM})')
    parser.add_argument('--predict', type=int, default=PREDICT_NUM,
                        help=f'预测天数 (默认 {PREDICT_NUM})')
    parser.add_argument('--dataset', type=str, default=DATASET_PATH,
                        help=f'数据集路径 (默认 {DATASET_PATH})')
    parser.add_argument('--weights', type=float, nargs='+', default=WEIGHTS,
                        help=f'类别权重 (默认全1.0, 长度=类别数)')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE,
                        help=f'学习率 (默认 {LEARNING_RATE})')
    parser.add_argument('--batch', type=int, default=BATCH_SIZE,
                        help=f'批大小 (默认 {BATCH_SIZE})')
    parser.add_argument('--epochs', type=int, default=MAX_EPOCHS,
                        help=f'最大epoch数 (默认 {MAX_EPOCHS})')
    parser.add_argument('--patience', type=int, default=PATIENCE,
                        help=f'早停patience (默认 {PATIENCE})')
    parser.add_argument('--hidden', type=int, default=HIDDEN_SIZE,
                        help=f'隐藏层大小 (默认 {HIDDEN_SIZE})')
    parser.add_argument('--layers', type=int, default=NUM_LAYERS,
                        help=f'层数 (默认 {NUM_LAYERS})')
    parser.add_argument('--dmodel', type=int, default=D_MODEL,
                        help=f'Transformer d_model (默认 {D_MODEL})')
    parser.add_argument('--checkpoint', type=str, default='checkpoint.pt',
                        help=f'保存/恢复checkpoint的路径 (默认 checkpoint.pt)')
    return parser.parse_args()

# 全局共享内存数据
_global_data = None

class DS:
    def __init__(self, data, start, end, previous=15, predict=3, transformer_format=False):
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
        change = raw[pp:2*pp]
        high_low = raw[2*pp:3*pp]
        change_delta = raw[3*pp:4*pp]

        price_pct = self._cal_pct(mean[:self.previous])
        pct_chg = (mean[-1] - mean[self.previous - 1]) / mean[self.previous - 1]
        # 4-class label
        if pct_chg < -0.1: label = 0
        elif pct_chg < 0: label = 1
        elif pct_chg <= 0.1: label = 2
        else: label = 3

        features = np.stack([price_pct, change[:self.previous], high_low[:self.previous], change_delta[:self.previous]], axis=-1)
        return features.astype(np.float32), label

class DSShared(Dataset):
    """使用共享内存tensor的Dataset"""
    def __init__(self, start, end, previous=15, predict=3, transformer_format=False):
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
        change = raw[pp:2*pp]
        high_low = raw[2*pp:3*pp]
        change_delta = raw[3*pp:4*pp]

        price_pct = self._cal_pct(mean[:self.previous])
        pct_chg = (mean[-1] - mean[self.previous - 1]) / mean[self.previous - 1]
        # 4-class label
        if pct_chg < -0.1: label = 0
        elif pct_chg < 0: label = 1
        elif pct_chg <= 0.1: label = 2
        else: label = 3

        features = np.stack([price_pct, change[:self.previous], high_low[:self.previous], change_delta[:self.previous]], axis=-1)
        return features.astype(np.float32), label

def main():
    args = parse_args()

    # 更新全局配置
    global PREVIOUS_NUM, PREDICT_NUM, DATASET_PATH, WEIGHTS, LEARNING_RATE
    global BATCH_SIZE, MAX_EPOCHS, PATIENCE, HIDDEN_SIZE, NUM_LAYERS, MODEL_TYPE, D_MODEL

    PREVIOUS_NUM = args.previous
    PREDICT_NUM = args.predict
    DATASET_PATH = args.dataset
    WEIGHTS = args.weights
    LEARNING_RATE = args.lr
    BATCH_SIZE = args.batch
    MAX_EPOCHS = args.epochs
    PATIENCE = args.patience
    HIDDEN_SIZE = args.hidden
    NUM_LAYERS = args.layers
    MODEL_TYPE = args.model
    D_MODEL = args.dmodel

    print(f'\n{"="*70}')
    print(f'训练配置')
    print(f'{"="*70}')
    print(f'模型: {MODEL_TYPE.upper()}')
    print(f'数据目录: {DATASET_PATH}')
    print(f'历史天数: {PREVIOUS_NUM}, 预测天数: {PREDICT_NUM}')
    print(f'类别权重: {" ".join([f"L{i}={WEIGHTS[i]}" for i in range(len(WEIGHTS))])}')
    print(f'学习率: {LEARNING_RATE}, Batch: {BATCH_SIZE}')
    if MODEL_TYPE in ("encoder", "decoder", "moe"):
        print(f'd_model: {D_MODEL}, layers: {NUM_LAYERS}')
    else:
        print(f'hidden: {HIDDEN_SIZE}, layers: {NUM_LAYERS}')
    print(f'{"="*70}')

    # 计算数据文件总大小，判断是否能放到共享内存
    files = sorted([f for f in os.listdir(DATASET_PATH) if f.endswith('.npy')])
    total_bytes = sum(os.path.getsize(os.path.join(DATASET_PATH, f)) for f in files)
    IN_MEMORY_THRESHOLD = 8 * 1024**3  # 8GB

    global _global_data

    if total_bytes < IN_MEMORY_THRESHOLD:
        print(f'数据大小: {total_bytes/1024**3:.2f}GB < {IN_MEMORY_THRESHOLD/1024**3:.0f}GB, 加载到共享内存')
        # 加载数据到共享内存
        all_data = []
        for f in files:
            all_data.append(np.load(os.path.join(DATASET_PATH, f)))
        all_data = np.concatenate(all_data, axis=0)
        print(f'Data shape: {all_data.shape}')
        # Shuffle
        np.random.seed(42)
        np.random.shuffle(all_data)
        _global_data = torch.from_numpy(all_data).share_memory_()
        del all_data

        total_num = len(_global_data)
        split = int(0.8 * total_num)
        print(f'\n=== 全量数据训练 ===')
        print(f'训练集: {split:,} 样本')
        print(f'测试集: {total_num - split:,} 样本')

        import platform
        if platform.system() == 'Windows':
            print('Windows detected, using single process')
            num_workers = 0
            prefetch_factor = None
        else:
            print(f'启用多进程预加载 (num_workers=4)')
            num_workers = 4
            prefetch_factor = 2
        train_ds = DSShared(0, split, PREVIOUS_NUM, PREDICT_NUM)
        test_ds = DSShared(split, total_num, PREVIOUS_NUM, PREDICT_NUM)
    else:
        print(f'数据大小: {total_bytes/1024**3:.2f}GB > {IN_MEMORY_THRESHOLD/1024**3:.0f}GB, 使用单进程')
        # 单进程模式，直接传numpy数组
        all_data = []
        for f in files:
            all_data.append(np.load(os.path.join(DATASET_PATH, f)))
        all_data = np.concatenate(all_data, axis=0)
        print(f'Data shape: {all_data.shape}')
        # Shuffle
        np.random.seed(42)
        np.random.shuffle(all_data)

        total_num = len(all_data)
        split = int(0.8 * total_num)
        print(f'\n=== 全量数据训练 ===')
        print(f'训练集: {split:,} 样本')
        print(f'测试集: {total_num - split:,} 样本')

        num_workers = 0
        prefetch_factor = None
        train_ds = DS(all_data, 0, split, PREVIOUS_NUM, PREDICT_NUM)
        test_ds = DS(all_data, split, total_num, PREVIOUS_NUM, PREDICT_NUM)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, prefetch_factor=prefetch_factor)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=num_workers, prefetch_factor=prefetch_factor)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # 类别数从 weights 推导
    n_classes = len(WEIGHTS)
    # 创建模型 (统一 [bs, seq_len, feature_dim] 输入)
    if MODEL_TYPE == "encoder":
        from models.sequence_models import EncoderModel
        print(f'输入维度: ({PREVIOUS_NUM}, 4) (Encoder)')
        model = EncoderModel(feature_dim=4, seq_len=PREVIOUS_NUM, num_class=n_classes,
                                 d_model=D_MODEL, num_layers=NUM_LAYERS).to(device)
    elif MODEL_TYPE == "decoder":
        from models.sequence_models import DecoderTransformerModel
        print(f'输入维度: ({PREVIOUS_NUM}, 4) (Decoder)')
        model = DecoderTransformerModel(feature_dim=4, seq_len=PREVIOUS_NUM, num_class=n_classes,
                                        d_model=D_MODEL, num_layers=NUM_LAYERS).to(device)
    elif MODEL_TYPE == "moe":
        from models.sequence_models import DecoderTransformerModelMoE
        print(f'输入维度: ({PREVIOUS_NUM}, 4) (MoE Decoder)')
        model = DecoderTransformerModelMoE(feature_dim=4, seq_len=PREVIOUS_NUM, num_class=n_classes,
                                           d_model=D_MODEL, num_layers=NUM_LAYERS).to(device)
    else:
        from models.FCmodel import FCmodel
        print(f'输入维度: ({PREVIOUS_NUM}, 4) (FC)')
        model = FCmodel(feature_dim=4, seq_len=PREVIOUS_NUM, num_class=n_classes,
                        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)

    criterion = nn.CrossEntropyLoss(weight=torch.tensor(WEIGHTS).float().to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f'模型参数量: {sum(p.numel() for p in model.parameters()):,}')

    best_score = -1  # L0 + last class AvgP
    best_epoch = 0
    best_confusion = None
    best_precision = [0] * n_classes
    best_recall = [0] * n_classes
    best_model_state = None
    no_improve = 0
    start_epoch = 0

    # 恢复训练
    if os.path.exists(args.checkpoint):
        print(f'从checkpoint恢复: {args.checkpoint}')
        ckpt = torch.load(args.checkpoint, map_location=device)
        if 'model_state' in ckpt:
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            start_epoch = ckpt['epoch'] + 1
            best_score = ckpt.get('best_score', ckpt.get('best_f1', 0))
            best_epoch = ckpt.get('best_epoch', 0)
            no_improve = ckpt.get('no_improve', 0)
            print(f'  恢复到 epoch {start_epoch}, best_score={best_score:.1f}%')
        else:
            model.load_state_dict(ckpt)
            print(f'  仅加载模型权重')

    p_heads = ' '.join([f'P_L{i}' for i in range(n_classes)])
    r_heads = ' '.join([f'R_L{i}' for i in range(n_classes)])
    print(f'\n{"="*70}')
    print(f'Epoch | Acc(%) | Score | [{p_heads}] | [{r_heads}] | Best')
    print(f'{"="*70}')

    for epoch in range(start_epoch, MAX_EPOCHS):
        t0 = time.time()
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{MAX_EPOCHS}', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            result = model(inputs.float())
            if MODEL_TYPE == "moe":
                outputs, aux_loss = result
                loss = criterion(outputs, labels) + 0.01 * aux_loss / NUM_LAYERS
            else:
                outputs = result
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        # 评估
        model.eval()
        confusion_matrix = [[0]*n_classes for _ in range(n_classes)]

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Evaluating', leave=False):
                inputs, labels = inputs.to(device), labels.to(device)
                result = model(inputs.float())
                if MODEL_TYPE == "moe":
                    outputs, _ = result
                else:
                    outputs = result
                _, predicted = torch.max(outputs, 1)
                for p, t in zip(predicted.cpu().numpy(), labels.cpu().numpy()):
                    confusion_matrix[p][t] += 1

        # 计算指标
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
        # 选模标准: L0 + 最后一类平均 Precision
        # 距离加权得分: 正确=1, 差1档=0.7, 差2档=0.3, 远=0
        score_mat = [[0.0]*n_classes for _ in range(n_classes)]
        for p in range(n_classes):
            for t in range(n_classes):
                d = abs(p - t)
                score_mat[p][t] = 1.0 if d == 0 else (0.7 if d == 1 else (0.3 if d == 2 else 0.0))
        total_weighted = sum(confusion_matrix[p][t] * score_mat[p][t] for p in range(n_classes) for t in range(n_classes))
        total_samples = sum(sum(r) for r in confusion_matrix)
        score = total_weighted / total_samples * 100 if total_samples > 0 else 0

        is_best = score > best_score
        if is_best:
            best_score = score
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
        print(f'{epoch+1:>5} | {accuracy:>6.1f} | {score:>5.1f} | [{p_str}] | [{r_str}] | {best_epoch}{marker}')

        if no_improve >= PATIENCE:
            print(f'\nEarly stop at epoch {epoch+1}')
            # 保存中断checkpoint
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_score': best_score,
                'best_epoch': best_epoch,
                'no_improve': no_improve,
            }
            torch.save(ckpt, args.checkpoint)
            print(f'Checkpoint saved: {args.checkpoint}')
            break

        # 每10个epoch保存一次checkpoint
        if (epoch + 1) % 10 == 0:
            ckpt = {
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'best_score': best_score,
                'best_epoch': best_epoch,
                'no_improve': no_improve,
            }
            torch.save(ckpt, args.checkpoint)

    # 最终结果
    print(f'\n{"="*70}')
    print(f'Best: Epoch {best_epoch}, L0L{n_classes-1}-AvgP={best_score:.1f}%')
    print(f'Precision: {" ".join([f"L{i}={best_precision[i]:.2f}" for i in range(n_classes)])}')
    print(f'Recall:    {" ".join([f"L{i}={best_recall[i]:.2f}" for i in range(n_classes)])}')
    print(f'\nConfusion Matrix (pred rows x true cols):')
    print(f'         {" ".join([f"True_L{i:>9}" for i in range(n_classes)])}')
    for i, row in enumerate(best_confusion):
        print(f'Pred_L{i}  {" ".join([f"{v:>9}" for v in row])}')

    # epoch跑满，保存checkpoint而不是best model
    if no_improve < PATIENCE:
        print(f'\n达到最大epoch {MAX_EPOCHS}，保存checkpoint')
        ckpt = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_score': best_score,
            'best_epoch': best_epoch,
            'no_improve': no_improve,
        }
        torch.save(ckpt, args.checkpoint)
        print(f'Checkpoint saved: {args.checkpoint}')
    else:
        # 生成带信息的模型文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        extra_info = f'd{D_MODEL}' if MODEL_TYPE in ("encoder", "decoder", "moe") else f'h{HIDDEN_SIZE}'
        model_path = f'best_{MODEL_TYPE}_{PREVIOUS_NUM}d_{extra_info}_l{NUM_LAYERS}_{timestamp}.pt'
        torch.save(best_model_state, model_path)
        print(f'\nModel saved: {model_path}')

        # 删除checkpoint
        if os.path.exists(args.checkpoint):
            os.remove(args.checkpoint)
            print(f'Checkpoint removed: {args.checkpoint}')

if __name__ == "__main__":
    main()