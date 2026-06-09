#!/usr/bin/env python3
"""
统一训练脚本 — 读取 JSON 配置文件
用法:
  python train/train_config.py --config configs/FC_1M.json
  python train/train_config.py --config configs/Transformer_d128_l4.json --override training.weights=[5,1,1,5]
  python train/train_config.py --config configs/FC_1M.json --two-stage
"""
import os, sys, json, time, argparse, copy
from datetime import datetime
sys.path.insert(0, '.')
os.environ['PYTHONUNBUFFERED'] = '1'

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from tqdm import tqdm

# ============ 默认值 ============
DEFAULTS = {
    "model": {
        "type": "fc",
        "feature_dim": 4,
        "seq_len": 30,
        "num_class": 4,
        "hidden_size": 256,
        "num_layers": 4,
        "d_model": 128,
    },
    "training": {
        "dataset": "baostock_dataset_30",
        "previous_num": 30,
        "predict_num": 3,
        "weights": [1.0, 1.0, 1.0, 1.0],
        "lr": 0.0001,
        "batch_size": 2048,
        "max_epochs": 500,
        "patience": 20,
    },
    "two_stage": {
        "stage1_weights": [5.0, 1.0, 1.0, 5.0],
        "stage2_weights": [1.0, 1.0, 1.0, 1.0],
        "stage1_epochs": 200,
        "stage2_epochs": 200,
    }
}


def deep_merge(base, override):
    """递归合并 override 到 base"""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path, overrides=None):
    """加载 JSON 配置, 合并默认值, 应用命令行 overrides"""
    with open(config_path, 'r') as f:
        cfg = json.load(f)
    # 合并默认值
    cfg = deep_merge(DEFAULTS, cfg)
    # 应用命令行 overrides (格式: model.hidden_size=512 training.lr=0.01)
    if overrides:
        for ov in overrides:
            key, val = ov.split('=', 1)
            keys = key.split('.')
            target = cfg
            for k in keys[:-1]:
                target = target[k]
            # 解析值
            import ast
            try:
                val = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                pass
            target[keys[-1]] = val
    return cfg


# ============ Data Pipeline ============
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
    def __init__(self, data, start, end, previous, predict):
        self.data = data[start:end]
        self.previous = previous; self.predict = predict
    def __len__(self): return len(self.data)
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
    def __init__(self, start, end, previous, predict):
        self.start = start; self.end = end
        self.previous = previous; self.predict = predict
    def __len__(self): return self.end - self.start
    def __getitem__(self, idx):
        raw = _global_data[idx + self.start].numpy()
        pp = self.previous + self.predict
        mean = raw[:pp]; change = raw[pp:2*pp]; hl = raw[2*pp:3*pp]; cd_ = raw[3*pp:4*pp]
        price_pct = _cal_pct(mean[:self.previous])
        pct_chg = (mean[-1] - mean[self.previous - 1]) / mean[self.previous - 1]
        label = _get_label(pct_chg)
        features = np.stack([price_pct, change[:self.previous], hl[:self.previous], cd_[:self.previous]], axis=0)
        return features.astype(np.float32), label


def load_data(dataset_path, previous, predict):
    global _global_data
    files = sorted([f for f in os.listdir(dataset_path) if f.endswith('.npy')])
    total_bytes = sum(os.path.getsize(os.path.join(dataset_path, f)) for f in files)
    THRESHOLD = 8 * 1024**3
    if total_bytes < THRESHOLD:
        print(f'数据大小: {total_bytes/1024**3:.2f}GB < 8GB, 共享内存模式')
        all_data = []
        for f in files: all_data.append(np.load(os.path.join(dataset_path, f)))
        all_data = np.concatenate(all_data, axis=0)
        np.random.seed(42); np.random.shuffle(all_data)
        _global_data = torch.from_numpy(all_data).share_memory_()
        del all_data
        total_num = len(_global_data)
        split = int(0.8 * total_num)
        train_ds = DSShared(0, split, previous, predict)
        test_ds = DSShared(split, total_num, previous, predict)
    else:
        print(f'数据大小: {total_bytes/1024**3:.2f}GB > 8GB, 单进程模式')
        all_data = []
        for f in files: all_data.append(np.load(os.path.join(dataset_path, f)))
        all_data = np.concatenate(all_data, axis=0)
        np.random.seed(42); np.random.shuffle(all_data)
        total_num = len(all_data)
        split = int(0.8 * total_num)
        train_ds = DS(all_data, 0, split, previous, predict)
        test_ds = DS(all_data, split, total_num, previous, predict)
    print(f'总样本: {total_num:,}, 训练: {split:,}, 测试: {total_num-split:,}')
    return train_ds, test_ds


def create_model(model_cfg, num_class, device):
    t = model_cfg['type']
    if t == 'encoder':
        from models.sequence_models import EncoderModel
        return EncoderModel(feature_dim=model_cfg['feature_dim'], seq_len=model_cfg['seq_len'],
                                num_class=num_class, d_model=model_cfg['d_model'],
                                num_layers=model_cfg['num_layers']).to(device)
    elif t == 'decoder':
        from models.sequence_models import DecoderTransformerModel
        return DecoderTransformerModel(feature_dim=model_cfg['feature_dim'], seq_len=model_cfg['seq_len'],
                                       num_class=num_class, d_model=model_cfg['d_model'],
                                       num_layers=model_cfg['num_layers']).to(device)
    elif t == 'moe':
        from models.sequence_models import DecoderTransformerModelMoE
        return DecoderTransformerModelMoE(feature_dim=model_cfg['feature_dim'], seq_len=model_cfg['seq_len'],
                                          num_class=num_class, d_model=model_cfg['d_model'],
                                          num_layers=model_cfg['num_layers']).to(device)
    else:
        from models.FCmodel import FCmodel
        return FCmodel(feature_dim=model_cfg['feature_dim'], seq_len=model_cfg['seq_len'],
                       num_class=num_class, hidden_size=model_cfg['hidden_size'],
                       num_layers=model_cfg['num_layers']).to(device)


def train_epochs(model, train_loader, test_loader, criterion, optimizer, cfg, device):
    n_classes = len(cfg['training']['weights'])
    previous = cfg['training']['previous_num']
    model_type = cfg['model']['type']
    best_score = -1
    best_epoch = 0
    best_state = None
    no_improve = 0
    best_precision = [0]*n_classes
    best_recall = [0]*n_classes
    best_confusion = None

    p_heads = ' '.join([f'P_L{i}' for i in range(n_classes)])
    r_heads = ' '.join([f'R_L{i}' for i in range(n_classes)])
    print(f'\n{"="*70}')
    print(f'Epoch | Acc(%) | Score | [{p_heads}] | [{r_heads}] | Best')
    print(f'{"="*70}')

    for epoch in range(cfg['training']['max_epochs']):
        t0 = time.time()
        model.train()
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.float().to(device), labels.long().to(device)
            result = model(inputs)
            if model_type == 'moe':
                outputs, aux_loss = result
                loss = criterion(outputs, labels) + 0.01 * aux_loss / cfg['model']['num_layers']
            else:
                outputs = result
                loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=f'{loss.item():.4f}')

        model.eval()
        cm = [[0]*n_classes for _ in range(n_classes)]
        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc='Eval', leave=False):
                inputs = inputs.float().to(device)
                result = model(inputs)
                outputs = result[0] if model_type == 'moe' else result
                _, pred = torch.max(outputs, 1)
                for p, t in zip(pred.cpu().numpy(), labels.numpy()): cm[p][t] += 1

        precision = [0.0]*n_classes; recall = [0.0]*n_classes; f1 = [0.0]*n_classes
        for c in range(n_classes):
            tp = cm[c][c]; fp = sum(cm[c]) - tp; fn = sum(row[c] for row in cm) - tp
            precision[c] = tp/(tp+fp) if (tp+fp)>0 else 0
            recall[c] = tp/(tp+fn) if (tp+fn)>0 else 0
            f1[c] = 2*precision[c]*recall[c]/(precision[c]+recall[c]) if (precision[c]+recall[c])>0 else 0

        acc = 100.0 * sum(cm[i][i] for i in range(n_classes)) / sum(sum(r) for r in cm)
        # 距离加权得分矩阵: 正确=1, 差1档=0.7, 差2档=0.3, 全反=0
        score_mat = [[1.0 - 0.3 * abs(p - t) if abs(p - t) <= 2 else 0.0 for t in range(n_classes)] for p in range(n_classes)]
        # 对3档及以上用 0, 其余线性衰减
        for p in range(n_classes):
            for t in range(n_classes):
                d = abs(p - t)
                if d == 0: score_mat[p][t] = 1.0
                elif d == 1: score_mat[p][t] = 0.7
                elif d == 2: score_mat[p][t] = 0.3
                else: score_mat[p][t] = 0.0
        total_weighted = 0
        total_samples = 0
        for t in range(n_classes):
            for p in range(n_classes):
                count = cm[p][t]
                total_weighted += score_mat[p][t] * count
                total_samples += count
        score = total_weighted / total_samples * 100 if total_samples > 0 else 0

        is_best = score > best_score
        if is_best:
            best_score = score; best_epoch = epoch+1; no_improve = 0
            best_precision = precision[:]; best_recall = recall[:]
            best_confusion = [r[:] for r in cm]
            best_state = copy.deepcopy(model.state_dict())
            marker = '*'
        else:
            no_improve += 1; marker = ''

        elapsed = time.time() - t0
        p_str = ' '.join([f'{precision[i]:.2f}' for i in range(n_classes)])
        r_str = ' '.join([f'{recall[i]:.2f}' for i in range(n_classes)])
        print(f'{epoch+1:>5} | {acc:>6.1f} | {score:>5.1f} | [{p_str}] | [{r_str}] | {best_epoch}{marker}')

        if no_improve >= cfg['training']['patience']:
            print(f'\nEarly stop at epoch {epoch+1}')
            break

    return best_state, best_epoch, best_score, best_precision, best_recall, best_confusion


def main():
    parser = argparse.ArgumentParser(description='统一训练脚本 (JSON配置)')
    parser.add_argument('--config', type=str, required=True, help='JSON配置文件路径')
    parser.add_argument('--override', type=str, nargs='*', default=[],
                        help='覆盖配置, 格式: training.lr=0.01 model.hidden_size=512')
    parser.add_argument('--two-stage', action='store_true', help='两阶段训练')
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"配置: {args.config}")
    print(f"{'='*60}")
    print(f"模型: {cfg['model']['type']}, 类别数: {len(cfg['training']['weights'])}")
    print(f"超参: {json.dumps(cfg['model'], indent=None)}")
    print(f"训练: {json.dumps(cfg['training'], indent=None)}")
    if args.two_stage:
        print(f"两阶段: {json.dumps(cfg['two_stage'], indent=None)}")

    train_ds, test_ds = load_data(cfg['training']['dataset'],
                                   cfg['training']['previous_num'],
                                   cfg['training']['predict_num'])
    train_loader = DataLoader(train_ds, batch_size=cfg['training']['batch_size'],
                              shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=cfg['training']['batch_size'],
                             shuffle=False, num_workers=0)

    if args.two_stage:
        # ── Two-Stage ──
        print(f"\n{'='*60}\nStage 1: 高权重拉Recall\n{'='*60}")
        cfg_s1 = copy.deepcopy(cfg)
        cfg_s1['training']['weights'] = cfg['two_stage']['stage1_weights']
        cfg_s1['training']['max_epochs'] = cfg['two_stage'].get('stage1_epochs', 200)

        model = create_model(cfg['model'], len(cfg_s1['training']['weights']), device)
        print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg_s1['training']['weights']).float().to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg_s1['training']['lr'])

        s1_state, s1_epoch, s1_score, s1_p, s1_r, s1_cm = train_epochs(
            model, train_loader, test_loader, criterion, optimizer, cfg_s1, device)
        print(f'\nStage1 Best: Epoch {s1_epoch}, Score={s1_score:.1f}')
        print(f'P: {" ".join([f"L{i}={s1_p[i]:.2f}" for i in range(len(s1_p))])}')
        print(f'R: {" ".join([f"L{i}={s1_r[i]:.2f}" for i in range(len(s1_r))])}')

        # Stage 2
        print(f"\n{'='*60}\nStage 2: 等权重收Precision\n{'='*60}")
        cfg_s2 = copy.deepcopy(cfg)
        cfg_s2['training']['weights'] = cfg['two_stage']['stage2_weights']
        cfg_s2['training']['max_epochs'] = cfg['two_stage'].get('stage2_epochs', 200)

        model.load_state_dict(s1_state)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg_s2['training']['weights']).float().to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg_s2['training']['lr'] / 2)

        s2_state, s2_epoch, s2_score, s2_p, s2_r, s2_cm = train_epochs(
            model, train_loader, test_loader, criterion, optimizer, cfg_s2, device)
        print(f'\nStage2 Best: Epoch {s2_epoch}, Score={s2_score:.1f}')
        print(f'P: {" ".join([f"L{i}={s2_p[i]:.2f}" for i in range(len(s2_p))])}')
        print(f'R: {" ".join([f"L{i}={s2_r[i]:.2f}" for i in range(len(s2_r))])}')

        # 最终对比
        print(f"\n{'='*60}")
        print(f'对比:')
        print(f'  Stage1  Score={s1_score:.1f}')
        print(f'  Stage2  Score={s2_score:.1f}')
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = f'best_{cfg["model"]["type"]}_twostage_{cfg["model"].get("hidden_size",cfg["model"].get("d_model"))}_{timestamp}.pt'
        torch.save(s2_state, model_path)
        print(f'\nModel: {model_path}')
    else:
        # ── Single-Stage ──
        model = create_model(cfg['model'], len(cfg['training']['weights']), device)
        print(f'参数量: {sum(p.numel() for p in model.parameters()):,}')
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(cfg['training']['weights']).float().to(device))
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['training']['lr'])

        best_state, best_epoch, best_score, best_p, best_r, best_cm = train_epochs(
            model, train_loader, test_loader, criterion, optimizer, cfg, device)

        print(f"\n{'='*60}")
        print(f'Best: Epoch {best_epoch}, Score={best_score:.1f}')
        print(f'Precision: {" ".join([f"L{i}={best_p[i]:.2f}" for i in range(len(best_p))])}')
        print(f'Recall:    {" ".join([f"L{i}={best_r[i]:.2f}" for i in range(len(best_r))])}')
        print(f'\nConfusion Matrix:')
        print(f'         {" ".join([f"True_L{i:>9}" for i in range(len(best_p))])}')
        for i, row in enumerate(best_cm):
            print(f'Pred_L{i}  {" ".join([f"{v:>9}" for v in row])}')

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        mtype = cfg['model']['type']
        extra = f'd{cfg["model"]["d_model"]}' if mtype in ('encoder','decoder','moe') else f'h{cfg["model"]["hidden_size"]}'
        model_path = f'best_{mtype}_30d_{extra}_l{cfg["model"]["num_layers"]}_{timestamp}.pt'
        torch.save(best_state, model_path)
        print(f'\nModel: {model_path}')


if __name__ == '__main__':
    main()
