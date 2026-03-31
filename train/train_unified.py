"""
通用训练脚本 - 支持多种模型 (FC, LSTM, Transformer)
用于对比性能
"""
import os
import sys
from pathlib import Path
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

project_root = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(project_root))

from dataloader.dockdataset import Dockdataset
from models.fcmodel import FCmodel
from models.sequence_models import LSTMSequenceModel, TransformerSequenceModel


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def create_model(model_type, input_size=15, num_classes=5):
    """创建模型"""
    if model_type == "fc":
        return FCmodel(input_size, num_class=num_classes)
    elif model_type == "lstm":
        return LSTMSequenceModel(num_classes=num_classes, hidden_size=256, num_layers=2)
    elif model_type == "transformer":
        return TransformerSequenceModel(num_classes=num_classes, d_model=128, num_layers=3)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def get_model_name(model_type):
    """获取模型的显示名称"""
    mapping = {
        "fc": "全连接网络 (FC)",
        "lstm": "LSTM",
        "transformer": "Transformer"
    }
    return mapping.get(model_type, model_type)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0.0

    pbar = tqdm(train_loader, desc="训练", leave=False)
    for inputs, labels in pbar:
        outputs = model(torch.tensor(inputs, device=device).float())
        loss = criterion(outputs, torch.tensor(labels, device=device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * inputs.size(0)
        
        # 更新进度条的损失值
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    epoch_loss /= len(train_loader.dataset)
    return epoch_loss


def evaluate(model, test_loader, criterion, device):
    """验证"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(test_loader, desc="验证", leave=False)
    with torch.no_grad():
        for inputs, labels in pbar:
            outputs = model(torch.tensor(inputs, device=device).float())
            loss = criterion(outputs, torch.tensor(labels, device=device))
            test_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted.to("cpu") == labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    test_loss /= len(test_loader.dataset)
    accuracy = 100 * correct / total
    return test_loss, accuracy


def train_model(model_type, num_epochs=50, batch_size=64, learning_rate=0.0001):
    """训练模型"""
    # 设置
    torch.manual_seed(42)
    np.random.seed(42)
    
    dataset_path = project_root / "dataset"
    total_num = 3805550
    split_index = int(0.8 * total_num)
    previous_num = 15
    class_num = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 创建数据加载器
    print("📁 创建数据加载器...")
    train_dataset = Dockdataset(dataset_path, 0, split_index, total_num)
    test_dataset = Dockdataset(dataset_path, split_index, total_num, total_num)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=16,
                              persistent_workers=True,
                              prefetch_factor=2)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=16,
                             persistent_workers=True,
                             prefetch_factor=2)

    # 创建模型
    print(f"🧠 创建模型: {get_model_name(model_type)}...")
    model = create_model(model_type, input_size=previous_num, num_classes=class_num)
    model.to(device)
    
    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   参数数: {total_params:,} (可训练: {trainable_params:,})")

    # 优化器和调度器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=learning_rate * 0.01
    )
    early_stopping = EarlyStopping(patience=5)

    # 打印配置
    print(f"\n⚙️  训练配置:")
    print(f"   模型: {get_model_name(model_type)}")
    print(f"   设备: {device}")
    print(f"   学习率: {learning_rate}")
    print(f"   Batch Size: {batch_size}")
    print(f"   Epochs: {num_epochs}")
    
    print("\n" + "="*80)
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Test Loss':<15} {'Accuracy':<12} {'LR':<15}")
    print("="*80)

    best_accuracy = 0.0
    results = {
        'model_type': model_type,
        'train_losses': [],
        'test_losses': [],
        'accuracies': [],
        'best_accuracy': 0.0
    }

    # 训练循环
    for epoch in range(num_epochs):
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        results['train_losses'].append(train_loss)

        # 验证
        test_loss, accuracy = evaluate(model, test_loader, criterion, device)
        results['test_losses'].append(test_loss)
        results['accuracies'].append(accuracy)

        # 学习率
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()

        # 打印进度
        print(f"{epoch+1:<8} {train_loss:<15.4f} {test_loss:<15.4f} {accuracy:<12.2f}% {current_lr:<15.8f}")

        # 保存最好的模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model_name = f"best_{model_type}_model.pt"
            torch.save(model.state_dict(), project_root / model_name)

        results['best_accuracy'] = best_accuracy

        # 早停
        if early_stopping(test_loss):
            print(f"\n⚠️  早停: 验证loss未改善")
            break

    print("="*80)
    print(f"\n✅ 训练完成!")
    print(f"   模型: {get_model_name(model_type)}")
    print(f"   最佳准确率: {best_accuracy:.2f}%")
    print(f"   最终准确率: {accuracy:.2f}%")
    print(f"   训练轮数: {len(results['train_losses'])}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="训练序列模型")
    parser.add_argument("--model", choices=["fc", "lstm", "transformer"], default="fc",
                       help="模型类型")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=64, help="批大小")
    parser.add_argument("--lr", type=float, default=0.0001, help="学习率")
    
    args = parser.parse_args()
    
    print("\n" + "="*80)
    print(f"🚀 开始训练 {get_model_name(args.model)}")
    print("="*80 + "\n")
    
    results = train_model(
        model_type=args.model,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
    
    print(f"\n💾 模型已保存到: {project_root / f'best_{args.model}_model.pt'}")


if __name__ == "__main__":
    main()
