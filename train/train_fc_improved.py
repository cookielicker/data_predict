"""
改进的训练脚本 v2
修复: 学习率 + 优化器 + 学习率调度 + 早停
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 路径设置
current_file = __file__
current_dir = os.path.dirname(os.path.abspath(current_file))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from dataloader.dockdataset import Dockdataset
from models.fcmodel import FCmodel

torch.manual_seed(42)
np.random.seed(42)

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
                return True  # 应该停止
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("改进的FC模型训练 v2 (学习率 + 调度 + 早停)")
    print("="*70 + "\n")

    dataset_path = os.path.join(project_root, "dataset")
    total_num = 3805550
    previous_num = 15
    predict_num = 3
    class_num = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}\n")

    # 创建数据加载器
    split_index = int(0.8 * total_num)
    train_dataset = Dockdataset(dataset_path, 0, split_index, total_num)
    test_dataset = Dockdataset(dataset_path, split_index, total_num, total_num)

    batch_size = 64
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
    model = FCmodel(previous_num, num_class=class_num)
    model.to(device)

    # ===== 改进1: 更低的学习率 =====
    learning_rate = 0.0001  # 原来是 0.001
    
    # ===== 改进2: 使用 AdamW 而不是 Adam =====
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    
    # ===== 改进3: 学习率调度 =====
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=50,  # 总epoch数
        eta_min=learning_rate * 0.01
    )
    
    # ===== 改进4: 早停 =====
    early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    num_epochs = 50
    train_losses = []
    test_losses = []
    test_accuracies = []

    print(f"训练参数:")
    print(f"  学习率: {learning_rate}")
    print(f"  优化器: AdamW")
    print(f"  学习率调度: CosineAnnealingLR")
    print(f"  早停耐心值: 5")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}\n")

    print("="*70)
    print(f"{'Epoch':<8} {'Train Loss':<15} {'Test Loss':<15} {'Accuracy':<12} {'LR':<12}")
    print("="*70)

    best_accuracy = 0.0

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        epoch_loss = 0.0

        for idx, (inputs, labels) in enumerate(train_loader):
            outputs = model(torch.tensor(inputs, device=device).float())
            loss = criterion(outputs, torch.tensor(labels, device=device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * inputs.size(0)

        epoch_loss /= len(train_loader.dataset)
        train_losses.append(epoch_loss)

        # 验证阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(torch.tensor(inputs, device=device).float())
                loss = criterion(outputs, torch.tensor(labels, device=device))
                test_loss += loss.item() * inputs.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted.to(torch.device("cpu")) == labels).sum().item()

        test_loss /= len(test_loader.dataset)
        accuracy = 100 * correct / total
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)

        # 获取当前学习率
        current_lr = scheduler.get_last_lr()[0]

        # 打印进度
        print(f"{epoch+1:<8} {epoch_loss:<15.4f} {test_loss:<15.4f} {accuracy:<12.2f}% {current_lr:<12.6f}")

        # 保存最好的模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(project_root, "best_fc_model.pt"))

        # 学习率调度
        scheduler.step()

        # 早停检查
        if early_stopping(test_loss):
            print(f"\n⚠️  早停: 验证loss在 {early_stopping.patience} 个epoch内未改善")
            print(f"最佳准确率: {best_accuracy:.2f}%")
            break

    print("="*70)
    print(f"\n最终结果:")
    print(f"  最佳准确率: {best_accuracy:.2f}%")
    print(f"  最终准确率: {test_accuracies[-1]:.2f}%")
    print(f"  总epoch数: {len(train_losses)}")
    print(f"\n✓ 模型已保存到: {os.path.join(project_root, 'best_fc_model.pt')}")
