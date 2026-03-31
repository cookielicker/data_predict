"""
改进的训练脚本
基于实际数据分析，重点关注学习率、优化器和训练策略
"""
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

# 设置路径
current_file = __file__
current_dir = os.path.dirname(os.path.abspath(current_file))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

from dataloader.dockdataset import Dockdataset
from models.fcmodel import FCmodel

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

class TrainingConfig:
    """训练配置"""
    # 数据参数
    dataset_path = os.path.join(project_root, "dataset")
    total_num = 3805550
    previous_num = 15
    predict_num = 3
    class_num = 5
    
    # 模型参数
    hidden_size = 256
    num_layers = 10
    
    # 训练参数
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.0001  # ← 降低学习率（原来是0.001）
    weight_decay = 1e-4     # ← 添加权重衰减
    
    # 可选：学习率调度
    use_scheduler = True
    scheduler_type = "cosine"  # "cosine" 或 "step"
    
    # 数据加载器
    num_workers = 16
    persistent_workers = True
    prefetch_factor = 2
    
    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_dataloaders(config):
    """创建数据加载器"""
    split_index = int(0.8 * config.total_num)
    
    train_dataset = Dockdataset(
        config.dataset_path, 
        0, 
        split_index, 
        config.total_num
    )
    test_dataset = Dockdataset(
        config.dataset_path, 
        split_index, 
        config.total_num, 
        config.total_num
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        persistent_workers=config.persistent_workers,
        prefetch_factor=config.prefetch_factor,
    )
    
    return train_loader, test_loader


def create_optimizer_and_scheduler(model, config):
    """创建优化器和学习率调度器"""
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scheduler = None
    if config.use_scheduler:
        if config.scheduler_type == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=config.num_epochs,
                eta_min=config.learning_rate * 0.01
            )
        elif config.scheduler_type == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5
            )
    
    return optimizer, scheduler


def train_epoch(model, train_loader, criterion, optimizer, config):
    """训练一个epoch"""
    model.train()
    epoch_loss = 0.0
    
    for idx, (inputs, labels) in enumerate(train_loader):
        # 前向传播
        outputs = model(torch.tensor(inputs, device=config.device).float())
        
        # 计算损失
        loss = criterion(outputs, torch.tensor(labels, device=config.device))
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * inputs.size(0)
        
        # 定期打印进度
        if idx % 1000 == 0:
            current_loss = epoch_loss / ((idx + 1) * inputs.size(0))
            print(f"  Batch {idx:5d} | Loss: {current_loss:.4f}")
    
    # 计算平均损失
    epoch_loss /= len(train_loader.dataset)
    return epoch_loss


def evaluate(model, test_loader, criterion, config):
    """评估模型"""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(torch.tensor(inputs, device=config.device).float())
            loss = criterion(outputs, torch.tensor(labels, device=config.device))
            
            total_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == torch.tensor(labels, device=config.device)).sum().item()
    
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = total_correct / len(test_loader.dataset)
    
    return avg_loss, accuracy


def main():
    """主训练函数"""
    config = TrainingConfig()
    
    print("="*70)
    print("改进的FC模型训练脚本")
    print("="*70)
    print(f"\n📊 配置信息:")
    print(f"  设备: {config.device}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  使用学习率调度: {config.use_scheduler}")
    if config.use_scheduler:
        print(f"  调度器类型: {config.scheduler_type}")
    
    # 创建数据加载器
    print(f"\n📁 创建数据加载器...")
    train_loader, test_loader = create_dataloaders(config)
    print(f"  训练集: {len(train_loader.dataset):,} 样本")
    print(f"  测试集: {len(test_loader.dataset):,} 样本")
    
    # 创建模型
    print(f"\n🧠 创建模型...")
    model = FCmodel(
        input_size=config.previous_num,
        num_class=config.class_num,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers
    )
    model.to(config.device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  总参数数: {total_params:,}")
    print(f"  可训练参数: {trainable_params:,}")
    
    # 创建损失函数和优化器
    print(f"\n⚙️  创建优化器...")
    criterion = nn.CrossEntropyLoss()  # 数据分布良好，无需加权
    optimizer, scheduler = create_optimizer_and_scheduler(model, config)
    print(f"  优化器: AdamW")
    print(f"  学习率: {config.learning_rate}")
    print(f"  权重衰减: {config.weight_decay}")
    
    # 训练循环
    print(f"\n🚀 开始训练...\n")
    print("="*70)
    
    train_losses = []
    test_losses = []
    test_accuracies = []
    best_accuracy = 0.0
    
    for epoch in range(config.num_epochs):
        print(f"Epoch {epoch+1:3d}/{config.num_epochs:3d}")
        
        # 训练
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config)
        train_losses.append(train_loss)
        print(f"  Training Loss: {train_loss:.4f}")
        
        # 评估
        test_loss, accuracy = evaluate(model, test_loader, criterion, config)
        test_losses.append(test_loss)
        test_accuracies.append(accuracy)
        
        print(f"  Test Loss: {test_loss:.4f} | Accuracy: {accuracy*100:.2f}%")
        
        # 保存最好的模型
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(project_root, "best_model.pt"))
            print(f"  ✓ 保存最好模型 (准确率: {accuracy*100:.2f}%)")
        
        # 学习率调度
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            print(f"  学习率: {current_lr:.6f}")
        
        print()
    
    print("="*70)
    print("✅ 训练完成!")
    print(f"最佳准确率: {best_accuracy*100:.2f}%")
    print("="*70)
    
    # 保存模型
    model_path = os.path.join(project_root, "final_model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n模型已保存到: {model_path}")


if __name__ == "__main__":
    main()
