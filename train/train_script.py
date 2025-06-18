import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# 获取项目根目录的绝对路径
current_file = __file__
current_dir = os.path.dirname(os.path.abspath(current_file))
project_root = os.path.dirname(current_dir)  # 只向上走一级到项目根目录

# 打印路径用于调试
print(f"当前脚本路径: {os.path.abspath(current_file)}")
print(f"项目根目录: {project_root}")
print("系统路径:")
for p in sys.path:
    print(f" - {p}")

# 将项目根目录添加到系统路径
sys.path.insert(0, project_root)

from dataloader.dockdataset import Dockdataset
from models.fcmodel import FCmodel

# 设置随机种子确保可复现性
torch.manual_seed(42)
np.random.seed(42)

if __name__ == "__main__":

  dataset_path = os.path.join(project_root, "dataset")
  total_num = 3805550
  previous_num = 15
  predict_num = 3
  class_num = 5

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  split_index = int(0.8 * total_num)
  train_dataset = Dockdataset(dataset_path, 0, split_index)
  test_dataset = Dockdataset(dataset_path, split_index, total_num)

  batch_size = 64
  train_loader = DataLoader(train_dataset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=16,
                            persistent_workers=True,
                            prefetch_factor=2,)
  test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

  model = FCmodel(previous_num, num_class=class_num)
  model.to(device)

  # ==================== 5. 设置训练参数 ====================
  criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数
  optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam优化器
  num_epochs = 50

  # 记录训练过程
  train_losses = []
  test_accuracies = []

  # ==================== 6. 训练循环 ====================
  for epoch in range(num_epochs):
      # 训练阶段
      model.train()
      epoch_loss = 0.0
      
      for idx, (inputs, labels) in enumerate(train_loader):
          # 前向传播
          outputs = model(torch.tensor(inputs, device=device).float())
          
          # 计算损失
          loss = criterion(outputs, torch.tensor(labels, device=device))
          
          # 反向传播和优化
          optimizer.zero_grad()
          loss.backward()
          optimizer.step()
          
          epoch_loss += loss.item() * inputs.size(0)

          print(idx)
      
      # 计算平均训练损失
      epoch_loss /= len(train_loader.dataset)
      train_losses.append(epoch_loss)
      
      # 评估阶段
      model.eval()
      correct = 0
      total = 0
      
      with torch.no_grad():
          for inputs, labels in test_loader:
              outputs = model(torch.tensor(inputs, device=device))
              _, predicted = torch.max(outputs, 1)  # 获取预测类别
              total += labels.size(0)
              correct += (predicted == labels).sum().item()
      
      accuracy = 100 * correct / total
      test_accuracies.append(accuracy)
      
      # 打印进度
      # if (epoch + 1) % 5 == 0:
      #     print(f"Epoch [{epoch+1}/{num_epochs}], "
      #           f"Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")
      print(f"Epoch [{epoch+1}/{num_epochs}], "
                f"Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.2f}%")