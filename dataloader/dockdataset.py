import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

class Dockdataset(Dataset):
  def __init__(self, dataset_path, start_id, end_id, total, class_num=5, previous_num=30, predict_num=3, data_limit=10000, use_3class=False):
    super().__init__()
    self.path = dataset_path
    self.start = start_id
    self.end = end_id
    self.total = total
    self.class_num = class_num
    self.previous = previous_num
    self.predict = predict_num
    self.limit = data_limit
    self.cache_name = None
    self.cache = None
    self.use_3class = use_3class

  def __len__(self):
    return self.end - self.start
  
  def _cal_pct(self, data):
    result = np.zeros_like(data)
    prev = data[:-1]
    cur = data[1:]
    pct = (cur - prev) / prev
    result[1:] = pct
    return result
  
  def _make_label(self, data):
    pct = (data[-1] - data[0]) / data[0]
    if pct < -0.1:
      label = 0
    elif -0.1 <= pct < -0.03:
      label = 1
    elif -0.03 <= pct <= 0.03:
      label = 2
    elif 0.03 < pct <= 0.1:
      label = 3
    else:
      label = 4
    return label

  def _make_label_3class(self, data):
    """三分类：下跌/震荡/上涨"""
    pct = (data[-1] - data[0]) / data[0]
    if pct < -0.1:
      return 0  # 大幅下跌
    elif pct > 0.1:
      return 2  # 大幅上涨
    else:
      return 1  # 震荡/小幅波动

  def _trans(self, raw_data):
    mean, change = raw_data[:self.previous+self.predict], raw_data[self.previous+self.predict:]
    # 价格变化率(15维) + 换手率(15维) = 30维
    price_pct = self._cal_pct(mean[:self.previous])
    label = self._make_label(mean[self.previous-1:])
    # 换手率直接用原始值
    data = np.concatenate([price_pct, change[:self.previous]])
    return data, label
  
  def __getitem__(self, index):
    index = index + self.start
    filename = ((index + self.limit) // self.limit) * self.limit
    if filename > self.total:
      filename = self.total
    filename = f"{filename:07d}.npy"
    filename = os.path.join(Path(self.path), filename)
    if self.cache_name == filename:
      file_data = self.cache
    else:
      file_data = np.load(filename)
      self.cache_name = filename
      self.cache = file_data
    data, label = self._trans(file_data[index%self.limit])
    # 使用三分类或五分类
    if self.use_3class:
      label = self._make_label_3class(file_data[index%self.limit][:self.previous+self.predict])
    return data, label
    