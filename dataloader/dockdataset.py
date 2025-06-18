import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class Dockdataset(Dataset):
  def __init__(self, dataset_path, start_id, end_id, class_num=5, previous_num=15, predict_num=3, data_limit=10000):
    super().__init__()
    self.path = dataset_path
    self.start = start_id
    self.end = end_id
    self.class_num = class_num
    self.previous = previous_num
    self.predict = predict_num
    self.limit = data_limit

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

  def _trans(self, raw_data):
    mean, change = raw_data[:self.previous+self.predict], raw_data[self.previous+self.predict:]
    pct = self._cal_pct(mean[:self.previous])
    label = self._make_label(mean[self.previous-1:])
    data = np.concatenate([pct, change[:self.previous]])
    return data, label
  
  def __getitem__(self, index):
    index = index + self.start
    filename = ((index + self.limit - 1) // self.limit) * self.limit
    filename = f"{filename:07d}.npy"
    filename = os.path.join(Path(self.path), filename)
    file_data = np.load(filename)
    data, label = self._trans(file_data[index%self.limit])
    return data, label
    