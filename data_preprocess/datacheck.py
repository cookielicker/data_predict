import os
import sys
import re
from pathlib import Path
import numpy as np
from preprocess import find_continue_data

def check_change_rate(seq:np.array, n=-1):
    # 使用assert进行严格验证
    assert seq.ndim == 1, "输入必须是一维数组"
    if n > 0:
      assert len(seq) % 2 == 0, "未指定换手率的情况下，输入必须是2的倍数"
      n = len(seq) // 2
    else:
      assert len(seq) > n, "换手率数据长度需小于总数据长度"
    # 检查前n个元素是否都小于1
    return np.all(seq[n:] < 1) and np.all(seq[:] > 0)

if __name__ == "__main__":

  pattern = re.compile(r'^(\d{7})\.npy$')
  dataset_path = Path("../dataset")
  # 遍历目录
  for filename in os.listdir(dataset_path):
      # 检查文件名是否匹配
      if pattern.match(filename):
        writed_data = np.load(os.path.join(dataset_path, filename))
        for i in range(len(writed_data)):
          if not check_change_rate(writed_data[i]):
            print(f"data idx : {int(filename[:7]) + i} check fail!")
  print("check finished!")


