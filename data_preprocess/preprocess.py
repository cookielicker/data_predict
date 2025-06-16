import os
import sys

import numpy as np

def check_nan_presence(sequence):
    """
    检查序列中是否存在 NaN 值
    
    参数:
        sequence: 可迭代对象 (列表、元组、NumPy 数组等)
        
    返回:
        bool: 存在 NaN 返回 True，否则返回 False
    """
    arr = np.asarray(sequence)  # 转换为 NumPy 数组
    return np.any(np.isnan(arr))  # 关键逻辑


def find_continue_data(frames:list,
                       previous_num:int = 15,
                       predict_num:int = 3,)
  num_frame = len(frames)
  frame_length = len(frames[0])
  num_data = previous_num + predict_num
  start_id = 0
  end_id = start_id + num_data - 1
  p_flag = False
  frame_data = None
  while end_id <= frame_length:
    new_seq = None
    # 判断当前连续段是否符合 不含有NaN
    if p_flag: # 前序段不含NaN
      flag = False
      for frame in frames:
        flag |= np.isnan(frame[end_id])
      if flag: # 出现NaN
        p_flag = False
        start_id = end_id + 1
        end_id = start_id + num_data - 1
      else: # 没有NaN
        new_seq = np.concatenate([frame[start_id:end_id] for frame in frames], axis=-1)
        start_id += 1
        end_id += 1
    else: # 前序段含NaN
      flag = False
      for frame in frames:
        flag |= check_nan_presence(frame[start_id:end_id])
      if flag:
        p_flag = False
        start_id = end_id + 1
        end_id = start_id + num_data - 1
      else:
        new_seq = np.concatenate([frame[start_id:end_id] for frame in frames], axis=-1)
        start_id += 1
        end_id += 1
    if new_seq is not None:
      if frame_data is not None:
        frame_data = 
        
