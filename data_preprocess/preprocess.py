import os
import sys
from pathlib import Path
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
                       predict_num:int = 3,):
  """
  输出: List 其中元素为多个frame的data直接拼接的np array
  """
  num_frame = len(frames)
  frame_length = len(frames[0])
  num_data = previous_num + predict_num
  start_id = 0
  end_id = start_id + num_data
  p_flag = False
  frame_data = []
  while end_id <= frame_length:
    new_seq = None
    flag = False
    for frame in frames:
      flag |= np.isnan(frame[end_id]) if p_flag else check_nan_presence(frame[start_id:end_id]) # 前序段是否含NaN
    if flag: # 出现NaN
      p_flag = False
      start_id = end_id + 1
      end_id = start_id + num_data
    else: # 没有NaN
      new_seq = np.concatenate([frame[start_id:end_id] for frame in frames], axis=-1)
      start_id += 1
      end_id += 1
    if new_seq is not None:
      # if frame_data is not None:
      #   frame_data = np.vstack([frame_data, new_seq])
      # else:
      #   frame_data = new_seq
      frame_data.append(new_seq)
  return frame_data

if __name__ == "__main__":
  middle_data_path = Path("../middle_data")
  middle_files = ["mean_adj.npy",
                  "change_rate.npy"]
  data_frames = []
  for file in middle_files:
    data_frames.append(np.load(os.path.join(middle_data_path, file)))

  length, number = data_frames[0].shape

  dataset_path = Path("../dataset")
  os.makedirs(dataset_path, exist_ok=True)

  count = 0
  dataset = []
  len_limit = 10000
  
  for i in range(number):
    # find continue data
    conti_data_list = find_continue_data([data_frame[:, i] for data_frame in data_frames])
    new_num = len(conti_data_list)
    print(f"number: {i}, new data num: {new_num}")
    # 
    count += new_num
    dataset.extend(conti_data_list)

    if len(dataset) > len_limit:
      dataname = f"{count//len_limit*len_limit:07d}.npy"
      datapath = os.path.join(dataset_path, dataname)
      np.save(datapath, np.vstack(dataset[:len_limit]))
      print(f"data saved, idx {count//len_limit*len_limit}")
      dataset = dataset[len_limit:]
  
  if len(dataset) > 0:
    dataname = f"{count:07d}.npy"
    datapath = os.path.join(dataset_path, dataname)
    np.save(datapath, np.vstack(dataset))
    print(f"data saved, idx {count}")
  
  # continue_data = find_continue_data([data_frames[0][:, 4], data_frames[1][:, 4]])
  # print(len(continue_data[0]))
  # for i in range(20):
  #   print(continue_data[i])

  # length, number = loaded_change_rate_data.shape

