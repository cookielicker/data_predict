import os
import sys
from pathlib import Path
sys.path.append(Path("./"))
import pandas as pd
import numpy as np

def convert_stock_code(stock_code):
    """
    将股票代码转换为指定格式
    """
    if stock_code.endswith('.XSHE'):
        return f'CS_SZSE_{stock_code[:6]}'
    elif stock_code.endswith('.XSHG'):
        return f'CS_SSE_{stock_code[:6]}'
    return stock_code

def load_trade_date(folder_list, file_name, father=None):
  date_df = None
  for folder in folder_list:
    if father is None:
      file_path = os.path.join(folder, file_name)
    else:
      file_path = os.path.join(father, folder, file_name)
    if date_df is None:
      date_df = pd.read_csv(file_path, header=None)
    else:
      date_df = pd.concat([date_df, pd.read_csv(file_path, header=None)])
  return date_df

def read_data_csv(file_list, data=None, axis=0):
  for file in file_list:
    data_file = np.loadtxt(file, delimiter=",")
    if data is None:
      data = data_file
    else:
      data = np.concatenate([data, data_file], axis=axis)
  return data

def load_data(folder_list, file_name, father=None, data=None, axis=0):
  file_list = []
  for folder in folder_list:
    if father is None:
      file_list.append(os.path.join(folder, file_name))
    else:
      file_list.append(os.path.join(father, folder, file_name))
  data_np = read_data_csv(file_list, data=data, axis=axis)
  return data_np

def safe_elementwise_divide(a, b, default=np.nan, handle_zero='nan'):
    """
    安全执行矩阵逐元素除法
    
    参数:
        a (ndarray): 分子矩阵
        b (ndarray): 分母矩阵
        default: 无效计算时的默认值（默认 NaN）
        handle_zero: 分母为零时的处理方式 ('nan', 'inf', 'zero')
    
    返回:
        ndarray: 结果矩阵
    """
    # 创建结果矩阵并初始化为默认值
    result = np.full_like(a, fill_value=default, dtype=float)
    
    # 创建有效分母掩码（非零且非NaN）
    valid_mask = (b != 0) & ~np.isnan(b)
    
    # 执行安全除法
    with np.errstate(divide='ignore', invalid='ignore'):
        # 计算有效位置的结果
        result[valid_mask] = a[valid_mask] / b[valid_mask]
        
        # 处理分母为零的情况
        if handle_zero == 'nan':
            result[b == 0] = np.nan
        elif handle_zero == 'inf':
            # 正无穷或负无穷
            result[(b == 0) & (a > 0)] = np.inf
            result[(b == 0) & (a < 0)] = -np.inf
            result[(b == 0) & (a == 0)] = np.nan  # 0/0 未定义
        elif handle_zero == 'zero':
            result[b == 0] = 0
    
    return result

def safe_elementwise_multiple(a, b, default=np.nan):
    """
    安全执行矩阵逐元素乘法
    
    参数:
        a (ndarray): 矩阵
        b (ndarray): 矩阵
        default: 无效计算时的默认值（默认 NaN）
    
    返回:
        ndarray: 结果矩阵
    """
    # 创建结果矩阵并初始化为默认值
    result = np.full_like(a, fill_value=default, dtype=float)
    
    # 创建有效分母掩码（非零且非NaN）
    valid_mask = (b != 0) & ~np.isnan(b)
    
    # 执行安全乘法
    result[valid_mask] = a[valid_mask] * b[valid_mask]
    
    return result

def find_element_numpy(df, target):
    """
    使用 NumPy 高效查找元素位置
    
    参数:
        df: Pandas DataFrame
        target: 要查找的值
        
    返回:
        list: (index, column) 元组列表
    """
    # 将 DataFrame 转换为 NumPy 数组
    arr = df.values
    
    # 处理 NaN
    if pd.isna(target):
        mask = pd.isna(arr)
    else:
        mask = (arr == target)
    
    # 获取匹配位置的行列索引
    rows, cols = np.where(mask)
    
    # 转换为 DataFrame 的索引和列名
    positions = []
    for i in range(len(rows)):
        row_idx = df.index[rows[i]]
        col_name = df.columns[cols[i]]
        positions.append((row_idx, col_name))
    
    return positions


if __name__ == "__main__":
  raw_data_path = Path("../raw_data")
  # folder_names
  folder_names = ["data_2021-11-01_2022-10-31",
                "data_2022-11-01_2023-10-31",
                "data_2023-11-01_2024-10-31",
                "data_2024-11-01_2025-02-14"]
  # data_names
  raw_names = ["mtx_open_1day.csv",
                "mtx_close_1day.csv",
                "mtx_high_1day.csv",
                "mtx_low_1day.csv",
                "mtx_vol_1day.csv",
                "mtx_amount_1day.csv"]
  
  adj_names = ["mtx_open_adj_1day.csv",
                "mtx_close_adj_1day.csv",
                "mtx_high_adj_1day.csv",
                "mtx_low_adj_1day.csv",]
  
  trade_date = "trade_date_list.csv"
  
  ### 计算adj后均价
  # 获取 adj 因子
  open_data = load_data(folder_names, raw_names[0], father=raw_data_path)
  # print(open_data.shape)
  open_adj_data = load_data(folder_names, adj_names[0], father=raw_data_path)

  adj_factor = safe_elementwise_divide(open_adj_data, open_data)
  # print(adj_factor.shape)
  # print(adj_factor[:, 0])

  # 成交额
  amount_data = load_data(folder_names, raw_names[5], father=raw_data_path)
  # 成交量
  vol_data = load_data(folder_names, raw_names[4], father=raw_data_path)
  # adj前均价
  mean_data = safe_elementwise_divide(amount_data, vol_data)
  # adj后均价
  mean_adj_data = safe_elementwise_multiple(mean_data, adj_factor)

  ### 计算换手率
  change_data = np.full_like(vol_data, fill_value=np.nan, dtype=float)

  ### circulating cap
  cir_cp_file = Path("../raw_data/circulating_cap.csv")
  cir_cp_df = pd.read_csv(cir_cp_file)

  ### sec_list
  sec_list_file = Path("../raw_data/sec_list.csv")
  sec_list_df = pd.read_csv(sec_list_file, header=None)

  ### trade_date_list
  trade_date_df = load_trade_date(folder_names, trade_date, father=raw_data_path)
  # print(vol_data.shape)
  # print(trade_date_df.shape)

  columns = cir_cp_df.columns
  for cir_index, code in enumerate(cir_cp_df[columns[0]]):
    stock_code = convert_stock_code(code)
    positions = find_element_numpy(sec_list_df, stock_code)
    #  print(positions)
    try:
      code_index = positions[0][0]
      # print(code_index)
    except:
      code_index = None
      print(f"stock code {stock_code} not find in sec list!!!")

    if code_index is not None:
      for date_index, date in enumerate(trade_date_df[0]):
        cap_data = cir_cp_df.get(date)
        if cap_data is None:
          print(f"{date} cap data missing!!!")
        else:
          change_data[date_index, code_index] = cap_data[cir_index]
  
  ### cir_cap 单位为10000
  change_data = safe_elementwise_multiple(change_data, np.full_like(change_data, fill_value=10000, dtype=float))
  # 计算换手率
  change_data = safe_elementwise_divide(vol_data, change_data)
  # print(change_data[:, 0])

  ### write data
  # create middle data
  middle_data_path = Path("../middle_data")
  os.makedirs(middle_data_path, exist_ok=True)
  middle_files = ["high_adj.npy",
                  "low_adj.npy",
                  "mean_adj.npy",
                  "change_rate.npy"]
  write_flag = False
  if write_flag:
    # adj均价
    np.save(os.path.join(middle_data_path, middle_files[2]), mean_adj_data)

    # 换手率
    np.save(os.path.join(middle_data_path, middle_files[3]), mean_adj_data)

    # adj最高价
    high_adj_data = load_data(folder_names, adj_names[2], father=raw_data_path)
    np.save(os.path.join(middle_data_path, middle_files[0]), high_adj_data)

    # adj最低价
    low_adj_data = load_data(folder_names, adj_names[3], father=raw_data_path)
    np.save(os.path.join(middle_data_path, middle_files[1]), high_adj_data)
       
  # loaded_adj_mean_data = np.load("../middle_data/mean_adj.npy")
  # from matplotlib import pyplot as plt
  # x = np.arange(0, len(loaded_adj_mean_data[:, 0]))
  # # plt.bar(x, data_np[:, 0])
  # plt.plot(x, loaded_adj_mean_data[:, 0])
  # plt.show()