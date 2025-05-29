import os
import sys
sys.path.append("./")
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

file = "../raw_data/circulating_cap.csv"

cir_cp_df = pd.read_csv(file)

print(cir_cp_df.columns)
print(cir_cp_df.shape)
# print(cir_cp_df.info())
print(cir_cp_df["2021-11-01"])
print(cir_cp_df["2021-11-01"][0])
# print(cir_cp_df.head())


data_file = "../raw_data/data_2021-11-01_2022-10-31/mtx_open_1day.csv"

# data_df = pd.read_csv(data_file)

# print(data_df.shape)
# print(data_df)

data_np = np.loadtxt(data_file, delimiter=",")

print(data_np.shape)
print(data_np[:, 1].shape)
print(data_np[:, 1])
print(data_np[:, 1][0])
print(type(data_np[:, 1][0]))
print(np.isnan(data_np[:, 1]))

sec_list_file = "../raw_data/sec_list.csv"
sec_list_df = pd.read_csv(sec_list_file, header=None)
print(sec_list_df[0][0])


# x = np.arange(0, len(data_np[:, 0]))

# plt.plot(x, data_np[:, 0])
# plt.show()

file_names = ["data_2021-11-01_2022-10-31",
              "data_2022-11-01_2023-10-31",
              "data_2023-11-01_2024-10-31",
              "data_2024-11-01_2025-02-14"]

for f_name in file_names:
  data_f = "../raw_data/" + f_name + "/mtx_open_1day.csv"
  data_np = np.loadtxt(data_f, delimiter=",")
  print(data_np.shape)
  print(data_np[:, 1].shape)
  print(data_np[:, 1])
  print(np.isnan(data_np[:, 1]))
