import os
import sys

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

file = "C:/Users/p_hzhongxu/Documents/GitHub/data_predict/raw_data/circulating_cap.csv"

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
print(data_np[:, 0])


x = np.arange(0, len(data_np[:, 0]))

plt.plot(x, data_np[:, 0])
plt.show()