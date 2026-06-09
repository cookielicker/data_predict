import os
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

# ============ 可配置参数 ============
PREVIOUS_NUM = 30   # 历史数据天数
PREDICT_NUM = 3     # 预测天数
# ==================================

def parse_args():
    parser = argparse.ArgumentParser(description='生成训练数据集')
    parser.add_argument('--previous', type=int, default=PREVIOUS_NUM,
                        help=f'历史数据天数 (默认 {PREVIOUS_NUM})')
    parser.add_argument('--predict', type=int, default=PREDICT_NUM,
                        help=f'预测天数 (默认 {PREDICT_NUM})')
    parser.add_argument('--middle', type=str, default='middle_data_baostock',
                        help='中间数据目录 (默认 middle_data_baostock)')
    parser.add_argument('--output', type=str, default=None,
                        help='输出目录 (默认 dataset_<previous_num>)')
    return parser.parse_args()

def check_nan_presence(sequence):
    """
    检查序列中是否存在 NaN 或 0 值

    参数:
        sequence: 可迭代对象 (列表、元组、NumPy 数组等)

    返回:
        bool: 存在 NaN 或 0 返回 True，否则返回 False
    """
    arr = np.asarray(sequence)
    return np.any(np.isnan(arr))


def find_continue_data(frames: list,
                        previous_num: int = PREVIOUS_NUM,
                        predict_num: int = PREDICT_NUM):
    """
    输出: List 其中元素为多个frame的data直接拼接的np array
    """
    num_data = previous_num + predict_num
    frame_length = len(frames[0])
    start_id = 0
    end_id = start_id + num_data
    p_flag = False
    frame_data = []
    while end_id <= frame_length:
        new_seq = None
        flag = False
        for i, frame in enumerate(frames):
            flag |= np.isnan(frame[end_id]) if p_flag else check_nan_presence(frame[start_id:end_id])
            if i == 0 and not p_flag:
                flag |= np.any(frame[start_id:end_id] == 0)  # mean_adj不能有0
        if flag:
            p_flag = False
            start_id = end_id + 1
            end_id = start_id + num_data
        else:  # 没有NaN
            new_seq = np.concatenate([frame[start_id:end_id] for frame in frames], axis=-1)
            start_id += 1
            end_id += 1
        if new_seq is not None:
            frame_data.append(new_seq)
    return frame_data


def main():
    args = parse_args()
    previous_num = args.previous
    predict_num = args.predict
    middle_data_path = Path(args.middle)
    dataset_path = Path(args.output) if args.output else Path(f"dataset_{previous_num}")

    os.makedirs(dataset_path, exist_ok=True)
    # 清空旧数据
    for f in Path(dataset_path).glob("*.npy"):
        f.unlink()

    print(f"=" * 60)
    print(f"数据生成配置")
    print(f"=" * 60)
    print(f"历史天数 (previous_num): {previous_num}")
    print(f"预测天数 (predict_num): {predict_num}")
    print(f"每样本特征数: ({previous_num} + {predict_num}) + {previous_num}*3 = {(previous_num + predict_num) + previous_num * 3}")
    print(f"  - mean (价格): {previous_num} + {predict_num}")
    print(f"  - change (换手率): {previous_num}")
    print(f"  - high_low (波动率): {previous_num}")
    print(f"  - change_delta (换手率变化量): {previous_num}")
    print(f"中间数据目录: {middle_data_path}")
    print(f"输出目录: {dataset_path}")
    print(f"=" * 60)

    # 加载原始数据
    middle_files = ["mean_adj.npy", "change_rate.npy", "high_low_ratio.npy", "change_delta.npy"]
    data_frames = []
    for file in middle_files:
        data_frames.append(np.load(middle_data_path / file))

    length, number = data_frames[0].shape

    # 加载股票代码
    codes_df = pd.read_csv(middle_data_path / "codes.csv")
    stock_codes = codes_df['code'].tolist()

    print(f"原始数据: {length}天 x {number}只股票")

    count = 0
    dataset = []
    code_list = []  # 每个样本对应的股票代码
    len_limit = 10000

    for i in range(number):
        conti_data_list = find_continue_data(
            [data_frame[:, i] for data_frame in data_frames],
            previous_num=previous_num,
            predict_num=predict_num
        )
        new_num = len(conti_data_list)
        print(f"number: {i}, code: {stock_codes[i]}, new data num: {new_num}")

        count += new_num
        dataset.extend(conti_data_list)
        code_list.extend([stock_codes[i]] * new_num)

        if len(dataset) > len_limit:
            batch_label = f"{count // len_limit * len_limit:07d}"
            datapath = os.path.join(dataset_path, f"{batch_label}.npy")
            np.save(datapath, np.vstack(dataset[:len_limit]))
            # 保存对应的股票代码
            with open(os.path.join(dataset_path, f"{batch_label}_codes.txt"), "w") as f:
                f.write("\n".join(code_list[:len_limit]))
            print(f"data saved, idx {batch_label}")
            dataset = dataset[len_limit:]
            code_list = code_list[len_limit:]

    if len(dataset) > 0:
        batch_label = f"{count:07d}"
        datapath = os.path.join(dataset_path, f"{batch_label}.npy")
        np.save(datapath, np.vstack(dataset))
        with open(os.path.join(dataset_path, f"{batch_label}_codes.txt"), "w") as f:
            f.write("\n".join(code_list))
        print(f"data saved, idx {batch_label}")

    print(f"\n生成完成！共 {count:,} 个样本")
    print(f"输出目录: {dataset_path}")

if __name__ == "__main__":
    main()