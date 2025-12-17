import json
from typing import Tuple, List

import pandas as pd
import numpy as np
import os, glob
import matplotlib.pyplot as plt
from datetime import timedelta
from processed_data import calculate_status_active_times

def data_preprocess(file_path):
    """
    对原始数据文件进行预处理，包括时间戳转换和保存处理后的数据，主要是将时间戳转化为yy-mm-dd hh:mm:ss的格式

    :param file_path: 原始数据文件路径 (str)
    :return: 无返回值，处理后的数据将直接保存为新文件
    """
    # 定义列标题
    headers = ['timestamp', 'current', 'voltage', 'apparent power', 'active power']

    # 读取原始数据，指定列标题
    df = pd.read_csv(file_path, names=headers)

    # 检查并处理NaN值
    print(f"处理前数据行数: {len(df)}")
    print(f"Timestamp列NaN值数量: {df['timestamp'].isna().sum()}")

    # 删除timestamp列中的NaN值
    df = df.dropna(subset=['timestamp'])

    # 检查timestamp是否为有限值（非inf）
    df = df[np.isfinite(df['timestamp'])]

    print(f"清理NaN和inf值后数据行数: {len(df)}")

    # 时间列预处理：去掉后六位微秒数，转换为秒级时间戳
    # 使用整数除法提取前10位（秒级时间戳）
    df['datetime'] = df['timestamp'].astype(np.int64) // 1000000

    # 转换为datetime并格式化
    df['datetime'] = pd.to_datetime(df['datetime'], unit='s')
    df['datetime'] = df['datetime'].dt.tz_localize('UTC').dt.tz_convert('Asia/Shanghai')
    df['datetime'] = df['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')

    print(df['datetime'].head(100))

    # 重新排列列顺序，将 datetime 放在第一列
    cols = ['datetime'] + headers
    df = df[cols]

    directory = os.path.dirname(file_path)  # 获取目录部分
    filename = os.path.basename(file_path)  # 获取文件名部分
    name, ext = os.path.splitext(filename)  # 拆分文件名和扩展名
    filename = f"processed_{name}{ext}"
    filepath = os.path.join(directory, filename)

    # 保存新CSV
    df.to_csv(filepath, index=False)

    print(f"已保存处理后的数据到 {filepath}")


def merge_csv_files(file_paths, output_path=None):
    """
    按照给定的csv文件路径顺序进行合并，并写入新文件

    参数:
        file_paths (list): CSV文件路径列表
        output_path (str, optional): 合并后输出的文件路径，若为None则自动生成

    返回:
        pd.DataFrame: 合并后的DataFrame
    """
    COLUMNS = ['col1', 'col2', 'col3', 'col4', 'col5']  # 自定义列名，确保5列
    dfs = []
    for file in file_paths:
        df = pd.read_csv(file, header=None, names=COLUMNS)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)

    # 自动生成输出路径（如果未指定）
    if output_path is None:
        directory = os.path.dirname(file_paths[0])
        base_name = "data.csv"
        output_path = os.path.join(directory, base_name)

    # 保存到CSV
    merged_df.to_csv(output_path, index=False)
    print(f"已保存合并后的数据到 {output_path}")

    return merged_df


def replace_first_two_lines_with_header(folder_path):
    """
    将指定文件夹下所有 .csv 文件的前两行替换为一行 'datetime,active_power'

    :param folder_path: 文件夹路径 (str)
    :return: 无返回值
    """
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # 删除前两行，并插入新行
            if len(lines) >= 2:
                lines = lines[2:]  # 删除前两行
            elif len(lines) == 1:
                lines = lines[1:]  # 只删一行
            else:
                lines = []  # 文件为空或不足一行

            # 插入新 header
            lines.insert(0, "datetime,active_power\n")

            # 写回文件
            with open(file_path, 'w', encoding='utf-8') as file:
                file.writelines(lines)

            print(f"已处理文件: {filename}")


def label_csv_by_list(csv_file, label_list):
    """
    根据时间范围列表为 CSV 文件中的数据打标签

    :param csv_file: CSV 文件路径 (str)
    :param label_list: 标签列表，每个元素为 [start_time, end_time, label] (list)
    :return: 无返回值，直接修改并保存 CSV 文件
    """
    df = pd.read_csv(csv_file)
    label_mapping = json.load(open('./label_mapping.json'))

    # 确保 datetime 列为 datetime 类型
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 如果 status 列不存在，则创建
    if 'status' not in df.columns:
        df['status'] = 0  # 默认值为 0

    if 'on/off status' not in df.columns:
        df['on/off status'] = 0

    for label_item in label_list:
        start_time = pd.to_datetime(label_item[0])
        end_time = pd.to_datetime(label_item[1])
        label_value = label_mapping[label_item[2]]

        # 为指定时间范围内的数据分配标签值
        mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
        df.loc[mask, 'status'] = label_value

        # 当标签值非0时，将on/off status置为1
        if label_value != 0:
            df.loc[mask, 'on/off status'] = 1
        else:
            df.loc[mask, 'on/off status'] = 0

    # 生成新文件名：在源文件名基础上加上'_label'
    directory = os.path.dirname(csv_file)
    filename = os.path.basename(csv_file)
    name, ext = os.path.splitext(filename)
    new_filename = f"{name}_labeled{ext}"
    new_filepath = os.path.join(directory, new_filename)

    # 保存修改后的数据
    df.to_csv(new_filepath, index=False)
    print(f"已为文件 {new_filepath} 完成标签分配")

    print_label_result(new_filepath)


def print_label_result(csv_file):
    """
    读取CSV文件，以timestamp为x轴，active power为y轴绘制可视化图，
    并根据status的值为不同数据标上不同颜色
    """
    print(f"可视化标签化文件:{csv_file}")
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 将timestamp转换为datetime格式用于更好的显示
    df['datetime'] = pd.to_datetime(df['datetime'])

    # 创建图形和轴
    plt.figure(figsize=(12, 8))

    # 定义颜色映射 - 根据status值分配不同颜色
    colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
    status_values = sorted(df['status'].unique())

    # 为每个status值绘制数据点
    for i, status in enumerate(status_values):
        # 筛选当前status的数据
        status_data = df[df['status'] == status]

        # 绘制散点图
        plt.scatter(status_data['timestamp'],
                    status_data['active power'],
                    c=colors[i % len(colors)],
                    label=f'Status {status}',
                    alpha=0.7,
                    s=10)

    # 设置图表标题和标签
    plt.title(csv_file)
    plt.xlabel('Timestamp')
    plt.ylabel('Active Power')

    # 添加图例
    plt.legend()

    # 格式化x轴显示
    plt.xticks(rotation=45)

    # 自动调整布局
    plt.tight_layout()

    # 显示图表
    plt.show()


def print_csv_in_dir(dir_path='./process_dataset', order='*resample*'):
    """

    :param dir_path:
    :param order:
    :return:
    """
    files = get_all_files(dir_path, order)  # 获取所有已经打了标签的csv文件

    for file in files:
        print_label_result(file)


def label_csv_by_json(csv_file, json_file):
    json_dict = json.load(open(json_file))[0]
    label_result = json_dict['annotations'][0]['result']
    label_list = []
    for label_item in label_result:
        start_time = label_item['value']['start']
        end_time = label_item['value']['end']
        label_value = label_item['value']['timeserieslabels'][0]
        label_list.append([start_time, end_time, label_value])
    print(label_list)
    label_csv_by_list(csv_file, label_list)


def down_sampling_data(csv_file, output_file, sampling_rate):
    """
    下采样一个csv文件，
    :param csv_file:
    :param output_file:
    :param sampling_rate: 采样率，以秒为单位
    :return:
    """
    df = pd.read_csv(csv_file)

    df['datetime'] = pd.to_datetime(df['datetime'])

    # 设置datetime为索引以便进行重采样
    df.set_index('datetime', inplace=True)

    # 定义重采样的规则
    rule = f'{sampling_rate}S'  # S表示秒

    # 对不同列采用不同的聚合方式
    resampled_data = df.resample(rule).agg({
        'timestamp': 'first',  # timestamp取第一个值
        'current': 'mean',  # current取平均值
        'voltage': 'mean',  # voltage取平均值
        'apparent power': 'mean',  # apparent power取平均值
        'active power': 'mean',  # active power取平均值
        'status': 'first',  # status取第一个值
        'on/off status': 'first'  # on/off status取第一个值
    })

    # 重置索引，使datetime重新成为一列
    resampled_data.reset_index(inplace=True)

    # 删除可能产生的空值行
    resampled_data.dropna(inplace=True)

    # 保存到新的CSV文件
    resampled_data.to_csv(output_file, index=False)

    print(f"已完成下采样，采样率: {sampling_rate}秒")
    print(f"原始数据点数: {len(df)}")
    print(f"下采样后数据点数: {len(resampled_data)}")
    print(f"结果已保存至: {output_file}")


def correct_csv_data(csv_file, output_file):
    """
    遍历CSV中的数据，根据条件修正数据值：
    1. 当on/off status == 0时：
       - current应小于0.4且大于0，否则修正
       - apparent power和active power应小于10且大于0，否则修正

    :param csv_file: 输入CSV文件路径
    :param output_file: 输出CSV文件路径
    :return: 无返回值
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file)

    # 确保列存在
    required_columns = ['current', 'apparent power', 'active power', 'on/off status']
    for col in required_columns:
        if col not in df.columns:
            raise ValueError(f"缺少必要的列: {col}")

    # 创建修正数据的副本
    df_corrected = df.copy()

    # 找到on/off status为0的行
    off_status_mask = df_corrected['on/off status'] == 0

    # 修正current值（当on/off status == 0时）
    current_values = df_corrected.loc[off_status_mask, 'current']
    # 对于小于等于0的值，设置为一个很小的正值
    # 对于大于等于0.4的值，使用取余操作将其限制在0.4以下
    corrected_current = np.where(current_values <= 0, 0.05,
                                 np.where(current_values >= 0.4, current_values % 0.4, current_values))
    df_corrected.loc[off_status_mask, 'current'] = corrected_current

    # 修正apparent power值（当on/off status == 0时）
    ap_values = df_corrected.loc[off_status_mask, 'apparent power']
    # 对于小于等于0的值，设置为一个很小的正值
    # 对于大于等于10的值，使用取余操作将其限制在10以下
    corrected_ap = np.where(ap_values <= 0, 0.1,
                            np.where(ap_values >= 10, ap_values % 10, ap_values))
    df_corrected.loc[off_status_mask, 'apparent power'] = corrected_ap

    # 修正active power值（当on/off status == 0时）
    active_power_values = df_corrected.loc[off_status_mask, 'active power']
    # 对于小于等于0的值，设置为一个很小的正值
    # 对于大于等于10的值，使用取余操作将其限制在10以下
    corrected_active_power = np.where(active_power_values <= 0, 0.1,
                                      np.where(active_power_values >= 10, active_power_values % 10,
                                               active_power_values))
    df_corrected.loc[off_status_mask, 'active power'] = corrected_active_power

    # 保存修正后的数据到新文件
    df_corrected.to_csv(output_file, index=False)

    # 打印统计信息
    total_rows = len(df)
    off_rows = off_status_mask.sum()

    print(f"数据修正完成:")
    print(f"总行数: {total_rows}")
    print(f"on/off status为0的行数: {off_rows}")
    print(f"修正后的数据已保存至: {output_file}")


def label_appliance_to_total_power(csv_file, output_file):
    """
    将标签和总功率提取出来，作为X和y，并且保存为新的.npz文件
    :param csv_file:
    :param output_file:
    :return:
    """


def get_all_files(folder, rule='*.csv'):
    """
    获取文件夹下所有满足规则匹配的文件，并返回文件路径列表

    :param folder: 文件夹路径 (str)
    :param rule: 文件匹配规则，支持通配符 (str), 默认为 '*.csv'
    :return: 匹配的文件路径列表 (list)
    """
    # 确保文件夹路径存在
    if not os.path.exists(folder):
        raise FileNotFoundError(f"文件夹不存在: {folder}")

    # 构建搜索模式
    search_pattern = os.path.join(folder, rule)

    # 获取所有匹配的文件
    files = glob.glob(search_pattern)

    # 按文件名排序（可选）
    files.sort()

    return files


def correct_datasets(file_dir='./process_dataset'):
    """
    将一个文件夹下的csv文件做矫正，注意，不会递归地寻找文件夹里的内容
    :param file_dir:
    :return:
    """
    files = get_all_files(file_dir, '*labeled.csv')  # 获取所有已经打了标签的csv文件

    for file_path in files:
        output_file = file_path.replace('labeled', 'correct')
        print(f'correct data file :{file_path}')
        correct_csv_data(file_path, output_file)


def resample_all_csv_files(file_dir='./process_dataset', order='*correct*', sampling_rate=1):
    """
    将一个文件夹下的csv文件做重采样，注意，不会递归地寻找文件夹里的内容
    :param order:
    :param sampling_rate:
    :param file_dir:
    :return:
    """
    files = get_all_files(file_dir, order)  # 获取所有已经打好标签的csv文件

    for file_path in files:
        output_file = file_path.replace('correct', f'resample_{sampling_rate}s')
        down_sampling_data(file_path, output_file, sampling_rate)
        print(f'resample data file :{file_path}')


def fill_lost_data(file_dir='./process_dataset', order='*resample*', sampling_rate=1):
    """
    将因为网络问题而丢失的时间帧都填充上0~10的随机浮点值
    :param sampling_rate:
    :param order:
    :param file_dir:
    :return:
    """
    print(f'按照{sampling_rate}s对丢失的数据进行插值补充')
    files = get_all_files(file_dir, order)  # 获取所有已经打好标签的csv文件

    for file_path in files:
        # df = pd.read_csv('.\\process_dataset\\Microwave\\processed_peek_data_20250819_resample_1s.csv')
        df = pd.read_csv(file_path)
        print(f'-------------fill lost data file :{file_path}, data size: {len(df)}--------------')

        directory = os.path.dirname(file_path)
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        output_filename = f"{name}_fill{ext}"
        output_filepath = os.path.join(directory, output_filename)

        df['datetime'] = pd.to_datetime(df['datetime'])
        # expected_interval = sampling_rate * 10000000     # 计算时间戳间隔

        new_rows = []
        # 获取第0号元素的datetime并且将时分秒清零
        # 这是防止一天刚开始的时候就出现了数据丢失
        current_time = pd.to_datetime(df.iloc[0]['datetime']).normalize()
        for i in range(len(df) - 1):
            next_time = df.iloc[i]['datetime']
            interval = (next_time - current_time).total_seconds()

            if interval > sampling_rate * 1.5:
                # 获取缺失的时间段
                missing_points = int(round(interval / sampling_rate)) - 1
                print(f'missing {missing_points} points between {current_time} and {next_time}')
                for j in range(1, missing_points + 1):
                    missing_time = current_time + timedelta(seconds=int(sampling_rate * j))

                    # 创建新行数据
                    new_row = {
                        'datetime': missing_time,
                        'timestamp': int(missing_time.timestamp() * 1_000_000),
                        'current': 0.0,
                        'voltage': 220.0,
                        'apparent power': np.random.uniform(0, 10),
                        'active power': np.random.uniform(0, 10),
                        'status': 0,
                        'on/off status': 0
                    }
                    new_rows.append(new_row)

            current_time = df.iloc[i]['datetime']

        # 如果有缺失的数据，将它们添加到原数据中
        if new_rows:
            # 创建包含新行的DataFrame
            new_df = pd.DataFrame(new_rows)
            # 合并原始数据和新数据
            result_df = pd.concat([df, new_df], ignore_index=True)
            # 按datetime重新排序
            result_df = result_df.sort_values('datetime').reset_index(drop=True)
        else:
            # 如果没有缺失数据，直接使用原始数据
            result_df = df.copy()

        print(f'数据插值填充完毕，总长度:{len(result_df)},准备存储到新文件中:{output_filepath}\n\n\n')
        result_df.to_csv(output_filepath, index=False)


def merge_all_csv_files(file_dir='./process_dataset', order='*resample*'):
    """
    扫描指定文件夹下所有匹配规则的CSV文件，并将其合并到同一个DataFrame中
    :param file_dir:
    :param order:
    :return:
    """
    files = get_all_files(file_dir, order)
    dataframes = []

    for i, csv_file in enumerate(files):
        print(f"正在读取文件 {i + 1}/{len(files)}: {csv_file}")
        file_df = pd.read_csv(csv_file)
        # 将datetime列转换为datetime类型并且设置为DataFrame的索引index
        file_df['datetime'] = pd.to_datetime(file_df['datetime'])
        file_df.set_index('datetime', inplace=True)
        dataframes.append(file_df)

    # 合并所有DataFrame
    print("开始合并所有数据，并且对数据按照时间进行排序...")
    total_df = pd.concat(dataframes, ignore_index=False)
    # 按照索引排序
    total_df = total_df.sort_index()
    # 检查是否有重复索引
    if total_df.index.duplicated().any():
        print("发现重复索引，正在进行处理...")
        # 可以选择删除重复项或聚合重复项
        total_df = total_df[~total_df.index.duplicated(keep='first')]  # 保留第一个

    print(f"合并完成！总数据行数: {len(total_df)}")
    print("\n合并后数据基本信息:")
    print(f"  - 总行数: {len(total_df)}")
    print(f"  - 列名: {list(total_df.columns)}")
    print(f"  - 索引名: {total_df.index.name}")
    print(f"  - 索引类型: {type(total_df.index)}")
    return total_df


def delete_data_by_time(file_path, start_time=None, end_time=None):
    """

    :param file_dir:
    :param order:
    :param sampling_rate:
    :return:
    """
    df = pd.read_csv(file_path)
    df['datetime'] = pd.to_datetime(df['datetime'])
    original_rows = len(df)

    # 根据时间范围筛选数据
    if start_time is not None and end_time is not None:
        # 同时有开始和结束时间
        mask = (df['datetime'] >= start_time) & (df['datetime'] <= end_time)
    elif start_time is not None:
        # 只有开始时间
        mask = df['datetime'] >= start_time
    else:  # end_time is not None
        # 只有结束时间
        mask = df['datetime'] <= end_time
    # 应用筛选条件
    df_filtered = df[mask]
    # 获取筛选后数据行数
    filtered_rows = len(df_filtered)
    deleted_rows = original_rows - filtered_rows
    # 保存回原文件
    df_filtered.to_csv(file_path, index=False)
    print(f"  原始数据行数: {original_rows}")
    print(f"  筛选后行数: {filtered_rows}")
    print(f"  删除行数: {deleted_rows}")
    print(f"已保存到文件: {df_filtered.head()}")


def save_dataframe_as_npz(df, filename_prefix='merged_data'):
    """
    将DataFrame保存为NPZ格式文件

    :param df: 要保存的DataFrame
    :param filename_prefix: 文件名前缀
    :return: 无返回值
    """
    # 准备保存为NPZ的数据字典
    data_dict = {
        'datetime': df.index.values  # 保存索引（datetime）
    }

    # 添加所有数值列到字典中
    for col in df.columns:
        data_dict[col.replace('/', '_')] = df[col].values  # 处理列名中的斜杠

    # 保存为NPZ文件
    npz_filename = f'{filename_prefix}.npz'
    np.savez(npz_filename, **data_dict)
    print(f"已保存为NPZ文件: {npz_filename}")

    print(f"总行数: {len(df)}")
    print(f"列名: {list(df.columns)}")


def process_negative_data(csv_file_path, output_file_path=None):
    """
    遍历CSV文件中的active power和apparent power列，将负数值修正为1
    Args:
        csv_file_path (str): CSV文件路径
        output_file_path (str, optional): 输出文件路径，如果为None则覆盖原文件
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 定义需要处理的列名（可根据实际文件调整大小写）
    power_columns = ['active power', 'apparent power']

    # 遍历指定的列
    for column in power_columns:
        if column in df.columns:
            # 将负数值修正为1
            negative_count = (df[column] < 0).sum()
            if negative_count > 0:
                print(f"列 '{column}' 中发现 {negative_count} 个负值，已修正为1")
                df.loc[df[column] < 0, column] = 1
        else:
            print(f"警告: 列 '{column}' 不存在于CSV文件中")

    # 保存处理后的数据
    output_path = output_file_path if output_file_path else csv_file_path
    df.to_csv(output_path, index=False)
    print(f"处理完成，数据已保存至: {output_path}")


def filter_data_by_status(csv_file_path, target_status, output_file_path=None):
    """
    根据指定状态过滤数据：
    - 保留status==target_status的数据
    - 对于status!=target_status的数据：
      * status==0: 保持不变
      * status!=0且status!=target_status: 将status和on/off status置为0，active power和apparent power置为0到10的随机值

    :param output_file_path: 输出文件路径，若为None则不保存文件 (str, optional)
    :param csv_file_path: 输入CSV文件路径 (str)
    :param target_status: 目标状态值 (int)
    :return: 处理后的DataFrame
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)

    # 确保datetime列格式正确
    if 'datetime' in df.columns:
        df['datetime'] = pd.to_datetime(df['datetime'])

    # 创建条件掩码
    keep_mask = df['status'] == target_status  # 需要保留的数据
    zero_mask = df['status'] == 0  # status为0的数据
    modify_mask = ~keep_mask & ~zero_mask  # 需要修改的数据(status既不等于目标值也不等于0)

    # 对需要修改的数据进行处理
    if modify_mask.any():
        # 将status和on/off status置为0
        df.loc[modify_mask, 'status'] = 0
        df.loc[modify_mask, 'on/off status'] = 0

        # 将active power和apparent power置为0到10的随机值
        df.loc[modify_mask, 'active power'] = 1
        df.loc[modify_mask, 'apparent power'] = 1

        print(f"已修改{modify_mask.sum()}行数据的状态和功率值")

    # 如果指定了输出路径，则保存为CSV和NPZ格式
    if output_file_path is not None:
        # 保存为CSV格式
        csv_output_path = output_file_path + '.csv'
        df.to_csv(csv_output_path, index=False)
        print(f"已保存CSV文件: {csv_output_path}")

        # 保存为NPZ格式
        # 准备保存为NPZ的数据字典
        data_dict = {}
        for col in df.columns:
            if col == 'datetime':
                data_dict['datetime'] = df['datetime'].values
            else:
                data_dict[col.replace('/', '_')] = df[col].values  # 处理列名中的斜杠

        # 生成NPZ文件名
        npz_output_path = output_file_path + '.npz'
        np.savez(npz_output_path, **data_dict)
        print(f"已保存NPZ文件: {npz_output_path}")

    return df


def clean_too_short_data(df: pd.DataFrame, active_periods: List[Tuple[pd.Timestamp, pd.Timestamp, int]],
                         min_duration=10):
    """
    根据active_periods中的数据，修改小于min_duration的数据：将on/off status和status都改为0，将active power,
    apparent power修改为1
    :param df:
    :param active_periods: 活动时间段列表，每个元素为(开始时间, 结束时间, 持续时间)的三元组
    :param min_duration: 最小持续时间(秒)，小于该时间的活动将被删除
    :return: None
    """
    # 读取CSV文件
    print(f"原始数据行数: {len(df)}")
    print(f"输入的活动段数量: {len(active_periods)}")

    # 筛选出持续时间小于min_duration的活动段（需要被修改的段）
    short_periods = [period for period in active_periods if period[2] < min_duration]
    print(f"需要修改的短活动段数量: {len(short_periods)}")

    # 对每个短活动段，修改对应时间段的数据
    for start_time, end_time, duration in short_periods:
        # 创建时间范围掩码
        mask = (df.index >= start_time) & (df.index <= end_time)
        modified_rows = mask.sum()

        # 打印活动段信息并等待用户确认
        print(f"\n发现短时间段:")
        print(f"  开始时间: {start_time}")
        print(f"  结束时间: {end_time}")
        print(f"  持续时间: {duration}秒")
        print(f"  匹配行数: {modified_rows}行")

        # 获取用户输入
        while True:
            user_input = input("是否要修改此时间段的数据？(y/n): ").lower().strip()
            if user_input in ['y', 'yes']:
                # 修改对应行的数据
                if 'on/off status' in df.columns:
                    df.loc[mask, 'on/off status'] = 0
                if 'status' in df.columns:
                    df.loc[mask, 'status'] = 0
                if 'active power' in df.columns:
                    df.loc[mask, 'active power'] = 1
                if 'apparent power' in df.columns:
                    df.loc[mask, 'apparent power'] = 1

                print(f"已修改时间段 {start_time} 到 {end_time} 的数据")
                break
            elif user_input in ['n', 'no']:
                print(f"跳过时间段 {start_time} 到 {end_time} 的修改")
                break
            else:
                print("请输入 'y' 或 'n'")
    return df


if __name__ == "__main__":
    # merge_csv_files(['dataset/AirCondition1/peek_data_20250707.csv', 'dataset/AirCondition1/peek_data_20250708.csv', 'dataset/AirCondition1/peek_data_20250709.csv'])
    # data_preprocess('dataset/Air-condition/peek_data_20250821.csv')
    # label_csv_by_json('dataset/Air-condition/processed_peek_data_20250803.csv', 'dataset/Air-condition/processed_peek_data_20250803.json')
    label_list = [
        ['2025-08-19 11:17:00', '2025-08-19 11:18:00', 'Microwave'],
        ['2025-08-19 12:07:30', '2025-08-19 12:13:40', 'Light Wave'],
        ['2025-08-19 12:18:00', '2025-08-19 12:19:00', 'Microwave'],
        ['2025-08-19 16:14:00', '2025-08-19 16:20:00', 'Thaw'],
        ['2025-08-19 18:18:00', '2025-08-19 18:19:20', 'Sterilize'],
    ]
    # 记得该文件日期
    # 给文件打标注
    # label_csv_by_list('./dataset/Microwave/processed_peek_data_20250819.csv', label_list)
    # 然后做矫正
    # correct_datasets('./process_dataset/Microwave/')
    # 以1s为频率做重采样
    # resample_all_csv_files('./process_dataset/Microwave')
    # 数据修整
    # delete_data_by_time('process_dataset/Microwave/processed_peek_data_20250804_resample_1s.csv', '2025-08-04 00:00:00')

    # # # 合并所有文件
    # df = merge_all_csv_files('./process_dataset/Air-condition')
    # # 保存为CSV文件
    # df.to_csv('./process_dataset/Air-condition/Air-condition_missing.csv', index=True)
    #
    # # # 填补丢失的数据
    # fill_lost_data('./process_dataset/Air-condition/', 'Air-condition*', 1)

    # process_negative_data('./process_dataset/WashMachine/WashMachine.csv')

    # # 保存为NPZ文件
    # df = pd.read_csv('./process_dataset/dataset/Air_condition.csv')
    # save_dataframe_as_npz(df, './process_dataset/dataset/Air-condition')

    # filter_data_by_status('./process_dataset/dataset/WashMachine.csv', 1, './process_dataset/dataset/WashMachine_Wash')

    file_path = r'./process_dataset/Air-condition/Air_condition.csv'
    df = pd.read_csv(file_path, index_col=0)
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    if 'status' in df.columns:
        df['status'] = df['status'].round()
    if 'on/off status' in df.columns:
        df['on/off status'] = df['on/off status'].round()
    active_count, total_duration, removed_periods = calculate_status_active_times(df, 3)
    df = clean_too_short_data(df, removed_periods, min_duration=10)

    print("处理完毕，打印处理后的数据")
    _, _, filter_periods = calculate_status_active_times(df, 3)
    for period in filter_periods:
        print(f"时间段: {period}")

    # # 保存处理后的数据，覆盖原始文件
    # df.to_csv(file_path, index=True)
    # print(f"已保存处理后的数据到: {file_path}")


