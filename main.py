from typing import Tuple, Any

import numpy as np
from nilmtk import DataSet, TimeFrame, Appliance
import matplotlib.pyplot as plt
import struct, os

import pandas as pd
from pandas import DataFrame
import utils

params_appliance = {
    'light': {
        'appliance_name': 'light',
        'building': {
            1: {
                'instances': [1]
            }
        },
        'start_time': '2013-06-05 00:00:00',
        'end_time': '2013-06-07 23:59:59'
    },
}


def print_dale_info(dataset_path: str):
    uk_dale = DataSet(dataset_path)
    print("-----------UKDALE metadata-------------")
    print(uk_dale.metadata)
    print("-----------UKDALE building-------------")
    print(uk_dale.buildings)
    print('-----------House1 metadata--------------')
    print(uk_dale.buildings[1].metadata)
    print('-----------House1 elections data--------------')
    elec = uk_dale.buildings[1].elec
    print(elec)


def print_all_appliances(dataset_path: str):
    uk_dale = DataSet(dataset_path)
    for i in range(1, 5):
        print(f'-----------House{i} appliances--------------')
        building = uk_dale.buildings[i]
        for app in building.elec.appliances:
            print(app.metadata['type'] + ' ' + str(app.metadata['instance']))  # +str(app.instance)+'\n'
        # print(uk_dale.buildings[i].elec.appliances)


def get_appliance_power_by_name(dataset_path: str, elec_name: str, building_id: int = 1, instance: int = None):
    """
    根据名字获取电器的active power数据，并且返回对应的DataFrame和电器的开始以及结束时间
    :param instance:
    :param building_id:
    :param dataset_path:
    :param elec_name:
    :return:
    """
    # 加载数据
    ukdale = DataSet(dataset_path)
    elec = ukdale.buildings[building_id].elec

    # 根据是否指定实例获取电器数据
    if instance is not None:
        # 获取指定实例的电器数据
        elec_meter = elec[elec_name, instance]
        elec_timeframe = elec_meter.metadata['timeframe']
    else:
        # 获取所有同名电器的数据
        elec_meter = elec[elec_name]
        elec_timeframe = elec_meter.metadata['timeframe']

    start_time = elec_timeframe['start']
    end_time = elec_timeframe['end']
    df = next(elec_meter.load(physical_quantity='power', ac_type='active'))

    instance_info = f" instance {instance}" if instance is not None else " all instances"
    print(f'get appliance data {elec_name}{instance_info}, start time: {start_time}, end time: {end_time}, '
          f'data shortcut\n {df.head()}')

    return df, start_time, end_time


def get_mains_power(dataset_path: str, building_id: int = 1):
    """
    提取指定建筑的主干线功率数据

    Parameters:
    dataset_path: 数据集路径
    building_id: 建筑ID，默认为1
    """
    # 加载数据集
    ukdale = DataSet(dataset_path)

    # 获取指定建筑
    building = ukdale.buildings[building_id]

    # 获取电力数据
    elec = building.elec

    # 获取主干线功率数据
    mains = elec.mains()  # 获取主干线
    mains_timeframe = mains.metadata['timeframe']
    # 加载主干线功率数据
    mains_power = next(mains.load(physical_quantity='power', ac_type='active'))
    print(f"get mains power data, start time: {mains_timeframe['start']}, end time: {mains_timeframe['end']}"
          f" data shortcut \n {mains_power.head()}")
    return mains_power, mains_timeframe['start'], mains_timeframe['end']


def merge_mains_and_appliance_power(mains_df, app_df, threshold=10):
    """
    将主干线功率和用电器功率和并在一个DataFrame中，并且根据用电器功率是否大于threshold标注on/off status列
    返回一个header为[datetime mains_power  appliance_power  on/off status]的DataFrame
    :param mains_df:
    :param app_df:
    :return:
    """
    # 时间对齐：将两个DataFrame的时间都四舍五入到秒
    mains_df.index = mains_df.index.tz_localize(None, ambiguous='infer').floor('S').tz_localize('UTC')
    app_df.index = app_df.index.tz_localize(None, ambiguous='infer').floor('S').tz_localize('UTC')

    # 检查并处理重复索引
    print("处理数据中的重复时间戳，保留最大值...")
    mains_df = mains_df.groupby(mains_df.index).max()
    app_df = app_df.groupby(app_df.index).max()

    # 确定共同的时间范围
    mains_start_time = mains_df.index.min()
    mains_end_time = mains_df.index.max()
    app_start_time = app_df.index.min()
    app_end_time = app_df.index.max()
    # 开始时间取较晚的那个
    common_start_time = max(pd.to_datetime(mains_start_time), pd.to_datetime(app_start_time))
    # 结束时间取较早的那个
    common_end_time = min(pd.to_datetime(mains_end_time), pd.to_datetime(app_end_time))

    print(f"共同时间范围: {common_start_time} 到 {common_end_time}")

    # 裁剪两个DataFrame到相同的时间范围
    mains_df = mains_df[(mains_df.index >= common_start_time) & (mains_df.index <= common_end_time)]
    app_df = app_df[(app_df.index >= common_start_time) & (app_df.index <= common_end_time)]

    print(f"裁剪后主干线数据大小: {len(mains_df)}")
    print(f"裁剪后电器数据大小: {len(app_df)}")

    # 确保两个DataFrame有相同的时间索引（可选）
    # 使用相同的时间索引进行对齐
    print("对齐数据时间戳...这一步可能会很慢")
    common_index = mains_df.index.intersection(app_df.index)
    mains_df_aligned = mains_df.loc[common_index]
    app_df_aligned = app_df.loc[common_index]

    print(f"mains_df 数据点数: {len(mains_df)}")
    print(f"app_df 数据点数: {len(app_df)}")
    print(f"common_index 数据点数: {len(common_index)}")

    # 重命名列以避免冲突并明确含义
    mains_df_aligned.columns = ['mains_power']
    app_df_aligned.columns = ['appliance_power']

    print("mains_df_aligned 列名:", mains_df_aligned.columns.tolist())
    print("app_df_aligned 列名:", app_df_aligned.columns.tolist(), '\n\n\n')

    # 合并数据
    combined_df = pd.concat([mains_df_aligned, app_df_aligned], axis=1)

    print(f"----------------合并完成------------------\n合并后的DataFrame:{combined_df.head()}")
    print("combined_df 列名:", combined_df.columns.tolist())

    # 添加新的on/off列，将功率>10的标记为on状态。
    combined_df['on/off status'] = np.where(combined_df['appliance_power'] > threshold, 1, 0)

    nums_of_on_status = np.sum(combined_df['on/off status'] == 1)
    nums_of_off_status = np.sum(combined_df['on/off status'] == 0)

    print("最终合并后的DataFrame:")
    print(combined_df.head(10))
    print(f"开关状态统计:")
    print(f"开启状态共计次数:{nums_of_on_status}, 关闭状态共计次数:{nums_of_off_status}")
    return combined_df


def align_data(source: DataFrame, target: DataFrame, datetime=""):
    """
    source首先会找到target中离datetime最近的时间，然后将自己的第一个时间戳与之对齐，然后
    使用对齐函数，只留下和target和source中的共同时间
    :param datetime:
    :param source:
    :param target:
    :return:
    """
    # 创建副本以避免修改原始数据
    source_copy = source.copy()
    target_copy = target.copy()

    # 将source和target的index都四舍五入到秒
    source_copy.index = source_copy.index.floor('S')
    target_copy.index = target_copy.index.floor('S')

    # 检查并处理重复索引
    source_copy = source_copy[~source_copy.index.duplicated(keep='first')]
    target_copy = target_copy[~target_copy.index.duplicated(keep='first')]

    # 如果提供了datetime，则找到target中离datetime最近的时间点
    if datetime != "":
        # 将datetime转换为pandas时间戳
        target_datetime = pd.to_datetime(datetime)
        # 找到target中离指定时间最近的时间戳
        try:
            nearest_time_in_target = target_copy.index.get_indexer([target_datetime], method='nearest')[0]
            target_alignment_time = target_copy.index[nearest_time_in_target]
            print(f"找到target中离{datetime}最近的时间点: {target_alignment_time}")
        except Exception as e:
            print(f"找不到最近时间点: {e}")
            target_alignment_time = target_copy.index[0]
    else:
        # 如果没有提供datetime，使用target的第一个时间戳作为对齐时间
        target_alignment_time = target_copy.index[0]
        print(f"未提供datetime，使用target的第一个时间点作为对齐时间: {target_alignment_time}")

    # 计算source需要的时间偏移量
    source_alignment_time = source_copy.index[0]
    time_offset = target_alignment_time - source_alignment_time
    print(f"计算得到的时间偏移量: {time_offset}")

    # 将source的时间戳进行对齐（加上时间偏移量）
    source_aligned = source_copy.copy()
    source_aligned.index = source_aligned.index + time_offset

    # 找到两个DataFrame共同的时间戳（交集操作）
    common_index = source_aligned.index.intersection(target_copy.index)
    print(f"对齐后的共同时间点数量: {len(common_index)}")

    # 根据共同时间戳筛选数据
    source_result = source_aligned.loc[common_index]
    target_result = target_copy.loc[common_index]

    # 打印部分对齐结果
    print("对齐后的部分结果 (source):")
    print(source_result.head())
    print("对齐后的部分结果 (target):")
    print(target_result.head())

    return source_result, target_result


def load_dat_file(dat_file_path: str, header_list=None) -> pd.DataFrame:
    """
    读取UK-DALE数据集的.dat文件为DataFrame

    参数:
    dat_file_path: .dat文件的路径

    返回:
    包含时间和功率数据的DataFrame，格式为[YYYY-MM-DD HH:MM:SS, POWER]
    """
    if header_list is None:
        header_list = ['timestamp', 'power']
    records = pd.read_csv(dat_file_path, sep=' ', header=None, names=header_list)

    # 对timestamp列进行向下取整处理
    records['timestamp'] = np.floor(records['timestamp']).astype(int)

    # 将时间戳转换为datetime格式
    records['datetime'] = pd.to_datetime(records['timestamp'], unit='s')
    records.set_index('datetime', inplace=True)
    # 删除timestamp列
    records = records.drop('timestamp', axis=1)
    return records


def remove_appliance_on_aggregate(mains_df: pd.DataFrame, app_df: pd.DataFrame) -> pd.DataFrame:
    """
    将指定用电器的功率从总线上移除
    """
    df = merge_mains_and_appliance_power(mains_df, app_df)
    # df, _, _ = get_appliance_power_by_name('./ukdale.h5', 'light')
    print(df)

    # 将df的第一列减去第二列的数据
    # 如果第二列数据是1.0的话等效为0.0
    first_col = df.columns[0]  # mains_power
    second_col = df.columns[1]  # appliance_power

    # 创建第二列的副本，将1.0替换为0.0
    appliance_power_data = df[second_col].copy()  # 获取第二列数据的副本
    appliance_power_data[appliance_power_data == 1.0] = 0.0  # 将1.0替换为0.0

    # 创建新列来存储第一列减去第二列的值
    new_column_name = 'mains_minus_appliance'
    df[new_column_name] = df[first_col] - appliance_power_data

    # 打印处理后的统计信息
    print("处理后数据统计信息:")
    print(f"数据长度: {len(df)}")
    print(f"最大值: {df[new_column_name].max()}")
    print(f"最小值: {df[new_column_name].min()}")
    print(f"平均值: {df[new_column_name].mean()}")
    print(f"标准差: {df[new_column_name].std()}")

    # 提取新列中的负值
    negative_values = df[df[new_column_name] < 0]

    # 打印负值统计信息
    print("负值统计信息:")
    print(f"负值数量: {len(negative_values)}")
    print(f"负值占比: {len(negative_values) / len(df) * 100:.2f}%")
    print("负值预览:")
    # 创建一个临时的DataFrame，包含所有需要的列
    temp_df = df[[first_col, new_column_name]].copy()
    temp_df[second_col] = appliance_power_data

    # 从临时DataFrame中提取负值
    negative_with_all_cols = temp_df[temp_df[new_column_name] < 0]

    # 打印 mains_power, mains_minus_appliance 和 appliance_power 三列
    print(negative_with_all_cols[[first_col, new_column_name, second_col]].head(10))

    # 如果需要，也可以获取负值的统计信息
    if len(negative_values) > 0:
        print("负值的统计信息:")
        print(f"负值的最大值: {negative_values[new_column_name].max()}")
        print(f"负值的最小值: {negative_values[new_column_name].min()}")
        print(f"负值的平均值: {negative_values[new_column_name].mean()}")

    print("处理后的数据预览:")

    # 只保留第一列和新列
    df = df[[first_col, new_column_name]]

    # 只保留第一列
    df = df[[first_col]]

    print(df)
    return df


def add_appliance_to_aggregate(mains_df: pd.DataFrame, app_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """

    :param mains_df:
    :param app_df:
    :return:合并后的merge_df，进行偏置处理后的app_df
    """
    mains_df.index = pd.to_datetime(mains_df.index, format='%Y-%m-%d %H:%M:%S')
    app_df.index = pd.to_datetime(app_df.index, format='%Y-%m-%d %H:%M:%S')
    # 筛选指定的列（根据实际列名调整）,列名包括
    # [datetime,timestamp,current,voltage,apparent power,active power,status,on/off status]
    app_df = app_df[['active power', 'on/off status']]
    app_df, mains_df = align_data(app_df, mains_df)
    print("合并前的app_df:")
    print(app_df.head())
    print("合并前的mains_df:")
    print(mains_df.head())

    # 做电器替换：比如要替换掉其中的Microwave，那么应该取得该电器的电器数据，在总干线数据上减去该电器的功率
    # 然后再将自己的Microwave的功率加到总干线上，但是替换会出现负值，所以暂缓一下
    # removed_file_path = "E:\\datasets\\NILM\\uk_dale\\house_1\\channel_1.dat"
    # removed_app_df = load_dat_file(removed_file_path)

    # 找到两个DataFrame共同的时间戳（交集操作）
    common_index = mains_df.index.intersection(app_df.index)

    # 根据共同时间戳筛选app_df数据，创建新的app_df
    app_df_common = app_df.loc[common_index]
    app_df_common['active power'] = app_df_common['active power'].round().astype(int)

    # 找到两个DataFrame共同的时间戳（交集操作）
    merged_df = pd.concat([mains_df, app_df], axis=1)

    print(
        f"合并完成，appliance形状：{app_df.shape}，aggregate形状：{mains_df.shape}，合并后的DataFrame形状: {merged_df.shape}")
    print("合并后的DataFrame列名:", merged_df.columns.tolist())
    print("合并后的DataFrame预览:")
    print(merged_df.head(10))

    # 将active power列四舍五入为整数
    active_power_col = 'active power'  # active power列名
    power_col = 'power'  # power列名

    # 确保active power列存在
    if active_power_col in merged_df.columns:
        # 四舍五入active power列为整数
        merged_df[active_power_col] = merged_df[active_power_col].round().astype(int)
        print(f"已将'{active_power_col}'列四舍五入为整数")

        # 用power列减去active power列，覆盖power列
        if power_col in merged_df.columns:
            merged_df[power_col] = merged_df[power_col] + merged_df[active_power_col]
            print(f"已用'{power_col}'列加上'{active_power_col}'列，结果存储在'{power_col}'列中")
        else:
            print(f"警告：列'{power_col}'不存在于DataFrame中")
    else:
        print(f"警告：列'{active_power_col}'不存在于DataFrame中")

    print('-----------------------合成后merge_df打印---------------------\n')
    print(merged_df.head())
    return merged_df, app_df_common


def save_with_timestamp(df, filepath, value_column_name):
    """
    保存DataFrame为两列格式：时间戳和值
    将索引转换为秒级时间戳，然后保存为两列格式的CSV文件
    Parameters:
    df: DataFrame，需要保存的数据
    filepath: 保存路径
    value_column_name: 值列的名称
    """
    # 创建包含时间戳的新DataFrame
    result_df = pd.DataFrame({
        'timestamp': df.index.astype('int64') // 10 ** 9,  # 转换为秒级时间戳
        value_column_name: df.iloc[:, 0].values  # 取第一列的值
    })

    # 保存为CSV格式
    result_df.to_csv(filepath, sep=' ', header=False, index=False)


def query_by_timestamp_value(target_df: pd.DataFrame, timestamp_value: int, window_seconds: int = 10):
    """
    根据时间戳数值查询数据

    参数:
    target_df: DataFrame，索引为datetime
    timestamp_value: 时间戳数值（秒）
    window_seconds: 查询窗口（秒）

    返回:
    时间窗口内的数据
    """
    # 将时间戳数值转换为datetime
    target_time = pd.to_datetime(timestamp_value, unit='s')

    # 查询前后数据
    start_time = target_time - pd.Timedelta(seconds=window_seconds)
    end_time = target_time + pd.Timedelta(seconds=window_seconds)

    window_data = target_df[start_time:end_time]

    return window_data, target_time


def find_intersection_data(source: pd.DataFrame, target: pd.DataFrame):
    """
    找到source和target的index相同的前1000个数据，并且将他们合并到同一个DataFrame中，
    提供交互式查看功能

    :param source: 源DataFrame
    :param target: 目标DataFrame
    :return: 合并后的DataFrame
    """
    # 确保索引是datetime类型
    if not isinstance(source.index, pd.DatetimeIndex):
        source.index = pd.to_datetime(source.index, format='%Y-%m-%d %H:%M:%S')

    if not isinstance(target.index, pd.DatetimeIndex):
        target.index = pd.to_datetime(target.index, format='%Y-%m-%d %H:%M:%S')

    print(f"打印source和target状态，准备查找交集")
    print("Source头部:")
    print(source.head())
    print("Target头部:")
    print(target.head())

    print(f"Source索引类型: {type(source.index)}, 数量: {len(source.index)}")
    print(f"Target索引类型: {type(target.index)}, 数量: {len(target.index)}")

    # 找到两个DataFrame中index的交集
    common_index = source.index.intersection(target.index)
    print(f"共同索引数量: {len(common_index)}")

    # 取前1000个共同索引
    common_index_limited = common_index[:1000]
    print(f"使用的共同索引数量: {len(common_index_limited)}")

    # 根据共同索引筛选数据
    source_filtered = source.loc[common_index_limited]
    target_filtered = target.loc[common_index_limited]

    # 去除重复索引，保留第一个出现的值
    source_filtered = source_filtered[~source_filtered.index.duplicated(keep='first')]
    target_filtered = target_filtered[~target_filtered.index.duplicated(keep='first')]

    print(f"筛选后的source形状: {source_filtered.shape}")
    print(f"筛选后的target形状: {target_filtered.shape}")

    # 给列名添加前缀以区分来源
    source_filtered = source_filtered.add_prefix('source_')
    target_filtered = target_filtered.add_prefix('target_')

    # 在合并前确保两个DataFrame有完全相同的索引
    # 重新索引以确保一致性
    common_index_final = source_filtered.index.intersection(target_filtered.index)
    source_filtered = source_filtered.loc[common_index_final]
    target_filtered = target_filtered.loc[common_index_final]

    # 合并两个DataFrame
    merged_df = pd.concat([source_filtered, target_filtered], axis=1)

    print(f"找到 {len(common_index)} 个共同索引，使用前1000个")
    print(f"合并后的DataFrame形状: {merged_df.shape}")

    # 交互式查看数据
    start_idx = 0
    while True:
        user_input = input("输入'n'查看接下来的50条数据，输入'q'退出: ").strip().lower()

        if user_input == 'n':
            end_idx = min(start_idx + 50, len(merged_df))
            # 使用to_string方法显示所有列
            print(f"\n显示第 {start_idx} 到 {end_idx - 1} 行数据:")
            print(merged_df.iloc[start_idx:end_idx].to_string())
            start_idx = end_idx

            # 如果已经显示完所有数据，则退出
            if start_idx >= len(merged_df):
                print("已显示所有数据")
                break

        elif user_input == 'q':
            print("退出查看")
            break
        else:
            print("无效输入，请输入'n'或'q'")

    return merged_df


if __name__ == '__main__':
    # mains_df, mains_start_time, mains_end_time = get_mains_power('ukdale.h5', building_id=1)
    # app_df, app_start_time, app_end_time = get_appliance_power_by_name('./ukdale.h5', 'microwave')
    # df = merge_mains_and_appliance_power(mains_df, app_df)
    # df, _, _ = get_appliance_power_by_name('./ukdale.h5', 'light')
    # print(df)
    timestamp = 1363547564
    # 总干线数据
    dat_file_path = "E:\\datasets\\NILM\\uk_dale\\house_1\\mains.dat"
    target_df = load_dat_file(dat_file_path, ['timestamp', 'active power', 'power', 'reactive power'])
    target_df = target_df.drop('active power', axis=1)
    target_df = target_df.drop('reactive power', axis=1)
    print(target_df.head())

    # 电器数据
    app_file_path = r'./process_dataset/dataset/Microwave_Sterilize.csv'
    source_df = pd.read_csv(app_file_path, index_col=0)
    source_df.index = pd.to_datetime(source_df.index, format='%Y-%m-%d %H:%M:%S')

    # 添加电器数据到总线数据中的代码
    df, app_df = add_appliance_to_aggregate(target_df, source_df)
    df = df.iloc[:, [0]]
    app_df_power = app_df.iloc[:, [0]]
    app_df_status = app_df.iloc[:, [1]]
    print(f'-----------------数据裁剪完成，预览：------------------')
    print(df.head())
    print(app_df_power.head())
    print(app_df_status.head())

    # # 分别保存为3个独立的.npz文件
    # np.savez('mains_power.npz', data=df.values)
    # np.savez('appliance_power.npz', data=app_df_power.values)
    # np.savez('appliance_status.npz', data=app_df_status.values)

    # 使用函数保存数据
    save_with_timestamp(df,
                        'process_dataset/experiment_dataset/BERT4NILM/microwave/test/mains_sterilize.dat', 'power')
    save_with_timestamp(app_df_power,
                        'process_dataset/experiment_dataset/BERT4NILM/microwave/test/microwave_sterilize.dat', 'active power')
    save_with_timestamp(app_df_status,
                        'process_dataset/experiment_dataset/BERT4NILM/microwave/test/microwave_sterilize_s.dat', 'on/off status')

    # 查找交集的代码，选择性开启
    # app_file_path = r'E:/datasets/NILM/uk_dale/house_1/channel_1_org.dat'
    # source_df = load_dat_file(app_file_path, ['timestamp', 'app power'])
    # print(source_df.head())
    # find_intersection_data(source_df, target_df)