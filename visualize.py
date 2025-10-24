import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

visualize_config = {
    'max_y_value': 800.0,
    'title': 'Data Visualization',
    'width': 800,
    'height': 600,
    'dpi': 100
}

def general_data_print_function(selected_data, title='Data Visualization', max_y_value=800.0):
    """
    通用数据可视化函数

    :param selected_data: 需要可视化的数据
    :param title: 图表标题
    :param max_y_value: Y轴最大值
    :return: None
    """
    # 可视化
    # 将像素转换为英寸（matplotlib使用英寸作为单位）
    fig_width = visualize_config['width'] / visualize_config['dpi']
    fig_height = visualize_config['height'] / visualize_config['dpi']

    # 可视化，设置指定尺寸
    plt.figure(figsize=(fig_width, fig_height), dpi=visualize_config['dpi'])

    # 如果数据有多列，可视化所有列
    for column in selected_data.columns:
        plt.plot(selected_data.index, selected_data[column], label=column)

    plt.xlabel('Timestamp')
    plt.ylabel('Active Power')
    plt.title(title)
    plt.ylim(top=max_y_value, bottom=0)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print(f"可视化数据范围: 从索引 {selected_data.index[0]} 到 {selected_data.index[-1]}")
    print(f"实际数据点数量: {len(selected_data)}")


def visualize_first_active_data(file_path: str, threshold: float = 10.0, length: int = 1000,
                      max_y_value: float = 800, title='Data Visualization'):
    """
    设置第一列为index，并且找到第二列大于threshold的值，
    然后往后读取length个值，并且使用可视化工具对其进行可视化

    :param file_path: 数据文件路径
    :param threshold: 阈值，用于筛选第二列的值
    :param length: 要读取和可视化的数据点数量
    :param max_y_value: Y轴最大值
    :param title: 图表标题
    :return: tuple (df, start_time) - 数据框和开始时间
    """
    # 读取数据，第一列作为索引
    df = pd.read_csv(file_path, index_col=0, sep=' ', header=None)
    df.index = pd.to_datetime(df.index, unit='s')

    # 获取第二列数据
    second_column = df.iloc[:, 0]  # 第一列是索引，所以第二列是第0列

    # 找到第二列大于threshold的值的索引
    above_threshold_indices = second_column[second_column > threshold].index

    start_time = None
    if len(above_threshold_indices) > 0:
        # 获取第一个大于threshold的值的索引位置
        start_index = df.index.get_loc(above_threshold_indices[0])
        start_time = df.index[start_index]

        # 计算结束位置
        end_index = min(start_index + length, len(df))

        # 提取数据
        selected_data = df.iloc[start_index:end_index]

        # 调用通用可视化函数
        visualization_title = f'{title} (Threshold: {threshold}, Length: {len(selected_data)})'
        general_data_print_function(selected_data, visualization_title, max_y_value)

        print(f"开始时间: {start_time}")
    else:
        print(f"没有找到第二列大于阈值 {threshold} 的数据")

    return df, start_time  # 返回数据框和开始时间


def visualize_data_by_time(file_path: str, start_time: str, end_time: str, max_y_value: float = 800,
                           title='Data Visualization'):
    """
    根据指定的时间范围可视化数据

    :param file_path: 数据文件路径
    :param start_time: 开始时间
    :param end_time: 结束时间
    :param max_y_value: Y轴最大值
    :param title: 图表标题
    :return: None
    """
    df = pd.read_csv(file_path, index_col=0, sep=' ', header=None)
    df.index = pd.to_datetime(df.index, unit='s')

    # 将输入的时间字符串转换为datetime对象
    start_datetime = pd.to_datetime(start_time)
    end_datetime = pd.to_datetime(end_time)

    # 根据时间范围筛选数据
    selected_data = df[(df.index >= start_datetime) & (df.index <= end_datetime)]

    if len(selected_data) > 0:
        # 通过调用general_print_function实现可视化
        visualization_title = f'{title} (Time Range: {start_time} to {end_time})'
        general_data_print_function(selected_data, visualization_title, max_y_value)
    else:
        print(f"在指定时间范围 {start_time} 到 {end_time} 内没有找到数据")

    return selected_data

if __name__ == '__main__':
   file_path = './process_dataset/experiment_dataset/BERT4NILM/aircondition/aircondition_freeze_low.dat'
   df, start_time = visualize_first_active_data(file_path, threshold=10.0, length=500, max_y_value=300, title='aircondition freeze low')
