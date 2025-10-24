import os, pandas as pd

from nilmtk import DataSet, TimeFrame, MeterGroup, ElecMeter, Appliance
import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
from sklearn.manifold import TSNE


def get_apps_each_buildings(app_list: list, start_time: str, end_time: str):
    """

    :param app_list:  需要提取达到电器列表
    :param start_time: 开始时间
    :param end_time: 结束时间
    :return:
    """
    ukdale = DataSet('../ukdale.h5')
    building_list = []
    app_list = []
    # 遍历每一个建筑
    for building_id, building in ukdale.buildings.items():
        building_list.append(building)
        print(f'-----------House{building_id} appliances--------------')
        for app in building.elec.appliances:
            if app.metadata['type'] in app_list:
                print(f'Try to get appliance {app}')
                # 查询该电器在指定时间段的数据
                df = app.get_dataframe(columns=['power', 'state'])

                # 筛选指定时间范围的数据
                if start_time is not None and end_time is not None:
                    df = df[start_time:end_time]
                elif start_time is not None:
                    df = df[start_time:]
                elif end_time is not None:
                    df = df[:end_time]
                app_list.append(app)
                print(df.head())


def get_elec_meters_by_type(dataset_path: str, target_appliance_type: str):
    """
    根据指定的电器类型，从数据集中获取对应的电表对象列表

    :param dataset_path: 数据集路径，字符串类型
    :param target_appliance_type: 目标电器类型，字符串类型
    :return: 返回匹配目标电器类型的电表对象列表
    """
    print(f'start get appliance :{target_appliance_type}')
    target_meters = []
    dataset = DataSet(dataset_path)
    for building_id, building in dataset.buildings.items():
        print(f'-----------House{building_id} appliances--------------')
        elec = building.elec  # MeterGroup 对象

        for meter in elec.meters:
            # 检查该 ElecMeter 是否有关联的 Appliance，并且 type 是 'light'，
            # 而且要求这个电表只监测一个电器，监测多个电器的我们用不上
            if meter.appliances and len(meter.appliances) == 1:
                appliance = meter.appliances[0]
                if appliance.metadata['type'] == target_appliance_type:
                    target_meters.append(meter)
                    print(meter)
                    break  # 只要有一个符合条件的 appliance 即可加入列表

    print(f'get all {target_appliance_type} appliances, total: {len(target_meters)}')
    return target_meters


def record_meters_info_to_csv(csv_file: str = 'meter_infor.csv'):
    """
    将数据集中的 meter 信息写入 CSV 文件，用于速览meter信息
    """
    ukdale = DataSet('../ukdale.h5')
    with open(csv_file, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        # 写入表头
        writer.writerow(['Building ID', 'Meter Identifier', 'Appliances'])
        for building_id, building in ukdale.buildings.items():
            print(f'-----------House{building_id} appliances--------------')
            elec = building.elec  # MeterGroup 对象

            for meter in elec.meters:
                # 将 identifier 转换为字符串
                identifier_str = f'ElecMeter({meter.identifier.instance})'

                # 将 appliances 转换为字符串（如果存在）
                if meter.appliances:
                    appliances_str = ', '.join(str(app) for app in meter.appliances)
                else:
                    appliances_str = 'None'

                # 写入 CSV 文件
                writer.writerow([building_id, identifier_str, appliances_str])
                print(f"写入: {identifier_str} -> {appliances_str}")

        print(f"数据已成功写入 {csv_file}")


def extract_time_series_features(df, window_size=100):
    """
    从一个时间序列 DataFrame 中提取多个统计特征

    :param df: 单个电器的 DataFrame，包含 'power' 列
    :param window_size: 滑动窗口大小（可选）
    :return: 包含提取特征的一维数组
    """
    series = df['power'].values

    # 基础统计特征
    features = [
        np.mean(series),
        np.std(series),
        np.max(series),
        np.min(series),
        np.median(series),
        np.percentile(series, 25),
        np.percentile(series, 75),
        skew(series),
        kurtosis(series),
        np.sum(series ** 2),  # 能量
        np.sqrt(np.mean(series ** 2)),  # RMS
        # np.mean(np.abs(np.diff(series)))  # 变化率
    ]

    # 滑动窗口统计特征（可选）
    if len(series) >= window_size:
        windows = [series[i:i + window_size] for i in range(0, len(series) - window_size + 1, window_size)]
        window_means = [np.mean(w) for w in windows]
        window_stds = [np.std(w) for w in windows]

        # 添加前几个窗口的 mean 和 std
        features += window_means[:5] + window_stds[:5]
    print(features)
    return np.array(features)


def prepare_data_for_pca(data_list, window_size=100):
    """
    将多个 DataFrame 转换为统一特征维度的 numpy 数组用于 PCA

    :param data_list: list of pd.DataFrame
    :param window_size: 滑动窗口大小
    :return: np.ndarray shape=(n_samples, n_features)
    """
    processed = []
    feature_dim = None

    for df in data_list:
        if len(df) < 1000:  # 过滤掉太短的数据
            print(f"警告：数据长度不足 1000，忽略该样本。")
            continue

        features = extract_time_series_features(df, window_size=window_size)

        if feature_dim is None:
            feature_dim = len(features)
            print(f"提取的特征维度为 {feature_dim}，后续样本将保持一致。")
        elif len(features) != feature_dim:
            print("警告：特征维度不一致，跳过该样本。")
            continue

        processed.append(features)

    print(f"共处理了 {len(processed)} 个有效样本，每个样本特征维度为 {feature_dim}。")
    return np.array(processed)


def save_data_to_csv(meters, output_dir=''):
    for meter in meters:
        appliance_name = f"{meter.appliances[0].metadata['type']}_{meter.appliances[0].metadata['instance']}"
        filename = f'{appliance_name}_building_{meter.identifier.building}.csv'
        file_path = os.path.join(output_dir, filename)
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['timestamp', 'power'])
        df = next(meter.load(physical_quantity='power', ac_type='active'))
        df.to_csv(file_path, index=True)
        print(f"已保存 {file_path}")


def data_visualization(file_path):
    # 定义一个统一的加载函数来提取2013年的数据为DataFrame
    def load_data_for_meter(meter, year='2014'):
        timeframe = TimeFrame(f'{year}-01-01', f'{year}-12-31')
        df = next(meter.load(physical_quantity='power', ac_type='active'))
        # df = df[f'{year}-01-01', f'{year}-12-31']
        return df

    # 获取指定电器类型的电表并加载数据
    kettle_meters = get_elec_meters_by_type('../ukdale.h5', 'kettle')
    freezer_meters = get_elec_meters_by_type('../ukdale.h5', 'freezer') + get_elec_meters_by_type('../ukdale.h5',
                                                                                           'fridge freezer')
    toster_meters = get_elec_meters_by_type('../ukdale.h5', 'toaster')
    hair_dryer_meters = get_elec_meters_by_type('../ukdale.h5', 'hair dryer')
    microwave_meters = get_elec_meters_by_type('../ukdale.h5', 'microwave')
    dish_washer_meters = get_elec_meters_by_type('../ukdale.h5', 'dish washer')
    washing_machine_meters = get_elec_meters_by_type('../ukdale.h5', 'washing machine')
    oven_meters = get_elec_meters_by_type('../ukdale.h5', 'oven')

    kettle_meter_data = [load_data_for_meter(meter) for meter in get_elec_meters_by_type('../ukdale.h5', 'kettle')]
    freezer_meter_data = [load_data_for_meter(meter) for meter in (get_elec_meters_by_type('../ukdale.h5', 'freezer') +
                                                                   get_elec_meters_by_type('../ukdale.h5',
                                                                                           'fridge freezer'))]
    toster_meter_data = [load_data_for_meter(meter) for meter in get_elec_meters_by_type('../ukdale.h5', 'toaster')]
    hair_dryer_meter_data = [load_data_for_meter(meter) for meter in
                             get_elec_meters_by_type('../ukdale.h5', 'hair dryer')]
    microwave_meter_data = [load_data_for_meter(meter) for meter in get_elec_meters_by_type('../ukdale.h5', 'microwave')]
    dish_washer_meter_data = [load_data_for_meter(meter) for meter in
                              get_elec_meters_by_type('../ukdale.h5', 'dish washer')]
    oven_meter_data = [load_data_for_meter(meter) for meter in get_elec_meters_by_type('../ukdale.h5', 'oven')]

    print(kettle_meter_data)
    record_meters_info_to_csv()

    # 准备各类电器数据
    kettle_data = prepare_data_for_pca(kettle_meter_data)
    freezer_data = prepare_data_for_pca(freezer_meter_data)
    toaster_data = prepare_data_for_pca(toster_meter_data)
    hair_dryer_data = prepare_data_for_pca(hair_dryer_meter_data)
    microwave_data = prepare_data_for_pca(microwave_meter_data)
    dish_washer_data = prepare_data_for_pca(dish_washer_meter_data)
    oven_data = prepare_data_for_pca(oven_meter_data)

    # 合并所有样本
    X = np.vstack([
        kettle_data,
        freezer_data,
        toaster_data,
        hair_dryer_data,
        microwave_data,
        dish_washer_data,
        oven_data
    ])

    # 添加标签（用于可视化）
    y = np.array(
        ['kettle'] * len(kettle_data) +
        ['freezer'] * len(freezer_data) +
        ['toaster'] * len(toaster_data) +
        ['hair dryer'] * len(hair_dryer_data) +
        ['microwave'] * len(microwave_data) +
        ['dish washer'] * len(dish_washer_data) +
        ['oven'] * len(oven_data)
    )

    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 应用 PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 可视化 PCA 结果
    plt.figure(figsize=(10, 8))
    for label in np.unique(y):
        idx = y == label
        plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=label, alpha=0.7)

    plt.title('PCA of Appliance Power Consumption')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



    tsne = TSNE(n_components=2, random_state=42, perplexity=10, n_iter=1000)
    X_tsne = tsne.fit_transform(X_scaled)

    # 可视化 t-SNE 结果
    plt.figure(figsize=(12, 8))
    unique_labels = np.unique(y)

    for label in unique_labels:
        idx = y == label
        plt.scatter(X_tsne[idx, 0], X_tsne[idx, 1], label=label, alpha=0.7, edgecolor='k')

    plt.title('t-SNE Visualization of Appliance Power Consumption Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend(title='Appliance Type')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def extract_partial_csv_files(folder_path, rows=10000):
    """
    提取指定文件夹下所有 CSV 文件的前指定行数，并保存为新文件到 partial 子目录下

    :param folder_path: CSV 文件所在文件夹路径 (str)
    :param rows: 要提取的行数，默认为 10000 (int)
    :return: 无返回值
    """
    # 检查文件夹是否存在
    if not os.path.exists(folder_path):
        print(f"错误：文件夹 {folder_path} 不存在")
        return

    # 创建 partial 子目录
    partial_dir = os.path.join(folder_path, 'partial')
    os.makedirs(partial_dir, exist_ok=True)

    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        # 只处理 CSV 文件
        if filename.endswith('.csv') and not filename.startswith('partial'):  # 避免处理 partial 目录中的文件
            file_path = os.path.join(folder_path, filename)

            try:
                # 读取前指定行数的数据
                df = pd.read_csv(file_path, nrows=rows)

                # 生成新文件名
                name, ext = os.path.splitext(filename)
                new_filename = f"{name}_partial{ext}"
                new_file_path = os.path.join(partial_dir, new_filename)

                # 保存到新文件
                df.to_csv(new_file_path, index=False)

                print(f"已处理文件: {filename} -> {os.path.join('partial', new_filename)} (提取了 {len(df)} 行数据)")

            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")


def visualize_csv_data(file_path, x_column, y_column, rows=None):
    """
    可视化 CSV 文件中指定列的数据

    :param file_path: CSV 文件路径
    :param x_column: 用作 x 轴的列名
    :param y_column: 用作 y 轴的列名
    :param rows: 要可视化的行范围，例如 (0, 1000) 表示前1000行
    """
    # 读取 CSV 文件
    df = pd.read_csv(file_path)

    # 如果指定了行范围，则截取相应行
    if rows:
        start_row, end_row = rows
        df = df.iloc[start_row:end_row]

    # 绘制图形
    plt.figure(figsize=(10, 6))
    plt.plot(df[x_column], df[y_column], marker='o', linestyle='-', markersize=2)
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.title(f'{y_column} vs {x_column}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# 示例调用
if __name__ == '__main__':
    # dataset_path = '../ukdale.h5'
    # data_visualization(dataset_path)
    # extract_partial_csv_files('../ukdale_disaggregate')
    visualize_csv_data('../ukdale_disaggregate/partial/dish washer_1_building_1_partial.csv', 'datetime', 'active_power')