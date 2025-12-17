import pandas as pd, numpy as np


def trans_status_data_to_vae_formulate(df: pd.DataFrame, status: int, data_len: int):
    """
    将数据转换成VAE的输入格式
    :param df: 用电器数据DataFrame
    :param status:需要采集的状态，特殊的，当status=5的时候会将所有on的状态都采集起来
    :param data_len:单次时间序列长度
    :return:
    """
    if status == 5:
        # 直接获取所有的处于active状态的数据
        if 'on/off status' not in df.columns:
            raise ValueError("DataFrame中必须包含'on/off status'列")

        # 提取'on/off status'为1的所有数据
        active_data = df[df['on/off status'] == 1]
    else:
        if 'status' not in df.columns:
            raise ValueError("DataFrame中必须包含'status'列")
        active_data = df[df['status'] == status]
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print("前10行active_data数据:")
            print(active_data.head(10))
    # 检查'active power'列是否存在
    if 'active power' not in active_data.columns:
        raise ValueError("active_data中必须包含'active power'列")

    # 将active_data的'active power'列转换为(n, 1)形状的numpy数组
    active_power_array = active_data['active power'].values.reshape(-1, 1)
    print(f"原始数据形状: {active_power_array.shape}")

    # 计算可以切分出多少个完整的序列
    data_size = len(active_power_array) // data_len
    print(f"数据总长度: {len(active_power_array)}, 时间序列长度: {data_len}, 可生成序列数: {data_size}")

    # 截取可以构成完整序列的部分
    trimmed_length = data_size * data_len
    trimmed_array = active_power_array[:trimmed_length]
    print(f"截取后数据长度: {trimmed_length}")

    # 重塑为(data_size, data_len, 1)的三维数组
    reshaped_array = trimmed_array.reshape(data_size, data_len, 1)
    print(f"最终三维数组形状: {reshaped_array.shape}")

    return reshaped_array

if __name__ == '__main__':
    # 请注意使用完整的数据集，以获取最多的训练数据
    file_path = r'Microwave/Microwave.csv'
    df = pd.read_csv(file_path, index_col=0)
    df = df[df['active power'] >= 500]
    # 空调建议长度为3600也就是一小时
    # 微波炉建议长度为60也就是一分钟
    # 洗衣机建议长度是600也就是十分钟
    np_array = trans_status_data_to_vae_formulate(df, 2, 60)
    print(f"生成的numpy数组信息:")
    print(f"  形状: {np_array.shape}")
    print(f"  数据类型: {np_array.dtype}")
    print(f"  数组大小: {np_array.size}")

    # 捕获用户输入
    user_input = input("是否保存数据到文件? (y/n): ")

    if user_input.lower() == 'y':
        # 保存为npz文件
        output_path = r'experiment_dataset/VAE/microwave_lightwave_bt500.npz'
        np.savez(output_path, data=np_array)
        print(f"数据已保存到 {output_path}")
    else:
        print("数据未保存")
