import os
import shutil
import pandas as pd


def move_datafile_to_dataset(main_file: str, app_file: str, dataset_path, channel: int):
    """
    检查dataset_path下是否存在channel1_org.dat，如果不存在则将channel1.dat复制并重命名为channel1_org.dat
    将main_file复制到dataset_path下并重命名为channel1.dat
    将app_file复制到dataset_path下并根据参数channel重命名

    :param main_file: 主文件路径
    :param app_file: 应用文件路径
    :param dataset_path: 数据集路径
    :param channel: 通道号
    :return: None
    """
    # 构建目标文件路径
    channel1_org_path = os.path.join(dataset_path, 'channel_1_org.dat')
    channel1_path = os.path.join(dataset_path, 'channel_1.dat')
    channel_app_path = os.path.join(dataset_path, f'channel_{channel}.dat')

    print(f"开始处理文件移动到数据集: {dataset_path}")
    print(f"主文件: {main_file}")
    print(f"应用文件: {app_file}")
    print(f"目标通道: {channel}")

    # 检查channel1_org.dat是否存在
    if not os.path.exists(channel1_org_path):
        print(f"文件 {channel1_org_path} 不存在，检查源文件...")
        # 检查源文件channel1.dat是否存在
        if os.path.exists(channel1_path):
            # 复制并重命名文件
            shutil.copy2(channel1_path, channel1_org_path)
            print(f"已复制 {channel1_path} 到 {channel1_org_path}")
        else:
            print(f"警告: 源文件 {channel1_path} 不存在")
    else:
        print(f"文件 {channel1_org_path} 已存在，跳过复制")

    # 复制main_file到dataset_path并重命名为channel1.dat
    if os.path.exists(main_file):
        shutil.copy2(main_file, channel1_path)
        print(f"已复制主文件 {main_file} 到 {channel1_path}")
    else:
        raise FileNotFoundError(f"主文件 {main_file} 不存在")

    # 复制app_file到dataset_path并重命名为channel_{channel}.dat
    if os.path.exists(app_file):
        shutil.copy2(app_file, channel_app_path)
        print(f"已复制应用文件 {app_file} 到 {channel_app_path}")
    else:
        raise FileNotFoundError(f"应用文件 {app_file} 不存在")

    print("文件处理完成！")


def calculate_status_active_times(app_df: pd.DataFrame, target_status: int):
    """
    计算某个状态工作了几次，app_df为一个index为datetime，包含power  active power  on/off status 和status列的DataFrame,
    某一段连续的status==target_status视作一次active，统计target_status了多少次，共计时常是多少，并且将每次active的开始、结束
    时间和持续时间长度（以s为单位）存入一个数组中返回
    :return: (次数, 总时长, 每次活动的详细信息列表)
    """
    # 检查必要的列是否存在
    if 'status' not in app_df.columns:
        raise ValueError("DataFrame中必须包含'status'列")
    print(f"计算状态{target_status} active的时间段")
    # 初始化变量
    active_count = 0  # 活动次数
    total_duration = 0  # 总时长
    active_periods = []  # 每次活动的详细信息 [(开始时间, 结束时间, 持续时间), ...]

    # 初始化状态跟踪变量
    in_target_status = False
    start_time = None

    # 遍历DataFrame的每一行
    for idx, row in app_df.iterrows():
        current_status = row['status']

        # 如果当前状态是目标状态且之前不是目标状态，则开始一个新的活动期
        if current_status == target_status and not in_target_status:
            in_target_status = True
            start_time = idx

        # 如果当前状态不是目标状态且之前是目标状态，则结束当前活动期
        elif current_status != target_status and in_target_status:
            in_target_status = False
            end_time = idx

            duration = (end_time - start_time).total_seconds()

            # 更新统计信息
            active_count += 1
            total_duration += duration
            active_periods.append((start_time, end_time, duration))

    # 处理最后一条记录仍在目标状态的情况
    if in_target_status:
        end_time = app_df.index[-1]
        duration = (end_time - start_time).total_seconds()
        active_count += 1
        total_duration += duration
        active_periods.append((start_time, end_time, duration))

    print(f"状态 '{target_status}' 的活动统计:")
    print(f"  活动次数: {active_count}")
    print(f"  总时长: {total_duration} 秒 ({total_duration / 3600:.2f} 小时)")
    print(f"  平均每次持续时间: {total_duration / active_count if active_count > 0 else 0:.2f} 秒")

    return active_count, total_duration, active_periods


def clean_data_by_list(df: pd.DataFrame, periods_list):
    """
    对于某一个状态的用电器数据，将periods_list中的每个时间段的status和on/off status置为0，将apparent power和active power置为1
    :param df:
    :param periods_list: 时间段列表，每个元素为(start_time, end_time, duration)的三元组
    :return: None
    """
    print(f"需要处理的时间段数量: {len(periods_list)}")

    # 对每个时间段进行处理
    for i, (start_time, end_time, duration) in enumerate(periods_list):
        print(f"处理时间段 {i + 1}/{len(periods_list)}: {start_time} 到 {end_time} (持续时间: {duration}秒)")

        # 创建时间范围掩码
        mask = (df.index >= start_time) & (df.index <= end_time)

        # 修改对应行的数据
        modified_rows = mask.sum()
        print(f"  匹配到 {modified_rows} 行数据")

        if 'status' in df.columns:
            df.loc[mask, 'status'] = 0
        if 'on/off status' in df.columns:
            df.loc[mask, 'on/off status'] = 0
        if 'active power' in df.columns:
            df.loc[mask, 'active power'] = 1
        if 'apparent power' in df.columns:
            df.loc[mask, 'apparent power'] = 1

    # 保存处理后的数据
    print("数据清理完成！")
    return df


def clean_data(df: pd.DataFrame, status: int):
    """

    :param status:
    :param df:
    :return:
    """
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    if 'status' in df.columns:
        df['status'] = df['status'].round()

    if 'on/off status' in df.columns:
        df['on/off status'] = df['on/off status'].round()
    active_count, total_duration, removed_periods = calculate_status_active_times(df, status)

    # removed_periods是需要删除的时间段，如果是需要留下来的数据，在接下来的步骤中，将这些时间段从removed_periods剔除
    while True:
        # 打印带下标的active_periods
        print("\n当前活动时间段列表:")
        for i, period in enumerate(removed_periods):
            print(f"[{i}] {period}")

        # 如果列表为空，退出循环
        if not removed_periods:
            print("活动时间段列表为空，退出循环。")
            break

        # 获取用户输入
        user_input = input("\n需要保留哪个下标的数据？(输入要移除的下标，输入'quit'退出): ")

        # 检查是否退出
        if user_input.lower() == 'quit':
            print("退出交互循环。")
            break

        # 尝试解析用户输入为整数
        try:
            index_to_remove = int(user_input)
            # 检查下标是否有效
            if 0 <= index_to_remove < len(removed_periods):
                removed_period = removed_periods.pop(index_to_remove)
                print(f"已移除下标为 {index_to_remove} 的元素: {removed_period}")
                # 计算剩余活动时间段的总时长
                remaining_total_duration = sum(period[2] for period in removed_periods)

                # 计算删除的百分比和剩余的百分比
                if total_duration > 0:
                    deleted_percentage = (remaining_total_duration / total_duration) * 100
                    remaining_percentage = ((total_duration - remaining_total_duration) / total_duration) * 100
                else:
                    deleted_percentage = 0
                    remaining_percentage = 0

                # 打印统计信息
                print(f"\n时长统计:")
                print(f"  原始总时长: {total_duration:.2f} 秒")
                print(f"  删除列表总时长: {remaining_total_duration:.2f} 秒")
                print(f"  剩余列表总时长: {total_duration - remaining_total_duration:.2f} 秒")
                print(f"  将会删除 {deleted_percentage:.2f}% 的数据")
                print(f"  剩余 {remaining_percentage:.2f}% 的数据")
            else:
                print(f"错误: 下标 {index_to_remove} 超出范围 [0, {len(removed_periods) - 1}]")
        except ValueError:
            print("错误: 请输入有效的整数或 'quit'")

    print(f"\n最终保留的活动时间段数量: {len(removed_periods)}")
    for i, period in enumerate(removed_periods):
        print(f"[{i}] {period}")

    clean_data_by_list(df, removed_periods)
    active_count, total_duration, periods = calculate_status_active_times(df, status)
    for i, period in enumerate(periods):
        print(f"[{i}] {period}")
    print(f'剩余的列表总时长: {total_duration:.2f} 秒')
    return df


def print_all_periods(df: pd.DataFrame, status_num: int):
    df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
    if 'status' in df.columns:
        df['status'] = df['status'].round()

    if 'on/off status' in df.columns:
        df['on/off status'] = df['on/off status'].round()
    for i in range(0, status_num+1):
        active_count, total_duration, periods = calculate_status_active_times(df, i)
        for j, period in enumerate(periods):
            print(f"[{j}] {period}")
        print(f'剩余的列表总时长: {total_duration:.2f} 秒')


if __name__ == '__main__':
    main_file = r'.\experiment_dataset\BERT4NILM\microwave\test\mains_sterilize.dat'
    app_file = r'.\experiment_dataset\BERT4NILM\microwave\test\microwave_sterilize.dat'
    dataset_path = r'E:\datasets\NILM\uk_dale\house_2'
    channel = 15
    move_datafile_to_dataset(main_file, app_file, dataset_path, channel)
    # file_path = r'./dataset/Microwave_Microwave.csv'
    # df = pd.read_csv(file_path, index_col=0)
    # df_after_process = clean_data(df, 2)
    # df_after_process.to_csv(file_path, index=True)
    # print(f"已保存处理后的数据到: {file_path}")
    # print_all_periods(df, 4)
    exit()
