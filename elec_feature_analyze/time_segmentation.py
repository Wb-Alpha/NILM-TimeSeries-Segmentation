# import stumpy
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
#
# # 读取时间序列数据
# time_series = pd.read_csv('../process_dataset/Air-condition/Air_condition.csv')
# # 确保 'active power' 列是数值类型，非数值数据会被转换为 NaN
# time_series['active power'] = pd.to_numeric(time_series['active power'], errors='coerce')
#
# # 删除或填充 NaN 值（这里选择删除）
# time_series = time_series.dropna(subset=['active power'])
#
# # 截取数据点：只截取前3600*24个点
# max_points = 6000
# time_series = time_series.head(max_points)
#
# # 提取处理后的数值数据
# active_power_data = time_series['active power'].values
#
# # 设置窗口大小（通常根据业务需求设定）
# window_size = 600
#
# # 计算矩阵剖面 - 传入处理后的数值数组而不是整个DataFrame
# matrix_profile = stumpy.stump(active_power_data, window_size)
#
# # 使用 FLUSS 算法进行时间序列分割
# # cac_threshold: 复杂度原子曲线阈值
# # n_regimes: 预期的段落数量
# cac, regime_locations = stumpy.fluss(
#     matrix_profile[:, 0],  # 矩阵剖面的第一列
#     window_size,
#     n_regimes=10,
#     excl_factor=1
# )
#
# # 可视化结果
# fig, axes = plt.subplots(2, 1, figsize=(12, 8))
#
# # 绘制原始时间序列 - 使用处理后的数值数据
# axes[0].plot(active_power_data)
# axes[0].set_title('Original Time Series')
# axes[0].set_xlabel('Time')
# axes[0].set_ylabel('Value')
#
# # 绘制复杂度原子曲线和分割点
# axes[1].plot(cac)
# axes[1].scatter(regime_locations, cac[regime_locations],
#                color='red', s=100, zorder=3)
# axes[1].set_title('FLUSS Complexity Atoms Curve')
# axes[1].set_xlabel('Window Index')
# axes[1].set_ylabel('Complexity')
#
# plt.tight_layout()
# plt.show()
#
# # 输出分割点位置
# print("Change points detected at indices:", regime_locations)
#
# # 可视化矩阵剖面
# fig, ax = plt.subplots(figsize=(12, 6))
#
# # 使用imshow进行可视化，高值红色，低值蓝色
# # 将数据转换为float类型
# mp_data = matrix_profile[:, 0].astype(float)
# cax = ax.imshow(mp_data.reshape(1, -1),
#                 cmap='RdBu_r',  # RdBu_r colormap: red(高值)到blue(低值)
#                 aspect='auto',
#                 interpolation='nearest')
#
# # 添加颜色条
# cbar = plt.colorbar(cax)
# cbar.set_label('Matrix Profile Values')
#
# # 设置标题和标签
# ax.set_title('Matrix Profile Visualization')
# ax.set_xlabel('Time Index')
# ax.set_yticks([])  # 隐藏y轴刻度
#
# plt.tight_layout()
# plt.show()


import numpy as np
import matplotlib.pyplot as plt


def compute_matrix_profile(ts, window_size, excl_zone=None):
    """
    手动计算矩阵轮廓（Matrix Profile）和索引（Matrix Profile Index）
    参数：
        ts: 时间序列（1D numpy数组）
        window_size: 子序列窗口大小
        excl_zone: 排除区域（默认window_size//2）
    返回：
        mp: 矩阵轮廓数组
        mpi: 矩阵轮廓索引数组
    """
    n = len(ts)
    num_subseq = n - window_size + 1
    if excl_zone is None:
        excl_zone = window_size // 2

    mp = np.full(num_subseq, np.inf)
    mpi = np.full(num_subseq, -1)

    for i in range(num_subseq):
        # 提取当前子序列
        subseq = ts[i:i + window_size]
        min_dist = np.inf
        min_idx = -1

        for j in range(num_subseq):
            # 跳过排除区域内的自匹配
            if abs(i - j) < excl_zone:
                continue
            # 提取对比子序列
            comp_subseq = ts[j:j + window_size]
            # 计算欧氏距离
            dist = np.sqrt(np.sum((subseq - comp_subseq) ** 2))
            if dist < min_dist:
                min_dist = dist
                min_idx = j

        mp[i] = min_dist
        mpi[i] = min_idx

    return mp, mpi


def compute_arc_curve(mpi, num_subseq, excl_zone):
    """
    计算弧曲线（Arc Curve）
    参数：
        mpi: 矩阵轮廓索引数组
        num_subseq: 子序列总数
        excl_zone: 排除区域
    返回：
        ac: 弧曲线数组
    """
    ac = np.zeros(num_subseq)
    for i in range(num_subseq):
        j = mpi[i]
        # 仅统计排除区域外的有效匹配
        if abs(i - j) >= excl_zone:
            ac[j] += 1
    return ac


def compute_cac(ac):
    """
    计算校正弧曲线（Corrected Arc Curve）
    参数：
        ac: 弧曲线数组
    返回：
        cac: 校正弧曲线数组（0-1区间，越小越可能是边界）
    """
    max_ac = np.max(ac)
    cac = 1 - (ac / max_ac)
    return cac


def find_boundaries(cac, n_regimes):
    """
    从校正弧曲线中选择边界点
    参数：
        cac: 校正弧曲线数组
        n_regimes: 期望分割段数
    返回：
        segments: 分割边界的索引（时间序列中的位置）
    """
    # 找到前n_regimes-1个最小的CAC值的位置
    boundary_indices = np.argsort(cac)[:n_regimes - 1]
    # 转换为时间序列中的绝对位置（子序列起始位置 + 窗口大小//2，近似中心位置）
    segments = boundary_indices + (window_size // 2)
    # 排序确保边界按时间顺序排列
    segments = np.sort(segments)
    return segments


# --------------------------
# 测试手动实现的FLUSS
# --------------------------
# 读取时间序列数据
import pandas as pd
time_series = pd.read_csv('../process_dataset/Air-condition/Air_condition.csv')
# 确保 'active power' 列是数值类型，非数值数据会被转换为 NaN
time_series['active power'] = pd.to_numeric(time_series['active power'], errors='coerce')

# 删除或填充 NaN 值（这里选择删除）
ts = time_series.dropna(subset=['active power'])
ts = time_series.head(6000)
ts = ts['active power'].values  # 提取数值数组

# 2. 手动计算矩阵轮廓
window_size = 600
excl_zone = window_size // 2
mp, mpi = compute_matrix_profile(ts, window_size, excl_zone)

# 3. 计算弧曲线和校正弧曲线
num_subseq = len(mp)
ac = compute_arc_curve(mpi, num_subseq, excl_zone)
cac = compute_cac(ac)

# 4. 检测边界
n_regimes = 2
segments = find_boundaries(cac, n_regimes)
print(f"手动实现FLUSS检测的边界：{segments}")

# 5. 可视化结果
plt.figure(figsize=(12, 8))

# 子图1：原始时间序列 + 边界
plt.subplot(3, 1, 1)
plt.plot(ts, label='原始时间序列')
for seg in segments:
    plt.axvline(x=seg, color='red', linestyle='--', linewidth=2, label='边界' if seg == segments[0] else "")
plt.title('手动FLUSS分割结果')
plt.legend()

# 子图2：矩阵轮廓
plt.subplot(3, 1, 2)
plt.plot(mp, label='矩阵轮廓')
plt.title('Matrix Profile')
plt.legend()

# 子图3：校正弧曲线
plt.subplot(3, 1, 3)
plt.plot(cac, label='校正弧曲线（CAC）')
for seg in segments - (window_size // 2):  # 转换为子序列索引
    plt.axvline(x=seg, color='red', linestyle='--', linewidth=2,
                label='边界（子序列索引）' if seg == segments[0] - (window_size // 2) else "")
plt.title('Corrected Arc Curve')
plt.legend()

plt.tight_layout()
plt.show()