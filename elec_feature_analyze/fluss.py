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
        j = int(mpi[i])
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


def find_boundaries(window_size, ac, cac, n_regimes):
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
    # 在 boundary_indices 计算之后添加
    print("Boundary indices:", boundary_indices)
    for idx in boundary_indices:
        start = max(0, idx - 5)
        end = min(len(ac), idx + 6)
        print(f"Index {idx} 周围的AC值: {ac[start:end]}")
        print(f"Index {idx} 周围的CAC值: {cac[start:end]}")

    # 转换为时间序列中的绝对位置（子序列起始位置 + 窗口大小//2，近似中心位置）
    segments = boundary_indices + (window_size // 2)
    # 排序确保边界按时间顺序排列
    segments = np.sort(segments)
    return segments


def fluss(ts, window_size,  n_regimes=3, excl_factor=1):
    excl_zone = window_size * excl_factor
    mp, mpi = compute_matrix_profile(ts, window_size, excl_zone)

    # 3. 计算弧曲线和校正弧曲线
    num_subseq = len(mp)
    ac = compute_arc_curve(mpi, num_subseq, excl_zone)
    cac = compute_cac(ac)

    segments = find_boundaries(window_size, ac, cac, n_regimes)
    print(f"手动实现FLUSS检测的边界：{segments}")

    fluss_visualize(ts, cac, ac, mp, mpi, segments)


def fluss_visualize(ts, cac, mp, ac, mpi, segments):
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun']
    plt.rcParams['axes.unicode_minus'] = False
    # 5. 可视化结果
    plt.figure(figsize=(12, 12))  # 进一步调整图形高度以容纳更多子图
    # 添加大标题
    plt.suptitle('FLUSS时间序列分割空调Auto状态', fontsize=16, fontweight='bold')

    # 子图1：原始时间序列 + 边界
    plt.subplot(5, 1, 1)
    plt.plot(ts, label='原始时间序列')
    for seg in segments:
        plt.axvline(x=seg, color='red', linestyle='--', linewidth=2, label='边界' if seg == segments[0] else "")
    plt.title('手动FLUSS分割结果')
    plt.legend()

    # 子图2：矩阵轮廓
    plt.subplot(5, 1, 2)
    plt.plot(mp, label='矩阵轮廓')
    plt.title('Matrix Profile')
    plt.legend()

    # 子图3：矩阵轮廓索引（新增）
    plt.subplot(5, 1, 3)
    plt.plot(mpi, label='矩阵轮廓索引（MPI）', color='green')
    plt.title('Matrix Profile Index')
    plt.legend()

    # 子图4：弧曲线
    plt.subplot(5, 1, 4)
    plt.plot(ac, label='弧曲线（AC）', color='orange')
    plt.title('Arc Curve')
    plt.legend()

    # 子图5：校正弧曲线
    plt.subplot(5, 1, 5)
    plt.plot(cac, label='校正弧曲线（CAC）')
    plt.title('Corrected Arc Curve')
    plt.legend()

    plt.tight_layout()
    plt.show()
