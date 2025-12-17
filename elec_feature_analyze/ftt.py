import numpy as np
import pandas as pd
from scipy import fftpack
from scipy.signal import welch
import matplotlib.pyplot as plt


class NILMFourierAnalyzer:
    def __init__(self, sampling_rate=1.0):
        self.sampling_rate = sampling_rate

    def apply_fft(self, signal):
        """
        对信号应用快速傅里叶变换

        Parameters:
        signal: 输入信号（一维数组）

        Returns:
        freqs: 频率轴
        magnitude: 幅度谱
        phase: 相位谱
        """
        # 确保信号长度为偶数（便于处理）
        n = len(signal)
        if n % 2 == 1:
            signal = signal[:-1]
            n = n - 1

        # 执行FFT
        fft_values = fftpack.fft(signal)

        # 计算频率轴
        freqs = fftpack.fftfreq(n, 1 / self.sampling_rate)

        # 只取正频率部分（因为实信号的FFT是对称的）
        positive_freq_idx = freqs >= 0
        freqs = freqs[positive_freq_idx]
        fft_values = fft_values[positive_freq_idx]

        # 计算幅度谱和相位谱
        magnitude = np.abs(fft_values) / n
        phase = np.angle(fft_values)

        return freqs, magnitude, phase

    def compute_power_spectrum(self, signal, nperseg=None):
        """
        计算功率谱密度

        Parameters:
        signal: 输入信号
        nperseg: 每段的样本数（用于Welch方法）

        Returns:
        freqs: 频率轴
        psd: 功率谱密度
        """
        if nperseg is None:
            nperseg = min(1024, len(signal) // 4)

        freqs, psd = welch(signal, fs=self.sampling_rate, nperseg=nperseg)
        return freqs, psd

    def extract_harmonics(self, signal, fundamental_freq=50):
        """
        提取谐波成分

        Parameters:
        signal: 输入信号
        fundamental_freq: 基波频率（通常为50Hz或60Hz）

        Returns:
        harmonics: 谐波信息字典
        """
        freqs, magnitude, phase = self.apply_fft(signal)

        harmonics = {}
        for i in range(1, 11):  # 前10次谐波
            harmonic_freq = i * fundamental_freq

            # 找到最接近的频率索引
            idx = np.argmin(np.abs(freqs - harmonic_freq))

            harmonics[f'harmonic_{i}'] = {
                'order': i,
                'frequency': freqs[idx],
                'magnitude': magnitude[idx],
                'phase': phase[idx]
            }

        return harmonics

    def extract_frequency_features(self, signal):
        """
        提取频域特征

        Parameters:
        signal: 输入信号

        Returns:
        features: 频域特征字典
        """
        freqs, magnitude, phase = self.apply_fft(signal)

        features = {}

        # 基本统计特征
        features['max_magnitude'] = np.max(magnitude)
        features['dominant_frequency'] = freqs[np.argmax(magnitude)]
        features['mean_magnitude'] = np.mean(magnitude)
        features['std_magnitude'] = np.std(magnitude)

        # 频谱质心
        spectral_centroid = np.sum(freqs * magnitude) / np.sum(magnitude)
        features['spectral_centroid'] = spectral_centroid

        # 频谱带宽
        spectral_bandwidth = np.sqrt(
            np.sum(((freqs - spectral_centroid) ** 2) * magnitude) / np.sum(magnitude)
        )
        features['spectral_bandwidth'] = spectral_bandwidth

        # 频谱滚降点（95%能量以下的频率）
        cumsum = np.cumsum(magnitude ** 2)
        if cumsum[-1] > 0:
            rolloff_idx = np.where(cumsum >= 0.95 * cumsum[-1])[0]
            if len(rolloff_idx) > 0:
                features['spectral_rolloff'] = freqs[rolloff_idx[0]]
            else:
                features['spectral_rolloff'] = freqs[-1]
        else:
            features['spectral_rolloff'] = 0

        # 总谐波失真（THD）
        fundamental_idx = np.argmin(np.abs(freqs - 50))  # 假设基波频率为50Hz
        fundamental_magnitude = magnitude[fundamental_idx]

        if fundamental_magnitude > 0:
            # 计算2-10次谐波的总能量
            harmonic_energy = 0
            for i in range(2, 11):
                harmonic_freq = i * 50
                idx = np.argmin(np.abs(freqs - harmonic_freq))
                harmonic_energy += magnitude[idx] ** 2

            features['thd'] = np.sqrt(harmonic_energy) / fundamental_magnitude
        else:
            features['thd'] = 0

        return features

    def plot_spectrum(self, signal, plot_type='magnitude'):
        """
        绘制频谱图

        Parameters:
        signal: 输入信号
        plot_type: 绘图类型 ('magnitude', 'power', 'phase')
        """
        if plot_type == 'magnitude':
            freqs, magnitude, _ = self.apply_fft(signal)
            plt.figure(figsize=(12, 6))
            plt.plot(freqs, magnitude)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Magnitude')
            plt.title('Magnitude Spectrum')
            plt.grid(True)

        elif plot_type == 'power':
            freqs, psd = self.compute_power_spectrum(signal)
            plt.figure(figsize=(12, 6))
            plt.plot(freqs, 10 * np.log10(psd))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power (dB)')
            plt.title('Power Spectral Density')
            plt.grid(True)

        elif plot_type == 'phase':
            freqs, _, phase = self.apply_fft(signal)
            plt.figure(figsize=(12, 6))
            plt.plot(freqs, phase)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Phase (radians)')
            plt.title('Phase Spectrum')
            plt.grid(True)

        plt.show()


def analyze_appliance_signature(signal, analyzer, appliance_name="Unknown"):
    """
    分析电器的频域特征签名

    Parameters:
    signal: 电器功率信号
    analyzer: NILMFourierAnalyzer实例
    appliance_name: 电器名称

    Returns:
    signature: 电器特征签名
    """
    print(f"分析 {appliance_name} 的频域特征...")

    # 提取频域特征
    features = analyzer.extract_frequency_features(signal)

    # 提取谐波成分
    harmonics = analyzer.extract_harmonics(signal)

    # 绘制频谱
    print("绘制幅度谱...")
    analyzer.plot_spectrum(signal, 'magnitude')

    print("绘制功率谱...")
    analyzer.plot_spectrum(signal, 'power')

    # 创建电器签名
    signature = {
        'appliance_name': appliance_name,
        'features': features,
        'harmonics': harmonics,
        'dominant_frequencies': get_dominant_frequencies(signal, analyzer)
    }

    return signature


def get_dominant_frequencies(signal, analyzer, top_n=5):
    """
    获取主要频率成分

    Parameters:
    signal: 输入信号
    analyzer: NILMFourierAnalyzer实例
    top_n: 返回前N个主要频率

    Returns:
    dominant_freqs: 主要频率列表
    """
    freqs, magnitude, _ = analyzer.apply_fft(signal)

    # 获取幅度最大的前N个频率
    top_indices = np.argsort(magnitude)[-top_n:][::-1]

    dominant_freqs = []
    for idx in top_indices:
        if magnitude[idx] > np.max(magnitude) * 0.1:  # 只考虑幅度大于最大值10%的频率
            dominant_freqs.append({
                'frequency': freqs[idx],
                'magnitude': magnitude[idx]
            })

    return dominant_freqs


def get_dominant_frequencies(signal, analyzer, top_n=5):
    """
    获取主要频率成分

    Parameters:
    signal: 输入信号
    analyzer: NILMFourierAnalyzer实例
    top_n: 返回前N个主要频率

    Returns:
    dominant_freqs: 主要频率列表
    """
    freqs, magnitude, _ = analyzer.apply_fft(signal)

    # 获取幅度最大的前N个频率
    top_indices = np.argsort(magnitude)[-top_n:][::-1]

    dominant_freqs = []
    for idx in top_indices:
        if magnitude[idx] > np.max(magnitude) * 0.1:  # 只考虑幅度大于最大值10%的频率
            dominant_freqs.append({
                'frequency': freqs[idx],
                'magnitude': magnitude[idx]
            })

    return dominant_freqs


def compare_appliance_signatures(signature1, signature2):
    """
    比较两个电器的频域特征签名

    Parameters:
    signature1, signature2: 两个电器的特征签名

    Returns:
    similarity: 相似性度量
    """
    features1 = signature1['features']
    features2 = signature2['features']

    # 计算特征相似性
    similarities = {}

    # 基本特征比较
    for key in ['dominant_frequency', 'spectral_centroid', 'spectral_bandwidth']:
        if key in features1 and key in features2:
            # 计算相对差异
            diff = abs(features1[key] - features2[key]) / max(features1[key], features2[key], 1e-10)
            similarities[key] = 1 - diff  # 转换为相似性（值越大越相似）

    # THD比较
    if 'thd' in features1 and 'thd' in features2:
        thd_diff = abs(features1['thd'] - features2['thd'])
        similarities['thd_similarity'] = max(0, 1 - thd_diff)  # THD差异越大相似性越低

    # 谐波比较
    harmonic_similarity = compare_harmonics(signature1['harmonics'], signature2['harmonics'])
    similarities['harmonic_similarity'] = harmonic_similarity

    # 综合相似性
    overall_similarity = np.mean(list(similarities.values()))

    return {
        'feature_similarities': similarities,
        'overall_similarity': overall_similarity
    }


def compare_harmonics(harmonics1, harmonics2):
    """
    比较两个谐波分析结果
    """
    # 比较前5次谐波的幅度
    magnitude_diffs = []
    for i in range(1, 6):
        key = f'harmonic_{i}'
        if key in harmonics1 and key in harmonics2:
            mag1 = harmonics1[key]['magnitude']
            mag2 = harmonics2[key]['magnitude']
            if mag1 > 0 and mag2 > 0:
                diff = abs(mag1 - mag2) / max(mag1, mag2)
                magnitude_diffs.append(1 - diff)

    return np.mean(magnitude_diffs) if magnitude_diffs else 0


# 使用示例
def main_analysis_example():
    """
    主分析示例
    """
    # 创建分析器（假设采样率为1Hz）
    analyzer = NILMFourierAnalyzer(sampling_rate=1.0)

    # 加载数据
    # 注意：需要根据实际数据调整文件路径
    # 加载洗碗机数据
    df_dishwasher = pd.read_csv('../dataset/Air-condition/processed_peek_data_20250808_labeled.csv')
    dishwasher_power = df_dishwasher.iloc[:, 5].values[:10000]  # 取前1000个点
    # 加载冰箱数据
    df_fridge = pd.read_csv('../dataset/Microwave/processed_peek_data_20250808_labeled.csv')
    fridge_power = df_fridge.iloc[:, 5].values[:100000]  # 取前1000个点
    # 分析洗碗机特征
    dishwasher_signature = analyze_appliance_signature(
        dishwasher_power, analyzer, "Dishwasher"
    )
    # 分析冰箱特征
    fridge_signature = analyze_appliance_signature(
        fridge_power, analyzer, "Fridge"
    )
    # 比较两种电器的相似性
    similarity_result = compare_appliance_signatures(
        dishwasher_signature, fridge_signature
    )
    print("\n=== 电器特征比较结果 ===")
    print(f"整体相似性: {similarity_result['overall_similarity']:.4f}")
    print("\n各特征相似性:")
    for feature, similarity in similarity_result['feature_similarities'].items():
        print(f"  {feature}: {similarity:.4f}")


def df_fft_visualize(df: pd.DataFrame, signal_col: str, sampling_freq, time_col=None, fig_title="傅里叶变换分析结果"):
    """
    对DataFrame中的一维信号列执行傅里叶变换并可视化（时间域+频率域）

    参数：
        df: pd.DataFrame - 包含信号数据的DataFrame
        signal_col: str - 需分析的一维信号列名（如'sensor_data'）
        sampling_freq: float - 信号采样频率（Hz，必需，用于计算频率轴）
        time_col: str, optional - 时间列名（若为None，自动生成时间轴）
        fig_title: str, optional - 图表总标题
    """
    # 1. 数据预处理：提取信号并处理缺失值
    # 提取信号列，删除缺失值（避免FFT计算错误）
    signal_1d = df[signal_col].dropna().values
    n = len(signal_1d)  # 有效信号数据点数量

    # 生成时间轴（若未提供time_col，按采样频率生成）
    if time_col and time_col in df.columns:
        # 若有时间列，同步删除信号缺失对应的时间值
        time = df[df[signal_col].notna()][time_col].values
    else:
        # 无时间列时，按“采样间隔=1/采样频率”生成时间轴
        time = np.arange(n) / sampling_freq

    # 2. 傅里叶变换计算
    # 实数信号用rfft（仅返回非负频率，减少计算量）
    fft_result = np.fft.rfft(signal_1d)
    # 计算频率轴（对应每个FFT结果的频率值）
    freq_axis = np.fft.rfftfreq(n, d=1 / sampling_freq)
    # 幅度谱归一化（除以数据长度，确保幅度与原始信号一致）
    amplitude_spectrum = np.abs(fft_result) / n

    # 3. 可视化配置
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)

    # 3.1 子图1：时间域 - 原始信号
    ax1.plot(time, signal_1d, color='#1f77b4', linewidth=1.2, label=f'原始信号（{signal_col}）')
    ax1.set_title(f'时间域：{signal_col} 原始信号', fontsize=12, fontweight='bold')
    ax1.set_xlabel('时间（s）' if time_col else '采样点', fontsize=10)
    ax1.set_ylabel('信号幅度', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 3.2 子图2：频率域 - 幅度谱（标注Top2峰值频率）
    ax2.plot(freq_axis, amplitude_spectrum, color='#ff7f0e', linewidth=1.2, label='归一化幅度谱')
    ax2.set_title(f'频率域：傅里叶变换幅度谱（采样频率：{sampling_freq}Hz）', fontsize=12, fontweight='bold')
    ax2.set_xlabel('频率（Hz）', fontsize=10)
    ax2.set_ylabel('归一化幅度', fontsize=10)
    ax2.set_xlim(0, sampling_freq / 2)  # 仅显示0~Nyquist频率（采样频率的1/2）
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # 标注Top2峰值频率（排除0Hz直流分量）
    non_zero_mask = freq_axis > 0  # 过滤0Hz
    if np.any(non_zero_mask):
        # 取幅度前2大的频率索引
        top2_idx = np.argsort(amplitude_spectrum[non_zero_mask])[-2:]
        top2_freq = freq_axis[non_zero_mask][top2_idx]
        top2_amp = amplitude_spectrum[non_zero_mask][top2_idx]
        # 逐个标注
        for freq, amp in zip(top2_freq, top2_amp):
            ax2.annotate(f'峰值: {freq:.1f}Hz\n幅度: {amp:.2f}',
                         xy=(freq, amp), xytext=(freq + sampling_freq / 20, amp + np.max(amplitude_spectrum) * 0.1),
                         arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                         fontsize=9, color='red', fontweight='bold')

    # 图表总标题与布局调整
    fig.suptitle(fig_title, fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.show()


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def fft_freq_separation_visualize(
        df, signal_col, sampling_freq, time_col=None,
        n_components=3, freq_bandwidth=0.5, include_dc=False
):
    """
    用FFT分离信号的主要频率成分，并可视化原始信号、各分离频率信号、叠加恢复信号

    参数：
        df: pd.DataFrame - 输入数据
        signal_col: str - 信号列名
        sampling_freq: float - 采样频率（Hz，必需）
        time_col: str - 时间列名（可选，无则自动生成）
        n_components: int - 要分离的主要频率成分数量（默认Top3）
        freq_bandwidth: float - 频率筛选带宽（围绕峰值频率的左右范围，避免单频点失真）
        include_dc: bool - 是否包含0Hz直流分量（默认不包含）
    """
    # ---------------------- 1. 数据预处理 ----------------------
    # 提取信号并处理缺失值
    signal_1d = df[signal_col].dropna().values
    n = len(signal_1d)

    # 生成时间轴
    if time_col and time_col in df.columns:
        time = df[df[signal_col].notna()][time_col].values
    else:
        time = np.arange(n) / sampling_freq

    # ---------------------- 2. FFT分解信号 ----------------------
    # 正向FFT（实数信号优化）
    fft_result = np.fft.rfft(signal_1d)
    freq_axis = np.fft.rfftfreq(n, d=1 / sampling_freq)  # 频率轴
    amplitude_spectrum = np.abs(fft_result) / n  # 归一化幅度谱

    # ---------------------- 3. 筛选主要频率成分 ----------------------
    # 筛选目标频率（排除/包含直流分量）
    if include_dc:
        valid_mask = np.ones_like(freq_axis, dtype=bool)
    else:
        valid_mask = freq_axis > 0  # 排除0Hz直流分量

    # 提取Top N幅度的频率（主要成分）
    valid_amplitude = amplitude_spectrum[valid_mask]
    valid_freq = freq_axis[valid_mask]
    top_n_idx = np.argsort(valid_amplitude)[-n_components:][::-1]  # Top N索引（降序）
    target_freqs = valid_freq[top_n_idx]  # 目标频率列表
    target_amps = valid_amplitude[top_n_idx]  # 对应幅度

    # ---------------------- 4. 分离各频率成分（逆FFT恢复时域信号） ----------------------
    separated_signals = []  # 存储各分离后的时域信号
    fft_separated_list = []  # 存储各分离后的FFT分量

    for target_freq in target_freqs:
        # 构造该频率的筛选掩码（保留目标频率±带宽内的分量，其他置零）
        freq_mask = np.abs(freq_axis - target_freq) <= freq_bandwidth
        fft_separated = fft_result.copy()
        fft_separated[~freq_mask] = 0  # 置零非目标频率分量

        # 逆FFT恢复时域信号（取实部，消除数值误差）
        signal_separated = np.fft.irfft(fft_separated).real
        separated_signals.append(signal_separated)
        fft_separated_list.append(fft_separated)

    # 所有分离成分叠加（验证是否接近原始信号）
    combined_signal = np.sum(separated_signals, axis=0)

    # ---------------------- 5. 可视化设计 ----------------------
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False

    # 子图数量：原始信号 + n_components个分离信号 + 叠加信号 + 幅度谱 = n_components+3
    fig, axes = plt.subplots(n_components + 3, 1, figsize=(12, 4 * (n_components + 3)), sharex=False)
    fig.suptitle(f'FFT频率分离分析（分离Top{n_components}主要频率）', fontsize=16, fontweight='bold', y=0.98)

    # 子图1：原始信号
    axes[0].plot(time, signal_1d, color='#1f77b4', linewidth=1.2, label='原始信号')
    axes[0].set_title('原始信号（含噪声+多频率叠加）', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('信号幅度', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # 子图2~n_components+1：各分离的频率成分
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']  # 颜色循环
    for i in range(n_components):
        ax = axes[i + 1]
        freq = target_freqs[i]
        amp = target_amps[i]
        ax.plot(time, separated_signals[i], color=colors[i % len(colors)], linewidth=1.2,
                label=f'分离成分：{freq:.1f}Hz（幅度：{amp:.2f}）')
        ax.set_title(f'分离频率成分：{freq:.1f}Hz', fontsize=12, fontweight='bold')
        ax.set_ylabel('信号幅度', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend()

    # 子图n_components+2：所有分离成分叠加（对比原始信号）
    axes[n_components + 1].plot(time, combined_signal, color='#e377c2', linewidth=1.5, label='分离成分叠加信号',
                                alpha=0.8)
    axes[n_components + 1].plot(time, signal_1d, color='#1f77b4', linewidth=0.8, label='原始信号（参考）', alpha=0.5)
    axes[n_components + 1].set_title('分离成分叠加 vs 原始信号', fontsize=12, fontweight='bold')
    axes[n_components + 1].set_ylabel('信号幅度', fontsize=10)
    axes[n_components + 1].grid(True, alpha=0.3)
    axes[n_components + 1].legend()

    # 子图n_components+3：幅度谱（标注分离的频率）
    axes[n_components + 2].plot(freq_axis, amplitude_spectrum, color='#1f77b4', linewidth=1.2, label='幅度谱')
    # 标注分离的频率
    for i, (freq, amp) in enumerate(zip(target_freqs, target_amps)):
        axes[n_components + 2].scatter(freq, amp, color=colors[i % len(colors)], s=50, zorder=5)
        axes[n_components + 2].annotate(f'{freq:.1f}Hz', xy=(freq, amp), xytext=(freq + 0.5, amp + 0.05),
                                        color=colors[i % len(colors)], fontsize=10, fontweight='bold')
    axes[n_components + 2].set_title('频率域：原始信号幅度谱（红点为分离频率）', fontsize=12, fontweight='bold')
    axes[n_components + 2].set_xlabel('频率（Hz）', fontsize=10)
    axes[n_components + 2].set_ylabel('归一化幅度', fontsize=10)
    axes[n_components + 2].set_xlim(0, sampling_freq / 2)
    axes[n_components + 2].grid(True, alpha=0.3)
    axes[n_components + 2].legend()

    # 统一设置x轴标签（最后一个子图）
    axes[-1].set_xlabel('时间（s）' if time_col else '采样点', fontsize=10)

    plt.tight_layout()
    plt.show()

    # 返回分离的频率和对应的时域信号（可选后续分析）
    return target_freqs, separated_signals, combined_signal


# 运行示例
if __name__ == '__main__':
    df = pd.read_csv(r'../process_dataset/Air-condition/Air_condition.csv')[:6000]
    # df_fft_visualize(
    #     df=df,
    #     signal_col='active power',  # 你的信号列名
    #     sampling_freq=1,  # 你的信号采样频率（必需）
    #     time_col=None,  # 你的时间列名（可选，无则传None）
    #     fig_title='电流信号傅里叶变换分析'  # 自定义图表标题
    # )

    # 分离Top3主要频率成分
    target_freqs, separated_signals, combined_signal = fft_freq_separation_visualize(
        df=df,
        signal_col='active power',  # 你的信号列名
        sampling_freq=1,  # 采样频率（必需）
        time_col=None,  # 时间列名（可选）
        n_components=5,  # 要分离的频率数量（默认3）
        freq_bandwidth=0.8,  # 频率筛选带宽（根据信号调整，越大包含越多谐波）
        include_dc=False  # 不包含直流分量
    )

    # 输出分离的频率
    print("分离的主要频率：", np.round(target_freqs, 1), "Hz")
