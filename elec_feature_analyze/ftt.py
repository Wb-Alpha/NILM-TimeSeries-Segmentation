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

# 运行示例
if __name__ == "__main__":
    main_analysis_example()
