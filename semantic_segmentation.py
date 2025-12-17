import pandas as pd
import numpy as np
from typing import List, Tuple, Optional
import os
import gc
from datetime import datetime


class ApplianceDataSegmenter:
    def __init__(self, appliance_name: str, power_threshold: float = 1.0,
                 min_duration_seconds: int = 30,
                 context_percent: float = 0.2):
        """
        电器数据切割器

        Args:
            appliance_name: 电器名称，用于文件命名
            power_threshold: 功率阈值，用于检测工作状态开始和结束
            min_duration_seconds: 最小持续时间(秒)，避免噪声误判
            context_percent: 上下文百分比，基于工作段长度的前后额外包含的数据比例
        """
        self.appliance_name = appliance_name
        self.power_threshold = power_threshold
        self.min_duration_seconds = min_duration_seconds
        self.context_percent = context_percent

    def process_dataset(self, input_file: str, output_dir: str) -> List[str]:
        """
        处理数据集并分割工作区间

        Args:
            input_file: 输入.dat文件路径
            output_dir: 输出目录

        Returns:
            生成的CSV文件路径列表
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        print(f"开始处理电器数据: {self.appliance_name}")
        print(f"输入文件: {input_file}")
        print(f"输出目录: {output_dir}")
        print(f"功率阈值: {self.power_threshold}")
        print(f"最小持续时间: {self.min_duration_seconds}秒")
        print(f"上下文百分比: {self.context_percent * 100}%")
        print("-" * 50)

        # 第一步：检测所有工作区间
        print("第一步：检测工作区间...")
        segments = self._detect_working_segments(input_file)

        print(f"检测到 {len(segments)} 个潜在工作区间")

        # 第二步：提取并保存工作区间数据
        print("第二步：提取并保存工作区间数据...")
        output_files = self._extract_all_segments(input_file, segments, output_dir)

        print(f"处理完成！共提取 {len(output_files)} 个工作区间")
        return output_files

    def _detect_working_segments(self, input_file: str) -> List[Tuple[int, int, int, int, int]]:
        """
        检测所有工作区间

        Returns:
            列表格式: [(start_idx, end_idx, start_time, end_time, duration), ...]
        """
        segments = []
        current_segment_start = None
        current_segment_start_time = None
        consecutive_above_count = 0

        print("正在扫描文件检测工作区间...")

        with open(input_file, 'r') as f:
            line_count = 0
            for line in f:
                line_count += 1
                if line_count % 1000000 == 0:
                    print(f"已扫描 {line_count} 行数据...")

                parts = line.strip().split()
                if len(parts) < 2:
                    continue

                try:
                    timestamp = int(parts[0])
                    power = float(parts[1])
                except (ValueError, IndexError):
                    continue

                if power >= self.power_threshold:
                    if current_segment_start is None:
                        current_segment_start = line_count - 1  # 0-based index
                        current_segment_start_time = timestamp
                    consecutive_above_count += 1
                else:
                    # 检查是否结束了一个有效的工作区间
                    if (current_segment_start is not None and
                            consecutive_above_count >= self.min_duration_seconds):
                        segment_end = line_count - 2  # 上一行是结束点
                        segment_end_time = timestamp
                        duration = consecutive_above_count
                        segments.append((
                            current_segment_start,
                            segment_end,
                            current_segment_start_time,
                            segment_end_time,
                            duration
                        ))

                    current_segment_start = None
                    current_segment_start_time = None
                    consecutive_above_count = 0

        # 处理文件末尾可能的工作区间
        if (current_segment_start is not None and
                consecutive_above_count >= self.min_duration_seconds):
            segments.append((
                current_segment_start,
                line_count - 1,
                current_segment_start_time,
                int(parts[0]),  # 最后一行的时间戳
                consecutive_above_count
            ))

        print(f"扫描完成，共 {line_count} 行数据")
        return segments

    def _extract_all_segments(self, input_file: str, segments: List[Tuple[int, int, int, int, int]],
                              output_dir: str) -> List[str]:
        """提取所有检测到的工作区间数据"""
        output_files = []

        for seg_idx, (start_idx, end_idx, start_time, end_time, duration) in enumerate(segments):
            try:
                output_file = self._extract_single_segment(
                    input_file, start_idx, end_idx, start_time, end_time, duration,
                    output_dir, seg_idx
                )
                if output_file:
                    output_files.append(output_file)

                if (seg_idx + 1) % 10 == 0:
                    print(f"已提取 {seg_idx + 1}/{len(segments)} 个工作区间")

            except Exception as e:
                print(f"提取区间 {seg_idx} 时出错: {e}")
                continue

        return output_files

    def _extract_single_segment(self, input_file: str, start_idx: int, end_idx: int,
                                start_time: int, end_time: int, duration: int,
                                output_dir: str, segment_id: int) -> Optional[str]:
        """提取单个工作区间数据（包含上下文）"""
        # 计算基于百分比的上下文边界
        context_points = int(duration * self.context_percent)
        context_start_idx = max(0, start_idx - context_points)
        context_end_idx = end_idx + context_points

        segment_data = []
        current_line = 0

        # 生成文件名
        start_dt = datetime.fromtimestamp(start_time)
        end_dt = datetime.fromtimestamp(end_time)

        # 格式化时间字符串，用于文件名
        start_str = start_dt.strftime("%Y%m%d_%H%M%S")
        end_str = end_dt.strftime("%Y%m%d_%H%M%S")

        # 创建文件名：{电器名称_起始时间_结束时间_持续时长}
        filename = f"{self.appliance_name}_{start_str}_{end_str}_{duration}s.csv"
        output_file = os.path.join(output_dir, filename)

        # 如果文件已存在，跳过
        if os.path.exists(output_file):
            print(f"文件已存在，跳过: {filename}")
            return output_file

        print(f"提取区间 {segment_id}: {start_str} - {end_str}, 持续 {duration}秒")
        print(f"  上下文: 前后各 {context_points} 个数据点 (工作段的 {self.context_percent * 100}%)")

        with open(input_file, 'r') as f:
            # 跳过前面的行，直到上下文开始
            for _ in range(context_start_idx):
                try:
                    next(f)
                    current_line += 1
                except StopIteration:
                    break

            # 读取区间数据
            while current_line <= context_end_idx:
                line = f.readline()
                if not line:
                    break

                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        timestamp = int(parts[0])
                        power = float(parts[1])
                        segment_data.append((timestamp, power))
                    except (ValueError, IndexError):
                        pass

                current_line += 1

        # 转换为DataFrame并保存
        if segment_data:
            df = pd.DataFrame(segment_data, columns=['timestamp', 'power'])

            # 添加可读时间列
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

            # 保存CSV文件
            df.to_csv(output_file, index=False)

            # 保存区间信息文件
            info_filename = f"{self.appliance_name}_{start_str}_{end_str}_{duration}s_info.txt"
            info_file = os.path.join(output_dir, info_filename)
            with open(info_file, 'w') as f:
                f.write(f"Appliance: {self.appliance_name}\n")
                f.write(f"Segment ID: {segment_id}\n")
                f.write(f"Original data file: {input_file}\n")
                f.write(f"Start index: {start_idx}\n")
                f.write(f"End index: {end_idx}\n")
                f.write(f"Start time: {start_time}\n")
                f.write(f"End time: {end_time}\n")
                f.write(f"Duration: {duration} seconds\n")
                f.write(f"Context percentage: {self.context_percent * 100}%\n")
                f.write(f"Context points: {context_points}\n")
                f.write(f"Power threshold: {self.power_threshold}\n")
                f.write(f"Extracted points: {len(df)}\n")
                f.write(f"Start datetime: {start_dt}\n")
                f.write(f"End datetime: {end_dt}\n")

            print(f"  保存了 {len(df)} 个数据点到 {filename}")
            return output_file

        return None


def main():
    """
    主函数 - 在这里设置您的参数
    """
    # ========== 在这里设置您的参数 ==========

    # 电器名称 (用于文件命名)
    APPLIANCE_NAME = "fridge"  # 例如: fridge, washing_machine, air_conditioner

    # 输入文件路径
    INPUT_FILE = "D:\Code-Program\TimeVAE_modified\pre_data\Kettle.dat"  # 替换为您的.dat文件路径

    # 输出目录
    OUTPUT_DIR = "D:\Code-Program\TimeVAE_modified\data\Kettle"  # 替换为您想要的输出目录

    # 三个超参数
    POWER_THRESHOLD = 30.0  # 功率阈值
    MIN_DURATION_SECONDS = 30  # 最小持续时间(秒)
    CONTEXT_PERCENT = 0.2  # 上下文百分比 (0.2 表示 20%)

    # ========== 参数设置结束 ==========

    # 创建分割器实例
    segmenter = ApplianceDataSegmenter(
        appliance_name=APPLIANCE_NAME,
        power_threshold=POWER_THRESHOLD,
        min_duration_seconds=MIN_DURATION_SECONDS,
        context_percent=CONTEXT_PERCENT
    )

    # 处理数据
    try:
        output_files = segmenter.process_dataset(INPUT_FILE, OUTPUT_DIR)
        print(f"\n成功提取 {len(output_files)} 个工作区间")
        print(f"所有文件保存在: {OUTPUT_DIR}")

    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# 批量处理多个电器的函数
def batch_process_appliances(configs: List[dict]):
    """
    批量处理多个电器的数据

    Args:
        configs: 配置字典列表，每个字典包含:
            - appliance_name: 电器名称
            - input_file: 输入文件路径
            - output_dir: 输出目录
            - power_threshold: 功率阈值
            - min_duration_seconds: 最小持续时间
            - context_percent: 上下文百分比
    """
    results = {}

    for config in configs:
        print(f"\n处理电器: {config['appliance_name']}")
        print("=" * 50)

        segmenter = ApplianceDataSegmenter(
            appliance_name=config['appliance_name'],
            power_threshold=config['power_threshold'],
            min_duration_seconds=config['min_duration_seconds'],
            context_percent=config['context_percent']
        )

        try:
            output_files = segmenter.process_dataset(
                input_file=config['input_file'],
                output_dir=config['output_dir']
            )
            results[config['appliance_name']] = {
                'output_dir': config['output_dir'],
                'segments_count': len(output_files),
                'status': 'success'
            }
        except Exception as e:
            results[config['appliance_name']] = {
                'output_dir': config['output_dir'],
                'segments_count': 0,
                'status': f'error: {e}'
            }

    # 打印汇总结果
    print("\n批量处理完成!")
    print("=" * 50)
    for appliance, result in results.items():
        print(f"{appliance}: {result['segments_count']} 个区间, 状态: {result['status']}")

    return results


# 参数调优帮助函数
def find_optimal_parameters(input_file: str, sample_size: int = 10000):
    """分析数据样本，帮助找到最优参数"""
    print("分析数据样本以找到最优参数...")

    timestamps = []
    powers = []

    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break

            parts = line.strip().split()
            if len(parts) >= 2:
                try:
                    timestamp = int(parts[0])
                    power = float(parts[1])
                    timestamps.append(timestamp)
                    powers.append(power)
                except (ValueError, IndexError):
                    continue

    powers = np.array(powers)

    print(f"数据样本分析结果 ({len(powers)} 个数据点):")
    print(f"  功率范围: {np.min(powers):.2f} - {np.max(powers):.2f}")
    print(f"  平均功率: {np.mean(powers):.2f}")
    print(f"  功率中位数: {np.median(powers):.2f}")
    print(f"  功率标准差: {np.std(powers):.2f}")

    # 计算不同阈值下的工作状态比例
    thresholds = [0.1, 0.5, 1.0, 2.0, 5.0]
    for threshold in thresholds:
        above_threshold = np.sum(powers >= threshold)
        percentage = (above_threshold / len(powers)) * 100
        print(f"  阈值 {threshold}: {above_threshold} 个点高于阈值 ({percentage:.2f}%)")

    return {
        'min_power': np.min(powers),
        'max_power': np.max(powers),
        'mean_power': np.mean(powers),
        'median_power': np.median(powers)
    }


if __name__ == "__main__":
    # 首先分析数据特性
    input_file = "appliance_data.dat"  # 替换为您的文件路径

    if os.path.exists(input_file):
        stats = find_optimal_parameters(input_file)

        # 根据分析结果建议参数
        print("\n参数建议:")
        if stats['max_power'] < 10:
            print("  检测到低功率设备，建议使用较低的功率阈值 (0.1-0.5)")
        else:
            print("  检测到正常功率设备，建议使用阈值 1.0-2.0")

    # 运行主程序
    main()