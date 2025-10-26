import pandas as pd
import fluss
time_series = pd.read_csv('../process_dataset/Microwave/Air_condition.csv')
# 确保 'active power' 列是数值类型，非数值数据会被转换为 NaN
time_series['active power'] = pd.to_numeric(time_series['active power'], errors='coerce')

# 删除或填充 NaN 值（这里选择删除）
ts = time_series.dropna(subset=['active power'])
ts = time_series.head(6000)
ts = ts['active power'].values  # 提取数值数组

fluss.fluss(ts, window_size=600, n_regimes=10, excl_factor=0.5)
