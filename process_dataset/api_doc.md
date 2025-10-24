# API 文档：process_data.py

## 概述

这个 Python 模块提供了几个函数，用于处理 NILM（非侵入式负荷监测）实验中的数据集。主要功能包括：

- 文件管理（移动和重命名）
- 状态活动时间统计
- 数据清洗操作
- 交互式数据处理

---

## 函数说明

### [move_datafile_to_dataset](file://D:\KnowledgeDatabase\ComputerSecience\PROJ_NILM\nilm_experiment\process_dataset\process_data.py#L5-L54)

```python
def move_datafile_to_dataset(main_file: str, app_file: str, dataset_path, channel: int)
```


#### 功能描述

将主文件和应用程序文件复制到指定的数据集目录，并根据规则进行重命名。

#### 参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| `main_file` | str | 主文件路径 |
| `app_file` | str | 应用程序文件路径 |
| `dataset_path` | str | 目标数据集目录路径 |
| `channel` | int | 通道编号 |

#### 工作流程

1. 检查 `channel_1_org.dat` 是否存在，如果不存在则备份原始 `channel_1.dat`
2. 将 `main_file` 复制并重命名为 `channel_1.dat`
3. 将 `app_file` 复制并重命名为 `channel_{channel}.dat`

---

### `calculate_status_active_times`

```python
def calculate_status_active_times(app_df: pd.DataFrame, target_status: int)
```


#### 功能描述

计算 DataFrame 中特定状态 (`status`) 的活跃时间段信息。

#### 参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| `app_df` | pandas.DataFrame | 包含时间索引和状态列的 DataFrame |
| `target_status` | int | 目标状态值 |

#### 返回值

返回一个三元组 `(次数, 总时长, 每次活动详情)`：
- **次数**：目标状态出现的总次数
- **总时长**：目标状态累计持续时间（秒）
- **每次活动详情**：包含起止时间和持续时间的列表

#### 注意事项

要求输入的 `app_df` 必须包含名为 `'status'` 的列。

---

### [clean_data_by_list](file://D:\KnowledgeDatabase\ComputerSecience\PROJ_NILM\nilm_experiment\process_dataset\process_data.py#L114-L145)

```python
def clean_data_by_list(df: pd.DataFrame, periods_list)
```


#### 功能描述

对 DataFrame 中指定时间段内的数据执行清理操作，将其状态和功率相关字段设置为默认值。

#### 参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| [df](file://D:\KnowledgeDatabase\ComputerSecience\PROJ_NILM\nilm_experiment\elec_feature_analyze\wavelet.py#L117-L117) | pandas.DataFrame | 待处理的 DataFrame |
| `periods_list` | list of tuples | 时间段列表，每个元素为 `(start_time, end_time, duration)` 形式的元组 |

#### 清理规则

对于每个时间段内匹配的所有行：
- `'status'` 和 `'on/off status'` 设置为 0
- `'active power'` 和 `'apparent power'` 设置为 1

---

### [clean_data](file://D:\KnowledgeDatabase\ComputerSecience\PROJ_NILM\nilm_experiment\process_dataset\process_data.py#L148-L222)

```python
def clean_data(df: pd.DataFrame, status: int)
```


#### 功能描述

交互式地让用户选择需要保留的状态活动时间段，并清除其余部分的数据。

#### 参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| [df](file://D:\KnowledgeDatabase\ComputerSecience\PROJ_NILM\nilm_experiment\elec_feature_analyze\wavelet.py#L117-L117) | pandas.DataFrame | 输入数据框 |
| `status` | int | 要分析的目标状态 |

#### 工作机制

1. 显示所有符合目标状态的时间段
2. 用户可通过输入序号逐个排除不需要删除的时间段
3. 最终输出清理后的新 DataFrame

> ⚠️ 此函数会进入交互模式等待用户输入

---

### [print_all_periods](file://D:\KnowledgeDatabase\ComputerSecience\PROJ_NILM\nilm_experiment\process_dataset\process_data.py#L225-L236)

```python
def print_all_periods(df: pd.DataFrame, status_num: int)
```


#### 功能描述

打印从状态 0 到给定最大状态编号之间的所有活动周期及其统计数据。

#### 参数说明

| 参数 | 类型 | 描述 |
|------|------|------|
| [df](file://D:\KnowledgeDatabase\ComputerSecience\PROJ_NILM\nilm_experiment\elec_feature_analyze\wavelet.py#L117-L117) | pandas.DataFrame | 输入数据框 |
| `status_num` | int | 最大状态编号 |

---

## 示例调用

```python
# 示例：移动数据文件到数据集目录
main_file = './experiment_dataset/BERT4NILM/microwave/test/mains_sterilize.dat'
app_file = './experiment_dataset/BERT4NILM/microwave/test/microwave_sterilize.dat'
dataset_path = 'E:/datasets/NILM/uk_dale/house_2'
channel = 15
move_datafile_to_dataset(main_file, app_file, dataset_path, channel)

# 示例：读取 CSV 数据并清理状态为 2 的活动时段
file_path = './dataset/Microwave_Microwave.csv'
df = pd.read_csv(file_path, index_col=0)
df_after_process = clean_data(df, 2)
df_after_process.to_csv(file_path, index=True)
```


--- 

## 注意事项与限制

- 所有涉及日期的操作都假定索引是 `%Y-%m-%d %H:%M:%S` 格式的字符串。
- 在使用 [clean_data()](file://D:\KnowledgeDatabase\ComputerSecience\PROJ_NILM\nilm_experiment\process_dataset\process_data.py#L148-L222) 进行交互式操作前，请确保运行环境支持标准输入。
- 所有文件操作基于本地磁盘路径，需保证程序具有相应权限。


# API 文档：vae_data_process.py