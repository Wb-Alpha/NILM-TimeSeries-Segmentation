# NILM-TimeSeries-Segmentation
The Semantic Segmentation for Non-invasive load monitoring


WashMachine
05有负值
06没打标
07有负值
09有负值

# 自建数据集

## 基本使用指令

### 数据标注
数据标注会给csv文件中添加两列数据，其中status为电器工作状态，on/off为电器是否工作，标签映射详见label_mapping.json和
下表。

采集到的数据按照电器类型放在./dataset/{Appliances Name}/文件夹下

运行utils.py中的data_process()函数进行数据处理，将时间戳转换为Datatime

对数据打标有两种办法：

1. 将转换后的csv文件导入label-studio中，进行标签标注，标注完毕后选择使用.json文件导出，然后使用utils.py中的
label_csv_by_json()函数将json文件中的标注信息转打在原来的csv文件上

2. python中按照[开始时间，结束时间，功能标签]写一个list,然后使用utils.py中的label_csv_by_list()打标签。

```python
label_list = [
    ['2025-08-18 22:04:00', '2025-08-18 22:33:00', 'Wash'],
    ['2025-08-18 22:33:00', '2025-08-18 23:00:00', 'Rinsing'],
]
# 记得改文件日期
label_csv_by_list('dataset/WashMachine/processed_peek_data_20250818.csv', label_list)
```

数据集标签映射

| 功能编号 | WashMachine | Air-Condition | Microwave |
|---------|-------------|---------------|-----------|
| 1       | Wash        | Freeze Auto   | Microwave |
| 2       | Rinsing     | Freeze Low Speed | Light Wave |
| 3       | Dehydrate   | Air Supply Only | Thaw      |
| 4       | Dry         | Dehumidification | Sterilize |


# UKDALE数据集

## 目录层级

building->1->appliance->metadata

dict类型，存储每个用电器的基本信息

## 常用代码
```python
# 获取一个电器的数据
ukdale = DataSet(dataset_path)
elec = ukdale.buildings[1].elec
dish_washer = elec['dish washer']
```

# 数据后处理和集成

对于新的电器数据嵌入到UKDALE里，主要有两种方法，一种是直接将新电器的起点和总干线起点对齐，然后将电器功率加总在总线上。另外一种

是用总干线减去原有的电器比如说Microwave的功率，将Microwave删除，然后再加上新的Microwave的功率。这有个问题，各个电器是1/6Hz采样率，但是总线
是1Hz。所以需要将相隔6s两个电器数据之间的总线数据都做减法。



## 附录

### Dataset所有用电器信息

#### UKDALE
-----------House1 appliances--------------
light 2
audio amplifier 1
light 8
light 5
external hard disk 1
toasted sandwich maker 1
fridge freezer 1
light 11
light 14
USB hub 1
food processor 2
kettle 1
toaster 1
clothes iron 1
audio system 1
mobile phone charger 1
computer 1
printer 1
fan 2
hair straighteners 1
laptop computer 3
water pump 1
television 1
light 1
radio 1
baby monitor 2
oven 1
ethernet switch 1
vacuum cleaner 1
light 4
light 10
light 7
wireless phone charger 1
light 13
soldering iron 1
food processor 1
light 16
immersion heater 1
computer monitor 1
fan 1
security alarm 1
charger 1
breadmaker 1
laptop computer 2
drill 1
tablet computer charger 1
radio 3
light 3
active subwoofer 1
microwave 1
broadband router 1
light 6
baby monitor 1
kitchen aid 1
desktop computer 1
washer dryer 1
light 9
light 15
light 12
coffee maker 1
boiler 1
HTPC 1
audio system 2
mobile phone charger 2
hair dryer 1
laptop computer 1
solar thermal pumping station 1
dish washer 1
radio 2
-----------House2 appliances--------------
computer monitor 1
cooker 1
games console 1
rice cooker 1
external hard disk 1
laptop computer 1
running machine 1
laptop computer 2
washing machine 1
fridge 1
kettle 1
toaster 1
computer 1
active speaker 1
modem 1
microwave 1
broadband router 1
dish washer 1
-----------House3 appliances--------------
projector 1
electric space heater 1
kettle 1
laptop computer 1
-----------House4 appliances--------------
set top box 1
television 1
light 1
DVD player 1
radio 1
boiler 1
washing machine 1
breadmaker 1
kettle 1
freezer 1
microwave 1

Process finished with exit code 0