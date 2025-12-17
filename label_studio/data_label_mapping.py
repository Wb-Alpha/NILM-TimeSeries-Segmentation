from label_studio_sdk import Client
import csv
from pathlib import Path
import uuid
from datetime import datetime

# -------------------------- 配置参数（需根据实际修改） --------------------------
LS_URL = "http://localhost:8080"  # Label Studio地址
LS_API_KEY = "f484abe046ed8ed77fc6f51b1a8c7914801670ac"  # 替换为你的API Key
PROJECT_NAME = "NILM_Segmentation"  # 替换为你的项目名称
DATA_FOLDER = Path("../ukdale_disaggregate/active/washing_machine")  # data文件夹路径
LABEL_FOLDER = Path("../ukdale_disaggregate/cps/washing_machine")  # label文件夹路径
LABEL_COLUMN = "datetime"  # label文件中存储时间戳的列名（如你的label文件列名是"datetime"则修改）
ANNOTATION_LABEL = "Change"  # 标注标签名（需与模板中的Label value一致）
# ------------------------------------------------------------------------------

# 连接Label Studio客户端
ls = Client(url=LS_URL, api_key=LS_API_KEY)
ls.check_connection()  # 验证连接是否成功

# 获取目标项目（通过项目名称）
projects = ls.get_projects()
target_project = next((p for p in projects if p.title == PROJECT_NAME), None)
if not target_project:
    raise ValueError(f"未找到项目：{PROJECT_NAME}")
project = target_project

# 遍历data文件夹中的所有CSV文件
data_files = list(DATA_FOLDER.glob("*.csv"))
print(f"发现{len(data_files)}个数据文件，开始批量导入...")

for idx, data_file in enumerate(data_files, 1):
    # 1. 获取对应label文件的路径（label_xxx.csv）
    label_filename = f"Changepoints_{data_file.name}"
    label_file = LABEL_FOLDER / label_filename

    # 跳过无对应label的文件
    if not label_file.exists():
        print(f"[{idx}/{len(data_files)}] 跳过 {data_file.name}：未找到对应label文件 {label_filename}")
        continue

    # try:
    # 2. 导入data.csv到Label Studio，生成Task
    print(f"[{idx}/{len(data_files)}] 导入数据：{data_file.name}")
    import_result = project.import_tasks([
        {
            "csv": f"file://{data_file.absolute()}",  # 本地文件路径（Label Studio需可访问）
            "meta": {"original_filename": data_file.name}  # 存储原文件名，便于后续排查
        }
    ])
    if not import_result:
        print(f"!  {data_file.name} 导入失败，跳过")
        continue
    task_id = import_result[0]["id"]  # 获取生成的Task ID
    # 3. 读取label文件中的时间戳（单条或多条）
    with open(label_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        label_rows = list(reader)  # 读取所有label行（支持单个文件多个时间戳）
    if not label_rows:
        print(f"!  {label_filename} 无数据，跳过标注导入")
        continue
    # 4. 构造Label Studio标注格式（start=end的零长度区间）
    annotation_results = []
    for row in label_rows:
        target_timestamp = row[LABEL_COLUMN]  # 获取label中的时间戳
        annotation_results.append({
            "id": str(uuid.uuid4()),  # 唯一ID
            "type": "timeserieslabels",  # 标注类型（与模板组件一致）
            "value": {
                "start": str(target_timestamp),  # 开始时间=目标时间戳
                "end": str(target_timestamp),  # 结束时间=开始时间（零长度区间→垂直线）
                "timeserieslabels": [ANNOTATION_LABEL]  # 标注标签
            },
            "from_name": "ts-label",  # 与模板中TimeSeriesLabels的name一致
            "to_name": "ts",  # 与模板中TimeSeries的name一致
            "origin": "prediction",  # 标记为"prediction"
            "readonly": False  # 可编辑
        })
    # 5. 将标注导入对应Task
    project.create_annotation(
        task_id=task_id,
        result=annotation_results,
        ground_truth=True,  # 设为True表示是真实标签（根据需求调整）
        created_at=datetime.utcnow().isoformat() + "Z"
    )
    print(f"✅ {data_file.name} 导入完成，关联{len(label_rows)}个标注点")

    # except Exception as e:
    #     print(f"Failed to label file: {data_file.name} 处理失败：{str(e)}")
    #     continue

print("\n批量导入完成！")
