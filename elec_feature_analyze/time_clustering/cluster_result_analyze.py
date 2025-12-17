import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ===============CONFIG=======================

# ================DeSTEC CONFIG=======================
CLUSTER_RESULT_FILE = f'cluster_data/detsec_clust_assignment.npy'
DATA_MAPPING_LIST = f'cluster_data/data_mapping_list.json'
ORIGINAL_DATA_FILE = f'cluster_data/data.npy'
SEQ_LEN_FILE = f'cluster_data/seq_length.npy'


def read_detsec_result():
    """
    读取DeTSEC的运行结果，并且进行结果映射
    :return:
    """
    cluster_result = np.load(CLUSTER_RESULT_FILE)
    data_len = np.load(SEQ_LEN_FILE)
    data = np.load(ORIGINAL_DATA_FILE)
    with open(DATA_MAPPING_LIST, 'r', encoding='utf-8') as file:
        data_info_list = json.load(file)

    print(cluster_result)
    cluster_dict = {}
    for i, data_info in enumerate(data_info_list):
        # add cluster res and original data
        data_info['cluster_id'] = cluster_result[i]
        data_info['data'] = pd.DataFrame(data[i][:data_len[i]])
        # create list if not exist (k,v)
        if cluster_result[i] not in cluster_dict:
            cluster_dict[cluster_result[i]] = []
        cluster_dict[cluster_result[i]].append(data_info)

    return data_info_list, cluster_dict


def cluster_result_analyze(data_info_list, cluster_dict):
    user_input = input("请输入命令:\n- show:遍历展示指定簇的数据 ")

    while user_input != 'e':
        if user_input == 'show':
            cluster_id = input("请输入簇ID: ")
            cluster_list = cluster_dict[int(cluster_id)]

            for i, item in enumerate(cluster_list):
                # 打印数据信息
                print(f"\n数据项 {i + 1}/{len(cluster_list)}")
                print(f"数据文件: {item.get('data_file', 'N/A')}")
                print(f"开始时间: {item.get('start_time', 'N/A')}")
                print(f"结束时间: {item.get('end_time', 'N/A')}")

                # 可视化数据
                plt.figure(figsize=(10, 6))
                plt.plot(item['data'])
                plt.title(f"Cluster {cluster_id} - Item {i + 1}")
                plt.xlabel("Time")
                plt.ylabel("Value")
                plt.show()

                # 等待用户输入继续
                input("按回车键查看下一个数据项...")
            print(f"簇 {cluster_id} 的所有数据项已展示完毕")
        else:
            print("无效的输入")


if __name__ == '__main__':
    data_info_list, cluster_dict = read_detsec_result()
    cluster_result_analyze(data_info_list, cluster_dict)
