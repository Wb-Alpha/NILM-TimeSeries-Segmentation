import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# 生成4个簇：2个大簇（各200点），2个小簇（各50点）
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

X, y_true = make_blobs(n_samples=[200, 250, 50, 10],
                       centers=None,
                       cluster_std=[0.8, 0.8, 0.6, 0.6],
                       random_state=42)

# 减少噪声点（从50个减到20个）
rng = np.random.RandomState(42)
noise = rng.uniform(low=-6, high=6, size=(10, 2))
X = np.vstack([X, noise])

# 标准化数据
X = StandardScaler().fit_transform(X)

# 调整DBSCAN参数
dbscan = DBSCAN(eps=0.35, min_samples=5)
labels = dbscan.fit_predict(X)

# # 可视化
# plt.figure(figsize=(13, 5))
#
# # 子图1：原始数据
# plt.subplot(1, 2, 1)
# plt.scatter(X[:, 0], X[:, 1], c='gray', s=25, alpha=0.6)
# plt.title('原始数据分布\n(2个大簇 + 2个小簇)', fontsize=13)
# plt.xlabel('特征 1')
# plt.ylabel('特征 2')
# plt.grid(True, alpha=0.3)

# 子图2：DBSCAN聚类结果
plt.plot()
unique_labels = np.unique(labels)

# 修复颜色索引问题：单独处理噪声点
n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
cluster_colors = plt.cm.Spectral(np.linspace(0, 1, max(n_clusters, 1)))

# 使用独立的颜色索引计数器
color_idx = 0
for label in unique_labels:
    mask = (labels == label)
    if label == -1:
        # 噪声点
        color = 'k'
        label_name = f'Noise)'
        marker = 'x'
        size = 35
    else:
        # 正常簇：按顺序取颜色
        color = cluster_colors[color_idx]
        label_name = f'Cluster {label} )'
        marker = 'o'
        size = 45
        color_idx += 1

    plt.scatter(X[mask, 0], X[mask, 1],
                c=color, marker=marker, s=size,
                label=label_name, alpha=0.7, edgecolors='k', linewidth=0.5)

plt.title('PCA\n', fontsize=13)
plt.xlabel('feature 1')
plt.ylabel('feature 2')
plt.legend(loc='best', fontsize=9, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 输出统计信息
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"识别出的簇数量: {n_clusters}")
print(f"噪声点数量: {n_noise}")
for label in sorted(set(labels)):
    if label != -1:
        print(f"簇 {label} 样本数: {list(labels).count(label)}")