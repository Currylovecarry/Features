import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# ------------------------------
# 🔧 1. 设置参数（这里可以自由删减 Q1~Q5）
# ------------------------------
file_path = "training_set.xlsx"
features = [ 'Q2', 'Q4','Q5',]  # 你可以删减，比如 ['Q1', 'Q2', 'Q3', 'Q4']

class_column = 'Programme'
n_components = 2 # 根据你的特征数自动调整，如最多只能设为 len(features)

# ------------------------------
# 2. 读取数据 & 处理
# ------------------------------
dataframe = pd.read_excel(file_path)
df_selected = dataframe[features + [class_column]].dropna()

# 标准化
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_selected[features])

# PCA
pca = PCA(n_components=min(n_components, len(features)), svd_solver='arpack')
df_pca = pca.fit_transform(df_scaled)

# ------------------------------
# 3. 定义你想要画的组合（按主成分索引，从1开始）
# ------------------------------
combinations = [
    (1, 2),
    (1, 3),
    (2, 3),
    (1, 4),
    (3, 4)
]

# ------------------------------
# 4. 绘制每个组合
# ------------------------------
classes = df_selected[class_column].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))
for dim_1, dim_2 in combinations:
    # 确保组合不超过当前 PCA 维度
    if dim_1 > df_pca.shape[1] or dim_2 > df_pca.shape[1]:
        print(f"跳过 Component {dim_1} vs {dim_2}，当前只计算了 {df_pca.shape[1]} 个主成分")
        continue

    plt.figure(figsize=(6, 5))
    for i, class_type in enumerate(classes):
        plt.scatter(df_pca[df_selected[class_column] == class_type, dim_1 - 1],
                    df_pca[df_selected[class_column] == class_type, dim_2 - 1],
                    color=colors[i], label=str(class_type), alpha=0.7)
    plt.xlabel(f'Component {dim_1}')
    plt.ylabel(f'Component {dim_2}')
    plt.title(f'PCA Scatter: Component {dim_1} vs {dim_2}\nFeatures used: {features}')
    plt.legend()
    plt.grid(True)
    plt.show()