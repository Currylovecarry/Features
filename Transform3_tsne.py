import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# 读取数据
file_path = "training_set.xlsx"  # 请替换为你的实际文件路径
df = pd.read_excel(file_path)

# 选择数值列进行 t-SNE（去除 Programme, Gender, Grade）
numeric_columns = [col for col in df.columns if col not in ["Programme", "Gender", "Grade"]]
df_numeric = df[numeric_columns]

# 标准化数据
df_scaled = StandardScaler().fit_transform(df_numeric)

# 执行 t-SNE 降维
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(df_scaled)

# 选择分类变量进行着色（假设分类列为 Grade）
class_column = "Programme"  # 你可以更改为其他分类列
classes = df[class_column].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

# 绘制 t-SNE 结果散点图
plt.figure(figsize=(8, 6))
for i, class_type in enumerate(classes):
    indices = df[class_column] == class_type
    plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], color=colors[i], label=str(class_type))

plt.xlabel('t-SNE Feature 1')
plt.ylabel('t-SNE Feature 2')
plt.title('t-SNE visualization of the data')
plt.legend()
plt.show()
