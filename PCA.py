import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 读取数据
file_path = "training_set.xlsx"  # 请替换为你的实际文件路径
df = pd.read_excel(file_path)

# 选择数值列进行 PCA（去除 Programme, Gender, Grade）
numeric_columns = [col for col in df.columns if col not in ["Programme", "Gender", "Grade"]]
df_numeric = df[numeric_columns]
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# 执行 PCA 分析
pca = PCA(n_components=4, svd_solver='arpack')
df_pca = pca.fit_transform(df_scaled)

# 绘制 PCA 系数分布图
plt.figure(figsize=(8, 5))
plt.boxplot(df_pca)
plt.title("Distribution of PCA Coefficients")
plt.show()

# 选择要绘制的 PCA 维度
dim_1 = 0  # 第一主成分
dim_2 = 1  # 第二主成分

# 获取分类标签（假设某列是分类列，需替换 class_column）
class_column = "Programme"  # 请替换为数据集中的分类列名
classes = df[class_column].unique()
colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))

# 绘制 PCA 结果散点图
plt.figure(figsize=(8, 6))
for i, class_type in enumerate(classes):
    plt.scatter(df_pca[df[class_column] == class_type, dim_1],
                df_pca[df[class_column] == class_type, dim_2],
                color=colors[i], label=str(class_type))

plt.xlabel(f'Component {dim_1 + 1}')
plt.ylabel(f'Component {dim_2 + 1}')
plt.title(f'Scatter plot of Component {dim_1 + 1} vs Component {dim_2 + 1} in PCA')
plt.legend()
plt.show()

# 输出 PCA 主要成分的系数
pca_components = pd.DataFrame(
    [pca.components_[dim_1], pca.components_[dim_2]],
    columns=df_numeric.columns,
    index=[f'Component {dim_1 + 1}', f'Component {dim_2 + 1}']
)
print(pca_components)