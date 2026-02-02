import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取数据
dataframe = pd.read_excel('training_set.xlsx')

# 计算 Q1 到 Q5 的总和
dataframe['Q1_to_Q5_sum'] = dataframe[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].sum(axis=1)

# 定义分类变量和坐标轴
class_col = 'Programme'  # 分类列
x_axis = 'Programme'  # X轴数据（分类变量）
y_axis = 'Q1_to_Q5_sum'  # Y轴数据（总分）

# 标准化处理
scaler = StandardScaler()
dataframe['Q1_to_Q5_sum_standardized'] = scaler.fit_transform(dataframe[[y_axis]])

# 强制按1,2,3,4顺序获取唯一类别（假设Programme是数值型）
classes = sorted(dataframe[class_col].unique())
# 如果Programme是字符串型数字（如"1","2"），改用：
# classes = sorted(dataframe[class_col].unique(), key=lambda x: int(x))

colors = plt.cm.rainbow(np.linspace(0, 1, len(classes)))  # 生成颜色

# 创建映射：Programme到x轴位置
programme_order = {prog: i for i, prog in enumerate(classes)}
jitter_strength = 0.15  # 控制jitter强度

# 创建图形
plt.figure(figsize=(10, 6))

# 绘制带jitter的散点图
for i, class_type in enumerate(classes):
    # 获取基准x位置并添加jitter
    x_base = programme_order[class_type]
    x_jittered = x_base + np.random.uniform(-jitter_strength, jitter_strength,
                                            size=len(dataframe[dataframe[class_col] == class_type]))

    plt.scatter(
        x_jittered,
        dataframe.loc[dataframe[class_col] == class_type, 'Q1_to_Q5_sum_standardized'],
        color=colors[i],
        label=f'Programme {class_type}',
        alpha=0.7,
        edgecolor='w',  # 添加白色边缘更清晰
        s=60  # 控制点的大小
    )

# 添加标签和标题
plt.xlabel('Programme', fontsize=12)
plt.ylabel('Standardized Q1-Q5 Total Score', fontsize=12)
plt.title('Standardized Total Score Distribution by Programme (with Jitter)', fontsize=14)

# 优化图形显示
plt.xticks(ticks=range(len(classes)), labels=classes)  # 确保刻度与Programme顺序一致
plt.legend(title='Programme', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, linestyle='--', alpha=0.6)

# 调整坐标轴范围，给jitter留出空间
plt.xlim(-0.5, len(classes) - 0.5)
plt.tight_layout()

plt.show()