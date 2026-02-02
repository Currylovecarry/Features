"""
男女比例约6:4。专业分布约为40:35:19:6。年级约为94:6
"""

import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('training_set.xlsx')

# 计算各列的百分比分布
data = {
    'Programme': df['Programme'].value_counts(normalize=True).sort_index() * 100,
    'Grade': df['Grade'].value_counts(normalize=True).sort_index() * 100,
    'Gender': df['Gender'].value_counts(normalize=True).sort_index() * 100
}

# 自定义标签名称
def get_label(category, value):
    if category == 'Gender':
        return 'Men' if value == 1 else 'Women'
    return f"{category}{value}"

# 准备绘图数据
labels = []
values = []
colors = []
color_map = {'Programme': 'skyblue', 'Grade': 'orange', 'Gender': 'lightgreen'}

for category in data:
    for value in data[category].index:
        labels.append(get_label(category, value))
        values.append(data[category][value])
        colors.append(color_map[category])

# 绘制条形图
plt.figure(figsize=(10, 5))
bars = plt.bar(labels, values, color=colors)

# 添加数值标签
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height+1,
             f'{height:.1f}%', ha='center', va='bottom')

# 图表美化
plt.title('Category Distribution (%)')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()