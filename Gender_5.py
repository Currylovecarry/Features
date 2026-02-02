import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 读取数据
df = pd.read_excel('training_set.xlsx')

# 2. 计算每个学生的Q1-Q5总分
df['Total'] = df[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].sum(axis=1)

# 3. 准备热力图数据
# 计算不同Gender和Programme的平均Total成绩
heatmap_data = df.pivot_table(values='Total',
                             index='Gender',
                             columns='Programme',
                             aggfunc="mean")

# 将Gender标签改为Men/Women
heatmap_data.index = ['Women' if x == 1 else 'Men' for x in heatmap_data.index]

# 4. 绘制热力图
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data,
            annot=True,
            fmt=".1f",  # 显示1位小数
            cmap="YlGnBu",  # 颜色方案
            linewidths=.5,
            cbar_kws={'label': 'Average Total Score'})

# 5. 图表美化
plt.title('Average Total Score by Gender and Programme')
plt.xlabel('Programme')
plt.ylabel('Gender')
plt.tight_layout()
plt.show()