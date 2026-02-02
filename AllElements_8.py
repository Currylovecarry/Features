import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_excel('training_set.xlsx')

# 计算总成绩
df['Total_Grade'] = df['Q1'] + df['Q2'] + df['Q3'] + df['Q4'] + df['Q5']

# 选择相关列
corr_df = df[['Programme', 'Gender', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Total_Grade']]

# 计算相关系数矩阵
corr_matrix = corr_df.corr()

# 绘制热力图
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt=".2f")
plt.title('Correlation Heatmap (Including Programme for Prediction)')
plt.show()