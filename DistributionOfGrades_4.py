import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 读取数据
df = pd.read_excel('training_set.xlsx')

# 计算总成绩
df['Total_Score'] = df['Q1'] + df['Q2'] + df['Q3'] + df['Q4'] + df['Q5']

# 映射专业代码为专业名称
programme_mapping = {
    1: 'Programme 1',
    2: 'Programme 2',
    3: 'Programme 3',
    4: 'Programme 4'
}
df['Programme'] = df['Programme'].map(programme_mapping)

# 绘制箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x='Programme', y='Total_Score', data=df)
plt.title('Distribution of Total Scores by Programme')
plt.xlabel('Programme')
plt.ylabel('Total Score (Q1+Q2+Q3+Q4+Q5)')
plt.grid(True)
plt.show()