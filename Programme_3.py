import pandas as pd
import matplotlib.pyplot as plt

# 读取数据
dataframe = pd.read_excel('training_set.xlsx')

# 计算Q1-Q5总分
dataframe['Total'] = dataframe[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].sum(axis=1)

# 绘制专业(Programme)的总分箱线图
plt.figure(figsize=(10, 6))
dataframe.boxplot(column='Total', by='Programme', grid=False)
plt.title('Total Score Distribution by Programme')
plt.suptitle('')  # 移除自动生成的副标题
plt.xlabel('Programme')
plt.ylabel('Total Score')
plt.show()