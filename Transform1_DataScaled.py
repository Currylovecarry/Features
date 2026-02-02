from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_excel('training_set.xlsx')

variable_names = ['Programme','Gender','Grade','Q1', 'Q2', 'Q3', 'Q4', 'Q5']

# 标准化数据
scaler = StandardScaler()
scaled_values = scaler.fit_transform(dataframe[variable_names])

# 将标准化后的数据转换为数据框
scaled_df = pd.DataFrame(scaled_values, columns=variable_names)

# 绘制标准化后的箱线图
plt.figure(figsize=(10, 6))
scaled_df.boxplot(grid=False)
plt.title('Box plot comparing the distributions of scaled features')
plt.show()