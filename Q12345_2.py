import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_excel('training_set.xlsx')

print(dataframe)

#draw picture 箱线图

column_names = dataframe.columns.tolist()#获取数据的列名 并将其转换为一个列表 column_names
class_names = column_names[0]#将列表中的第一个列名赋值给 class_names
variable_names = column_names[0:8]#将列表中第2到第5个列名赋值给 variable_names


plt.figure(figsize=(10, 6))#图大小不用管
dataframe[variable_names].boxplot(grid=False)
plt.title('Box plot comparing the distributions of each variable')
plt.show()