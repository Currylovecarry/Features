import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# 读取数据
dataframe = pd.read_excel('training_set.xlsx')

class_1 = 'Programme'
x_axis_1 = 'Programme'
# 计算 Q1 到 Q5 的和，并作为新的 Y 轴数据
dataframe['Q1_to_Q5_sum'] = dataframe[['Q1', 'Q2', 'Q3', 'Q4', 'Q5']].sum(axis=1)
y_axis_1 = 'Q1_to_Q5_sum'  # 使用 Q1 到 Q5 的和作为 Y 轴

# 强制按 1,2,3,4 顺序排列（假设 Programme 是数值类型）
classes_1 = sorted(dataframe[class_1].unique())
# 如果 Programme 是字符串类型（如 "1","2"），使用以下代码：
# classes_1 = sorted(dataframe[class_1].unique(), key=lambda x: int(x))

colors_1 = plt.cm.rainbow(np.linspace(0, 1, len(classes_1)))  # generate a color for each class

# Create a mapping from Programme to x-axis positions
programme_order = {prog: i for i, prog in enumerate(classes_1)}

# Add jitter to x-axis positions
jitter_strength = 0.2  # adjust this value to control the amount of jitter

for i, class_type in enumerate(classes_1):
    # Get the base x position from the mapping
    x_base = programme_order[class_type]

    # Create jitter by adding random noise
    x_jittered = x_base + np.random.uniform(-jitter_strength, jitter_strength,
                                            size=len(dataframe.loc[dataframe[class_1] == class_type]))

    plt.scatter(
        x_jittered,
        dataframe.loc[dataframe[class_1] == class_type, y_axis_1],
        color=colors_1[i],  # define the color for each class in the scatter plot
        alpha=0.7,  # slightly transparent to better see overlapping points
        label=class_type  # add label for legend
    )

# Set x-ticks to show the Programme names at their original positions
plt.xticks(range(len(classes_1)), classes_1)
plt.xlabel(x_axis_1)
plt.ylabel(y_axis_1)
plt.title('Scatter plot of ' + x_axis_1 + ' vs ' + y_axis_1 + ' (with jitter)')
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(title='Programme')  # add legend to show Programme categories
plt.show()