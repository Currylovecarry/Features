import pandas as pd
import matplotlib.pyplot as plt

dataframe = pd.read_excel('training_set.xlsx')

print(dataframe)

# Obtain the variable name of variables and labels
column_names = dataframe.columns.tolist()
class_names = column_names[0]
variable_names = column_names[1:5]
