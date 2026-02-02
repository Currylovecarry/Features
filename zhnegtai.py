import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Load the data
df = pd.read_excel('training_set.xlsx')

# List of variables to plot
variables = ['Programme', 'Gender', 'Grade', 'Q1', 'Q2', 'Q3', 'Q4', 'Q5']

# Create a figure with subplots
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(15, 20))
fig.suptitle('Normal Distribution Plots for Each Variable', fontsize=16)

# Flatten the axes array for easy iteration
axes = axes.flatten()

for i, var in enumerate(variables):
    ax = axes[i]

    # Plot histogram with density curve
    sns.histplot(df[var], kde=True, stat='density', ax=ax, color='skyblue')

    # Plot normal distribution curve
    mu, std = df[var].mean(), df[var].std()
    xmin, xmax = ax.get_xlim()
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, label='Normal dist.')

    # Add title and labels
    ax.set_title(f'Distribution of {var}')
    ax.set_xlabel(var)
    ax.set_ylabel('Density')
    ax.legend()

# Adjust layout and show plot
plt.tight_layout()
plt.show()