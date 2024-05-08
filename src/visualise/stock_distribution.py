import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data from CSV file
file_path = '../visualise/historical_stock_distribution.csv'
data = pd.read_csv(file_path)


# set aesthetic of the plots
sns.set_style("whitegrid")

# create figure to hold the subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# bar chart, sector distribution
sector_counts = data['Sector'].value_counts()
sns.barplot(x=sector_counts.values, y=sector_counts.index, ax=axes[0, 0])
axes[0, 0].set_title('Sector Distribution')
axes[0, 0].set_xlabel('Frequency')
axes[0, 0].set_ylabel('Sector')

# histogram, accuracy distribution
sns.histplot(data['Accuracy'], bins=30, kde=True, ax=axes[0, 1])
axes[0, 1].set_title('Accuracy Distribution')
axes[0, 1].set_xlabel('Accuracy')
axes[0, 1].set_ylabel('Frequency')

# boxplot, accuracy by sector
sns.boxplot(x='Accuracy', y='Sector', data=data, ax=axes[1, 0])
axes[1, 0].set_title('Accuracy by Sector')
axes[1, 0].set_xlabel('Accuracy')
axes[1, 0].set_ylabel('Sector')

# scatter plot, accuracy by sector
sns.stripplot(x='Accuracy', y='Sector', data=data, jitter=True, ax=axes[1, 1])
axes[1, 1].set_title('Scatter Chart of Accuracy by Sector')
axes[1, 1].set_xlabel('Accuracy')
axes[1, 1].set_ylabel('Sector')

# tight layout, display the plots
plt.tight_layout()
plt.show()
