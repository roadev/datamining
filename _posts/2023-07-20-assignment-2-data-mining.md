---
layout: post
title: Data Loading and Preliminary Analysis
subtitle: Loading the data and identifying missing values and data types
gh-repo: your-github-username/your-repo-name
gh-badge: [star, fork, follow]
tags: [data-analysis]
comments: true
---

```python
# Preliminary Analysis

import pandas as pd
import matplotlib.pyplot as plt

def load_and_analyze_data(path):
    df = pd.read_csv(path)
    
    # Display the percentage of missing values in each column
    missing_values_percentage = df.isnull().mean() * 100
    print(f'Percentage of missing values in {path.split("/")[-1]}:')
    print(missing_values_percentage)
    
    # Display the data type of each column
    data_types = df.dtypes
    print(f'\nData types of the columns in {path.split("/")[-1]}:')
    print(data_types)
    
    return df

df_tableA = load_and_analyze_data('tableA/tableA.csv')
df_tableB = load_and_analyze_data('tableB/tableB.csv')
```



{: .box-note}
**Note:** The csv files were loaded into pandas dataframes. We then performed a preliminary analysis on the datasets.
This included identifying missing values and understanding the data types of each column.

Here's a code chunk:

~~~python
# Display the percentage of missing values in each column
missing_values_percentage = df.isnull().mean() * 100
print(f'Percentage of missing values in {path.split("/")[-1]}:')
print(missing_values_percentage)
    
# Display the data type of each column
data_types = df.dtypes
print(f'\nData types of the columns in {path.split("/")[-1]}:')
print(data_types)
~~~

{: .box-note}
**Note:** For columns of type 'object' (strings), we calculated the average, minimum, and maximum lengths.

Here's a code chunk:

~~~python
# If a column is of object type (string), report the average, minimum, and maximum length
for column in df.select_dtypes(include='object').columns:
    average_length = df[column].str.len().mean()
    minimal_length = df[column].str.len().min()
    maximal_length = df[column].str.len().max()

    print(f"\nFor column '{column}' in {df_name}:")
    print(f"Average length: {average_length:.3f}")
    print(f"Minimal length: {minimal_length}")
    print(f"Maximal length: {maximal_length}")
~~~

## Data Visualization and Outlier Detection

{: .box-note}
**Note:** Histograms were created for the 'title' and 'abstract' columns of both dataframes to get an understanding of the distribution of lengths and to identify potential outliers.

Here's a code chunk:

~~~python
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(df['title'].str.len(), bins=20, color='skyblue', edgecolor='black')
plt.title(f'Histogram of Title Lengths in {df_name}')
plt.xlabel('Length')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(df['abstract'].str.len(), bins=20, color='skyblue', edgecolor='black')
plt.title(f'Histogram of Abstract Lengths in {df_name}')
plt.xlabel('Length')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
~~~

## Outlier Identification

{: .box-note}
**Note:** Outliers were identified in the 'ID' column using the Interquartile Range (IQR) method.

Here's a code chunk:

~~~python
Q1 = df['ID'].quantile(0.25)
Q3 = df['ID'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds for outliers
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Identify outliers
outliers = df[(df['ID'] < lower_bound) | (df['ID'] > upper_bound)]
print(f"Outliers in {df_name}")
print(outliers)
~~~

## Comparing Distributions Between Datasets

{: .box-note}
**Note:** Box plots were created to compare the distributions of the 'title' column in both dataframes. These visualizations help to understand if there are significant differences between the datasets.

Here's a code chunk:

~~~python
data_to_plot = [df1['title'], df2['title']]

# Create a figure instance
fig = plt.figure(figsize=(10, 5))

# Create an axes instance
ax = fig.add_axes([0, 0, 1, 1])

# Create the boxplot
bp = ax.boxplot(data_to_plot, patch_artist=True, notch=True, vert=0)

colors = ['#0000FF', '#00FF00']

for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

# Changing color and linewidth of whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#8B008B', linewidth=1.5)

# Changing color and linewidth of caps
for cap in bp['caps']:
    cap.set(color ='#8B008B', linewidth = 2)

# Changing color and linewidth of medians
for median in bp['medians']:
    median.set(color='red', linewidth=3)

# x-axis labels
ax.set_yticklabels(['tableA', 'tableB'])

# Adding title
plt.title('Comparison of Title Distribution between TableA and TableB')

# Removing top axes and right axes ticks
ax.get_xaxis().tick_bottom()
ax.get_yaxis().tick_left()

# Show the plot
plt.show()
~~~
