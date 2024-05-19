import pandas as pd 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the CSV file
csv_filename = "stress_data.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))

data = pd.read_csv(csv_filepath)

# Drop any rows with missing data
data = data.dropna()

features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Average Chest Breathing Rate', 
            'Average Chest RVT', 'Average Chest Symmetry Peak-Trough', 'Average Chest Symmetry Rise-Decay', 
            'Average Abdomen Breathing Rate', 'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 
            'Number of SDA Peaks', 'Average SDA Amplitudes', 'Average CO2 Exhaled ', 'Average VOC Exhaled']

num_features = len(features)

# Separating out the features
x = data.loc[:, features].values

# Separating out the target
y = data.loc[:,['Breathing Type']].values

# Standardize the data (Mean = 0, Variance = 1)
x = StandardScaler().fit_transform(x)

# Perform LDA on the data
# Compute the maximum allowed number of components
num_components = min(x.shape[1], len(np.unique(y)) - 1) 
lda = LDA(n_components=num_components)
lda_components = lda.fit_transform(x, y)

# Concatenate the LDA data with the target data
lda_data = pd.DataFrame(data=lda_components, columns=['LDA Component 1', 'LDA Component 2'])
print("Shape of lda_data:", lda_data.shape)
final_data = pd.concat([lda_data, data[['Breathing Type']]], axis=1)

# Save the concatenated data to CSV
final_data.to_csv("breathing_lda_data.csv", index=False)

# 1D Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('LDA Component', fontsize=15)
ax.set_title('1 Component LDA', fontsize=20)

targets = ['Rest', 'S1', 'S2']
colors = ['lightgreen', 'blue', 'mediumpurple']

for target, color in zip(targets, colors):
    indicesToKeep = final_data['Breathing Type'] == target
    ax.scatter(final_data.loc[indicesToKeep, 'LDA Component 1'], np.zeros_like(final_data.loc[indicesToKeep, 'LDA Component 1']), c=color, label=target, s=50)

ax.legend()
ax.grid()
# plt.show()

# 2D Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('LDA Component 1', fontsize=15)
ax.set_ylabel('LDA Component 2', fontsize=15)
ax.set_title('2 Component LDA', fontsize=20)

targets = ['Rest', 'S1', 'S2']
colors = ['lightgreen', 'blue', 'mediumpurple']

for target, color in zip(targets, colors):
    indicesToKeep = final_data['Breathing Type'] == target
    ax.scatter(final_data.loc[indicesToKeep, 'LDA Component 1'], final_data.loc[indicesToKeep, 'LDA Component 2'], c=color, label=target, s=50)

ax.legend()
ax.grid()
plt.show()