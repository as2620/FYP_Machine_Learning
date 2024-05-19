import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the CSV file
csv_filename = "breathing_rate_data_3_5_30_40.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))

data = pd.read_csv(csv_filepath)
print("Shape of data:", data.shape)

# Drop any rows with missing data
data = data.dropna()
print("Shape of data:", data.shape)

# All Features
# features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Average Chest Breathing Rate', 
#             'Average Chest RVT', 'Average Chest Symmetry Peak-Trough', 'Average Chest Symmetry Rise-Decay', 
#             'Average Chest Inhale Time', 'Average Chest Exhale Time','Average Chest Inhale-Exhale Ratio', 
#             'Average Abdomen Breathing Rate', 'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 
#             'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Time', 'Average Abdomen Exhale Time', 
#             'Average Abdomen Inhale Exhale Ratio', 'Number of SDA Peaks', 'Average SDA Amplitudes', 
#             'Average CO2 Exhaled ', 'Average VOC Exhaled']

# Breathing Features
# features = ['Average Chest Breathing Rate', 'Average Chest RVT', 'Average Chest Symmetry Peak-Trough',  'Average Chest Symmetry Rise-Decay', 
#             'Average Chest Inhale Time', 'Average Chest Exhale Time','Average Chest Inhale-Exhale Ratio', 'Average Abdomen Breathing Rate', 
#             'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Time', 
#             'Average Abdomen Exhale Time', 'Average Abdomen Inhale Exhale Ratio']

# Non Breathing Features
# features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Number of SDA Peaks', 'Average SDA Amplitudes', 'Average CO2 Exhaled ', 'Average VOC Exhaled']

# Heart Features
features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV']

# SDA Features
# features = ['Number of SDA Peaks', 'Average SDA Amplitudes']

# CO2/VOC Features
# features = ['Average CO2 Exhaled ', 'Average VOC Exhaled']

# Separating out the features
x = data.loc[:, features].values

# Separating out the target
y = data.loc[:,['Breathing Type']].values
print(y)

# Standardizing the features
x = StandardScaler().fit_transform(x)

# Perform LDA on the data
# Compute the maximum allowed number of components
num_components = min(x.shape[1], len(np.unique(y)) - 1) 
print("Number of components:", num_components)

lda = LDA(n_components=num_components)
lda_components = lda.fit_transform(x, y)

# Print LDA eigenvectors
print("LDA Eigenvectors:")
print(lda.scalings_)

# Concatenate the LDA data with the target data
lda_data = pd.DataFrame(data=lda_components, columns=['LDA Component 1', 'LDA Component 2',
                                                       'LDA Component 3'])
print("Shape of lda_data:", lda_data.shape)
final_data = pd.concat([lda_data, data[['Breathing Type']]], axis=1)

# Save the concatenated data to CSV
final_data.to_csv("breathing_lda_data.csv", index=False)

# 1D Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('LDA Component', fontsize=15)
ax.set_title('1 Component LDA', fontsize=20)

targets = ['3bpm', '5bpm', '10bpm', '15bpm', '20bpm', '30bpm', '40bpm']
colors = ['red', 'orange', 'yellow', 'lightgreen', 'cyan', 'blue', 'mediumpurple']

for target, color in zip(targets, colors):
    indicesToKeep = final_data['Breathing Type'] == target
    ax.scatter(final_data.loc[indicesToKeep, 'LDA Component 1'], np.zeros_like(final_data.loc[indicesToKeep, 'LDA Component 1']), c=color, label=target, s=50)

ax.legend()
ax.grid()

# 2D Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('LDA Component 1', fontsize=15)
ax.set_ylabel('LDA Component 2', fontsize=15)
ax.set_title('2 Component LDA', fontsize=20)

targets = ['3bpm', '5bpm', '10bpm', '15bpm', '20bpm', '30bpm', '40bpm']
colors = ['red', 'orange', 'yellow', 'lightgreen', 'cyan', 'blue', 'mediumpurple']

for target, color in zip(targets, colors):
    indicesToKeep = final_data['Breathing Type'] == target
    ax.scatter(final_data.loc[indicesToKeep, 'LDA Component 1'], final_data.loc[indicesToKeep, 'LDA Component 2'], c=color, label=target, s=50)

ax.legend()
ax.grid()
plt.show()

# 3D Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('LDA Component 1', fontsize=15)
ax.set_ylabel('LDA Component 2', fontsize=15)
ax.set_zlabel('LDA Component 3', fontsize=15)

ax.set_title('3 Component LDA', fontsize=20)

targets = ['3bpm', '5bpm', '10bpm', '15bpm', '20bpm', '30bpm', '40bpm']
colors = ['red', 'orange', 'yellow', 'lightgreen', 'cyan', 'blue', 'mediumpurple']

for target, color in zip(targets, colors):
    indicesToKeep = final_data['Breathing Type'] == target
    ax.scatter(final_data.loc[indicesToKeep, 'LDA Component 1'], final_data.loc[indicesToKeep, 'LDA Component 2'], final_data.loc[indicesToKeep, 'LDA Component 3'], c=color, label=target, s=50)

ax.legend()
ax.grid()
plt.show()