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
print("Shape of data:", data.shape)

# All Features
# features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Average Chest Breathing Rate', 
#             'Average Chest RVT', 'Average Chest Symmetry Peak-Trough', 'Average Chest Symmetry Rise-Decay', 
#             'Average Chest Inhale Time', 'Average Chest Exhale Time','Average Chest Inhale-Exhale Ratio', 
#             'Average Abdomen Breathing Rate', 'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 
#             'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Time', 'Average Abdomen Exhale Time', 
#             'Average Abdomen Inhale Exhale Ratio', 'Number of SDA Peaks', 'Average SDA Amplitudes', 
#             'Average CO2 Exhaled', 'Average VOC Exhaled']

# Breathing Features
# features = ['Average Chest Breathing Rate', 'Average Chest RVT', 'Average Chest Symmetry Peak-Trough',  'Average Chest Symmetry Rise-Decay', 
#             'Average Chest Inhale Time', 'Average Chest Exhale Time','Average Chest Inhale-Exhale Ratio', 'Average Abdomen Breathing Rate', 
#             'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Time', 
#             'Average Abdomen Exhale Time', 'Average Abdomen Inhale Exhale Ratio']

# Non Breathing Features
# features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Number of SDA Peaks', 'Average SDA Amplitudes', 'Average CO2 Exhaled', 'Average VOC Exhaled']

# Heart Features
# features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV']

# SDA Features
# features = ['Number of SDA Peaks', 'Average SDA Amplitudes']

# CO2/VOC Features
# features = ['Average CO2 Exhaled', 'Average VOC Exhaled']

# Features from Histogram Analysis
# features = ['Heart Rate', 'HRV', 'Average Chest Symmetry Rise-Decay', 'Average Chest Inhale Time', 'Average Chest Exhale Time','Average Chest Inhale-Exhale Ratio',
#             'Average CO2 Exhaled', 'Average VOC Exhaled']

# Features for Classification
features = ['Heart Rate', 'Average Systolic Amplitude', 'HRV', 'Average Chest Breathing Rate', 'Average Chest RVT', 'Average Chest Symmetry Rise-Decay', 
            'Average Chest Inhale Time', 'Average Chest Inhale-Exhale Ratio', 'Average Abdomen RVT', 'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Exhale Ratio', 
            'Number of SDA Peaks', 'Average SDA Amplitudes', 'Average CO2 Exhaled', 'Average VOC Exhaled']

# Specify the desired order for the classes
class_order = ['Rest', 'S1', 'S2']

# Outlier Removal for each feature for each class
for feature in features:
    for class_value in class_order:
        # Filter data for the current class
        class_data = data[data['Breathing Type'] == class_value][feature]
        # Calculate Q1, Q3, and IQR
        Q1 = class_data.quantile(0.25)
        Q3 = class_data.quantile(0.75)
        IQR = Q3 - Q1
        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Remove outliers
        data = data.drop(data[(data['Breathing Type'] == class_value) & ((data[feature] < lower_bound) | (data[feature] > upper_bound))].index)

# Drop any rows with missing data
data = data.dropna()

# Print the shape of the data after outlier removal
print("Shape of data after outlier removal:", data.shape)

# Separating out the features
x = data.loc[:, features].values

# Separating out the target
y = data.loc[:,['Breathing Type']].values.ravel()  # Using ravel() to convert to 1d array

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
lda_data = pd.DataFrame(data=lda_components, columns=['LDA Component ' + str(i+1) for i in range(num_components)])
final_data = pd.concat([lda_data, data.reset_index(drop=True)[['Breathing Type']]], axis=1)

# Print the LDA data
print(lda_data)

# Plot 1D LDA
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('LDA Component 1', fontsize=15)
ax.set_title('1 Component LDA', fontsize=20)

targets = class_order
colors = ["#ffa500", "#79c314", "#36cedc"]

for target, color in zip(targets, colors):
    indicesToKeep = final_data['Breathing Type'] == target
    ax.scatter(final_data.loc[indicesToKeep, 'LDA Component 1'], np.zeros_like(final_data.loc[indicesToKeep, 'LDA Component 1']), c=color, label=target, s=50)

ax.legend()
ax.grid()
plt.show()

# Plot 2D LDA
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('LDA Component 1', fontsize=15)
ax.set_ylabel('LDA Component 2', fontsize=15)
ax.set_title('2 Component LDA', fontsize=20)

for target, color in zip(targets, colors):
    indicesToKeep = final_data['Breathing Type'] == target
    ax.scatter(final_data.loc[indicesToKeep, 'LDA Component 1'], final_data.loc[indicesToKeep, 'LDA Component 2'], c=color, label=target, s=50)

ax.legend()
ax.grid()
plt.show()

# Plot 3D LDA
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('LDA Component 1', fontsize=15)
ax.set_ylabel('LDA Component 2', fontsize=15)
ax.set_zlabel('LDA Component 3', fontsize=15)
ax.set_title('3 Component LDA', fontsize=20)

for target, color in zip(targets, colors):
    indicesToKeep = final_data['Breathing Type'] == target
    ax.scatter(final_data.loc[indicesToKeep, 'LDA Component 1'], final_data.loc[indicesToKeep, 'LDA Component 2'], final_data.loc[indicesToKeep, 'LDA Component 3'], c=color, label=target, s=50)

ax.legend()
ax.grid()
plt.show()
