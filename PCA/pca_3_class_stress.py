import pandas as pd 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the CSV file
csv_filename = "stress_data.csv"
# csv_filename = "breathing_rate_data.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))

data = pd.read_csv(csv_filepath)

# All Features
features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Average Chest Breathing Rate', 
            'Average Chest RVT', 'Average Chest Symmetry Peak-Trough', 'Average Chest Symmetry Rise-Decay', 
            'Average Chest Inhale Time', 'Average Chest Exhale Time','Average Chest Inhale-Exhale Ratio', 
            'Average Abdomen Breathing Rate', 'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 
            'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Time', 'Average Abdomen Exhale Time', 
            'Average Abdomen Inhale Exhale Ratio', 'Number of SDA Peaks', 'Average SDA Amplitudes', 
            'Average CO2 Exhaled', 'Average VOC Exhaled']

# Breathing Features
# features = ['Average Chest Breathing Rate', 'Average Chest RVT', 'Average Chest Symmetry Peak-Trough',  'Average Chest Symmetry Rise-Decay', 
#             'Average Chest Inhale Time', 'Average Chest Exhale Time','Average Chest Inhale-Exhale Ratio', 'Average Abdomen Breathing Rate', 
#             'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Time', 
#             'Average Abdomen Exhale Time', 'Average Abdomen Inhale Exhale Ratio']

# Non Breathing Features
# features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Number of SDA Peaks', 'Average SDA Amplitudes', 'Average CO2 Exhaled', 'Average VOC Exhaled']

# # Heart Features
# features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV']

# SDA Features
# features = ['Number of SDA Peaks', 'Average SDA Amplitudes']

# CO2/VOC Features
# features = ['Average CO2 Exhaled', 'Average VOC Exhaled']

# Features for Classification
# features = ['Heart Rate', 'Average Systolic Amplitude', 'HRV', 'Average Chest Breathing Rate', 'Average Chest RVT', 'Average Chest Symmetry Rise-Decay', 
#             'Average Chest Inhale Time', 'Average Chest Inhale-Exhale Ratio', 'Average Abdomen RVT', 'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Exhale Ratio', 
#             'Number of SDA Peaks', 'Average SDA Amplitudes', 'Average CO2 Exhaled', 'Average VOC Exhaled']

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

# Separating out the features
x = data.loc[:, features].values

# Separating out the target
y = data.loc[:,['Breathing Type']].values

# Standardise the data (Mean = 0, Variance = 1)
x = StandardScaler().fit_transform(x)

# Perform PCA on the data for data visualisation
pca = PCA(n_components=len(features))
principal_components = pca.fit_transform(x)

# Print PCA Eigenvectors for interpretation
print("PCA Eigenvectors:")
print(pca.components_)

# Skree plot
pc_values = np.arange(pca.n_components_) + 1
plt.plot(pc_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()

# Percentage of total variance explained by each principal component
print("Explained Variance Ratio:")
print(pca.explained_variance_ratio_)

# Plot 2D PCA
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets = class_order
colors = ["#ffa500", "#79c314", "#36cedc"]

for target, color in zip(targets,colors):
    indicesToKeep = data['Breathing Type'] == target
    ax.scatter(principal_components[indicesToKeep, 0]
               , principal_components[indicesToKeep, 1]
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

# Plot 3D PCA
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_zlabel('Principal Component 3', fontsize = 15)
ax.set_title('3 Component PCA', fontsize = 20)

targets = class_order
colors = ["#ffa500", "#79c314", "#36cedc"]

for target, color in zip(targets,colors):
    indicesToKeep = data['Breathing Type'] == target
    ax.scatter(principal_components[indicesToKeep, 0]
               , principal_components[indicesToKeep, 1]
               , principal_components[indicesToKeep, 2]
               , c = color
               , s = 50)
    
ax.legend(targets)
ax.grid()
plt.show()