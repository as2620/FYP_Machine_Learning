import pandas as pd 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the CSV file
csv_filename = "breathing_rate_data_3_5_30_40.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))

data = pd.read_csv(csv_filepath)

# Drop any rows with missing data
data = data.dropna()

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
# features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV']

# SDA Features
# features = ['Number of SDA Peaks', 'Average SDA Amplitudes']

# CO2/VOC Features
features = ['Average CO2 Exhaled ', 'Average VOC Exhaled']

num_features = len(features)

# Separating out the features
x = data.loc[:, features].values

# Separating out the target
y = data.loc[:,['Breathing Type']].values

# Standardise the data (Mean = 0, Variance = 1)
x = StandardScaler().fit_transform(x)

# Perform PCA on the data for data visualisation
pca = PCA(n_components=num_features)
principal_components = pca.fit_transform(x)

print("PCA Eigenvectors:")
print(pca.components_)

# Skree plot
pc_values = np.arange(pca.n_components_) + 1
plt.plot(pc_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')

# Percentage of total variance explained by each principal component
print(pca.explained_variance_ratio_)

pca_data = pd.DataFrame(data = principal_components, columns = ['principal component 1', 
                                                                'principal component 2'])

# TODO: I lose rows due to the merge. Why?
# Concatenate the PCA data with the target data
print("Shapes before concatenation:")
print("PCA data shape:", pca_data.shape)
print("Target data shape:", data[['Breathing Type']].shape)

# Reset index of pca_data and data[['Breathing Type']]
pca_data.reset_index(drop=True, inplace=True)
data[['Breathing Type']].reset_index(drop=True, inplace=True)

# Merge the PCA data with the target data based on row order
final_data = pd.merge(pca_data, data[['Breathing Type']], left_index=True, right_index=True)

# Check the shape of the final concatenated data
print("Shape of final concatenated data:", final_data.shape)

# Save the concatenated data to CSV
final_data.to_csv("breathing_pca_data.csv", sep=',', index=False, encoding='utf-8')

# 2D Plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets = ['3bpm', '5bpm', '10bpm', '15bpm', '20bpm', '30bpm', '40bpm']
colors = ["#e81416", "#ffa500", "#faeb36", "#79c314", "#36cedc" ,"#487de7", "#70369d"]

for target, color in zip(targets,colors):
    indicesToKeep = final_data['Breathing Type'] == target
    ax.scatter(final_data.loc[indicesToKeep, 'principal component 1']
               , final_data.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

# 3D Plot
# fig = plt.figure(figsize = (8,8))
# ax = fig.add_subplot(projection='3d') 
# ax.set_xlabel('PC 1', fontsize = 15)
# ax.set_ylabel('PC 2', fontsize = 15)
# ax.set_zlabel('PC 3', fontsize = 15)
# ax.set_title('3 Component PCA', fontsize = 20)

# targets = ['3bpm', '5bpm', '10bpm', '15bpm', '20bpm', '30bpm', '40bpm']
# colors = ["#e81416", "#ffa500", "#faeb36", "#79c314", "#36cedc" ,"#487de7", "#70369d"]

# for target, color in zip(targets,colors):
#     indicesToKeep = final_data['Breathing Type'] == target
#     ax.scatter(final_data.loc[indicesToKeep, 'principal component 1']
#                , final_data.loc[indicesToKeep, 'principal component 2']
#                 , final_data.loc[indicesToKeep, 'principal component 3']
#                , c = color
#                , s = 50)
# ax.legend(targets)
# ax.grid()
# plt.show()