import pandas as pd 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the CSV file
csv_filename = "box_data.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))

data = pd.read_csv(csv_filepath)

# Drop any rows with missing data
data = data.dropna()

# features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Average Chest Breathing Rate', 
#             'Average Chest RVT', 'Average Chest Symmetry Peak-Trough', 'Average Chest Symmetry Rise-Decay', 
#             'Average Chest Inhale Time', 'Average Chest Exhale Time','Average Chest Inhale-Exhale Ratio', 
#             'Average Abdomen Breathing Rate', 'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 
#             'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Time', 'Average Abdomen Exhale Time', 
#             'Average Abdomen Inhale Exhale Ratio', 'Number of SDA Peaks', 'Average SDA Amplitudes', 
#             'Average CO2 Exhaled ', 'Average VOC Exhaled']

features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Number of SDA Peaks', 'Average SDA Amplitudes', 
            'Average CO2 Exhaled ', 'Average VOC Exhaled']

# features = ['Average CO2 Exhaled ', 'Average VOC Exhaled']

num_features = len(features)

# Separating out the features
x = data.loc[:, features].values

# Separating out the target
y = data.loc[:,['Breathing Type']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

# Perform PCA on the data 
pca = PCA(n_components=num_features)
pca_fit = pca.fit(x)
pca_transformed_data = pca.transform(x)

# Skree plot
pc_values = np.arange(pca.n_components_) + 1
plt.plot(pc_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
# plt.show()

# Percentage of total variance explained by each principal component
print(pca.explained_variance_ratio_)

# 2D Plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('PC 1', fontsize = 15)
ax.set_ylabel('PC 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets = ['Box', 'Rest']
colors = ['lightgreen', 'mediumpurple']

for i in range(len(y)): 
    if y[i] == "Box": 
        ax.scatter(pca_transformed_data[i, 0], pca_transformed_data[i, 1], c='lightgreen', s=50)
    elif y[i] == "Rest": 
        ax.scatter(pca_transformed_data[i, 0], pca_transformed_data[i, 1], c='mediumpurple', s=50)

ax.legend(targets)
ax.grid()
# plt.show()

# 3D Plot
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(projection='3d') 
ax.set_xlabel('PC 1', fontsize = 15)
ax.set_ylabel('PC 2', fontsize = 15)
ax.set_zlabel('PC 3', fontsize = 15)
ax.set_title('3 Component PCA', fontsize = 20)

targets = ['Box', 'Rest']
colors = ['lightgreen', 'mediumpurple']

for i in range(len(y)): 
    if y[i] == "Box": 
        ax.scatter(pca_transformed_data[i, 0], pca_transformed_data[i, 1], pca_transformed_data[i,2], c='lightgreen', s=50)
    elif y[i] == "Rest": 
        ax.scatter(pca_transformed_data[i, 0], pca_transformed_data[i, 1], pca_transformed_data[i,2], c='mediumpurple', s=50)

ax.legend(targets)
ax.grid()
plt.show()