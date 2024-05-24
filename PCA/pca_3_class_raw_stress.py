import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the CSV file
csv_filename = "stress_raw_data.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))

data = pd.read_csv(csv_filepath)
print("Shape of Data: ", data.shape)

# All Features
features = ['CO2', 'VOC', 'abdomen_coil', 'chest_coil', 'gsr', 'ppg_ir', 'ppg_red']

# Specify the desired order for the classes
class_order = ['Rest', 'S1', 'S2']

# Drop any rows with missing data
data = data.dropna()
print("Shape of Data after dropping NaNs: ", data.shape)

# Check for non-numeric values in each feature and replace them with NaNs
for feature in features:
    data[feature] = pd.to_numeric(data[feature], errors='coerce')

# Check if there are any NaNs in the dataset after replacing non-numeric values
if data.isnull().values.any():
    print("NaNs detected after attempting to convert non-numeric values to float. Please check the data.")
    # Handle NaNs based on your requirements, such as removing rows with NaNs or imputing missing values
    # For example, to remove rows with NaNs:
    data = data.dropna()

print("Shape of Data after removing NaNs from conversions: ", data.shape)

# Outlier Removal for each feature for each class
for feature in features:
    for class_value in class_order:
        # Filter data for the current class
        class_data = data[data['classification'] == class_value][feature]
        # Calculate Q1, Q3, and IQR
        Q1 = class_data.quantile(0.25)
        Q3 = class_data.quantile(0.75)
        IQR = Q3 - Q1
        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Remove outliers
        data = data.drop(data[(data['classification'] == class_value) & ((data[feature] < lower_bound) | (data[feature] > upper_bound))].index)

print("Shape of Data after Outlier Removal: ", data.shape)

# Separating out the features
x = data.loc[:, features].values

# Separating out the target
y = data.loc[:, ['classification']].values

# Standardise the data (Mean = 0, Variance = 1)
x = StandardScaler().fit_transform(x)

# Perform PCA on the data for data visualisation
pca = PCA(n_components=len(features))
principal_components = pca.fit_transform(x)

# Print PCA Eigenvectors for interpretation
print("PCA Eigenvectors:")
print(pca.components_)

# Scree plot
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
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_title('2 Component PCA', fontsize=20)

targets = class_order
colors = ["#ffa500", "#79c314", "#36cedc"]

for target, color in zip(targets, colors):
    indicesToKeep = data['classification'] == target
    ax.scatter(principal_components[indicesToKeep, 0],
               principal_components[indicesToKeep, 1],
               c=color,
               s=50)
ax.legend(targets)
ax.grid()
plt.show()

# Plot 3D PCA
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(projection='3d')
ax.set_xlabel('Principal Component 1', fontsize=15)
ax.set_ylabel('Principal Component 2', fontsize=15)
ax.set_zlabel('Principal Component 3', fontsize=15)
ax.set_title('3 Component PCA', fontsize=20)

for target, color in zip(targets, colors):
    indicesToKeep = data['classification'] == target
    ax.scatter(principal_components[indicesToKeep, 0],
               principal_components[indicesToKeep, 1],
               principal_components[indicesToKeep, 2],
               c=color,
               s=50)

ax.legend(targets)
ax.grid()
plt.show()
