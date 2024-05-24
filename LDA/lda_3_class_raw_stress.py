import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
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

# Drop any rows with missing data
data = data.dropna()

# Print the shape of the data after outlier removal
print("Shape of data after outlier removal:", data.shape)

# Separating out the features
x = data.loc[:, features].values

# Separating out the target
y = data.loc[:,['classification']].values.ravel()  # Using ravel() to convert to 1d array

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
final_data = pd.concat([lda_data, data.reset_index(drop=True)[['classification']]], axis=1)

# Print the LDA data
print(lda_data)

# Plot 1D LDA
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlabel('LDA Component 1', fontsize=15)
ax.set_title('1 Component LDA', fontsize=20)

targets = class_order
colors = ["#ffa500", "#79c314", "#36cedc"]

for target, color in zip(targets, colors):
    indicesToKeep = final_data['classification'] == target
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
    indicesToKeep = final_data['classification'] == target
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
    indicesToKeep = final_data['classification'] == target
    ax.scatter(final_data.loc[indicesToKeep, 'LDA Component 1'], final_data.loc[indicesToKeep, 'LDA Component 2'], final_data.loc[indicesToKeep, 'LDA Component 3'], c=color, label=target, s=50)

ax.legend()
ax.grid()
plt.show()
