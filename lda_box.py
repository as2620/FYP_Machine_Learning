import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np

# Read the data from the CSV file
csv_filename = "box_data.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))

data = pd.read_csv(csv_filepath)

# Drop any rows with missing data
data = data.dropna()

# Non Breathing Features
# features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV',
#             'Number of SDA Peaks', 'Average SDA Amplitudes', 
#             'Average CO2 Exhaled ', 'Average VOC Exhaled']

# Heart Features - Note the seperation is NOT good when you zoom in
features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV']

# GSR Features
# features = ['Number of SDA Peaks', 'Average SDA Amplitudes']

# CO2 Features
# features = ['Average CO2 Exhaled ', 'Average VOC Exhaled']

# Separating out the features
x = data.loc[:, features].values

# Separating out the target
y = data.loc[:,['Breathing Type']].values

# Standardizing the features
x = StandardScaler().fit_transform(x)

# Perform LDA on the data
# Compute the maximum allowed number of components
num_components = min(x.shape[1], len(np.unique(y)) - 1) 
lda = LDA(n_components=num_components)
print("Number of components:", num_components)

lda_transformed_data = lda.fit_transform(x, y)

print("Shape of lda_transformed_data:", lda_transformed_data.shape)


# 1D Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('LD 1', fontsize=15)
ax.set_title('1 Component LDA', fontsize=20)

targets = ['Box', 'Rest']
colors = ['lightgreen', 'mediumpurple']

# Reshape y to ensure it's a 1D array
y = np.squeeze(y)

for i, target in enumerate(targets):
    indices = np.where(y == target)
    ax.scatter(lda_transformed_data[indices, 0], np.zeros_like(lda_transformed_data[indices]), c=colors[i], label=target, s=50)

ax.legend()
ax.grid()
plt.show()
