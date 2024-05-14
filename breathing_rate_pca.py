import pandas as pd 
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Read the data from the CSV file
csv_filename = "breathing_rate_data_10_to_20.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))

data = pd.read_csv(csv_filepath)

# Drop any rows with missing data
data = data.dropna()

# Drop Stress Box and At Rest columns
data = data.drop(data[data['Breathing Type'] == 'Rest'].index)
data = data.drop(data[data['Breathing Type'] == 'Box'].index)
data = data.drop(data[data['Breathing Type'] == 'S1'].index)
data = data.drop(data[data['Breathing Type'] == 'S2'].index)

# features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Average Chest Breathing Rate', 
#             'Average Chest RVT', 'Average Chest Symmetry Peak-Trough', 'Average Chest Symmetry Rise-Decay', 
#             'Average Chest Inhale Time', 'Average Chest Exhale Time','Average Chest Inhale-Exhale Ratio', 
#             'Average Abdomen Breathing Rate', 'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 
#             'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Time', 'Average Abdomen Exhale Time', 
#             'Average Abdomen Inhale Exhale Ratio', 'Number of SDA Peaks', 'Average SDA Amplitudes', 
#             'Average CO2 Exhaled ', 'Average VOC Exhaled']


# features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV',
#             'Number of SDA Peaks', 'Average SDA Amplitudes', 
#             'Average CO2 Exhaled ', 'Average VOC Exhaled']

features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Number of SDA Peaks', 'Average SDA Amplitudes']


# features = ['Average Chest RVT', 'Average Chest Symmetry Peak-Trough', 'Average Chest Symmetry Rise-Decay', 
#             'Average Chest Inhale Time', 'Average Chest Exhale Time','Average Chest Inhale-Exhale Ratio', 
#             'Average Abdomen Breathing Rate', 'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 
#             'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Time', 'Average Abdomen Exhale Time', 
#             'Average Abdomen Inhale Exhale Ratio']

# Separating out the features
x = data.loc[:, features].values

# Separating out the target
y = data.loc[:,['Breathing Type']].values

# Standardise the data (Mean = 0, Variance = 1)
x = StandardScaler().fit_transform(x)

# Perform PCA on the data for data visualisation
pca = PCA(n_components=2)
principal_components = pca.fit_transform(x)
pca_data = pd.DataFrame(data = principal_components, columns = ['principal component 1', 'principal component 2'])

# Concatenate the PCA data with the target data
final_data = pd.concat([pca_data, data[['Breathing Type']]], axis = 1)

# Plot the PCA results for visualisation
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 Component PCA', fontsize = 20)

targets = ['3bpm', '5bpm', '10bpm', '15bpm', '20bpm', '30bpm', '40bpm']
colors = ['red', 'orange', 'yellow', 'lightgreen', 'cyan', 'blue', 'mediumpurple']

for target, color in zip(targets,colors):
    indicesToKeep = final_data['Breathing Type'] == target
    ax.scatter(final_data.loc[indicesToKeep, 'principal component 1']
               , final_data.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
plt.show()

# Plot the Skree Plot 

# Plot the PCA results
