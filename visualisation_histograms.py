import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# Read the data from the CSV file
csv_filename = "stress_data.csv"
# csv_filename = "box_data.csv"
# csv_filename = "breathing_rate_data_10_to_20.csv"
# csv_filename = "breathing_rate_data_3_to_40.csv"

csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))

# Define column names
column_names = ['Breathing Type','Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Average Chest Breathing Rate', 
                'Average Chest RVT', 'Average Chest Symmetry Peak-Trough', 'Average Chest Symmetry Rise-Decay', 
                'Average Chest Inhale Time', 'Average Chest Exhale Time','Average Chest Inhale-Exhale Ratio', 
                'Average Abdomen Breathing Rate', 'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 
                'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Time', 'Average Abdomen Exhale Time', 
                'Average Abdomen Inhale Exhale Ratio', 'Number of SDA Peaks', 'Average SDA Amplitudes', 
                'Average CO2 Exhaled ', 'Average VOC Exhaled']


data = pd.read_csv(csv_filepath)

# Drop any rows with missing data
data = data.dropna()

# Box Plots
# Visualize the distribution of each feature to detect and remove outliers
for feature in column_names:
    if feature != "Breathing Type":
        # sns.boxplot(data[feature])
        # plt.show()

        # Outlier Removal
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

        # Box plot after removing outliers
        # sns.boxplot(data[feature])
        # plt.show()

# Divide the data set into features (X) and target variable (y)
x = data.iloc[:, 1:24].values
y = data.iloc[:, 0].values

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Visualize the distribution of each feature using histograms.
for i, feature in enumerate(column_names[:-1]):
    plt.figure(figsize=(12, 6))
    sns.histplot(data=data, x=feature, hue='Breathing Type', kde=True)
    plt.title(f'{feature} Distribution')

plt.tight_layout()
plt.show()
