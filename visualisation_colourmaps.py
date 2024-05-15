import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder


# Read the data from the CSV file
csv_filename = "pure_s2_data.csv"

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
        # Outlier Removal
        Q1 = data[feature].quantile(0.25)
        Q3 = data[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

correlation_matrix = data.corr(numeric_only = True)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()