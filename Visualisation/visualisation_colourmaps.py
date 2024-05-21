import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Read the data from the CSV file
csv_filename = "stress_data.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))

# Define column names for the first heatmap
feature_columns = [
    'Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 
    'Average Chest Breathing Rate', 'Average Chest RVT', 'Average Chest Symmetry Peak-Trough', 
    'Average Chest Symmetry Rise-Decay', 'Average Chest Inhale Time', 'Average Chest Exhale Time', 
    'Average Chest Inhale-Exhale Ratio', 'Average Abdomen Breathing Rate', 'Average Abdomen RVT', 
    'Average Abdomen Symmetry Peak-Trough', 'Average Abdomen Symmetry Rise-Decay', 
    'Average Abdomen Inhale Time', 'Average Abdomen Exhale Time', 'Average Abdomen Inhale Exhale Ratio', 
    'Number of SDA Peaks', 'Average SDA Amplitudes', 'Average CO2 Exhaled', 'Average VOC Exhaled'
]

# Define column names for the second heatmap
rating_columns = ['Rating', 'Reaction Time (ms)', 'Score']

# Read the data
data = pd.read_csv(csv_filepath)

# Drop any rows with missing data
data = data.dropna()

# Box Plots and Outlier Removal
for feature in feature_columns:
    Q1 = data[feature].quantile(0.25)
    Q3 = data[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data = data[(data[feature] >= lower_bound) & (data[feature] <= upper_bound)]

# Compute the correlation matrix for the features
feature_correlation_matrix = data[feature_columns].corr(numeric_only=True)

# Compute the correlation matrix for the rating, reaction time, and score
rating_correlation_matrix = data[rating_columns].corr(numeric_only=True)

# Plot the correlation heatmap for the features
plt.figure(figsize=(10, 8))
sns.heatmap(feature_correlation_matrix, annot=True, fmt=".2f", cmap='Spectral', linewidths=0.5, cbar_kws={"shrink": .75})
plt.title("Correlation Heatmap for Features")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()

# Plot the correlation heatmap for the rating, reaction time, and score
plt.figure(figsize=(8, 6))
sns.heatmap(rating_correlation_matrix, annot=True, fmt=".2f", cmap='Spectral', linewidths=0.5, cbar_kws={"shrink": .75})
plt.title("Correlation Heatmap for Rating, Reaction Time, and Score")
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.show()
