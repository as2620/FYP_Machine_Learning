import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Read the data from the CSV file
# csv_filename = "breathing_rate_data.csv"
# csv_filename = "breathing_rate_data_3_5_30_40.csv"
# csv_filename = "box_data.csv"
csv_filename = "stress_data.csv"

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

# Specify the desired order for the classes
# class_order = ['3bpm', '5bpm', '10bpm', '15bpm', '20bpm', '30bpm', '40bpm']
# class_order = ['Rest', 'Box']
class_order = ['Rest', 'S1', 'S2']

data['Breathing Type'] = pd.Categorical(data['Breathing Type'], categories=class_order, ordered=True)

# Group the data by the class column
grouped_data = data.groupby('Breathing Type')

# Initialize an empty DataFrame to collect the processed data
cleaned_data = pd.DataFrame()

# Outlier removal for each feature within each class
for name, group in grouped_data:
    for feature in column_names:
        if feature != "Breathing Type":
            # Outlier Removal for each group
            Q1 = group[feature].quantile(0.25)
            Q3 = group[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            group = group[(group[feature] >= lower_bound) & (group[feature] <= upper_bound)]
    
    # Append the cleaned group to the cleaned_data DataFrame
    cleaned_data = pd.concat([cleaned_data, group])

# Reset index of the cleaned_data DataFrame
cleaned_data.reset_index(drop=True, inplace=True)

# Box Plots
# Optionally, plot box plots before and after outlier removal
# for name, group in grouped_data:
#     for feature in column_names:
#         if feature != "Breathing Type":
#             # Box plot before removing outliers
#             sns.boxplot(x=group['Breathing Type'], y=group[feature])
#             plt.title(f"Before removing outliers - {feature} for {name}")
#             plt.show()

# for name, group in cleaned_data.groupby('Breathing Type'):
#     for feature in column_names:
#         if feature != "Breathing Type":
#             # Box plot after removing outliers
#             sns.boxplot(x=group['Breathing Type'], y=group[feature])
#             plt.title(f"After removing outliers - {feature} for {name}")
#             plt.show()

# Divide the cleaned data set into features (X) and target variable (y)
x = cleaned_data.iloc[:, 1:24].values
y = cleaned_data.iloc[:, 0].values

# Encode the target variable
le = LabelEncoder()
y = le.fit_transform(y)

# Define a custom color palette
# palette = sns.color_palette(["#e81416", "#ffa500", "#faeb36", "#79c314", "#36cedc" ,"#487de7", "#70369d"], len(class_order))
# palette = sns.color_palette(["#ffa500", "#36cedc"], len(class_order))
palette = sns.color_palette(["#ffa500", "#79c314", "#36cedc"], len(class_order))

# Visualize the distribution of each feature using histograms.
for i, feature in enumerate(column_names[1:]):
    plt.figure(figsize=(12, 6))
    sns.histplot(data=cleaned_data, x=feature, hue='Breathing Type', palette=palette, kde=True)
    plt.title(f'{feature} Distribution')

plt.tight_layout()
plt.show()