import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Read the data from the CSV file
csv_filename = "breathing_rate_data.csv"
# csv_filename = "breathing_rate_data_3_5_30_40.csv"

csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))

# Define column names
column_names = ['Breathing Type', 'Participant ID', 'Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Average Chest Breathing Rate', 
                'Average Chest RVT', 'Average Chest Symmetry Peak-Trough', 'Average Chest Symmetry Rise-Decay', 
                'Average Chest Inhale Time', 'Average Chest Exhale Time', 'Average Chest Inhale-Exhale Ratio', 
                'Average Abdomen Breathing Rate', 'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 
                'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Time', 'Average Abdomen Exhale Time', 
                'Average Abdomen Inhale Exhale Ratio', 'Number of SDA Peaks', 'Average SDA Amplitudes', 
                'Average CO2 Exhaled', 'Average VOC Exhaled']

data = pd.read_csv(csv_filepath)

# Print column names to debug
print("Column Names:", data.columns.tolist())

# Print the first few rows of the data to debug
print("Data Sample:\n", data.head())

# Drop any rows with missing data
data = data.dropna()

# Specify the desired order for the classes
class_order = ['3bpm', '5bpm', '10bpm', '15bpm', '20bpm', '30bpm', '40bpm']
data['Breathing Type'] = pd.Categorical(data['Breathing Type'], categories=class_order, ordered=True)

# Group the data by the class column
grouped_data = data.groupby('Breathing Type', observed=True)

# Get the list of unique participants
participants = data['Participant ID'].unique()

# Define a custom color palette for participants
palette = plt.cm.get_cmap("gist_rainbow", len(participants))

# Plotting all participants with the same breathing rate on the same graph
for breathing_type in class_order:
    subset = data[data['Breathing Type'] == breathing_type]
    subset = subset.sort_values(by='Participant ID')  # Sort by participant ID for consistent color assignment
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, participant_id in enumerate(participants):
        participant_data = subset[subset['Participant ID'] == participant_id]
        y_values = participant_data['Average CO2 Exhaled']
        x_values = range(len(y_values))
        ax.plot(x_values, y_values, label=f'Participant {participant_id}', color=palette(i))
        
    ax.set_title(f'Breathing Type: {breathing_type}')
    ax.set_xlabel('Time')
    ax.set_ylabel('Average CO2 Exhaled')
    ax.legend()
    
    plt.show()
