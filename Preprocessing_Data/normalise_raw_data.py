import numpy as np
import pandas as pd
import heartpy as hp
import scipy.signal
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

# Specify the path to your CSV file
file_path = "filtered_stress_data.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Define features to be processed
features = ['chest_coil', 'abdomen_coil', 'ppg_ir', 'ppg_red', 'gsr', 'CO2', 'VOC']

# Drop any rows with missing data
df = df.dropna()

# Check for non-numeric values in each feature and replace them with NaNs
for feature in features:
    df[feature] = pd.to_numeric(df[feature], errors='coerce')

# Check if there are any NaNs in the dataset after replacing non-numeric values
if df.isnull().values.any():
    print("NaNs detected after attempting to convert non-numeric values to float. Please check the data.")
    # Handle NaNs by removing rows with NaNs
    df = df.dropna()

# Ensure we are using the cleaned data for further processing
df_cleaned = df.copy()

print(df_cleaned["abdomen_coil"].shape)
print(df_cleaned["chest_coil"].shape)

# Get the unique class orders and user IDs
class_order = ['S2', 'S1', 'Rest']
user_ids = df_cleaned['user_id'].unique()

# Colors for chest and abdomen coils
colour_chest = ["#ffa500", "#79c314", "#36cedc"]
colour_abdomen = ["r", "g", "b"]

# Create a new DataFrame to store the filtered and normalized data
filtered_normalized_data = pd.DataFrame()

for user in user_ids:
    user_data = df_cleaned[df_cleaned['user_id'] == user]
    for feature in features:
        # Normalize the signal for the user across all classes
        mean = user_data[feature].mean()
        std = user_data[feature].std()
        user_data[feature] = (user_data[feature] - mean) / std
    
    # Append the normalized data to the new DataFrame
    filtered_normalized_data = pd.concat([filtered_normalized_data, user_data], ignore_index=True)

    colour = ["#ffa500", "#79c314", "#36cedc"]

    # Plot the normalized signals for this user
    plt.figure(figsize=(10, 6))
    
    for class_to_filter in class_order:
        user_class_data = user_data[user_data['classification'] == class_to_filter]
        
        if not user_class_data.empty:
            plt.plot(user_class_data['chest_coil'].values, label=f'Chest Coil - {class_to_filter}', c=colour[class_order.index(class_to_filter)])
            # plt.plot(user_class_data['abdomen_coil'].values, label=f'Abdomen Coil - {class_to_filter}')
            # plt.plot(user_class_data['ppg_ir'].values, label=f'PPG IR - {class_to_filter}')
            # plt.plot(user_class_data['ppg_red'].values, label=f'PPG Red - {class_to_filter}')
            # plt.plot(user_class_data['gsr'].values, label=f'GSR - {class_to_filter}')
            # plt.plot(user_class_data['CO2'].values, label=f'CO2 - {class_to_filter}')
            # plt.plot(user_class_data['VOC'].values, label=f'VOC - {class_to_filter}')
    
    plt.xlabel('Sample')
    plt.ylabel('Normalized Value')
    plt.title(f'User {user} - Normalized Signals')
    plt.legend()
    plt.tight_layout()
    plt.show()

# # Save the filtered and normalized data to a new CSV file
# output_file_path = 'filtered_normalized_stress_data.csv'
# filtered_normalized_data.to_csv(output_file_path, index=False)
# print(f"Filtered and normalized data saved to {output_file_path}")
