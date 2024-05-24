import numpy as np
import pandas as pd
import heartpy as hp
import scipy.signal
from scipy.ndimage import median_filter
import matplotlib.pyplot as plt

# Specify the path to your CSV file
file_path = 'stress_raw_data.csv'

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Define features to be processed
features = ['chest_coil', 'abdomen_coil']

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

# Filter data
df['chest_coil'] = scipy.signal.detrend(df['chest_coil'])
df['chest_coil'] = hp.filter_signal(df['chest_coil'], cutoff=1, sample_rate=20.0, filtertype='lowpass', return_top=False)

df['abdomen_coil'] = scipy.signal.detrend(df['abdomen_coil'])
df['abdomen_coil'] = hp.filter_signal(df['abdomen_coil'], cutoff=1, sample_rate=20.0, filtertype='lowpass', return_top=False)

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

# Create a new DataFrame to store the filtered data
filtered_data = pd.DataFrame()

for user in user_ids:
    for class_to_filter in class_order:
        # Filter the DataFrame by user ID and classification
        user_df = df_cleaned[(df_cleaned['user_id'] == user) & (df_cleaned['classification'] == class_to_filter)].copy()

        if not user_df.empty:
            # Apply median filter to smooth out spikes
            median_filter_size = 20
            user_df['chest_coil'] = median_filter(user_df['chest_coil'], size=median_filter_size)
            user_df['abdomen_coil'] = median_filter(user_df['abdomen_coil'], size=median_filter_size)

            threshold = 2  # Number of standard deviations

            # Detect outliers - CHEST
            mean = np.mean(user_df['chest_coil'])
            std = np.std(user_df['chest_coil'])
            chest_outliers = np.abs(user_df['chest_coil'] - mean) > threshold * std

            # Detect outliers - ABDOMEN
            mean = np.mean(user_df['abdomen_coil'])
            std = np.std(user_df['abdomen_coil'])
            abdomen_outliers = np.abs(user_df['abdomen_coil'] - mean) > threshold * std

            # Combine outliers for both signals
            combined_outliers = chest_outliers | abdomen_outliers

            # Remove outliers from user_df
            user_df = user_df[~combined_outliers]

            # Append the filtered data to the new DataFrame
            filtered_data = pd.concat([filtered_data, user_df], ignore_index=True)

            if not user_df.empty:
                # Plot the data for chest_coil
                plt.plot(user_df['chest_coil'].values, label=f'Chest Coil - {class_to_filter}', c=colour_chest[class_order.index(class_to_filter)])

                # Plot the data for abdomen_coil
                plt.plot(user_df['abdomen_coil'].values, label=f'Abdomen Coil - {class_to_filter}', c=colour_abdomen[class_order.index(class_to_filter)])
    
    plt.xlabel('Sample')
    plt.ylabel('Value')
    plt.title(f'User {user}')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Save the filtered data to a new CSV file
output_file_path = 'filtered_stress_data.csv'
filtered_data.to_csv(output_file_path, index=False)
print(f"Filtered data saved to {output_file_path}")
