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

# Filter data
df['chest_coil'] = scipy.signal.detrend(df['chest_coil'])
df['chest_coil'] = hp.filter_signal(df['chest_coil'], cutoff=1, sample_rate=20.0, filtertype='lowpass', return_top=False)

df['abdomen_coil'] = scipy.signal.detrend(df['abdomen_coil'])
df['abdomen_coil'] = hp.filter_signal(df['abdomen_coil'], cutoff=1, sample_rate=20.0, filtertype='lowpass', return_top=False)

df['gsr'] = hp.filter_signal(df['gsr'], cutoff=1, sample_rate=20.0, filtertype='lowpass', return_top=False)

df['ppg_red'] = scipy.signal.detrend(df['ppg_red'])
df['ppg_red'] = hp.filter_signal(df['ppg_red'], cutoff=[0.5, 4.5], sample_rate=20.0, filtertype='bandpass', return_top=False)

df['ppg_ir'] = scipy.signal.detrend(df['ppg_ir'])
df['ppg_ir'] = hp.filter_signal(df['ppg_ir'], cutoff=[0.5, 4.5], sample_rate=20.0, filtertype='bandpass', return_top=False)

df['CO2'] = hp.filter_signal(df['CO2'], cutoff=1.5, sample_rate=20.0, filtertype='lowpass', return_top=False)

df['VOC'] = hp.filter_signal(df['VOC'], cutoff=1.5, sample_rate=20.0, filtertype='lowpass', return_top=False)

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
            user_df['CO2'] = median_filter(user_df['CO2'], size=median_filter_size)
            user_df['VOC'] = median_filter(user_df['VOC'], size=median_filter_size)

            threshold = 2  # Number of standard deviations

            for feature in features:
                # Detect outliers
                mean = np.mean(user_df[feature])
                std = np.std(user_df[feature])
                outliers = np.abs(user_df[feature] - mean) > threshold * std

                # Remove outliers from user_df
                user_df = user_df[~outliers]
                

            # Append the filtered data to the new DataFrame
            filtered_data = pd.concat([filtered_data, user_df], ignore_index=True)

            if not user_df.empty:
                # Plot the data for chest_coil
                plt.plot(user_df['chest_coil'].values, label=f'Chest Coil - {class_to_filter}', c=colour_chest[class_order.index(class_to_filter)])

                # # Plot the data for abdomen_coil
                # plt.plot(user_df['abdomen_coil'].values, label=f'Abdomen Coil - {class_to_filter}', c=colour_abdomen[class_order.index(class_to_filter)])
                
                # Plot the data for ppg_ir
                # plt.plot(user_df['ppg_ir'].values, label=f'PPG IR - {class_to_filter}', c=colour_chest[class_order.index(class_to_filter)])

                # Plot the data for ppg_red
                # plt.plot(user_df['ppg_red'].values, label=f'PPG Red - {class_to_filter}' , c=colour_abdomen[class_order.index(class_to_filter)])

                # Plot the data for gsr
                # plt.plot(user_df['gsr'].values, label=f'GSR - {class_to_filter}', c=colour_chest[class_order.index(class_to_filter)])

                # Plot the data for CO2
                # plt.plot(user_df['CO2'].values, label=f'CO2 - {class_to_filter}', c=colour_chest[class_order.index(class_to_filter)])

                # Plot the data for VOC
                # plt.plot(user_df['VOC'].values, label=f'VOC - {class_to_filter}', c=colour_abdomen[class_order.index(class_to_filter)])

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
