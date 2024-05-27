import pandas as pd
import matplotlib.pyplot as plt
import heartpy as hp
import scipy.signal
import numpy as np

# Specify the path to your CSV file
file_path = "filtered_normalized_stress_data.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Specify the columns you want to extract
columns_to_extract = ['user_id', 'classification', 'chest_coil', 'abdomen_coil']
features = ['chest_coil', 'abdomen_coil']

# Extract the specified columns
extracted_columns = df[columns_to_extract]

# Drop any rows with missing data
extracted_columns = extracted_columns.dropna()

# Ensure we are using the cleaned data for further processing
df = extracted_columns

# Get the unique class orders and user IDs
class_order = ['S2', 'S1', 'Rest']
user_ids = df['user_id'].unique()

# Define the window length for the spectrogram
window_length = 200

# Function to plot spectrogram for a given feature and class
def plot_spectrogram(feature, class_label, title):
    num_users = len(user_ids)
    num_rows = 2
    num_cols = min(5, (num_users + 1) // num_rows)
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(20, 15), constrained_layout=True)
    
    # Flatten axs for easy iteration
    axs = axs.flatten()
    
    for user_idx, user in enumerate(user_ids):
        # Filter the DataFrame by user ID and classification
        user_df = df[(df['user_id'] == user) & (df['classification'] == class_label)]
        
        if not user_df.empty:
            f, t, Sxx = scipy.signal.spectrogram(user_df[feature].values, fs=20.0, nperseg=window_length)
            im = axs[user_idx].pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud', cmap='magma')
            axs[user_idx].set_title(f'User {user}')
            axs[user_idx].set_ylabel('Frequency [Hz]')
            axs[user_idx].set_xlabel('Time [sec]')
        else:
            axs[user_idx].set_visible(False)
    
    # If there are more subplots than users, hide the unused subplots
    for ax in axs[num_users:]:
        ax.set_visible(False)
    
    fig.colorbar(im, ax=axs[:num_users], orientation='vertical', label='Intensity [dB]')
    plt.suptitle(title, fontsize=16)

# Plot spectrograms for each combination of coil type and classification
for class_label in class_order:
    plot_spectrogram('chest_coil', class_label, f'Chest Coil - {class_label}')
    plot_spectrogram('abdomen_coil', class_label, f'Abdomen Coil - {class_label}')

plt.show()
