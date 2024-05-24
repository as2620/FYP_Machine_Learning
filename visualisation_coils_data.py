import pandas as pd
import matplotlib.pyplot as plt
import heartpy as hp
import scipy.signal

# Specify the path to your CSV file
# file_path = 'stress_raw_data.csv'
file_path ="filtered_stress_data.csv"

# Read the CSV file into a DataFrame
df = pd.read_csv(file_path)

# Specify the columns you want to extract
columns_to_extract = ['user_id', 'classification', 'chest_coil', 'abdomen_coil']
features = ['chest_coil', 'abdomen_coil']

# Extract the specified columns
extracted_columns = df[columns_to_extract]

# Drop any rows with missing data
extracted_columns = extracted_columns.dropna()

# Check for non-numeric values in each feature and replace them with NaNs
for feature in features:
    extracted_columns[feature] = pd.to_numeric(extracted_columns[feature], errors='coerce')

# Check if there are any NaNs in the dataset after replacing non-numeric values
if extracted_columns.isnull().values.any():
    print("NaNs detected after attempting to convert non-numeric values to float. Please check the data.")
    # Handle NaNs based on your requirements, such as removing rows with NaNs or imputing missing values
    # For this example, we will remove rows with NaNs
    extracted_columns = extracted_columns.dropna()

# Filter data 
extracted_columns['chest_coil'] = scipy.signal.detrend(extracted_columns['chest_coil'] )
extracted_columns['chest_coil']  = hp.filter_signal(extracted_columns['chest_coil'] , cutoff=1, sample_rate=20.0, filtertype='lowpass', return_top=False)

extracted_columns['abdomen_coil']  = scipy.signal.detrend(extracted_columns['abdomen_coil'] )
extracted_columns['abdomen_coil']  = hp.filter_signal(extracted_columns['abdomen_coil'] , cutoff=1, sample_rate=20.0, filtertype='lowpass', return_top=False)

print(extracted_columns["abdomen_coil"].shape)
print(extracted_columns["chest_coil"].shape)

# Ensure we are using the cleaned data for further processing
df = extracted_columns

# Get the unique class orders and user IDs
class_order = ['S2', 'S1', 'Rest']
user_ids = df['user_id'].unique()

# Colors for chest and abdomen coils
colour_chest = ["#ffa500", "#79c314", "#36cedc"]
colour_abdomen = ["r", "g", "b"]

for user in user_ids:
    plt.figure(figsize=(10, 5))
    
    for class_to_filter in class_order:
        # Filter the DataFrame by user ID and classification
        user_df = df[(df['user_id'] == user) & (df['classification'] == class_to_filter)]
        
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