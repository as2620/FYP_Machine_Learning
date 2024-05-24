import os
import pandas as pd
import optuna
from sklearn.model_selection import GroupKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

HYPERPARAMETER_TUNING = True
BEST_MODEL = False

# Best hyperparameters from hyperparameter tuning
_C = 36.65884868943965
_kernel = 'rbf'
_gamma = 'auto'

def objective(trial):
    c = trial.suggest_float('C', 1e-6, 1e2, log=True)
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
    gamma = trial.suggest_categorical('gamma', ['scale', 'auto'])
    
    model = SVC(C=c, kernel=kernel, gamma=gamma)
    
    # Use GroupKFold for cross-validation
    group_kfold = GroupKFold(n_splits=5)
    
    # Compute cross-validation F1 score
    scores = cross_val_score(model, x_train_scaled, y_train, cv=group_kfold, groups=train_data['user_id'], scoring="f1_macro")
    
    return scores.mean()

def remove_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

# Read the data from the CSV file
csv_filename = "stress_raw_data.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))
data = pd.read_csv(csv_filepath)

# Define features and target variable
features = ['CO2', 'VOC', 'abdomen_coil', 'chest_coil', 'gsr', 'ppg_ir', 'ppg_red']
target = 'classification'

# Drop any rows with missing data
data = data.dropna()
print("Shape of Data after dropping NaNs: ", data.shape)

# Check for non-numeric values in each feature and replace them with NaNs
for feature in features:
    data[feature] = pd.to_numeric(data[feature], errors='coerce')

# Check if there are any NaNs in the dataset after replacing non-numeric values
if data.isnull().values.any():
    print("NaNs detected after attempting to convert non-numeric values to float. Please check the data.")
    # Handle NaNs based on your requirements, such as removing rows with NaNs or imputing missing values
    # For example, to remove rows with NaNs:
    data = data.dropna()

print("Shape of Data after removing NaNs from conversions: ", data.shape)

# Combine S1 and S2 classes into one "Stress" class
data['classification'] = data['classification'].replace({'S1': 'Stress', 'S2': 'Stress'})

# Remove outliers for each feature within each class
for feature in features:
    for label in data[target].unique():
        subset = data[data[target] == label]
        # Remove outliers
        data = data.drop(subset.index)
        subset = remove_outliers(subset, feature)
        data = pd.concat([data, subset])

# Separate features and target variable
x = data[features]
y = data[target]

# Split the data into training and test sets based on participant IDs
participant_ids = data['user_id'].unique()
train_participant_ids, test_participant_ids = train_test_split(participant_ids, test_size=0.2, random_state=42)

print("Train Participant IDs:", train_participant_ids)
print("Test Participant IDs:", test_participant_ids)

# Split the data based on participant IDs
train_data = data[data['user_id'].isin(train_participant_ids)]
test_data = data[data['user_id'].isin(test_participant_ids)]

# Separate features and target variable for training and test sets
x_train = train_data[features]
y_train = train_data[target]
x_test = test_data[features]
y_test = test_data[target]

# Standardize the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

if HYPERPARAMETER_TUNING:
    # Create a study and optimize the objective function
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    trial = study.best_trial
    print("  Value: {}".format(trial.value))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    # Train the final model with the best hyperparameters
    best_model = SVC(**trial.params)

elif BEST_MODEL:
    # Train the final model with the best hyperparameters
    best_model = SVC(C=_C, kernel=_kernel, gamma=_gamma)

sample_weights = np.zeros(len(y_train))
sample_weights[y_train == "Rest"] = 0.4
sample_weights[y_train == "Stress"] = 0.6

best_model.fit(x_train_scaled, y_train, sample_weight = sample_weights)
y_pred = best_model.predict(x_test_scaled)

# Evaluate the final model
accuracy = accuracy_score(y_test, y_pred)
print("Final Model Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot confusion matrix
conf_mat = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=best_model.classes_, yticklabels=best_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
