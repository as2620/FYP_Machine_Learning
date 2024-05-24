import os
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import optuna
import optuna.visualization as vis
from imblearn.over_sampling import SMOTE
import numpy as np

HYPERPARAMETER_TUNING = True
BEST_MODEL = False

# Best hyperparameters from hyperparameter tuning
_n_estimators = 174
_max_depth = 12
_learning_rate = 0.07696877089912839
_min_samples_split = 4
_min_samples_leaf = 2

cross_val_scores = []

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)

    # Create and train the model
    gbc_model = GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    # Use GroupKFold for cross-validation
    group_kfold = GroupKFold(n_splits=5)

    scores = cross_val_score(gbc_model, x_train_balanced, y_train_balanced, cv=group_kfold, groups=train_data_resampled['Participant ID'], scoring='f1_macro')
    
    global cross_val_scores
    cross_val_scores.append(scores.mean())

    return scores.mean()

# Read the data from the CSV file
csv_filename = "stress_data.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))
data = pd.read_csv(csv_filepath)

# Drop any rows with missing data
data = data.dropna()

print(data['Breathing Type'].value_counts())

# Define features and target variable
# All Features
features = ['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Average Chest Breathing Rate', 
            'Average Chest RVT', 'Average Chest Symmetry Peak-Trough', 'Average Chest Symmetry Rise-Decay', 
            'Average Chest Inhale Time', 'Average Chest Exhale Time','Average Chest Inhale-Exhale Ratio', 
            'Average Abdomen Breathing Rate', 'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 
            'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Time', 'Average Abdomen Exhale Time', 
            'Average Abdomen Inhale Exhale Ratio', 'Number of SDA Peaks', 'Average SDA Amplitudes', 
            'Average CO2 Exhaled', 'Average VOC Exhaled']

target = 'Breathing Type'

# Specify the desired order for the classes
class_order = ['Rest', 'S1', 'S2']

# Remove outliers for each feature within each class
for feature in features:
    for class_value in class_order:
        # Filter data for the current class
        class_data = data[data['Breathing Type'] == class_value][feature]
        # Calculate Q1, Q3, and IQR
        Q1 = class_data.quantile(0.25)
        Q3 = class_data.quantile(0.75)
        IQR = Q3 - Q1
        # Define the lower and upper bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Remove outliers
        data = data.drop(data[(data['Breathing Type'] == class_value) & ((data[feature] < lower_bound) | (data[feature] > upper_bound))].index)

print(data['Breathing Type'].value_counts())

# Separate features and target variable
x = data[features]
y = data[target]

# Split the data into training and test sets based on participant IDs
participant_ids = data['Participant ID'].unique()
train_participant_ids, test_participant_ids = train_test_split(participant_ids, test_size=0.2, random_state=42)

print("Train Participant IDs:", train_participant_ids)
print("Test Participant IDs:", test_participant_ids)

# Split the data based on participant IDs
train_data = data[data['Participant ID'].isin(train_participant_ids)]
test_data = data[data['Participant ID'].isin(test_participant_ids)]

# Separate features and target variable for training and test sets
x_train = train_data[features]
y_train = train_data[target]
x_test = test_data[features]
y_test = test_data[target]

# Standardize the features
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42)
x_train_resampled, y_train_resampled = smote.fit_resample(x_train_scaled, y_train)

# Assign random participant IDs to the synthetic samples
synthetic_indices = np.arange(len(x_train), len(x_train_resampled))
synthetic_participant_ids = np.random.choice(train_participant_ids, len(synthetic_indices))

# Create a DataFrame for the synthetic samples and append to train_data
x_train_resampled_df = pd.DataFrame(x_train_resampled, columns=features)
y_train_resampled_df = pd.DataFrame(y_train_resampled, columns=[target])
synthetic_data = pd.concat([x_train_resampled_df, y_train_resampled_df], axis=1)
synthetic_data['Participant ID'] = np.concatenate([train_data['Participant ID'].values, synthetic_participant_ids])

# Ensure train_data includes both original and synthetic samples
train_data_resampled = synthetic_data
x_train_balanced = train_data_resampled[features]
y_train_balanced = train_data_resampled[target]

if HYPERPARAMETER_TUNING:
    # Create a study and optimize the objective function
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=100)

    # Print the best hyperparameters
    print("Best hyperparameters: ", study.best_params)

    # Plot the cross-validation scores
    plt.figure(figsize=(8, 6))
    plt.plot(cross_val_scores)
    plt.xlabel('Trial')
    plt.ylabel('Cross-Validation Score')
    plt.title('Gradient Boosting Cross-Validation Scores for Each Trial')
    plt.show()

    vis.plot_optimization_history(study)
    vis.plot_param_importances(study)
    vis.plot_slice(study)
    plt.show()


    # Train the model with the best hyperparameters on the entire training set
    best_gbc_model = GradientBoostingClassifier(
        n_estimators=study.best_params['n_estimators'],
        max_depth=study.best_params['max_depth'],
        learning_rate=study.best_params['learning_rate'],
        min_samples_split=study.best_params['min_samples_split'],
        min_samples_leaf=study.best_params['min_samples_leaf'],
        random_state=42
    )
elif BEST_MODEL:
    best_gbc_model = GradientBoostingClassifier(n_estimators=_n_estimators, max_depth=_max_depth, learning_rate=_learning_rate, min_samples_split=_min_samples_split, min_samples_leaf=_min_samples_leaf, random_state=42)

# Train the model on the entire training set
sample_weights = np.zeros(len(y_train))
sample_weights[y_train == "Rest"] = 0.46
sample_weights[y_train == "S1"] = 0.32
sample_weights[y_train == "S2"] = 0.22

best_gbc_model.fit(x_train_scaled, y_train, sample_weight = sample_weights)

# Predict on the test set
gbc_y_pred = best_gbc_model.predict(x_test_scaled)

# Evaluate the model on the test set
accuracy = accuracy_score(y_test, gbc_y_pred)
print("Gradient Boosting Accuracy (Test Set):", accuracy)

print("Gradient Boosting Classification Report (Test Set):")
print(classification_report(y_test, gbc_y_pred))

# Plot confusion matrix for the test set
conf_mat = confusion_matrix(y_test, gbc_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=best_gbc_model.classes_, yticklabels=best_gbc_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Gradient Boosting Confusion Matrix (Test Set)')
plt.show()
