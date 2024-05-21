import os
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split, cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def remove_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 3, 20)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-4, 1e-1)
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
    scores = cross_val_score(gbc_model, x_train_scaled, y_train, cv=group_kfold, groups=train_data['Participant ID'])
    return scores.mean()

# Read the data from the CSV file
csv_filename = "stress_data.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))
data = pd.read_csv(csv_filepath)

# Drop any rows with missing data
data = data.dropna()

# Remove Non Breathing Features
data = data.drop(['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Number of SDA Peaks', 'Average SDA Amplitudes', 'Average CO2 Exhaled', 'Average VOC Exhaled'], axis=1)

print(data)

print(data['Breathing Type'].value_counts())

# Define features and target variable
# Breathing Features
features = ['Average Chest Breathing Rate', 'Average Chest RVT', 'Average Chest Symmetry Peak-Trough',  'Average Chest Symmetry Rise-Decay', 
            'Average Chest Inhale Time', 'Average Chest Exhale Time','Average Chest Inhale-Exhale Ratio', 'Average Abdomen Breathing Rate', 
            'Average Abdomen RVT', 'Average Abdomen Symmetry Peak-Trough', 'Average Abdomen Symmetry Rise-Decay', 'Average Abdomen Inhale Time', 
            'Average Abdomen Exhale Time', 'Average Abdomen Inhale Exhale Ratio']

target = 'Breathing Type'

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

# Create a study and optimize the objective function
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)

# Print the best hyperparameters
print("Best hyperparameters: ", study.best_params)

# Train the model with the best hyperparameters on the entire training set
best_gbc_model = GradientBoostingClassifier(
    n_estimators=study.best_params['n_estimators'],
    max_depth=study.best_params['max_depth'],
    learning_rate=study.best_params['learning_rate'],
    subsample=study.best_params['subsample'],
    min_samples_split=study.best_params['min_samples_split'],
    min_samples_leaf=study.best_params['min_samples_leaf'],
    random_state=42
)

best_gbc_model.fit(x_train_scaled, y_train)

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
