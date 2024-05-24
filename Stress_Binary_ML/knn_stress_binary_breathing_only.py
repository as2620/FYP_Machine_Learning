import os
import pandas as pd
import optuna
from sklearn.model_selection import GroupKFold, train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

def objective(trial):
    # Suggest hyperparameters
    n_neighbors = trial.suggest_int('n_neighbors', 1, 30)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])
    
    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm)
    
    # Use GroupKFold for cross-validation
    group_kfold = GroupKFold(n_splits=5)
    
    # Compute cross-validation F1 score
    scores = cross_val_score(model, x_train_scaled, y_train, cv=group_kfold, groups=train_data['Participant ID'], scoring="accuracy")
    
    return scores.mean()

def remove_outliers(df, feature):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[feature] >= lower_bound) & (df[feature] <= upper_bound)]

# Read the data from the CSV file
csv_filename = "stress_data.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))
data = pd.read_csv(csv_filepath)

# Drop any rows with missing data
data = data.dropna()

# Remove Non Breathing Features
data = data.drop(['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Number of SDA Peaks', 'Average SDA Amplitudes', 'Average EDA Tonic SD', 'Average CO2 Exhaled', 'Average VOC Exhaled'], axis=1)
print(data)

# Combine S1 and S2 classes into one "Stress" class
data['Breathing Type'] = data['Breathing Type'].replace({'S1': 'Stress', 'S2': 'Stress'})

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
study.optimize(objective, n_trials=50)

print("Best trial:")
trial = study.best_trial
print("  Value: {}".format(trial.value))
print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

# Train the final model with the best hyperparameters
best_model = KNeighborsClassifier(**trial.params)
best_model.fit(x_train_scaled, y_train)
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
