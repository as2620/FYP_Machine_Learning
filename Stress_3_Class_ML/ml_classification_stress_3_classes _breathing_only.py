import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
import xgboost as xgb
from sklearn.neighbors import KNeighborsClassifier
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

# Read the data from the CSV file
csv_filename = "stress_data.csv"
csv_filepath = os.path.abspath(os.path.join("..", "..", "Machine_Learning_Data", csv_filename))
data = pd.read_csv(csv_filepath)

# Drop any rows with missing data
data = data.dropna()

# Remove Non Breathing Features
data = data.drop(['Heart Rate', 'SpO2', 'Average Systolic Amplitude', 'HRV', 'Number of SDA Peaks', 'Average SDA Amplitudes', 'Average EDA Tonic SD', 'Average CO2 Exhaled', 'Average VOC Exhaled'], axis=1)

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
        
        # Display box plot before outlier removal
        # plt.figure(figsize=(10, 6))
        # sns.boxplot(x=subset[target], y=subset[feature])
        # plt.title(f'Box plot of {feature} before outlier removal for {label}')
        # plt.show()
        
        # Remove outliers
        data = data.drop(subset.index)
        subset = remove_outliers(subset, feature)
        data = pd.concat([data, subset])
        
        # Display box plot after outlier removal
        # plt.figure(figsize=(10, 6))
        # sns.boxplot(x=subset[target], y=subset[feature])
        # plt.title(f'Box plot of {feature} after outlier removal for {label}')
        # plt.show()

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

# Initialize the Logistic Regression model
lr_model = LogisticRegression()

# Initialize the SVM model
svm_model = SVC()

# Initialize the Random Forest model
rf_model = RandomForestClassifier()

# Initialize the Gradient Boosting model
gbc_model = GradientBoostingClassifier()

# Initialize the AdaBoost model
ada_model = AdaBoostClassifier()

# Initialize the K-Nearest Neighbors model
knn_model = KNeighborsClassifier()

# Train the model
lr_model.fit(x_train_scaled, y_train)
svm_model.fit(x_train_scaled, y_train)
rf_model.fit(x_train_scaled, y_train)
gbc_model.fit(x_train_scaled, y_train)
ada_model.fit(x_train_scaled, y_train)
knn_model.fit(x_train_scaled, y_train)


# Predict on the test set
lr_y_pred = lr_model.predict(x_test_scaled)
svm_y_pred = svm_model.predict(x_test_scaled)
rf_y_pred = rf_model.predict(x_test_scaled)
gbc_y_pred = gbc_model.predict(x_test_scaled)
ada_y_pred = ada_model.predict(x_test_scaled)
knn_y_pred = knn_model.predict(x_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, lr_y_pred)
print("Logistic Regression Accuracy:", accuracy)

accuracy = accuracy_score(y_test, svm_y_pred)
print("SVM Accuracy:", accuracy)

accuracy = accuracy_score(y_test, rf_y_pred)
print("Random Forest Accuracy:", accuracy)

accuracy = accuracy_score(y_test, gbc_y_pred)
print("Gradient Bootsting Accuracy:", accuracy)

accuracy = accuracy_score(y_test, ada_y_pred)
print("AdaBoost Accuracy:", accuracy)

accuracy = accuracy_score(y_test, knn_y_pred)
print("KNN Accuracy:", accuracy)

# Print classification report
print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_y_pred))

print("SVM Classification Report:")
print(classification_report(y_test, svm_y_pred))

print("Random Forest Classification Report:")
print(classification_report(y_test, rf_y_pred))

print("Gradient Bootsting Classification Report:")
print(classification_report(y_test, gbc_y_pred))

print("AdaBoost Classification Report:")
print(classification_report(y_test, ada_y_pred))

print("KNN Classification Report:")
print(classification_report(y_test, knn_y_pred))

# Plot confusion matrix
conf_mat = confusion_matrix(y_test, lr_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=lr_model.classes_, yticklabels=lr_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Logistic Regression Confusion Matrix')
plt.show()

conf_mat = confusion_matrix(y_test, svm_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=svm_model.classes_, yticklabels=svm_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('SVM Confusion Matrix')
plt.show()

conf_mat = confusion_matrix(y_test, rf_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=rf_model.classes_, yticklabels=rf_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Random Forest Confusion Matrix')
plt.show()

conf_mat = confusion_matrix(y_test, gbc_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=gbc_model.classes_, yticklabels=gbc_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Gradient Bootsting Confusion Matrix')
plt.show()

conf_mat = confusion_matrix(y_test, ada_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=ada_model.classes_, yticklabels=ada_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('AdaBoost Confusion Matrix')
plt.show()

conf_mat = confusion_matrix(y_test, knn_y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues', xticklabels=knn_model.classes_, yticklabels=knn_model.classes_)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('KNN Confusion Matrix')
plt.show()