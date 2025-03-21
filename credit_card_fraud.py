import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Load Dataset
file_path = "creditcard.csv"

# Check if file exists
if not os.path.exists(file_path):
    raise FileNotFoundError(f"The dataset file '{file_path}' was not found.Please download it from Kaggle and save it in the same folder as this script.")

df = pd.read_csv(file_path)

# Feature Engineering
df['Transaction_Frequency'] = df.groupby('Time')['Amount'].transform('count')
df['Average_Spending'] = df.groupby('Time')['Amount'].transform('mean')
df['High_Transaction'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)

# Handle Imbalanced Data
X = df.drop(columns=['Class'])
y = df['Class']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE for oversampling
smote = SMOTE(sampling_strategy=0.5, random_state=42)
X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

# Apply Undersampling
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=42)
X_train_us, y_train_us = undersample.fit_resample(X_train, y_train)

# Scale Data
scaler = StandardScaler()
X_train_sm = scaler.fit_transform(X_train_sm)
X_train_us = scaler.fit_transform(X_train_us)
X_test = scaler.transform(X_test)

# Train Models
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_sm, y_train_sm)

log_model = LogisticRegression()
log_model.fit(X_train_us, y_train_us)

# Predictions
rf_pred = rf_model.predict(X_test)
log_pred = log_model.predict(X_test)

# Evaluation
print("Random Forest Performance:")
print("Accuracy:", accuracy_score(y_test, rf_pred))
print("Precision:", precision_score(y_test, rf_pred))
print("Recall:", recall_score(y_test, rf_pred))
print("F1-Score:", f1_score(y_test, rf_pred))
print("Classification Report:\n", classification_report(y_test, rf_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_pred))

print("\nLogistic Regression Performance:")
print("Accuracy:", accuracy_score(y_test, log_pred))
print("Precision:", precision_score(y_test, log_pred))
print("Recall:", recall_score(y_test, log_pred))
print("F1-Score:", f1_score(y_test, log_pred))
print("Classification Report:\n", classification_report(y_test, log_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, log_pred))

# Visualize Results
plt.figure(figsize=(12,6))
plt.subplot(1, 2, 1)
sns.heatmap(confusion_matrix(y_test, rf_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Random Forest')

plt.subplot(1, 2, 2)
sns.heatmap(confusion_matrix(y_test, log_pred), annot=True, fmt='d', cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Logistic Regression')

plt.show()
