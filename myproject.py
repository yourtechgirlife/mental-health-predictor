# Step 1: Import Libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Load Dataset
df = pd.read_csv("myml.csv")

# Step 3: Clean Column Names
df.columns = df.columns.str.strip()

# Step 4: Create Target Column - Risk
df['Risk_Score'] = df[['Anxiety', 'Academic_pressure', 'Financial_pressure']].mean(axis=1)
df['Risk'] = df['Risk_Score'].apply(lambda x: 1 if x >= 3 else 0)

# Step 5: Drop Irrelevant Columns (Target contributors + Risk Score)
X = df.drop(columns=['Risk', 'Risk_Score'], errors='ignore')
y = df['Risk']

# Step 6: One-Hot Encode Categorical Variables
X = pd.get_dummies(X, drop_first=True)

# Step 7: Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 8: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 9: Handle Imbalanced Data with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print("Original dataset shape:", X.shape, "Labels:", set(y))
print("Resampled dataset shape:", X_resampled.shape, "Labels:", set(y_resampled))

# Step 10: Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 11: Evaluate Model
y_pred = model.predict(X_test)

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 12: Save Model and Scaler
joblib.dump(model, "mental_health_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(list(X.columns), "feature_names.pkl")  # save feature order for interface

# Plot 1: Distribution of target variable
plt.figure(figsize=(6,4))
sns.countplot(x=y)
plt.title('Target Variable Distribution')
plt.xlabel('Target')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Plot 2: Correlation heatmap
plt.figure(figsize=(10,8))
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, fmt=".2f", cmap='Blues')
plt.title('Feature Correlation Heatmap')
plt.tight_layout()
plt.show()

# Plot 3: Feature distributions (first 4 features as example)
feature_cols = X.columns[:4]
df[feature_cols].hist(bins=20, figsize=(12,8))
plt.suptitle('Feature Distributions')
plt.tight_layout()
plt.show()