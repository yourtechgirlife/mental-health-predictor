# Step 1: Import Libraries
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix

# Step 2: Load Dataset
df = pd.read_csv("myml.csv")

# Step 3: Clean Column Names
df.columns = df.columns.str.strip()

# Step 4: Create Target Column - Risk
df['Risk_Score'] = df[['Anxiety', 'Academic_pressure', 'Financial_pressure']].mean(axis=1)
df['Risk'] = df['Risk_Score'].apply(lambda x: 1 if x >= 3 else 0)

# Step 5: Drop Irrelevant Columns (Target contributors + Risk Score)
X = df.drop(columns=['Risk', 'Risk_Score', 'Anxiety', 'Academic_pressure', 'Financial_pressure'], errors='ignore')
y = df['Risk']

# Step 6: One-Hot Encode Categorical Variables
X = pd.get_dummies(X, drop_first=True)

# Step 7: Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Create pipeline using RandomForest
pipeline = make_pipeline(
    SimpleImputer(strategy='mean'),
    StandardScaler(),
    RandomForestClassifier(n_estimators=100, random_state=42)
)

# Step 9: Fit pipeline
pipeline.fit(X_train, y_train)

# Step 10: Evaluate Model
y_pred = pipeline.predict(X_test)

cm = confusion_matrix(y_test, y_pred)

print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 11: Visualize Confusion Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Low Risk', 'High Risk'],
            yticklabels=['Low Risk', 'High Risk'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# Save the model as mental_health_model.pkl
joblib.dump(pipeline, 'mental_health_model.pkl')
print("Model saved as mental_health_model.pkl")
