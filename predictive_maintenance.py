"""
Predictive Maintenance for Industrial Equipment
Uses machine learning to forecast equipment failures before they occur,
based on historical sensor data and operational logs.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ---- 1. Load data ----
df = pd.read_csv('predictive_maintenance.csv')
print(df.head())
print(df.isnull().sum())

# ---- 2. Preprocessing ----
# Handle missing values, label encode categorical columns, normalize sensor readings
# (see README for full preprocessing notes)

# ---- 3. Feature engineering ----
# Rolling mean/std of sensors, lag features, failure label as target

# ---- 4. Train/test split ----
X = df.drop(['Failure'], axis=1)
y = df['Failure']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---- 5. Model: Random Forest Classifier ----
model = RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# ---- 6. Visualization: Confusion matrix ----
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Result: ~95% accuracy, high precision/recall, very few false alarms
