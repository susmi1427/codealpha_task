# ==========================================
# IRIS FLOWER CLASSIFICATION - MACHINE LEARNING
# ==========================================

# Import Required Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ==========================================
# LOAD DATASET
# ==========================================

# Load Iris Dataset CSV File
data = pd.read_csv("/Users/susmitha/Downloads/Iris.csv")
# Display First 5 Rows
print("First 5 Rows of Dataset:\n")
print(data.head())

# ==========================================
# DATA PREPROCESSING
# ==========================================

# Remove Id column if present
if 'Id' in data.columns:
    data = data.drop('Id', axis=1)

# Features and Target
X = data.drop('Species', axis=1)
y = data['Species']

# Convert Species Names into Numbers
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# ==========================================
# SPLIT DATASET
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# TRAIN MACHINE LEARNING MODEL
# ==========================================

model = DecisionTreeClassifier()

# Train Model
model.fit(X_train, y_train)

# ==========================================
# MAKE PREDICTIONS
# ==========================================

y_pred = model.predict(X_test)

# ==========================================
# MODEL EVALUATION
# ==========================================

# Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n===================================")
print("IRIS FLOWER CLASSIFICATION RESULTS")
print("===================================")

print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

# ==========================================
# TEST WITH NEW FLOWER DATA
# ==========================================

# Example Flower Measurements:
# [SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]

sample = [[5.1, 3.5, 1.4, 0.2]]

prediction = model.predict(sample)

# Convert Predicted Number back to Flower Name
flower_name = encoder.inverse_transform(prediction)

print("\nPredicted Flower Species:", flower_name[0])
