import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
data = pd.read_csv("/Users/susmitha/Downloads/Iris.csv")
print("First 5 Rows of Dataset:\n")
print(data.head())
if 'Id' in data.columns:
    data = data.drop('Id', axis=1)
X = data.drop('Species', axis=1)
y = data['Species']
encoder = LabelEncoder()
y = encoder.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("\n===================================")
print("IRIS FLOWER CLASSIFICATION RESULTS")
print("===================================")
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))
sample = [[5.1, 3.5, 1.4, 0.2]]
prediction = model.predict(sample)
flower_name = encoder.inverse_transform(prediction)
print("\nPredicted Flower Species:", flower_name[0])
