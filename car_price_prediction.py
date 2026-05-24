import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
data = pd.read_csv("/Users/susmitha/Downloads/car data.csv")
print("\nFirst 5 Rows of Dataset:\n")
print(data.head())
print("\nMissing Values:\n")
print(data.isnull().sum())
data = pd.get_dummies(data, drop_first=True)
X = data.drop('Selling_Price', axis=1)
y = data['Selling_Price']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("\n===================================")
print("CAR PRICE PREDICTION RESULTS")
print("===================================")
print(f"\nMean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted Car Prices")
plt.show()
plt.figure(figsize=(12,8))
sns.heatmap(data.corr(), cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()
print("\nSample Predictions:\n")
results = pd.DataFrame({
    'Actual Price': y_test,
    'Predicted Price': y_pred
})
print(results.head())
