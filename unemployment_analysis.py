import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv("/Users/susmitha/Downloads/Unemployment in India.csv")
print("\nFirst 5 Rows of Dataset:\n")
print(data.head())
data = data.dropna()
data.columns = data.columns.str.strip()
print("\nColumn Names:\n")
print(data.columns)
data['Date'] = pd.to_datetime(data['Date'])
print("\nDataset Information:\n")
print(data.info())
print("\nStatistical Summary:\n")
print(data.describe())
plt.figure(figsize=(12,6))
sns.lineplot(
    x='Date',
    y='Estimated Unemployment Rate (%)',
    data=data
)
plt.title("Unemployment Rate Over Time")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.show()
covid_data = data[data['Date'] >= '2020-01-01']
plt.figure(figsize=(12,6))
sns.lineplot(
    x='Date',
    y='Estimated Unemployment Rate (%)',
    data=covid_data,
    color='red'
)
plt.title("Covid-19 Impact on Unemployment")
plt.xlabel("Date")
plt.ylabel("Unemployment Rate (%)")
plt.xticks(rotation=45)
plt.show()
plt.figure(figsize=(14,8))
state_avg = data.groupby('Region')['Estimated Unemployment Rate (%)'].mean().sort_values()
state_avg.plot(kind='barh')
plt.title("Average Unemployment Rate by State")
plt.xlabel("Unemployment Rate (%)")
plt.show()
print("\n===================================")
print("KEY INSIGHTS")
print("===================================")
print("""
1. Unemployment rates increased sharply during Covid-19.
2. Some states showed higher unemployment than others.
3. Lockdowns caused economic slowdown and job losses.
4. Gradual recovery can be observed after pandemic peaks.
5. Data analysis helps governments improve employment policies.
""")
