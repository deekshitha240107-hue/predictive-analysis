import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("C:/users/24331/Date_Sales.csv")
print(df.head())
df['Date'] = pd.to_datetime(df['Date'])
print(df.dtypes)
df['Month_Number'] = range(1, len(df) + 1)
print(df.head())
X = df[['Month_Number']]
y = df['Sales']
model = LinearRegression()
model.fit(X, y)
df['Predicted_Sales'] = model.predict(X)
print(df[['Sales', 'Predicted_Sales']].head())
mae = mean_absolute_error(y, df['Predicted_Sales'])
r2 = r2_score(y, df['Predicted_Sales'])

print("MAE:", mae)
print("R² Score:", r2)
future_months = np.array(range(len(df)+1, len(df)+7)).reshape(-1,1)

future_predictions = model.predict(future_months)

for i, pred in enumerate(future_predictions, start=1):
    print(f"Month {len(df)+i}: {pred:.2f}")
plt.figure(figsize=(10,5))

plt.plot(df['Month_Number'], df['Sales'], label='Actual Sales')
plt.plot(df['Month_Number'], df['Predicted_Sales'], label='Predicted Sales')

plt.xlabel('Month Number')
plt.ylabel('Sales')
plt.title('Sales Forecasting')
plt.legend()

plt.show()
