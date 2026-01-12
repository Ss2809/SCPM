import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor

# Step 1: Download Stock Data from yfinance
stock_symbol = input("Enter the stock symbol (e.g., AAPL, MSFT): ")
data = yf.download(stock_symbol, start="2020-01-01", end="2025-04-12")
print(data)

# Use only the 'Close' prices for analysis
prices = data['Close'].dropna().values

# Define window size
window_size = 100  # Using past 5 days to predict the next day

# Create input (X) and output (y) for training
X, y = [], []
for i in range(len(prices) - window_size):
    X.append(prices[i:i + window_size].flatten())  # ✅ Flatten to 1D before adding
    y.append(prices[i + window_size])  # Next day's price

X = np.array(X)  # Shape: (samples, window_size)
y = np.array(y)  # Shape: (samples,)

# Step 2: Apply Random Forest Regression
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save the model as 'model.pkl'
joblib.dump(model, 'model.pkl')

# Step 3: Predict the next 5 days
future_predictions = []
last_window = prices[-window_size:].flatten()  # Last 5 days as input

for _ in range(5):  # Predict 5 days into the future
    next_price = model.predict([last_window])[0]
    future_predictions.append(next_price)
    last_window = np.append(last_window[1:], next_price)  # Shift the window



# Print predicted future prices
print("\nPredicted Future Prices:")
for i, price in enumerate(future_predictions, 1):
    print(f"Day {i}: {float(price):.2f}")


# Step 4: Plot Results
dates = np.arange(len(prices))
future_dates = np.arange(len(prices), len(prices) + 5)

plt.figure(figsize=(10, 6))
plt.plot(dates, prices, label='Actual Prices', color='blue')
plt.plot(dates[window_size:], model.predict(X), label='Regression Line (Random Forest)', color='red')
plt.scatter(future_dates, future_predictions, label='Predicted Prices', color='green')
plt.title(f'Random Forest Regression with Future Predictions for {stock_symbol}')
plt.xlabel('Time (Days)')
plt.ylabel('Stock Price')
plt.legend()
plt.show()



