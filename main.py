import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.ensemble import RandomForestRegressor


stock_symbol = input("Enter the stock symbol (e.g., AAPL, MSFT): ")
data = yf.download(stock_symbol, period="5y", interval="1d")


print(data)


prices = data['Close'].dropna().values


window_size = 100  

X, y = [], []
for i in range(len(prices) - window_size):
    X.append(prices[i:i + window_size].flatten()) 
    y.append(prices[i + window_size])  

X = np.array(X)  
y = np.array(y)  

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)


joblib.dump(model, 'model.pkl')

future_predictions = []
last_window = prices[-window_size:].flatten() 

for _ in range(5):  
    next_price = model.predict([last_window])[0]
    future_predictions.append(next_price)
    last_window = np.append(last_window[1:], next_price) 


print("\nPredicted Future Prices:")
for i, price in enumerate(future_predictions, 1):
    print(f"Day {i}: {float(price):.2f}")



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



