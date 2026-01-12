
from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import RandomForestRegressor
from datetime import timedelta

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    stock_symbol = request.form['stock_symbol'].upper()

    try:
        #df = yf.download(stock_symbol, period="5y", interval="1d")

        df = yf.download(stock_symbol, period="5y", interval="1d")

        prices = df['Close'].dropna().values

        if len(prices) < 101:
            raise Exception("Not enough data to make predictions (need at least 101 days).")

        window_size = 100

        X, y = [], []
        for i in range(len(prices) - window_size):
            X.append(prices[i:i + window_size].flatten())  
            y.append(prices[i + window_size])

        X = np.array(X)
        y = np.array(y)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Predict next 5 days
        future_predictions = []
        last_window = prices[-window_size:].flatten() 

        for _ in range(5):
            next_price = model.predict([last_window])[0]  
            future_predictions.append(next_price)
            last_window = np.append(last_window[1:], next_price)  

        
        future_dates = []
        last_date = df.index[-1].to_pydatetime()
  
        while len(future_dates) < 5:
            last_date += timedelta(days=1)
            if last_date.weekday() < 5: 
                future_dates.append(last_date)

        future_dates_str = [date.strftime('%d %b') for date in future_dates]
        prediction_text = "Predicted prices for the next 5 days: "
        prediction_text += ", ".join([f"{date} - ${price:.2f}" for date, price in zip(future_dates_str, future_predictions)])

       
        dates = df.index[-len(prices):]  
        plt.figure(figsize=(10, 6))
        # Historical
        plt.plot(dates, prices, label='Actual Prices', color='blue')
        
        model_fit_dates = dates[window_size:]
        model_fit_prices = model.predict(X)
        plt.plot(model_fit_dates, model_fit_prices, label='Model Fit', color='red')
        
        plt.scatter(future_dates, future_predictions, label='Predicted Prices', color='green')

        plt.title(f'Random Forest 5-Day Forecast for {stock_symbol}')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.legend()
        plt.tight_layout()

       
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('index.html', prediction_text=prediction_text, plot_url=plot_url)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=10000)
