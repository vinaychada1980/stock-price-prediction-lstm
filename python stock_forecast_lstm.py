# Stock Price Forecast with LSTM (NVDA, AAPL, MSFT, TSLA)

!pip install yfinance tensorflow scikit-learn matplotlib

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# -------------------
# Parameters
# -------------------
tickers = ["NVDA", "AAPL", "MSFT", "TSLA"]
lookback = 60   # number of past days to look at
forecast_horizon = 7  # predict next 7 days
epochs = 20
batch_size = 32

def download_data(ticker):
    df = yf.download(ticker, period="5y", interval="1d")
    df = df[["Close"]]
    df.dropna(inplace=True)
    return df

def prepare_data(df, lookback):
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(lookback, len(scaled)-forecast_horizon):
        X.append(scaled[i-lookback:i, 0])
        y.append(scaled[i:i+forecast_horizon, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

def build_lstm(input_shape, forecast_horizon):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(forecast_horizon))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def forecast_stock(ticker):
    print(f"\n=== {ticker} ===")
    df = download_data(ticker)
    X, y, scaler = prepare_data(df, lookback)
    
    model = build_lstm((X.shape[1], 1), forecast_horizon)
    model.fit(X, y, epochs=epochs, batch_size=batch_size, verbose=0)
    
    # last sequence for forecasting
    last_seq = df[-lookback:].values
    scaled_seq = scaler.transform(last_seq)
    scaled_seq = np.reshape(scaled_seq, (1, lookback, 1))

    pred_scaled = model.predict(scaled_seq)
    pred = scaler.inverse_transform(pred_scaled).flatten()

    # Plot
    plt.figure(figsize=(10,5))
    plt.plot(df.index, df["Close"], label="History")
    future_dates = pd.bdate_range(start=df.index[-1], periods=forecast_horizon+1)[1:]
    plt.plot(future_dates, pred, "r--", marker="o", label="Forecast (7d)")
    plt.title(f"{ticker} LSTM Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid()
    plt.show()

    return pred

# -------------------
# Run forecasts
# -------------------
results = {}
for t in tickers:
    results[t] = forecast_stock(t)

print("\nForecast results (next 7 trading days):")
for t, preds in results.items():
    print(f"{t}: {np.round(preds, 2)}")
