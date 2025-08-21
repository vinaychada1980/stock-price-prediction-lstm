# stock-price-prediction-lstm

This project uses deep learning (LSTM) to predict the next 7 trading days of stock prices
for **NVDA, AAPL, MSFT, TSLA** using 5 years of historical data from Yahoo Finance.

## Features
- Fetches 5 years of daily stock data with `yfinance`
- Trains an LSTM model for each ticker
- Forecasts the next 7 trading days
- Plots history + forecast

## Installation
```bash
git clone https://github.com/vinaychada1980/stock-price-prediction-lstm.git
cd stock-price-prediction-lstm
pip install -r requirements.txt
