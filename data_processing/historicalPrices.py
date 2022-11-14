import pandas as pd
import yfinance as yf

df = pd.read_csv("data/S&P100_symbols.csv")

price_history = yf.download(list(df['Symbol']), start="2021-10-22", end="2022-10-22")
price_history = price_history['Close']
price_history = price_history.iloc[::-1]
price_history.to_csv("data/historical_prices.csv", index=True)
