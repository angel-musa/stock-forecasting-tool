import pandas as pd
import yfinance as yf

filepath_stockdata = r"C:\Users\xange\OneDrive\Desktop\Angel\halo documentation\data new.xlsx"
sheet_name1 = 'short'

def load_tickers():
    df = pd.read_excel(filepath_stockdata, sheet_name=sheet_name1, usecols=['Symbol'])
    return df['Symbol'].tolist()

def load_stock_data(tickers):
    stock_data = {}
    for ticker in tickers:
        data = yf.download(ticker, period="1y")  # Default to 1 year
        stock_data[ticker] = data
    return stock_data
