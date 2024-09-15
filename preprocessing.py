import pandas as pd
from datetime import datetime 
import yfinance as yf 
import matplotlib.pyplot as plt 
import numpy as np
import preprocess_data as ppd
import visualize


def get_historical_data(symbol,start_date,end_date):
    ''' Daily quotes from Google. Date format='yyyy-mm-dd' '''
    symbol = symbol.upper()
    start = datetime(int(start_date[0:4]), int(start_date[5:7]), int(start_date[8:10]))
    end = datetime(int(end_date[0:4]), int(end_date[5:7]), int(end_date[8:10]))

    # get the data 
    df = yf.download(symbol, start = start_date, end = end_date) 
    df = df.drop('Adj Close', axis=1)
    return df

df = get_historical_data('GOOGL','2005-01-01','2017-06-30')
df.to_csv('google.csv', index=False)

data = pd.read_csv('google.csv')

# print("\n")
# print("Open   --- mean :", np.mean(data['Open']),  "  \t Std: ", np.std(data['Open']),  "  \t Max: ", np.max(data['Open']),  "  \t Min: ", np.min(data['Open']))
# print("High   --- mean :", np.mean(data['High']),  "  \t Std: ", np.std(data['High']),  "  \t Max: ", np.max(data['High']),  "  \t Min: ", np.min(data['High']))
# print("Low    --- mean :", np.mean(data['Low']),   "  \t Std: ", np.std(data['Low']),   "  \t Max: ", np.max(data['Low']),   "  \t Min: ", np.min(data['Low']))
# print("Close  --- mean :", np.mean(data['Close']), "  \t Std: ", np.std(data['Close']), "  \t Max: ", np.max(data['Close']), "  \t Min: ", np.min(data['Close']))
# print("Volume --- mean :", np.mean(data['Volume']),"  \t Std: ", np.std(data['Volume']),"  \t Max: ", np.max(data['Volume']),"  \t Min: ", np.min(data['Volume']))


stocks = data.drop('High', axis=1)
stocks = stocks.drop('Low', axis=1)
stocks['Item'] = np.arange(len(stocks))

print(stocks.head())
print("---")
print(stocks.tail())

visualize.plot_basic(stocks)

stocks = ppd.get_normalised_data(stocks)
print(stocks.head())

print("\n")
print("Open   --- mean :", np.mean(stocks['Open']),  "  \t Std: ", np.std(stocks['Open']),  "  \t Max: ", np.max(stocks['Open']),  "  \t Min: ", np.min(stocks['Open']))
print("Close  --- mean :", np.mean(stocks['Close']), "  \t Std: ", np.std(stocks['Close']), "  \t Max: ", np.max(stocks['Close']), "  \t Min: ", np.min(stocks['Close']))
print("Volume --- mean :", np.mean(stocks['Volume']),"  \t Std: ", np.std(stocks['Volume']),"  \t Max: ", np.max(stocks['Volume']),"  \t Min: ", np.min(stocks['Volume']))

visualize.plot_basic(stocks)
stocks.to_csv('google_preprocessed.csv',index= False)
