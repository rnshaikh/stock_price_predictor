import json
import os
import pandas as pd
import argparse
import pathlib
import stock_data as sd
from datetime import datetime 
import yfinance as yf 
import matplotlib.pyplot as plt 
import numpy as np
import preprocess_data as ppd
import stock_data as sd
import visualize


def get_historical_data(symbol,start_date,end_date):
    symbol = symbol.upper()
    start = datetime(int(start_date[0:4]), int(start_date[5:7]), int(start_date[8:10]))
    end = datetime(int(end_date[0:4]), int(end_date[5:7]), int(end_date[8:10]))

    # get the data 
    df = yf.download(symbol, start = start_date, end = end_date) 
    df = df.drop('Adj Close', axis=1)
    return df



if __name__ == "__main__":

    #parser = argparse.ArgumentParser()
    #parser.add_argument("model-type", type=str, default="linear_regression")
    #args, _ = parser.parse_known_args()

    path = "opt/ml/processing/"

    df = get_historical_data('GOOGL','2005-01-01','2017-06-30')
    raw_file_path = os.path.join(path, "google.csv")
    df.to_csv(raw_file_path, index=False)
    data = pd.read_csv(raw_file_path)


    stocks = data.drop('High', axis=1)
    stocks = stocks.drop('Low', axis=1)
    stocks['Item'] = np.arange(len(stocks))

    stocks = ppd.get_normalised_data(stocks)
    print(stocks.head())

    print("\n")
    print("Open   --- mean :", np.mean(stocks['Open']),  "  \t Std: ", np.std(stocks['Open']),  "  \t Max: ", np.max(stocks['Open']),  "  \t Min: ", np.min(stocks['Open']))
    print("Close  --- mean :", np.mean(stocks['Close']), "  \t Std: ", np.std(stocks['Close']), "  \t Max: ", np.max(stocks['Close']), "  \t Min: ", np.min(stocks['Close']))
    print("Volume --- mean :", np.mean(stocks['Volume']),"  \t Std: ", np.std(stocks['Volume']),"  \t Max: ", np.max(stocks['Volume']),"  \t Min: ", np.min(stocks['Volume']))

    #visualize.plot_basic(stocks)

    processed_file_path = os.path.join(path, "google_preprocessed.csv")
    stocks.to_csv(processed_file_path, index= False)

    X_train, X_test, y_train, y_test, label_range= sd.train_test_split_linear_regression(stocks)


    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)

    y_train = pd.DataFrame(y_train)
    y_test = pd.DataFrame(y_test)


    train_df = pd.concat([y_train, X_train], axis = 1)
    test_df = pd.concat([y_test, X_test], axis = 1)

    train_output_path = os.path.join(f"{path}/train", "linear_train.csv")
    test_output_path = os.path.join(f"{path}/test", "linear_test.csv")
    train_label_path = os.path.join(f"{path}/test", "linear_label.json")

    train_df.to_csv(train_output_path, index=False)
    test_df.to_csv(test_output_path, index=False)

    with open(train_label_path, 'w') as f:
        json.dump(label_range, f)


    # stocks_data = stocks.drop(['Item'], axis =1)
    # X_train, X_test,y_train, y_test = sd.train_test_split_lstm(stocks_data, 5)   
    
    # train_x_output_path = os.path.join(f"{path}/train", "lstm_train_x.csv")
    # train_y_output_path = os.path.join(f"{path}/train", "lstm_train_y.csv")

    # test_x_output_path = os.path.join(f"{path}/test", "lstm_test_x.csv")
    # test_y_output_path = os.path.join(f"{path}/test", "lstm_test_y.csv")

    # X_train = pd.DataFrame(X_train)
    # X_test = pd.DataFrame(X_test)

    # y_train = pd.DataFrame(y_train)
    # y_test = pd.DataFrame(y_test)

    
    # X_train.to_csv(train_x_output_path, index=False)
    # y_train.to_csv(train_y_output_path, index=False)


    # X_test.to_csv(test_x_output_path, index=False)
    # y_test.to_csv(test_y_output_path, index=False)
