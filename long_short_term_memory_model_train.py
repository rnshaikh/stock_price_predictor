import json
import os
import argparse
import math
import pandas as pd
import pathlib
import joblib
import numpy as np
from IPython.display import display

from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.models import Sequential
from keras.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

import lstm, time #helper libraries

import visualize as vs
import stock_data as sd
import LinearRegressionModel



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    #parser.add_argument("--preprocessed_data_dir", type=str, default=os.environ.get("SM_CHANNEL_PREPROCESSED"))
    parser.add_argument("--input_data_dir", type=str, default=os.environ.get("SM_CHANNEL_INPUT"))
    #parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()

    
    # stocks = pd.read_csv(f"{args.input_data_dir}/lstm_train.csv")
    # display(stocks.head())

    X_train = pd.read_csv(f"{args.input_data_dir}/lstm_train_x.csv")
    y_train = pd.read_csv(f"{args.input_data_dir}/lstm_train_y.csv")

    import pdb
    pdb.set_trace()
    X_train = X_train[X_train.columns[1:]].to_numpy()
    y_train = y_train[y_train.columns[0]].to_numpy()

    #stocks = pd.read_csv(f"{args.preprocessed_data_dir}/google_preprocessed.csv")
    #stocks_data = stocks.drop(['Item'], axis =1)

    #display(stocks_data.head())

    import pdb
    pdb.set_trace()

    #X_train, X_test,y_train, y_test = sd.train_test_split_lstm(stocks_data, 5)

    unroll_length = 50
    X_train = sd.unroll(X_train, unroll_length)
    #X_test = sd.unroll(X_test, unroll_length)
    y_train = y_train[-X_train.shape[0]:]
    #y_test = y_test[-X_test.shape[0]:]

    print("x_train", X_train.shape)
    print("y_train", y_train.shape)
    #print("x_test", X_test.shape)
    #print("y_test", y_test.shape)



    # build basic lstm model
    model = lstm.build_basic_model(input_dim = X_train.shape[-1], output_dim = unroll_length, return_sequences=True)

    # Compile the model
    start = time.time()
    model.compile(loss='mean_squared_error', optimizer='adam')
    print('compilation time : ', time.time() - start)


    model.fit(
        X_train,
        y_train,
        epochs=1,
        validation_split=0.05)


    #predictions = model.predict(X_test)

    #vs.plot_lstm_prediction(y_test,predictions)


    trainScore = model.evaluate(X_train, y_train, verbose=0)
    print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

    #testScore = model.evaluate(X_test, y_test, verbose=0)
    #print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))

    #testStd = np.std(y_test - predictions)
    #print('Std Deviation',testStd)

    model_location = args.model_dir + "/lstm-model"

    with open(model_location, "wb") as f:
        joblib.dump(model, f)


    # report_dict = {
    #     "regression_metrics": {
    #         "mse": {"value": testScore, "standard_deviation": testStd},
    #     },
    # }
    # evaluation_path = f"{args.output_data_dir}/lstm_evaluation.json"

    # with open(evaluation_path, "w") as f:
    #     f.write(json.dumps(report_dict))


    # # Set up hyperparameters
    # batch_size = 100
    # epochs = 5

    # # build improved lstm model
    # model = lstm.build_improved_model( X_train.shape[-1],output_dim = unroll_length, return_sequences=True)

    # start = time.time()
    # #final_model.compile(loss='mean_squared_error', optimizer='adam')
    # model.compile(loss='mean_squared_error', optimizer='adam')
    # print('compilation time : ', time.time() - start)


# model.fit(X_train, 
#           y_train, 
#           batch_size=batch_size,
#           epochs=epochs,
#           verbose=2,
#           validation_split=0.05
#          )

# predictions = model.predict(X_test, batch_size=batch_size)

# vs.plot_lstm_prediction(y_test,predictions)


# trainScore = model.evaluate(X_train, y_train, verbose=0)
# print('Train Score: %.8f MSE (%.8f RMSE)' % (trainScore, math.sqrt(trainScore)))

# testScore = model.evaluate(X_test, y_test, verbose=0)
# print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))
