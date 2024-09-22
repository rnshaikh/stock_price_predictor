import json
import os
import argparse
import math
import pandas as pd
import pathlib
import joblib
import numpy as np
from IPython.display import display
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

import visualize as vs
import stock_data as sd


def build_model(X, y):
    """
    build a linear regression model using sklearn.linear_model
    :param X: Feature dataset
    :param y: label dataset
    :return: a linear regression model
    """
    linear_mod = linear_model.LinearRegression()  # defining the linear regression model
    X = np.reshape(X, (X.shape[0], 1))
    y = np.reshape(y, (y.shape[0], 1))
    linear_mod.fit(X, y)  # fitting the data points in the model

    return linear_mod


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_dir", type=str, default=os.environ.get("SM_CHANNEL_INPUT"))
    #parser.add_argument("--output_data_dir", type=str, default=os.environ.get("SM_OUTPUT_DATA_DIR"))
    parser.add_argument("--model_dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    args = parser.parse_args()

    stocks = pd.read_csv(f"{args.input_data_dir}/linear_train.csv")
    display(stocks.head())

    X_train = stocks[stocks.columns[0]].to_numpy()
    y_train = stocks[stocks.columns[1]].to_numpy()

    #X_train, X_test, y_train, y_test, label_range= sd.train_test_split_linear_regression(stocks)

    # print("x_train", X_train.shape)
    # print("y_train", y_train.shape)
    # print("x_test", X_test.shape)
    # print("y_test", y_test.shape)
    #print("label_range", label_range)

    model = build_model(X_train,y_train)

    #predictions = LinearRegressionModel.predict_prices(model,X_test, label_range)
    
    #vs.plot_prediction(y_test,predictions)

    trainScore = mean_squared_error(X_train, y_train)
    print('Train Score: %.4f MSE (%.4f RMSE)' % (trainScore, math.sqrt(trainScore)))

    # testScore = mean_squared_error(predictions, y_test)
    # print('Test Score: %.8f MSE (%.8f RMSE)' % (testScore, math.sqrt(testScore)))

    # testStd = np.std(y_test - predictions)
    # print("Test Std Deviation", testStd)

    
    # report_dict = {
    #     "regression_metrics": {
    #         "mse": {"value": testScore, "standard_deviation": testStd},
    #     },
    # }
    # evaluation_path = f"{args.output_data_dir}/linear_evaluation.json"

    # with open(evaluation_path, "w") as f:
    #     f.write(json.dumps(report_dict))

    model_location = args.model_dir + "/linearregression-model"

    with open(model_location, "wb") as f:
        joblib.dump(model, f)