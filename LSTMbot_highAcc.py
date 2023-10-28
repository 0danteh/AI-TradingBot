import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing
import numpy as np
import keras
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
from keras.callbacks import History
from sklearn.metrics import mean_squared_error

def prepare_train_test_split(new_df, data_set_points, train_split):
    new_df = new_df.reset_index().drop(0)

    #Split the data into train and test sets.
    split_index = int(new_df.shape[0] * train_split)

    train_data = new_df.iloc[:split_index]
    test_data = new_df.iloc[split_index:].reset_index(drop=True)

    train_diff = np.diff(train_data['Adj Close'].values)
    test_diff = np.diff(test_data['Adj Close'].values)

    X_train = np.array([train_diff[i : i + data_set_points] for i in range(len(train_diff) - data_set_points)])

    y_train = np.array([train_diff[i + data_set_points] for i in range(len(train_diff) - data_set_points)])

    y_valid = np.array([train_data['Adj Close'].tail(len(y_train) // 10)])

    y_valid = y_valid.reshape(-1, 1)

    X_test = np.array([test_diff[i : i + data_set_points] for i in range(len(test_diff) - data_set_points)])

    y_test = np.array([test_data['Adj Close'][i + data_set_points] for i in range(len(test_diff) - data_set_points)])

    return X_train, y_train, X_test, y_test, test_data