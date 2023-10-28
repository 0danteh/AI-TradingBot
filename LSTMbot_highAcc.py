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

start_date = datetime(2010, 9, 1)
end_date = datetime(2020, 8, 31)
ticker_symbol = "TSLA"

#Download the stock data from Yahoo Finance
stock_df = yf.download(tickers=ticker_symbol, start=start_date, end=end_date)

new_df = stock_df[['Adj Close']].copy()

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

def create_lstm_model(X_train, y_train, data_set_points):
    #Set the random seed for reproducibility
    tf.random.set_seed(20)
    np.random.seed(10)

    lstm_input = Input(shape=(data_set_points, 1), name='lstm_input')

    inputs = LSTM(21, name='lstm_0', return_sequences=True)(lstm_input)

    inputs = Dropout(0.1, name='dropout_0')(inputs)
    inputs = LSTM(32, name='lstm_1')(inputs)
    inputs = Dropout(0.05, name='dropout_1')(inputs) #Dropout layers to prevent overfitting
    inputs = Dense(32, name='dense_0')(inputs)
    inputs = Dense(1, name='dense_1')(inputs)
    output = Activation('linear', name='output')(inputs)

    model = Model(inputs=lstm_input, outputs=output)
    adam = optimizers.Adam(lr=0.002)

    model.compile(optimizer=adam, loss='mse')
    model.fit(x=X_train, y=y_train, batch_size=15, epochs=25, shuffle=True, validation_split=0.1)

    return model

train_split = 0.7

data_set_points = 21

#Prepare the train and test data sets
X_train, y_train, X_test, y_test, test_data = prepare_train_test_split(new_df, data_set_points, train_split)

#Create and train the LSTM model
model = create_lstm_model(X_train, y_train, data_set_points)

#Predict the test data using the model
y_pred = model.predict(X_test)

y_pred = y_pred.flatten()

#Get the actual prices of the test data
actual = np.array([test_data['Adj Close'][i + data_set_points] for i in range(len(test_data) - data_set_points)])

temp_actual = actual[:-1]

#Adding each actual price at time t with the predicted difference to get a predicted price at time t + 1
new = np.add(temp_actual, y_pred)

plt.gcf().set_size_inches(12, 8, forward=True)
plt.title('Plot of real price and predicted price against number of days for test set')
plt.xlabel('Number of days')
plt.ylabel('Adjusted Close Price($)')

plt.plot(actual[1:], label='Actual Price')
plt.plot(new, label='Predicted Price')

print(mean_squared_error(actual[1:], new, squared = False))

#plotting of model
plt.legend(['Actual Price', 'Predicted Price'])


plt.show()
