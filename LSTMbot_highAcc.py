# Importing the necessary libraries
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
from sklearn.metrics import mean_squared_error

if __name__ == "__main__":
    # Setting the params of the stock data intended to download (e.g. TSLA)
    start_date = datetime(2010, 1, 1)
    end_date = datetime(2020, 1, 1)
    ticker_symbol = "TLSA"
    # Define the train and test split ratio
    train_split = 0.7
    # Define the number of data points to use as input features for the model
    data_set_points = 21

# Download the stock data from Yahoo Finance
stock_df = yf.download(tickers=ticker_symbol, start=start_date, end=end_date)
# Create a new dataframe with only the adjusted close price column
new_df = stock_df[['Adj Close']].copy()

# Define a function to prepare the train and test data for the model
def prepare_train_test_split(new_df, data_set_points, train_split):
    # Reset the index of the dataframe and drop the first row
    new_df.reset_index(inplace=True)
    new_df.drop(0, inplace=True)
    # Split the data into train and test sets based on the train_split ratio
    split_index = int(len(new_df) * train_split)
    train_data = new_df[:split_index]
    test_data = new_df[split_index:].reset_index(drop=True)
    # Calculate the difference between consecutive values of the adjusted close price for both train and test sets
    train_diff = train_data['Adj Close'].diff().dropna().values
    test_diff = test_data['Adj Close'].diff().dropna().values
    # Create the input features for the model by taking data_set_points number of values from the train_diff array
    X_train = np.array([train_diff[i : i + data_set_points] for i in range(len(train_diff) - data_set_points)])
    # Create the output labels for the model by taking the next value after data_set_points number of values from the train_diff array
    y_train = np.array([train_diff[i + data_set_points] for i in range(len(train_diff) - data_set_points)])
    # Create a validation set from the last 10% of the train_data
    y_valid = train_data['Adj Close'].tail(len(y_train) // 10).values
    # Reshape the validation set to match the output shape of the model
    y_valid = y_valid.reshape(-1, 1)
    # Create the input features for the test set by taking data_set_points number of values from the test_diff array
    X_test = np.array([test_diff[i : i + data_set_points] for i in range(len(test_diff) - data_set_points)])
    # Create the output labels for the test set by taking the next value after data_set_points number of values from the test_data dataframe
    y_test = test_data['Adj Close'].shift(-data_set_points).dropna().values
    # Return the train and test sets as numpy arrays
    return X_train, y_train, X_test, y_test, test_data

# Define a function to create and train an LSTM model for the stock price prediction
def create_lstm_model(X_train, y_train, data_set_points):
    # Set the random seed for reproducibility
    tf.random.set_seed(20)
    np.random.seed(10)
    # Define the input layer of the model with the shape of the input features
    lstm_input = Input(shape=(data_set_points, 1), name='lstm_input')
    # Add the first LSTM layer with 21 units and return the full sequence of outputs
    inputs = LSTM(21, name='lstm_0', return_sequences=True)(lstm_input)
    # Add a dropout layer with 0.1 dropout rate to prevent overfitting
    inputs = Dropout(0.1, name='dropout_0')(inputs)
    # Add the second LSTM layer with 32 units and return only the last output
    inputs = LSTM(32, name='lstm_1')(inputs)
    # Add another dropout layer with 0.05 dropout rate to prevent overfitting
    inputs = Dropout(0.05, name='dropout_1')(inputs)
    # Add a dense layer with 32 units and a linear activation function
    inputs = Dense(32, name='dense_0')(inputs)
    # Add another dense layer with 1 unit and a linear activation function
    inputs = Dense(1, name='dense_1')(inputs)
    # Add the output layer with a linear activation function
    output = Activation('linear', name='output')(inputs)
    # Create the model object with the input and output layers
    model = Model(inputs=lstm_input, outputs=output)
    # Define the Adam optimizer with a learning rate of 0.002
    adam = optimizers.Adam(lr=0.002)
    # Compile the model with the optimizer and the mean squared error loss function
    model.compile(optimizer=adam, loss='mse')
    # Fit the model on the train data with a batch size of 15, 25 epochs, and a validation split of 0.1
    model.fit(x=X_train, y=y_train, batch_size=15, epochs=25, shuffle=True, validation_split=0.1)
    # Return the trained model
    return model

# Prepare the train and test data sets by calling the prepare_train_test_split function
X_train, y_train, X_test, y_test, test_data = prepare_train_test_split(new_df, data_set_points, train_split)
# Create and train the LSTM model by calling the create_lstm_model function
model = create_lstm_model(X_train, y_train, data_set_points)
# Predict the test data using the model
y_pred = model.predict(X_test)
# Flatten the prediction array to a one-dimensional array
y_pred = y_pred.flatten()
# Get the actual prices of the test data by taking the adjusted close price column from the test_data dataframe
actual1 = np.array([test_data['Adj Close'][i + data_set_points] for i in range(len(test_data) - data_set_points)])
# Get the actual prices of the test data except the last one
actual2 = actual1[:-1]
# Adding each actual price at time t with the predicted difference to get a predicted price at time t + 1
data = np.add(actual2, y_pred)
# Set the size of the plot
plt.gcf().set_size_inches(12, 8, forward=True)
# Set the title of the plot
plt.title('Plot of real price and predicted price against number of days for test set')
# Set the x-axis label of the plot
plt.xlabel('Number of days')
# Set the y-axis label of the plot
plt.ylabel('Adjusted Close Price($)')
# Plot the actual prices of the test data starting from the second one
plt.plot(actual1[1:], label='Actual Price')
# Plot the predicted prices of the test data
plt.plot(data, label='Predicted Price')
# Print the root mean squared error between the actual and predicted prices
print(mean_squared_error(actual1[1:], data, squared = False))
# Add a legend to the plot
plt.legend(['Actual Price', 'Predicted Price'])
# Show the plot
plt.show()