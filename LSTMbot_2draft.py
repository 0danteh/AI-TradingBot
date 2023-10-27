import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from keras.models import Sequential 
from keras.layers import Dense, LSTM, Dropout, Activation
from keras.layers import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit

# Get the ticker symbol 
ticker_symbol = 'NFLX'

# Set the start and end date for the data
end_date = datetime.today()
start_date = end_date - timedelta(365*100)

# Download the data
df = yf.download(tickers=ticker_symbol, start=start_date, end=end_date)

# Setting the variables since the multivariate nature of LSTM
# Set the target value
target_value_y = df['Close']
# Set the "input" values
X_starting_values = df.iloc[:,0:3]

# Scaling (testing with SS)
ss = StandardScaler()
X_scaled = ss.fit_transform(X_starting_values)
X_scaled = pd.DataFrame(columns=X_starting_values.columns, data=X_scaled, index=X_starting_values.index)

def nnLSTM_split_func(data, n_steps):
    # Initialize empty lists to store input (X) and output (y) sequences
    X, y = [], []
    # Iterate through the 'data' array, considering 'n_steps' elements at a time
    for i in range(len(data) - n_steps + 1):
        # Extract a sequence of 'n_steps' rows, excluding the last column, for input (X)
        X.append(data[i:i + n_steps, :-1])
        # Extract the last value in the 'i + n_steps - 1' row for output (y)
        y.append(data[i + n_steps - 1, -1])
    # Convert the lists into NumPy arrays for efficient computation
    return np.array(X), np.array(y)

# Preparing data
X_1, y_1 = nnLSTM_split_func(df.values, n_steps=10)

# Define the train-test split ratio (80% for training, 20% for testing)
train_split = 0.8

# Calculate the index at which the dataset will be split
split_idx = int(np.ceil(len(X_1) * train_split))

# Get the date index from the DataFrame
date_index = df.index

# Split the feature data (X) into training and testing sets
X_train, X_test = X_1[:split_idx], X_1[split_idx:]

# Split the target data (y) into training and testing sets
y_train, y_test = y_1[:split_idx], y_1[split_idx:]

# Split the date index into training and testing sets for reference
X_train_date, X_test_date = date_index[:split_idx], date_index[split_idx:]

lstm = Sequential()
lstm.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), activation='relu', return_sequences=True))
lstm.add(Dense(1))
lstm.compile(loss='mean_squared_error', optimizer='adam')

arch=lstm.fit(X_train, y_train, epochs=100, batch_size=4, verbose=2, shuffle=False)
y_pred = lstm.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
mape = mean_absolute_percentage_error(y_test, y_pred)

print("RMSE: ", rmse)
print("MAPE: ", mape)