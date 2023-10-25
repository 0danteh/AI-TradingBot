import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import keras.api._v2.keras as keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, LSTM

# Get the ticker symbol 
ticker_symbol = 'TSLA'

# Set the start and end date for the data
end_date = datetime.today()
start_date = end_date - timedelta(365*50)

# Download the data
df = yf.download(tickers=ticker_symbol, start=start_date, end=end_date)

# Preprocess the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df["Close"].values.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

# Define a function to create input sequences and labels
def create_sequences(data, window):
    # Use list comprehension to generate input sequences and labels
    return np.array([data[i-window:i, 0] for i in range(window, len(data))]), np.array([data[i, 0] for i in range(window, len(data))])

# Create input sequences and labels
window = 60
train_x, train_y = create_sequences(train_data, window)
test_x, test_y = create_sequences(test_data, window)

# Step 3: Building Your Stock Trading Bot
# Define the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(train_x.shape[1], 1)))
model.add(LSTM(units=50))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy")

# Train the model
model.fit(train_x, train_y, epochs=10, batch_size=32)

# Step 4: Evaluating Your Stock Trading Bot
# Evaluate the model on the test set
test_loss = model.evaluate(test_x, test_y)

# Step 5: Using Your Stock Trading Bot
# Make predictions on the test set
predictions = model.predict(test_x)
predictions = scaler.inverse_transform(predictions)

# Convert the predictions and actual values to pandas DataFrame
predictions_df = pd.DataFrame(predictions, index=df.index[train_size+window:])
actual_df = pd.DataFrame(scaler.inverse_transform(test_y.reshape(-1, 1)), index=df.index[train_size+window:])

# Compare the predictions and actual values
comparison_df = pd.concat([actual_df, predictions_df], axis=1)
comparison_df.columns = ["Actual", "Predicted"]
print(comparison_df)

# Plot the actual vs. predicted values
plt.plot(actual_df, label="Actual")
plt.plot(predictions_df, label="Predicted")
plt.xlabel("Date")
plt.ylabel("Stock Price")
plt.title("Actual vs. Predicted Stock Prices")
plt.legend()
plt.show()