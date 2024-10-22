!pip install tpot
!pip install ta

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tpot import TPOTRegressor
import yfinance as yf
from datetime import datetime
from sklearn.model_selection import train_test_split
import ta

# User input for ticker
ticker = input("Enter the stock ticker (e.g., 'MSFT'): ")

# Automatically update the end date to today's date
current_date = datetime.now().strftime('%Y-%m-%d')

# Load stock data from Yahoo Finance
stock_data = yf.download(ticker, start='2010-07-26', end=current_date)
stock_data['Date'] = stock_data.index
stock_data.set_index('Date', inplace=True)

# Generate Moving Averages (20 and 50 days)
stock_data['SMA_20'] = stock_data['Close'].rolling(window=20).mean()
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()

# Remove rows with missing values
stock_data.dropna(inplace=True)

# Calculate RSI and MACD indicators
stock_data['RSI'] = ta.momentum.RSIIndicator(stock_data['Close'], window=14).rsi()
macd_calc = ta.trend.MACD(stock_data['Close'])
stock_data['MACD'] = macd_calc.macd()
stock_data['Signal_Line'] = macd_calc.macd_signal()

# Define trading signals based on RSI and MACD values
stock_data['Buy_Signal'] = (stock_data['RSI'] < 40) & (stock_data['MACD'] > stock_data['Signal_Line'])
stock_data['Sell_Signal'] = (stock_data['RSI'] > 60) & (stock_data['MACD'] < stock_data['Signal_Line'])

# Plot closing prices along with moving averages
plt.figure(figsize=(16, 8))
plt.plot(stock_data['Close'], label='Closing Price', color='blue')
plt.plot(stock_data['SMA_20'], label='20-Day SMA', color='orange')
plt.plot(stock_data['SMA_50'], label='50-Day SMA', color='green')
plt.title(f'{ticker} Stock Price with Moving Averages')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Highlight Buy/Sell signals on the price chart
plt.figure(figsize=(16, 8))
plt.plot(stock_data['Close'], label='Closing Price', color='blue')
plt.scatter(stock_data.index[stock_data['Buy_Signal']], stock_data['Close'][stock_data['Buy_Signal']], label='Buy', marker='^', color='green', alpha=1)
plt.scatter(stock_data.index[stock_data['Sell_Signal']], stock_data['Close'][stock_data['Sell_Signal']], label='Sell', marker='v', color='red', alpha=1)
plt.title(f'Buy and Sell Signals for {ticker}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Scale the feature data for the TPOT model
input_features = stock_data[['Close', 'SMA_20', 'SMA_50']].values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(input_features)

# Create training and testing datasets
train_size = int(len(scaled_features) * 0.8)
train_set, test_set = scaled_features[:train_size], scaled_features[train_size:]

# Function to prepare the dataset for model training
def prepare_data(dataset, step_size):
    X, y = [], []
    for i in range(len(dataset) - step_size - 1):
        X.append(dataset[i:i + step_size].flatten())
        y.append(dataset[i + step_size, 0])
    return np.array(X), np.array(y)

# Set the time step and prepare the data
time_step = 30
X_train, y_train = prepare_data(train_set, time_step)
X_test, y_test = prepare_data(test_set, time_step)

# Split the training data into training and validation sets
X_train_part, X_val, y_train_part, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Initialize and train the TPOT AutoML model
tpot_model = TPOTRegressor(generations=5, population_size=50, verbosity=2, random_state=42)
tpot_model.fit(X_train_part, y_train_part)

# Evaluate the model on validation data
validation_score = tpot_model.score(X_val, y_val)
print("Validation R^2 Score:", validation_score)

# Use the model to predict test data
y_pred_test = tpot_model.predict(X_test)

# Reverse the scaling to get the original values
y_pred_test_rescaled = scaler.inverse_transform(np.concatenate((y_pred_test.reshape(-1, 1), np.zeros((y_pred_test.shape[0], 2))), axis=1))[:, 0]
y_test_rescaled = scaler.inverse_transform(np.concatenate((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 2))), axis=1))[:, 0]

# Plot the predicted vs actual closing prices
plt.figure(figsize=(16, 8))
plt.plot(stock_data.index[len(stock_data)-len(y_pred_test_rescaled):], y_pred_test_rescaled, label='Predicted Prices', color='purple')
plt.plot(stock_data.index, stock_data['Close'], label='Actual Prices', color='blue')
plt.title(f'{ticker} Stock Price Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Error metrics calculation
mse = mean_squared_error(y_test_rescaled, y_pred_test_rescaled)
mae = mean_absolute_error(y_test_rescaled, y_pred_test_rescaled)
r2 = r2_score(y_test_rescaled, y_pred_test_rescaled)
rmse = np.sqrt(mse)
mape = np.mean(np.abs((y_test_rescaled - y_pred_test_rescaled) / y_test_rescaled)) * 100

# Print performance metrics
print(f"MSE: {mse}")
print(f"MAE: {mae}")
print(f"R^2 Score: {r2}")
print(f"RMSE: {rmse}")
print(f"MAPE: {mape}%")

# Predict future stock prices using the trained model
def forecast_future_prices(model, last_window, time_step, days_ahead):
    future_preds = []
    input_data = last_window[-time_step:].flatten()

    for _ in range(days_ahead):
        next_pred = model.predict(input_data.reshape(1, -1))
        future_preds.append(next_pred[0])
        input_data = np.append(input_data[1:], next_pred[0])

    return future_preds

# Predict stock price for the next day
days_ahead = 1
last_window_data = scaled_features[-time_step:]
predicted_future_prices = forecast_future_prices(tpot_model.fitted_pipeline_, last_window_data, time_step, days_ahead)
predicted_future_prices_rescaled = scaler.inverse_transform(np.concatenate((np.array(predicted_future_prices).reshape(-1, 1), np.zeros((days_ahead, 2))), axis=1))[:, 0]

# Plot future predictions
last_stock_date = stock_data.index[-1]
future_dates = pd.date_range(start=last_stock_date, periods=days_ahead + 1) # future_dates has 2 elements

# Adjust future_dates to match the length of predicted_future_prices_rescaled
future_dates = future_dates[1:] # Now future_dates has 1 element

plt.figure(figsize=(16, 8))
plt.plot(stock_data.index, stock_data['Close'], label='Actual Prices', color='blue')
plt.plot(future_dates, predicted_future_prices_rescaled, label='Predicted Future Prices', color='orange')
plt.title(f'Future {ticker} Stock Price Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Display the forecasted price
print(f"Current Stock Price: {stock_data['Close'].iloc[-1]}")
print("\nPredicted Stock Price for the Next Day:")
for i, price in enumerate(predicted_future_prices_rescaled):
    print(f"Day {i+1}: {price:.2f}")