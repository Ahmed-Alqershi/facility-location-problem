import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import matplotlib.pyplot as plt


# Function to split data into train and test sets
def split_data(series, train_size):
    train, test = series[:train_size], series[train_size:]
    return train, test

# Function to calculate Mean Squared Error (MSE)
def calculate_rmse(actual, predicted):
    return np.sqrt(mean_squared_error(actual, predicted))

# Function for Simple Moving Average (SMA) forecasting
def sma_forecast(train, test):
    history = list(train)
    predictions = []
    for t in range(len(test)):
        yhat = np.mean(history)
        predictions.append(yhat)
        history.append(test[t])
    return predictions

# Function for Exponential Smoothing forecasting
def exponential_smoothing_forecast(train, test):
    model = ExponentialSmoothing(train)
    model_fit = model.fit()
    predictions = model_fit.predict(start=len(train), end=len(train) + len(test) - 1)
    return predictions

# Function for Linear Regression forecasting
def linear_regression_forecast(train, test):
    X_train, y_train = np.arange(len(train)).reshape(-1, 1), np.array(train)
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# Function for Random Forest forecasting
def random_forest_forecast(train, test):
    X_train, y_train = np.arange(len(train)).reshape(-1, 1), np.array(train)
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# Function for XGBoost forecasting
def xgboost_forecast(train, test):
    X_train, y_train = np.arange(len(train)).reshape(-1, 1), np.array(train)
    X_test = np.arange(len(train), len(train) + len(test)).reshape(-1, 1)

    model = XGBRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return predictions

# Function to choose the best forecasting method
def choose_best_method(train, test):
    methods = [sma_forecast, exponential_smoothing_forecast, linear_regression_forecast,
               random_forest_forecast, xgboost_forecast]

    best_method = None
    best_mse = float('inf')

    for method in methods:
        predictions = method(train, test)
        mse = calculate_rmse(test, predictions)

        if mse < best_mse:
            best_mse = mse
            best_method = method

    return best_method

# Function to plot the real data and predictions
def plot_results(train, test, predictions, method_name):
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(len(train)), train, label='Train Data', marker='o')
    plt.plot(np.arange(len(train), len(train) + len(test)), test, label='Test Data', marker='o')
    plt.plot(np.arange(len(train), len(train) + len(test)), predictions, label=f'{method_name} Predictions', marker='o')
    plt.title('Time Series Forecasting')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.show()


# Main script
if __name__ == "__main__":
    # Replace 'your_time_series_data' with your actual time series data
    time_series_data = [3130, 5854, 7697, 11000, 12084, 15453, 10433, 12873, 15890, 15768, 18232, 18103, 10270, 9858, 11249, 13030, 14448, 12513, 16213, 17028]

    train_size = 14
    train, test = split_data(time_series_data, train_size)

    best_method = choose_best_method(train, test)
    best_predictions = best_method(train, test)

    print(f"Best forecasting method: {best_method.__name__}")
    print(f"Predictions: {best_predictions}")
    print(f"Mean Squared Error: {calculate_rmse(test, best_predictions)}")

    # Plotting results
    plot_results(train, test, best_predictions, best_method.__name__)
