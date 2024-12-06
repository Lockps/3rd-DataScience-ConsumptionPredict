from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
import numpy as np


class ArimaModel:
    def __init__(self, data, time_column, value_columns):
        self.data = data
        self.time_column = time_column
        # List of value columns (energy consumption types)
        self.value_columns = value_columns
        self.forecasts = {}
        self.percentage_changes = {}
        print("Starting ARIMA model for multiple columns")

    def preprocess_data(self):
        if self.time_column not in self.data.columns:
            raise KeyError(
                f"The '{self.time_column}' column is missing from the dataset.")

        # Ensure all value columns exist in the dataset
        for value_column in self.value_columns:
            if value_column not in self.data.columns:
                raise KeyError(
                    f"The '{value_column}' column is missing from the dataset.")

        # Sort by time and set index
        self.data = self.data.sort_values(self.time_column)
        self.data.set_index(self.time_column, inplace=True)

    def split_data(self, test_size=5):
        self.train = {}
        self.test = {}

        # Split data for each value column (energy type)
        for value_column in self.value_columns:
            time_series = self.data[value_column]
            self.train[value_column] = time_series[:-test_size]
            self.test[value_column] = time_series[-test_size:]

    def fit_model(self, order=(2, 1, 2)):
        self.models = {}
        self.fitted_models = {}

        # Fit ARIMA model for each energy consumption column
        for value_column in self.value_columns:
            print(f"Fitting ARIMA model for {value_column}...")
            model = ARIMA(self.train[value_column], order=order)
            fitted_model = model.fit()
            self.models[value_column] = model
            self.fitted_models[value_column] = fitted_model
            print(fitted_model.summary())

    def forecast_values(self, steps=None):
        if steps is None:
            steps = max([len(self.test[value_column])
                        for value_column in self.value_columns])

        # Forecast for each energy consumption column
        for value_column in self.value_columns:
            forecast = self.fitted_models[value_column].forecast(steps=steps)
            self.forecasts[value_column] = forecast

            # Calculate percentage change
            percentage_change = (
                (forecast - self.test[value_column].values) / self.test[value_column].values) * 100
            self.percentage_changes[value_column] = percentage_change

    def plot_forecasts(self):
        num_columns = len(self.value_columns)
        fig, axes = plt.subplots(num_columns, 1, figsize=(10, 6 * num_columns))

        if num_columns == 1:
            axes = [axes]

        for i, value_column in enumerate(self.value_columns):
            ax = axes[i]
            ax.plot(self.train[value_column].index,
                    self.train[value_column], label="Train", color="blue")
            ax.plot(self.test[value_column].index,
                    self.test[value_column], label="Test", color="orange")
            ax.plot(self.test[value_column].index,
                    self.forecasts[value_column], label="Forecast", color="green")

            # Adding titles and labels for better clarity
            ax.set_title(f"ARIMA Model - {value_column}", fontsize=14)
            ax.set_xlabel(self.time_column, fontsize=12)
            ax.set_ylabel(f"{value_column} Consumption", fontsize=12)
            ax.legend()

        plt.tight_layout()  # Automatically adjust subplot spacing
        plt.show()

    def save_forecasts(self, filename="forecasts.csv"):
        all_forecasts = []
        for value_column in self.value_columns:
            forecast_df = pd.DataFrame({
                self.time_column: self.test[value_column].index,
                "actual": self.test[value_column].values,
                "forecast": self.forecasts[value_column],
                "percentage_change": self.percentage_changes[value_column]
            })
            forecast_df["energy_type"] = value_column
            all_forecasts.append(forecast_df)

        final_forecast_df = pd.concat(all_forecasts)
        final_forecast_df.to_csv(filename, index=False)
        print(f"Forecast results saved to {filename}")

    def calculate_accuracy(self):
        accuracy_metrics = {}

        for value_column in self.value_columns:
            # Get actual and predicted values
            actual = self.test[value_column].values
            forecast = self.forecasts[value_column]

            # Calculate MAE, MSE, and RMSE
            mae = mean_absolute_error(actual, forecast)
            mse = mean_squared_error(actual, forecast)
            rmse = np.sqrt(mse)

            # Store the accuracy metrics
            accuracy_metrics[value_column] = {
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse
            }

        # Print the accuracy for each energy type
        for value_column, metrics in accuracy_metrics.items():
            print(f"Accuracy for {value_column}:")
            print(f"  MAE: {metrics['MAE']:.4f}")
            print(f"  MSE: {metrics['MSE']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}\n")

        return accuracy_metrics
