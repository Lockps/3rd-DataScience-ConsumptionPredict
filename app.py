from cleandata import EnergyDataCleaner
import pandas as pd
from Arima import ArimaModel
import numpy as np


def Cleaner():
    print("Start clean program")

    data = pd.read_csv("./data/Data.csv")
    country_data = pd.read_csv("./data/Continents.csv")

    data.columns = data.columns.str.strip().str.lower()
    country_data.columns = country_data.columns.str.strip().str.lower()

    if 'country' not in data.columns or 'country' not in country_data.columns:
        raise KeyError(
            "The 'country' column is missing from one of the datasets.")

    cleaner = EnergyDataCleaner(data)
    cleaner.clean_column_names()
    cleaner.fill_missing_values(method='mean')
    cleaner.delete_years_before(1965)
    cleaned_data = cleaner.get_cleaned_data()

    merged_data = cleaned_data.merge(country_data, on="country", how="left")

    continent_counts = merged_data['continent'].value_counts()
    most_data_continent = continent_counts.idxmax()
    print(
        f"The continent with the most data is {most_data_continent}, with {continent_counts[most_data_continent]} rows.")

    merged_data.to_csv("cleaned_energy_data_with_continent.csv", index=False)
    print("Finish program")

    create_asia_file(merged_data)


def create_asia_file(data):
    """Create a CSV file containing only Asia's data."""
    print("Creating Asia.csv...")
    asia_data = data[data['continent'] == 'Asia']
    asia_data.to_csv("Asia.csv", index=False)
    print("Asia.csv has been created.")


def run_arima_model():
    asia_data = pd.read_csv("Asia.csv")

    energy_columns = [
        "biofuel_consumption",
        "coal_consumption",
        "gas_consumption",
        "renewables_consumption",
        "solar_consumption"
    ]

    arima_model = ArimaModel(
        asia_data, time_column="year", value_columns=energy_columns)

    arima_model.preprocess_data()
    arima_model.split_data(test_size=5)

    arima_orders = [
        (1, 1, 1),
        (2, 1, 2),
        (3, 1, 3),
        (4, 1, 4),
        (5, 1, 1),
        (5, 1, 2)
    ]

    best_accuracy = None
    best_order = None
    best_metrics = None

    for order in arima_orders:
        print(f"\nTesting ARIMA order: {order}")

        arima_model.fit_model(order=order)
        arima_model.forecast_values()

        accuracy_metrics = arima_model.calculate_accuracy()

        rmse_values = []
        for metrics in accuracy_metrics.values():
            if 'RMSE' in metrics:
                rmse_values.append(metrics['RMSE'])
            else:
                print(f"Warning: RMSE key missing for one of the energy types!")

        if rmse_values:
            average_rmse = np.mean(rmse_values)
            print(f"Average RMSE for order {order}: {average_rmse}")

            if best_accuracy is None or average_rmse < best_accuracy:
                best_accuracy = average_rmse
                best_order = order
                best_metrics = accuracy_metrics

    if best_order:
        print("\nBest ARIMA Order: ", best_order)
        print("Best Accuracy (Average RMSE): ", best_accuracy)
        print("Best Model Accuracy Metrics:")
        for value_column, metrics in best_metrics.items():
            print(f"  {value_column}: RMSE = {metrics['RMSE']:.4f}")

        arima_model.save_forecasts(f"Asia_ARIMA_Forecast_{best_order}.csv")
        arima_model.plot_forecasts()
    else:
        print("No valid RMSE values found for any ARIMA order.")


if __name__ == '__main__':
    run_arima_model()
