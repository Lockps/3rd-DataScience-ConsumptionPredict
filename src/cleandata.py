import pandas as pd
import numpy as np


class EnergyDataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data

    def clean_column_names(self):
        self.data.columns = (
            self.data.columns.str.strip().str.lower().str.replace(" ", "_")
        )

    def fill_missing_values(self, method: str = "mean", columns: list = None):
        if columns is None:
            columns = self.data.columns

        for col in columns:
            if method == "mean" and self.data[col].dtype in [np.float64, np.int64]:
                self.data[col].fillna(self.data[col].mean(), inplace=True)
            elif method == "median" and self.data[col].dtype in [np.float64, np.int64]:
                self.data[col].fillna(self.data[col].median(), inplace=True)
            elif method == "mode":
                self.data[col].fillna(self.data[col].mode()[0], inplace=True)
            elif method == "ffill":
                self.data[col].fillna(method="ffill", inplace=True)

    def convert_data_types(self, column_types: dict):
        for col, dtype in column_types.items():
            self.data[col] = self.data[col].astype(dtype)

    def remove_outliers(self, columns: list = None, method: str = "zscore", threshold: float = 3.0):
        if columns is None:
            columns = self.data.select_dtypes(
                include=[np.float64, np.int64]).columns

        for col in columns:
            if method == "zscore":
                z_scores = np.abs(
                    (self.data[col] - self.data[col].mean()) / self.data[col].std())
                self.data = self.data[z_scores < threshold]
            elif method == "iqr":
                Q1 = self.data[col].quantile(0.25)
                Q3 = self.data[col].quantile(0.75)
                IQR = Q3 - Q1
                self.data = self.data[(
                    self.data[col] >= Q1 - 1.5 * IQR) & (self.data[col] <= Q3 + 1.5 * IQR)]

    def delete_years_before(self, year: int = 1965):
        self.data = self.data[self.data['year'] >= year]

    def get_cleaned_data(self):
        return self.data
