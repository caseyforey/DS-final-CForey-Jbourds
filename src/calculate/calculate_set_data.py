"""
calculate_market_data.py

Module containing functions to do calculations on the set data.

Author: Jordan Bourdeau
Date Created: 4/17/24
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def find_set_outliers(
        set_dataframe: pd.DataFrame,
        columns: list[str] = ['total_count', 'num_banned', 'set_size', 'mean_price', 'median_price', 'std_price'],
        contamination: float = 0.1, 
    ) -> pd.DataFrame:
    np.random.seed(0)
    data_for_model: pd.DataFrame = set_dataframe[columns]

    # Initialize Isolation Forest model
    # Contamination is the approximate proportion of outliers we expect
    isolation_forest = IsolationForest(contamination=contamination)

    # Fit the model to the data
    isolation_forest.fit(data_for_model)

    # Predict outliers
    outlier_predictions: np.array = isolation_forest.predict(data_for_model)

    # Display outliers
    outliers: pd.DataFrame = set_dataframe.iloc[outlier_predictions == -1]

    return outliers
