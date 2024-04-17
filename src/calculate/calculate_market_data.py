"""
calculate_market_data.py

Module containing functions to do calculations on the market data.

Author: Jordan Bourdeau
Date Created: 4/14/24
"""

import pandas as pd

def calculate_aggregate_set_prices(card_price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Function to calculate aggregate set prices from the market price data.

    :param card_price_df: Pandas dataframe with market price information.

    :returns: Dataframe with aggregate statistics.
    """
    # Don't count promos- ignore sets with a s
    agg_set_df: pd.DataFrame = card_price_df \
        .groupby(['set'])['price'].agg(['mean', 'median', 'std']) \
        .reset_index() \
        .sort_values(by=['mean', 'median'], ascending=False) \
        .round(3)
    agg_set_df.rename(columns={
        'set': 'set_code',
        'mean': 'mean_price',
        'median': 'median_price',
        'std': 'std_price',
    }, inplace=True)
    return agg_set_df[~agg_set_df['std_price'].isnull()]
