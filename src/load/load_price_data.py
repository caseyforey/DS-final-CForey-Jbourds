"""
load_price_data.py

Module for loading price data.

Authors: Jordan Bourdeau, Casey Forey
Date Created: 4/7/2024
"""

import numpy as np
import os
import pandas as pd

from src import constants as c
from src.load import load_utils
import src.load.load_set_data as lsd

def load_card_price_df(data_directory: str = c.DATA_DIRECTORY, cache_directory: str = c.CACHE, use_cache: bool = True) -> pd.DataFrame:
    """
    Function which loads information about a card's cheapest printing from the Magic the Gathering Online
    market data, along with information about the printing and set it's from.

    :param data_directory:  String for the directory data is stored in.
    :param cache_directory: String for the directory to store cached results in.
    :param use_cache:       Boolean flag for whether the function should use cached files.

    :returns: Pandas dataframe with card price data.
    """

    # Filepaths
    # Data files
    all_prices_filepath: str = os.path.join(data_directory, 'AllPricesToday.json')
    all_printings_filepath: str = os.path.join(data_directory, 'AllPrintings.json')
    set_list_filepath: str = os.path.join(data_directory, 'SetList.json')
    # Cache files
    market_price_filepath: str = os.path.join(data_directory, cache_directory, 'market_prices.csv')
    unique_card_names_filepath: str = os.path.join(data_directory, cache_directory, 'unique_card_names')
    lowest_price_printing_filepath: str = os.path.join(data_directory, cache_directory, 'lowest_price_printings.csv')
    set_release_year_filepath: str = os.path.join(data_directory, cache_directory, 'set_release_years.csv')

    # If we can, skip all the intermediary steps entirely
    if os.path.exists(market_price_filepath) and use_cache:
        return pd.read_csv(market_price_filepath)

    # Load unique card_names if available, otherwise compute and cache it
    if os.path.exists(unique_card_names_filepath) and use_cache:
        unique_card_names: np.array = np.load(unique_card_names_filepath)
    else:
        all_printings: dict = load_utils.load_json_data(all_printings_filepath)
        unique_card_names: np.array = lsd.get_unique_card_names(all_printings)
        np.save(unique_card_names_filepath, unique_card_names)

    # Load lowest price printings if available, otherwise compute and cache it
    if os.path.exists(lowest_price_printing_filepath) and use_cache:
        lowest_price_printing_df: pd.DataFrame = pd.read_csv(lowest_price_printing_filepath)
    else:
        all_prices: dict = load_utils.load_json_data(all_prices_filepath)
        card_set_printings: dict[str, list[tuple[str, str]]] = lsd.get_card_printings_info(all_printings, unique_card_names)
        lowest_price_printing_df: pd.DataFrame = get_lowest_price_printing(all_prices, card_set_printings)
        lowest_price_printing_df.to_csv(lowest_price_printing_filepath, index=False)

    # Get the release year for all the sets
    if os.path.exists(set_release_year_filepath) and use_cache:
        set_release_year_df: pd.DataFrame = lsd.load_set_and_release_year()
    else:
        set_release_year_df: pd.DataFrame = lsd.get_set_release_year(set_list_filepath)
        set_release_year_df.to_csv(set_release_year_filepath, index=False)

    # Join the set release year with the set the lowest price was from
    # This also gets right of all the rows without price data as well
    full_market_df: pd.DataFrame = pd.merge(lowest_price_printing_df, set_release_year_df, left_on='set', right_on='set')
    full_market_df.to_csv(market_price_filepath, index=False)
    return full_market_df

def get_lowest_price_printing(
        all_prices_today_dataset: pd.DataFrame, 
        card_sets_uuids: dict[str: list[tuple[str, str]]]
    ) -> pd.DataFrame:
    """
    Function which finds the lowest price set and UUID printing for every card and compiles
    the data into a dataframe.
    
    NOTE: For now only looks at Magic the Gathering Online data.

    :param all_prices_today_dataset: JSON dataset with all card prices from a certain data.
    :param card_set_uuids:           Dictionary mapping every card_name to a list with each set/UUID printing pair.

    :returns: Pandas dataframe with the card_name and its lowest set printing.
    """
    num_cards: int = len(card_sets_uuids)
    columns: list[tuple[str, np.array]] = [
        ('set', [np.nan] * num_cards),
        ('uuid', [np.nan] * num_cards),
        ('price', [np.inf] * num_cards),
        ('rarity', [np.nan] * num_cards),
        ('currency', [np.nan] * num_cards),
        ('date', [np.nan] * num_cards),
        ('foil', [np.nan] * num_cards),
    ]
    
    df: pd.DataFrame = pd.DataFrame(
        {'card': list(card_sets_uuids.keys()),
         **{column_name: array for column_name, array in columns}
        } 
    )

    for card, printings in card_sets_uuids.items():
        for set, uuid, rarity in printings:
            card_prices: dict[str: dict] = all_prices_today_dataset['data'].get(uuid, None)
            # There is no price data available
            if card_prices is None:
                continue

            # Check if there is Magic The Gathering Online data
            mtgo_listing: dict[str: dict[str: float]] = card_prices.get('mtgo', None)
            if mtgo_listing is None:
                continue

            # Drill into the JSON structure some more
            mtgo_listing = mtgo_listing['cardhoarder']
            mtgo_retail: dict[str: dict[str: str]] = mtgo_listing['retail']
            target_column_names: list[str] = [column for column, _ in columns]

            # Update the DataFrame with the lowest price information
            for foil, listing in mtgo_retail.items():
                date, price = list(listing.items())[0]  # A dictionary which is basically a pair of date and price
                card_index = df[df['card'] == card].index
                if price < df.loc[card_index, 'price'].iloc[0]:
                    df.loc[card_index, target_column_names] = [set, uuid, price, rarity, mtgo_listing['currency'], date, foil]

    return df

