"""
load_card_data.py

Module containing functions to load card data.

Author: Jordan Bourdeau
Date Created: 4/14/24
"""

import numpy as np
import os
import pandas as pd

import src.constants as c
from src.load import load_set_data as lsd

def load_first_card_printing_in_format(
        format: str, 
        all_printings: dict,
        data_directory: str = c.DATA_DIRECTORY, 
        cache_directory: str = c.CACHE
    ) -> pd.DataFrame:
    """
    Function which returns a dataframe with the columns: 'card_name', 'format', 'set_code', 'set_name'
    where each row is the first set in the specified format the card appeared in.

    :param format:          Format string to look for. 
    :param all_printings:   Dictionary from the all printings dataset.
    :param data_directory:  String path for the data directory. Defaults to constant.
    :param cache_directory: String path for the cache directory. Defaults to constant.

    :returns: Pandas dataframe with first printing of a card in a format.
    """
    # Load all unique card_names
    unique_card_names_filepath: str = os.path.join(data_directory, cache_directory, 'unique_card_names.npy')
    unique_card_names: np.array = np.load(unique_card_names_filepath)

    # Load legal sets for the format
    legal_sets: pd.DataFrame = lsd.load_legal_format_sets(format, c.DATA_DIRECTORY)

    # Get all the card printings 
    card_printings_info = get_card_printings_info(all_printings, unique_card_names)
    df_printings = convert_card_printings_to_df(card_printings_info)

    # Filter out the basic lands
    basic_lands: list[str] = ['plains', 'island', 'swamp', 'mountain', 'forest']
    cleaned = df_printings[~df_printings['card_name'].str.lower().isin(basic_lands)]

    return filter_first_printing_in_sets(cleaned, legal_sets)

def load_format_banned_printings(
        format: str,
        data_directory: str= c.DATA_DIRECTORY,
        cache_directory: str = c.CACHE,
    ) -> pd.DataFrame:
    """
    Function to load the banned printings for a specific format.

    :param format:          String for the format being calculated.
    :param data_directory:  Directory for cache. Defaults to constant.
    :param cache_directory: Directory for cache. Defaults to constant.
    """
    return pd.read_csv(os.path.join(data_directory, cache_directory, f'{format}_banned_printings.csv'))

def filter_first_printing_in_sets(card_printings: pd.DataFrame, legal_sets: pd.DataFrame) -> pd.DataFrame:
    """
    Function which will filter the card printings into their first printing (if applicable) within
    a set of legal sets and include the set_name, code, and release year along with the card_name.

    :param card_printings: Pandas Dataframe with the card printings.
    :param legal_sets:     Sets to compare against for a format.

    :returns: Pandas Dataframe with the first legal set a card appeared in.
    """

    # Filter to only include printings from legal sets
    df_legal_printings: pd.DataFrame = card_printings[card_printings['set_code'].isin(legal_sets['set_code'])]

    # Merge with legal sets to get release years
    df_legal_printings: pd.DataFrame = df_legal_printings.merge(legal_sets, on='set_code')
    # Sort and group to get the first printing by release year
    df_first_printing: pd.DataFrame = df_legal_printings.sort_values(by=['release_year', 'release_month']).groupby('card_name', as_index=False).first()

    # Convert unique card_names into a DataFrame
    # Load all unique card_names
    unique_card_names_filepath: str = os.path.join(c.DATA_DIRECTORY, c.CACHE, 'unique_card_names.npy')
    unique_card_names: np.array = np.load(unique_card_names_filepath)
    df_unique_cards = pd.DataFrame(unique_card_names, columns=['card_name'])

    # Merge to include all cards, marking those without a legal printing as NaN
    return df_unique_cards.merge(df_first_printing[['card_name', 'set_name', 'set_code', 'release_year', 'release_month']], on='card_name', how='left')

def convert_card_printings_to_df(card_printings: dict) -> pd.DataFrame:
    """
    Function to convert dictionary of all card printings into DF.

    :param card_printings: Dictionary of all cards, and associated list of 
                           3-tuples containing set code, UUID, and rarity.

    :returns: Converted Dataframe.
    """

    # Flatten card printings into DF format
    flattened_data = []
    for card, printings in card_printings.items():
        for set_code, card_id, rarity in printings:
            flattened_data.append({
                'card_name': card,
                'set_code': set_code,
                'card_id': card_id,
                'rarity': rarity
            })
    return pd.DataFrame(flattened_data)

def load_bannded_cards_array(
        format: str, 
        data_directory: str = c.DATA_DIRECTORY, 
        cache_directory: str = c.CACHE
    ) -> np.array:
    """
    Function to load a Numpy array of banned cards for a given format.
    Relies on there being a file with the convention <format>_banned_cards.npy
    in the data directory.

    :param format:          String format the banned cards are in.
    :param data_directory:  String path for the data directory. Defaults to constant.
    :param cache_directory: String path for the data directory. Defaults to constant.

    :returns: Numpy array with banned cards.
    """
    banned_cards_path: str = os.path.join(data_directory, cache_directory, f'{format}_banned_cards.npy')
    modern_banned_cards: np.array = np.load(banned_cards_path)
    return modern_banned_cards

def get_unique_card_names(all_printings_dataset: dict) -> np.array:
    """
    Function which returns a numpy array of all unique card_names,

    :param all_printings_dataset: Dictionary containing information with all the card printings.

    :returns: Numpy array with all unique card_names.
    """
    card_names: set[str] = set()
    set_names: list[str] = list(all_printings_dataset['data'].keys())
    for set_name in set_names:
        set_cards: list[dict] = all_printings_dataset['data'][set_name]['cards']
        for card in set_cards:
            card_names.add(card['name'])
    return np.array(list(card_names))

def get_card_printings_info(all_printings_dataset: dict, unique_card_names: np.array) -> dict[str: list[str]]:
    """
    Function to get a list of sets, UUIDs, and rarities for every card_name based on the UUID in every set
    they were released in.

    :param all_printings_dataset: Dictionary containing information with all the card printings.
    :param unique_card_names:     Numpy array with all unique card_names.

    :returns: Dictionary mapping each card_name to a list of its set_names, UUIDs, and rarities.
    """
    card_sets_uuids: dict[str: list[tuple[str, str]]] = {card_name: [] for card_name in unique_card_names}
    set_names: list[str] = list(all_printings_dataset['data'].keys())

    for set_name in set_names:
        set_cards: list[dict] = all_printings_dataset['data'][set_name]['cards']
        for card in set_cards:
            card_sets_uuids[card['name']].append((set_name, card['uuid'], card['rarity']))
    return card_sets_uuids