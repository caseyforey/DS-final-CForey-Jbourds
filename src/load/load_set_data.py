"""
load_set_data.py

Module for loading the set data for cards.

Authors: Jordan Bourdeau, Casey Forey
Date Created: 4/7/2024
"""

import numpy as np
import os
import pandas as pd
import json

import src.constants as c
from src.load import load_card_data as lcd
from src.load import load_tournament_data as ltd

def load_augmented_set_data(
        all_printings: dict, 
        format: str, 
        year: list[str] = '2023',
        data_directory: str = c.DATA_DIRECTORY,
    ) -> pd.DataFrame:
    """
    Function to create an augmented set dataframe which includes usage of cards
    based on the tournament data and banned cards.

    :param all_printings:  Dictionary with data for all printings.
    :param format:         Magic format to load data for.
    :param year:           Year to load format data for.
    :param data_directory: String for the data directory path. Defaults to a constant.

    :returns: Pandas dataframe with set usage across years.
    """
    # Load the legal sets
    legal_sets: pd.DataFrame = load_legal_format_sets(format)
    first_format_printings: pd.DataFrame = lcd.load_first_card_printing_in_format(format, all_printings)
    
    # Load tournament usages
    base_path: str = os.path.join(data_directory, year)
    card_tournament_usage: pd.DataFrame = ltd.load_format_card_counts(base_path, format)

    # Get the usage for each card, and sum the total number of used cards per set
    format_card_usage: pd.DataFrame = first_format_printings.merge(card_tournament_usage, on=['card_name'])
    set_counts: pd.DataFrame = format_card_usage.groupby(['set_code'])['total_count'].sum().reset_index()

    # Get all the banned cards
    banned_set_counts: pd.DataFrame = load_format_set_ban_counts(format)

    # Merge banned set cards into set counts DF
    return set_counts.merge(banned_set_counts, on=['set_code'])

def save_format_set_ban_counts(
        all_printings: dict,
        format: str,
        data_directory: str= c.DATA_DIRECTORY,
        cache_directory: str = c.CACHE,
    ):
    """
    Function to save the format set ban counts for a specific format.

    :param all_printings:   Dictionary with all printings dataset.
    :param format:          String for the format being calculated.
    :param data_directory:  Directory for cache. Defaults to constant.
    :param cache_directory: Directory for cache. Defaults to constant.
    """

    banned_cards = lcd.load_bannded_cards_array(format)
    banned_cards_df: pd.DataFrame = pd.DataFrame({
        'card_name': banned_cards,
    })

    # Load all unique card_names
    unique_card_names_filepath: str = os.path.join(c.DATA_DIRECTORY, c.CACHE, 'unique_card_names.npy')
    unique_card_names: np.array = np.load(unique_card_names_filepath)

    # Get all the card printings 
    card_printings_info = lcd.get_card_printings_info(all_printings, unique_card_names)
    df_printings = lcd.convert_card_printings_to_df(card_printings_info)

    # Figure out which rows correspond to banned cards
    banned_printings: pd.DataFrame = df_printings.merge(banned_cards_df, on=['card_name'], how='inner')

    # Drop duplicates based on 'card_name' and 'set_code' to ensure uniqueness per set
    banned_printings = banned_printings.drop_duplicates(subset=['card_name', 'set_code'])
    banned_printings.to_csv(os.path.join(data_directory, cache_directory, f'{format}_banned_printings.csv'), index=False)

    # Get all legal sets
    legal_sets: pd.DataFrame = load_legal_format_sets(format)

    # Figure out which rows correspond to banned cards
    banned_printings = df_printings.merge(banned_cards_df, on='card_name', how='inner')

    # Count number of banned cards per set
    set_ban_counts: pd.DataFrame = banned_printings['set_code'].value_counts().reset_index()
    set_ban_counts.columns = ['set_code', 'num_banned']

    # Merge with legal sets to include all sets, even those with zero banned cards
    legal_sets: pd.DataFrame = legal_sets[['set_name', 'set_code']]  # Ensure these columns exist
    set_ban_counts: pd.DataFrame = legal_sets.merge(set_ban_counts, on='set_code', how='left').fillna(0)

    # Convert num_banned to integer (it might be float due to fillna)
    set_ban_counts['num_banned'] = set_ban_counts['num_banned'].astype(int)
    set_ban_counts.sort_values(by=['num_banned'], ascending=False, inplace=True)

    # Save to CSV
    output_path = os.path.join(data_directory, cache_directory, f'{format}_set_ban_counts.csv')
    set_ban_counts.to_csv(output_path, index=False)

def load_format_set_ban_counts(
        format: str,
        data_directory: str= c.DATA_DIRECTORY,
        cache_directory: str = c.CACHE,
    ) -> pd.DataFrame:
    """
    Function to load the format set ban counts for a specific format.

    :param format:          String for the format being calculated.
    :param data_directory:  Directory for cache. Defaults to constant.
    :param cache_directory: Directory for cache. Defaults to constant.
    """
    return pd.read_csv(os.path.join(data_directory, cache_directory, f'{format}_set_ban_counts.csv'))

def get_set_and_release_year(path: str = os.path.join(c.DATA_DIRECTORY, 'SetList.json')) -> pd.DataFrame:
    """
    File to load all the sets and release years into a Pandas dataframe.

    :param path: Path to load the data at. By default is within the data directory as 'SetList.json'.

    :returns: Pandas dataframe.
    """
    with open(path, 'r') as file:
        set_list = json.load(file)
        set_df = pd.DataFrame(
            [(set_id['code'], set_id['name'], set_id['releaseDate'][0:4], set_id['releaseDate'][5:7], set_id['baseSetSize']) for set_id in set_list['data']],
            columns=['set_code', 'set_name', 'release_year', 'release_month', 'set_size']
        )
        return set_df
    
def save_set_and_release_year(data_directory: str= c.DATA_DIRECTORY, cache_directory: str = c.CACHE):
    """
    Function to save the set and realase years dataframe to a cached csv.

    :param data_directory:  String path for the data directory. Defaults to constant.
    :param cache_directory: String path for the cache directory. Defaults to constant.
    """
    df: pd.DataFrame = get_set_and_release_year()
    df.to_csv(os.path.join(data_directory, cache_directory, 'set_release_years.csv'), index=False)

def load_set_and_release_year(data_directory: str = c.DATA_DIRECTORY, cache_directory: str = c.CACHE) -> pd.DataFrame:
    """
    Function to save the set and realase years dataframe to a cached csv.

    :param data_directory:  String path for the data directory. Defaults to constant.
    :param cache_directory: String path for the cache directory. Defaults to constant.
    """
    df: pd.DataFrame = pd.read_csv(os.path.join(data_directory, cache_directory, 'set_release_years.csv'))
    return df
    
def load_legal_format_sets(format: str, data_directory: str = c.DATA_DIRECTORY, cache_directory: str = c.CACHE) -> np.array:
    """
    Function to load a pandas dataframe of legal sets for a given format.
    Relies on there being a file with the convention <format>_legal_sets.csv
    in the data directory.

    :param format:          String format the banned cards are in.
    :param data_directory:  String path for the data directory. Defaults to constant.
    :param cache_directory: String path for the cache directory. Defaults to constant.

    :returns: Pandas dataframe with all the modern legal sets.
    """
    legal_sets_path: str = os.path.join(data_directory, cache_directory, f'{format.lower()}_legal_sets.csv')
    modern_legal_sets: pd.DataFrame = pd.read_csv(legal_sets_path)
    return modern_legal_sets

def get_set_release_year(path: str) -> pd.DataFrame:
    """
    :param path: The path to the set json file to be read

    :returns: Pandas DataFrame with every set and year it was released.
    """
    f = open(path, encoding= "utf8") 
    set_list = json.load(f)
    set_year = {}

    for set_id in set_list['data']:
        set_year[set_id['code']] = set_id['releaseDate'][0:4]
    set_year_df = pd.DataFrame.from_dict(set_year, orient='index')
    set_year_df.reset_index(inplace= True) 
    set_year_df.rename(columns={'index': 'set', 0: 'release_year'}, inplace=True)
    return set_year_df
