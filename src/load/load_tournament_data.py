"""
load_tournament_data.py

Module for loading tournament data.

Authors: Jordan Bourdeau, Casey Forey
Date Created: 4/7/2024
"""

import json
import os
import pandas as pd

def get_tournament_files(base_path: str, format: str = '') -> list[str]:
    """
    Function to return a list of file paths which use the structure of the base path
    and a specified format for each file in the corresponding tournament data.

    :param base_path: String for the base path to start from (e.g. 2023)
    :param format:    Format to look for in the tournament. Defaults to the empty string to select all data.

    :returns: List of all filepaths fitting criteria.
    """
    target_filenames: list[str] = []
    for root, directory_names, file_names in os.walk(base_path):
        target_filenames += [os.path.join(root, file_name) for file_name in file_names if format in file_name]
    return target_filenames


def update_dictionary_card_counts(data_dict: dict, tournament_file: str):
    """
    Function to update the counts in the data dictionary with the counts in a tournament file.

    :param data_dict:       Dictionary containing counts for usages of cards.
    :param tournament_file: String for the JSON file name to look at.

    :returns: None. Updates dictionary by reference.
    """
    with open(tournament_file, 'r') as infile:
        json_data = json.load(infile)
        for deck in json_data['Decks']:
            for main_card in deck['Mainboard']:
                if main_card['CardName'] in data_dict:
                    data_dict[main_card['CardName']] = main_card['Count'] + data_dict[main_card['CardName']]
                else:
                    data_dict[main_card['CardName']] = main_card['Count']
            for side_card in deck['Sideboard']:
                if side_card['CardName'] in data_dict:
                    data_dict[side_card['CardName']] = side_card['Count'] + data_dict[side_card['CardName']]
                else:
                    data_dict[side_card['CardName']] = side_card['Count']


def load_format_card_counts(path: str, format: str = '') -> dict:
    """
    Function to load a dictionary with the counts of how frequently each card is used
    in a given year of tournament data.

    :param path:   String path for the outermost folder of the tournament data.
    :param format: Game format to load data for. Defaults to all of them with an empty string.

    :returns: Pandas dataframe with card names and counts for the specified format.
    """
    data_dict: dict = {}
    tournament_files = get_tournament_files(path, format)
    for tournament_file in tournament_files:
        update_dictionary_card_counts(data_dict, tournament_file)
    card_counts_df: pd.DataFrame = pd.DataFrame.from_dict(data_dict, orient='index')  
    card_counts_df.reset_index(inplace= True) 
    card_counts_df.rename(columns={'index': 'Card Name', 0: 'Total Count'}, inplace=True) 

    return card_counts_df