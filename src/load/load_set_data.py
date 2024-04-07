"""
load_set_data.py

Module for loading the set data for cards.

Authors: Jordan Bourdeau, Casey Forey
Date Created: 4/7/2024
"""

import numpy as np
import pandas as pd

def get_unique_card_names(all_printings_dataset: dict) -> np.array:
    """
    Function which returns a numpy array of all unique card names,

    :param all_printings_dataset: Dictionary containing information with all the card printings.

    :returns: Numpy array with all unique card names.
    """
    card_names: set[str] = set()
    set_names: list[str] = list(all_printings_dataset['data'].keys())
    for set_name in set_names:
        set_cards: list[dict] = all_printings_dataset['data'][set_name]['cards']
        for card in set_cards:
            card_names.add(card['name'])
    return np.array(list(card_names))

def get_card_sets_uuids(all_printings_dataset: dict, unique_card_names: np.array) -> dict[str: list[str]]:
    """
    Function to get a list of sets and corresponding UUIDs for every card name based on the UUID in every set
    they were released in.

    :param all_printings_dataset: Dictionary containing information with all the card printings.
    :param unique_card_names:     Numpy array with all unique card names.

    :returns: Dictionary mapping each card name to a list of its set UUIDs.
    """
    card_uuids: dict[str: list[tuple[str, str]]] = {card_name: [] for card_name in unique_card_names}
    set_names: list[str] = list(all_printings_dataset['data'].keys())
    for set_name in set_names:
        set_cards: list[dict] = all_printings_dataset['data'][set_name]['cards']
        for card in set_cards:
            card_uuids[card['name']].append((set_name, card['uuid']))

    return card_uuids

