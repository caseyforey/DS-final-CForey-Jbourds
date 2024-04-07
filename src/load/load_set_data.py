"""
load_set_data.py

Module for loading the set data for cards.

Authors: Jordan Bourdeau, Casey Forey
Date Created: 4/7/2024
"""

import pandas as pd
import json

def get_first_printings() -> pd.DataFrame:
    """
    Function to get the first set printings for every card.

    :param ???:

    :returns: Pandas DataFrame with every card and its first printing.
    """
    pass
    

def get_set_data(path) -> pd.DataFrame:
    """
    Function to get data about all of the sets into a Pandas DataFrame.

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
    set_year_df.rename(columns={'index': 'Set Name', 0:'Release Year'},inplace= True) 

