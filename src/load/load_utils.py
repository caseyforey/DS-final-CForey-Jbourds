"""
load_utils.py

Module containing utility functions for use in the various load modules.

Author: Jordan Bourdeau
Date Created: 4/7/24
"""

import json

def load_json_data(filepath: str) -> dict:
    """
    Helper function to read in JSON file contents.

    :param filepath: File path to read in.

    :returns: Dictionary representation of the JSON data.
    """
    with open(filepath, 'r',encoding='utf8') as infile:
        json_data = json.load(infile)
        return json_data