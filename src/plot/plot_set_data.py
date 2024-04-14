"""
plot_set_data.py

Module to plot set information.

Author: Jordan Bourdeau
Date Created: 4/14/24
"""

from matplotlib import pyplot as plt
import os
import pandas as pd

import src.constants as c

def plot_stacked_set_counts(card_counts_df: pd.DataFrame, set_year_df: pd.DataFrame, format: str):
    """
    Function which creates a stacked histogram for the prevalence of card usage based on the
    number of times cards from the set get used based on the set release year.

    :param card_counts_df: Dataframe with the card counts.
    :param set_year_df:    Dataframe with the set and release year.
    :param format:         String for the format file (for saving file).
    """
    # Grouping by 'set_year' and 'set_name', and summing up 'total_count'
    grouped_data = card_counts_df.groupby(['set_year', 'set_name'])['total_count'].sum().unstack().fillna(0)

    # Creating a DataFrame with 'set_name' as a column
    sets = pd.DataFrame({'set_name': grouped_data.columns})

    # Merging with 'set_year_df' to get the release year
    sets = pd.merge(sets, set_year_df, on='set_name', how='left')

    # Plotting the data as a stacked bar chart
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed
    grouped_data.plot(kind='bar', stacked=True, legend=None)

    # Title and axis labels for the stacked bar chart
    plt.title('Card Counts by Set Name and Year')
    plt.xlabel('Year')
    plt.ylabel('Total Card Count')
    plt.tight_layout()
    plt.savefig(os.path.join(c.IMAGE_DIRECTORY, f'{format}_stacked_set_counts.png'))
    plt.show()

def plot_set_table(set_card_usages_and_bans: pd.DataFrame):
    """
    Create a table from the set card usage and bans data.

    :param set_card_usages_and_bands: Pandas Dataframe.
    """
    df = set_card_usages_and_bans.sort_values(by=['release_year', 'set_name'])

    # Rename columns
    df = df.rename(columns={
        'set_code': 'Set Code', 
        'total_count': 'Times Used in 2023 Tournaments', 
        'set_name': 'Set Name', 
        'num_banned': '# Banned Cards in Set',
        'release_year': 'Release Year',
        'release_month': 'Month Released',
        'set_size': '# Cards in Set',
    })

    # Reorder columns
    df = df[['Set Name', 'Set Code', 'Release Year', 'Times Used in 2023 Tournaments', '# Banned Cards in Set']]

    # Render DataFrame as a table
    fig, ax = plt.subplots(figsize=(10, 6))  # Adjust the figure size as needed
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center')

    # Adjust font size
    table.auto_set_font_size(False)
    table.set_fontsize(12)

    # Adjust cell heights
    table.scale(2.1, 1.5)  # Adjust the scale as needed

    plt.savefig(os.path.join(c.IMAGE_DIRECTORY, 'sets.png'), bbox_inches='tight')  # Save the figure with tight bounding box
    plt.show()