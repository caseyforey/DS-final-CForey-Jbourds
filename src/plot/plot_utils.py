"""
plot_utils.py

File containing utility functions for plotting a Pandas dataframe.

Author: Jordan Bourdeau
Date Created: 4/22/24
"""

from matplotlib import pyplot as plt
import pandas as pd


def plot_dataframe_as_table(dataframe: pd.DataFrame, filepath: str):
    """
    Function to plot a pandas dataframe as a table.

    :param dataframe: Pandas dataframe to plot.
    :param filepath:  File path to save the plot to.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=dataframe.values, colLabels=dataframe.columns, loc='center')
    plt.savefig(filepath)
    plt.show()