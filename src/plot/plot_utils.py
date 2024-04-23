"""
plot_utils.py

File containing utility functions for plotting a Pandas dataframe.

Author: Jordan Bourdeau
Date Created: 4/22/24
"""

from matplotlib import pyplot as plt
import pandas as pd


def plot_dataframe_as_table(dataframe: pd.DataFrame, filepath: str, font_size=12):
    """
    Function to plot a pandas dataframe as a table.

    :param dataframe: Pandas dataframe to plot.
    :param filepath: File path to save the plot to.
    :param font_size: Font size for the table.
    """
    fig, ax = plt.subplots(figsize=(len(dataframe.columns), len(dataframe) * 0.5))
    ax.axis('tight')
    ax.axis('off')
    
    # Calculate equal column widths
    col_widths = [2.0 / len(dataframe.columns)] * len(dataframe.columns)
    
    table = ax.table(cellText=dataframe.values,
                     colLabels=dataframe.columns,
                     colWidths=col_widths,
                     loc='center')
    
    # Set font properties
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    
    # Center cell values
    for key, cell in table.get_celld().items():
        cell.set_text_props(ha='center', va='center')
    
    plt.savefig(filepath)
    plt.show()