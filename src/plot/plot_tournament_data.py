"""
plot_tournament_data.py

Module for creating visualizations for tournament data.

Authors: Jordan Bourdeau, Casey Forey
Date Created: 4/7/2024
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns 

def plot_card_counts(df, rgb = (0.5,0.5,0.5)):
    grouped_year = df.groupby(['release_year']).sum().reset_index()
    plt.figure(figsize=(10, 6))
    sns.set_style('ticks')
    sns.barplot(data=grouped_year, x='release_year', y='total_count', color=rgb)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.margins(x=0)
    plt.xlabel('Release Year')
    plt.ylabel('Total Card Count')
    plt.axvline(x=19.2, color='blue', linestyle='--')
    plt.text(16, 70000, 'Pioneer Legal', color = 'black', fontsize = 10)
    plt.axvline(x=25.5, color='black', linestyle='--')
    plt.text(22.7, 70000, 'Fire Design', color = 'black', fontsize = 10)
    plt.show()