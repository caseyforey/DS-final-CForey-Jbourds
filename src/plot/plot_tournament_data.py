"""
plot_tournament_data.py

Module for creating visualizations for tournament data.

Authors: Jordan Bourdeau, Casey Forey
Date Created: 4/7/2024
"""
import seaborn as sns 
import matplotlib.pyplot as plt
def create_base_graph(df,rgb = (0.5,0.5,0.5) ):
    grouped_year = df.groupby(['Release Year']).sum()
    plt.figure(figsize=(10, 6))
    sns.set_style('ticks')
    sns.barplot(data = grouped_year, x = 'Release Year', y = 'Total Count',color = rgb)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.margins(x=0)
    return plt