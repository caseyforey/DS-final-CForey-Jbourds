"""
plot_tournament_data.py

Module for creating visualizations for tournament data.

Authors: Jordan Bourdeau, Casey Forey
Date Created: 4/7/2024
"""
import seaborn as sns 
import matplotlib.pyplot as plt
def create_base_graph(df,rgb = (0.5,0.5,0.5) ):
    grouped_year = df.groupby(['set_year']).sum()
    plt.figure(figsize=(10, 6))
    sns.set_style('ticks')
    sns.barplot(data = grouped_year, x = 'set_year', y = 'total_count',color = rgb)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.margins(x=0)
    plt.xlabel('Release Year')
    plt.ylabel('Total Card Count')
    return plt