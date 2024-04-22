"""
plot_set_data.py

Module to plot set information.

Author: Jordan Bourdeau
Date Created: 4/14/24
"""

import datetime
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd

import src.constants as c

def get_set_metrics_per_year(
        full_set_data: pd.DataFrame, 
        metric: str = 'mean_price', 
        use_max: bool = True,
    ) -> pd.DataFrame:
    """
    Function which gets the list of sets with the max metric from each year,

    :param full_set_data:   Full data on the set, including all augmentations.
    :param metric:          String name of the column to check.
    :param use_max:         Boolean flag for whether to use idxmax or idxmin

    :return: Returns Pandas Dataframe with selected set per year.
    """
    if use_max:
        df: pd.DataFrame = full_set_data.loc[full_set_data.groupby(['release_year'])[metric].idxmax()]
    else:
        df: pd.DataFrame = full_set_data.loc[full_set_data.groupby(['release_year'])[metric].idxmin()]
    return df

def plot_superimposed_max_min_metrics_by_year(
        augmented_data: pd.DataFrame, 
        metrics: list[str],
        listing_date: datetime.date, 
        format: str,
        legend_params: list[tuple[int, str, str, str]] = [],
    ):
    """
    Function which creates the metric plots which extract the max/min of each metric
    among the sets for each year.

    :param augmented_data: Pandas dataframe with the full, augmented data.
    :param metrics:        List of column names to create plots for.
    :param listing_date:   Date which card listings are gathered from. 
    :param format:         Format being plotted. 
    :param legend_params:  List of 4-tuples with year, line color, line style, and label to plot on to graph.
    """

    plt.style.use('seaborn-white')
    plt.figure(figsize=(10, 8))
    plt.title(f'{listing_date} - Max/Min Metrics For Sets Per Year in {format.capitalize()} Format')
    plt.tight_layout(pad=4)

    # Get internal list of colors, excluding the ones we are using for the lines
    steel_blue: str = f'#1f77b4'
    coral: str = '#ff7f0e'
    # Define colorblind-friendly colors
    colors = ['#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    # Get handles to manually make legend
    handles, _ = plt.gca().get_legend_handles_labels()

    earliest_year: int = augmented_data['release_year'].min()
    latest_year: int = augmented_data['release_year'].max()

    for year, color, linestyle, label in legend_params:
        if year > latest_year or year < earliest_year:
                continue
        plt.axvline(year - 0.5, color=color, linestyle=linestyle, label=label)
        handles.append(plt.Line2D([], [], color=color, linestyle=linestyle, label=label))

    # Manually define patches
    max_line_patch = mpatches.Patch(color=coral, label='Max Value Line', linestyle='-')
    min_line_patch = mpatches.Patch(color=steel_blue, label='Min Value Line', linestyle='-')

    # handles is a list, so append manual patch
    handles += [max_line_patch, min_line_patch]

    for color, metric in zip(colors, metrics):
        min_df: pd.DataFrame = get_set_metrics_per_year(augmented_data, metric, False)
        max_df: pd.DataFrame = get_set_metrics_per_year(augmented_data, metric, True)

        label: str = ' '.join([word.capitalize() for word in metric.split('_')])
        
        plt.xlabel('Set Release Year')
        plt.ylabel(label)

        plt.plot(min_df['release_year'], min_df[metric], color=steel_blue, alpha=0.5)
        plt.scatter(min_df['release_year'], min_df[metric], color=color)
        
        plt.plot(max_df['release_year'], max_df[metric], color=coral, alpha=0.5)
        plt.scatter(max_df['release_year'], max_df[metric], color=color)

        metric_patch  = mpatches.Patch(color=color, label=label)
        handles.append(metric_patch)

        for i, txt in enumerate(min_df['set_code']):
            plt.annotate(txt, (min_df['release_year'].iloc[i], min_df[metric].iloc[i]), 
                         ha='center', va='top', xytext=(0, -5), textcoords='offset points')
        for i, txt in enumerate(max_df['set_code']):
            plt.annotate(txt, (max_df['release_year'].iloc[i], max_df[metric].iloc[i]), 
                         ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

        plt.xticks(min_df['release_year'], rotation=45)
        plt.savefig(os.path.join(c.IMAGE_DIRECTORY, f'superimposed_metrics_plot.png'))

    plt.legend(handles=handles)
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)

def plot_max_min_metrics_by_year(
        augmented_data: pd.DataFrame, 
        metrics: list[str],
        listing_date: datetime.date, 
        format: str,
        legend_params: list[tuple[int, str, str, str]] = [],
    ):
    """
    Function which creates the metric plots which extract the max/min of each metric
    among the sets for each year.

    :param augmented_data: Pandas dataframe with the full, augmented data.
    :param metrics:        List of column names to create plots for.
    :param listing_date:   Date which card listings are gathered from. 
    :param format:         Format being plotted. 
    :param legend_params:  List of 4-tuples with year, line color, line style, and label to plot on to graph.
    """

    # Get internal list of colors, excluding the ones we are using for the lines
    steel_blue: str = f'#1f77b4'
    coral: str = '#ff7f0e'

    earliest_year: int = augmented_data['release_year'].min()
    latest_year: int = augmented_data['release_year'].max()

    for metric in metrics:
        min_df: pd.DataFrame = get_set_metrics_per_year(augmented_data, metric, False)
        max_df: pd.DataFrame = get_set_metrics_per_year(augmented_data, metric, True)

        label: str = ' '.join([word.capitalize() for word in metric.split('_')])

        # Manually define patches
        max_line_patch = mpatches.Patch(color=coral, label=f'Max {label} Value', linestyle='-')
        min_line_patch = mpatches.Patch(color=steel_blue, label=f'Min {label} Value', linestyle='-')

        plt.style.use('seaborn-white')
        plt.figure(figsize=(10, 8))
        plt.title(f'{listing_date} - Max/Min {label} Metrics For Sets Per Year in {format.capitalize()} Format')
        plt.tight_layout(pad=4)

        # Get handles to manually make legend
        handles, _ = plt.gca().get_legend_handles_labels() 

        for year, color, linestyle, lab in legend_params:
            if year > latest_year or year < earliest_year:
                continue
            plt.axvline(year - 0.5, color=color, linestyle=linestyle, label=lab)
            handles.append(plt.Line2D([], [], color=color, linestyle=linestyle, label=lab))

        # Handles is a list, so append manual patch
        handles += [max_line_patch, min_line_patch]

        plt.xlabel('Set Release Year')
        plt.ylabel(label)

        plt.plot(min_df['release_year'], min_df[metric], color=steel_blue, alpha=0.5)
        plt.scatter(min_df['release_year'], min_df[metric], color=steel_blue)
        
        plt.plot(max_df['release_year'], max_df[metric], color=coral, alpha=0.5)
        plt.scatter(max_df['release_year'], max_df[metric], color=coral)

        for i, txt in enumerate(min_df['set_code']):
            plt.annotate(txt, (min_df['release_year'].iloc[i], min_df[metric].iloc[i]), 
                         ha='center', va='top', xytext=(0, -5), textcoords='offset points')
            
        for i, txt in enumerate(max_df['set_code']):
            plt.annotate(txt, (max_df['release_year'].iloc[i], max_df[metric].iloc[i]), 
                         ha='center', va='bottom', xytext=(0, 5), textcoords='offset points')

        plt.xticks(min_df['release_year'], rotation=45)
        plt.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, borderaxespad=1.0)

        plt.savefig(os.path.join(c.IMAGE_DIRECTORY, f'{metric}_min_max_plot.png'), bbox_inches='tight') # Adjusted to include bbox_inches='tight'
        plt.show()
        plt.rcParams.update(plt.rcParamsDefault)

def plot_average_card_price_over_time(
        card_price_df: pd.DataFrame, 
        listing_date: datetime.date, 
        start_year: int = 1991, 
        end_year: int = 2023,
        legend_params: list[tuple[int, str, str, str]] = [],
    ):
    """
    Function which plots the average card price based on the year of the set 
    it is the least expensive in came out.

    :param card_price_df: Pandas dataframe containing market price data.
    :param listing_date:  Date which card listings are gathered from.        
    :param start_year:    Start of the year window to look at (inclusive). Defaults to 1993 (when Magic came out).
    :param end_year:      End of the year window to look at (inclusive). Defaults to 2023 (whole year increment from analysis). 
    :param legend_params:  List of 4-tuples with year, line color, line style, and label to plot on to graph.
    
    :returns: Outputs plot and saves it.
    """
    agg_df: pd.DataFrame = card_price_df.groupby(['release_year'])['price'].agg(['mean', 'median', 'std']).reset_index()
    trimmed_df: pd.DataFrame = agg_df[(agg_df['release_year'] >= start_year) & (agg_df['release_year'] <= end_year)]

    fig, (mean_plot, median_plot) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Mean plot
    mean_plot.set_ylabel('Mean Price (USD)')
    mean_plot.set_title(f'Mean Card Price from Cheapest Set Release Based on {listing_date} Listing (USD)')
    mean_plot.plot(trimmed_df['release_year'], trimmed_df['mean'])

    # Median plot
    median_plot.set_xlabel('Release Year')
    median_plot.set_ylabel('Median Price (USD)')
    median_plot.set_title(f'Median Card Price from Cheapest Set Release Based on {listing_date} Listing (USD)')
    median_plot.plot(trimmed_df['release_year'], trimmed_df['median'])

    # Add additional context to plots
    handles: list = []

    for year, color, linestyle, label in legend_params:
        mean_plot.axvline(year - 0.5, color=color, linestyle=linestyle, label=label)
        median_plot.axvline(year - 0.5, color=color, linestyle=linestyle, label=label)
        handles.append(plt.Line2D([], [], color=color, linestyle=linestyle, label=label))

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, borderaxespad=-1.0)

    fig.tight_layout()
    fig.savefig(os.path.join(c.IMAGE_DIRECTORY, 'average_card_price_over_time.png'))
    plt.show()

def plot_stacked_set_counts(
        card_counts_df: pd.DataFrame, 
        set_year_df: pd.DataFrame, 
        format: str, 
        relative: bool = True,
        legend_params: list[tuple[int, str, str, str]] = [],
        tournament_year: int = 2023,
    ):
    """
    Function which creates a stacked histogram for the prevalence of card usage based on the
    number of times cards from the set get used based on the set release year.

    :param card_counts_df:  Dataframe with the card counts.
    :param set_year_df:     Dataframe with the set and release year.
    :param format:          String for the format file (for saving file).
    :param relative:        Boolean flag for if the histogram should be in relative measures (%).
    :param legend_params:   List of 4-tuples with year, line color, line style, and label to plot on to graph.
    :param tournament_year: Year the tournament data is from.
    """
    
    # Grouping by 'set_year' and 'set_name', and summing up 'total_count'
    grouped_data = card_counts_df.groupby(['release_year', 'set_code'])['total_count'].sum().unstack().fillna(0)

    xlabel: str = 'Year'
    if relative:
        # Convert to percentage
        total_count: int = card_counts_df['total_count'].sum()
        grouped_data /= total_count
        grouped_data *= 100
        # Labels
        title: str = f'% of Total Cards Played by Set in {tournament_year} {format.capitalize()} Format Tournaments'
        ylabel: str = '% of Total Cards Played' 
        file_name: str = f'{format}_stacked_set_relative.png'
    else:
        title: str = f'Card Play Counts by Set and Year in {format.capitalize()}'
        ylabel: str = 'Total Card Count' 
        file_name: str = f'{format}_stacked_set_counts.png'

    # Creating a DataFrame with 'set_name' as a column
    sets = pd.DataFrame({'set_name': grouped_data.columns})

    # Merging with 'set_year_df' to get the release year
    sets = pd.merge(sets, set_year_df, on='set_name', how='left')

    # Plotting the data as a stacked bar chart
    plt.style.use('seaborn-white')
    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust the figure size as needed

    grouped_data.reset_index(inplace=True)
    grouped_data['release_year'] = grouped_data['release_year'].astype('int32')

    grouped_data.plot(kind='bar', x='release_year', stacked=True, ax=ax, legend=None)

    min_year = grouped_data['release_year'].min()

    # Add additional context to plot
    handles: list = []

    for year, color, linestyle, label in legend_params:
        # 0.5 is a slight spacer before the bar
        ax.axvline(year - min_year - 0.5, color=color, linestyle=linestyle)
        handles.append(plt.Line2D([], [], color=color, linestyle=linestyle, label=label))

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_xlim(1991 - min_year, 2025 - min_year)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis='x', rotation=45)

    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, -0.05), ncol=2, borderaxespad=-1.0)
    fig.tight_layout()    
    fig.savefig(os.path.join(c.IMAGE_DIRECTORY, file_name), bbox_inches='tight')
    plt.rcParams.update(plt.rcParamsDefault)
    fig.show()

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
    plt.rcParams.update(plt.rcParamsDefault)

def plot_outlier_distribution(
        outliers: pd.DataFrame,
        format: str,
    ):
    """
    Function to plot the outlier distribution histogram.

    :param outliers: Dataframe with outliers.
    :param format: Set format the outliers are from.
    """
    # Generate a histogram of outlier release years
    plt.style.use('seaborn-white')
    plt.figure(figsize=(10, 6))
    plt.hist(outliers['release_year'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title(f'Histogram of Outlier Release Years from {format.capitalize()} Data')
    plt.xlabel('Set Release Year')
    plt.ylabel('Number of Outlier Sets')

    # Set x-axis ticks to integer years only
    min_year = 1992
    max_year = 2023
    plt.xticks(np.arange(min_year, max_year + 1, 1), rotation=45)

    # Add additional context to plot with axvlines
    handles = []
    legend_params = [
        (1993, 'green', '--', 'Magic First Comes Out (1993)'),
        (2019, 'red', '--', 'Fire Design Principle Implemented (2019)'),
    ]

    for year, color, linestyle, label in legend_params:
        # 0.5 is a slight spacer before the bar
        plt.axvline(year - 0.5, color=color, linestyle=linestyle)
        handles.append(plt.Line2D([], [], color=color, linestyle=linestyle, label=label))

    plt.legend(handles=handles)

    # Calculate total number of outliers
    total_outliers = len(outliers)

    # Calculate percentage released after fire design principle
    outliers_after_fire_design = outliers[outliers['release_year'] >= 2019]
    percentage_after_fire_design = (len(outliers_after_fire_design) / total_outliers) * 100

    # Add caption
    caption = f"Total outliers: {total_outliers}\nPercentage released after fire design principle: {percentage_after_fire_design:.2f}%"
    plt.text(0.5, -0.2, caption, horizontalalignment='center', verticalalignment='center', transform=plt.gca().transAxes)


    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)