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
    ):
    """
    Function which creates the metric plots which extract the max/min of each metric
    among the sets for each year.

    :param augmented_data: Pandas dataframe with the full, augmented data.
    :param metrics:        List of column names to create plots for.
    :param listing_date:  Date which card listings are gathered from. 
    """

    plt.style.use('seaborn-white')
    plt.figure(figsize=(10, 8))
    plt.title(f'{listing_date} - Max/Min Metrics For Sets Per Year')
    plt.tight_layout(pad=4)

    # Get internal list of colors, excluding the ones we are using for the lines
    steel_blue: str = f'#1f77b4'
    coral: str = '#ff7f0e'
    colors: list[str] = [color for color in plt.rcParams["axes.prop_cycle"].by_key()["color"] if color not in [coral, steel_blue]]
    
    # Get handles to manually make legend
    handles, _ = plt.gca().get_legend_handles_labels()

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

        plt.xticks(min_df['release_year'], rotation=90)
        plt.savefig(os.path.join(c.IMAGE_DIRECTORY, f'superimposed_metrics_plot.png'))

    plt.legend(bbox_to_anchor=(1.25, .5), handles=handles)
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)

def plot_max_min_metrics_by_year(
        augmented_data: pd.DataFrame, 
        metrics: list[str],
        listing_date: datetime.date, 
    ):
    """
    Function which creates the metric plots which extract the max/min of each metric
    among the sets for each year.

    :param augmented_data: Pandas dataframe with the full, augmented data.
    :param metrics:        List of column names to create plots for.
    :param listing_date:  Date which card listings are gathered from. 
    """

    # Get internal list of colors, excluding the ones we are using for the lines
    steel_blue: str = f'#1f77b4'
    coral: str = '#ff7f0e'

    for metric in metrics:
        min_df: pd.DataFrame = get_set_metrics_per_year(augmented_data, metric, False)
        max_df: pd.DataFrame = get_set_metrics_per_year(augmented_data, metric, True)

        label: str = ' '.join([word.capitalize() for word in metric.split('_')])

        # Manually define patches
        max_line_patch = mpatches.Patch(color=coral, label=f'Max {label} Value', linestyle='-')
        min_line_patch = mpatches.Patch(color=steel_blue, label=f'Min {label} Value', linestyle='-')

        plt.style.use('seaborn-white')
        plt.figure(figsize=(10, 8))
        plt.title(f'{listing_date} - Max/Min {label} Metrics For Sets Per Year')
        plt.tight_layout(pad=4)

        # Get handles to manually make legend
        handles, _ = plt.gca().get_legend_handles_labels()

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

        plt.xticks(min_df['release_year'], rotation=90)
        plt.savefig(os.path.join(c.IMAGE_DIRECTORY, f'{metric}_min_max_plot.png'))

        plt.legend(bbox_to_anchor=(1.275, .5), handles=handles)
        plt.show()
        plt.rcParams.update(plt.rcParamsDefault)

def plot_average_card_price_over_time(
        card_price_df: pd.DataFrame, 
        listing_date: datetime.date, 
        start_year: int = 1991, 
        end_year: int = 2023,
    ):
    """
    Function which plots the average card price based on the year of the set 
    it is the least expensive in came out.

    :param card_price_df: Pandas dataframe containing market price data.
    :param listing_date:  Date which card listings are gathered from.        
    :param start_year:    Start of the year window to look at (inclusive). Defaults to 1993 (when Magic came out).
    :param end_year:      End of the year window to look at (inclusive). Defaults to 2023 (whole year increment from analysis). 

    :returns: Outputs plot and saves it.
    """
    agg_df: pd.DataFrame = card_price_df.groupby(['release_year'])['price'].agg(['mean', 'median', 'std']).reset_index()
    trimmed_df: pd.DataFrame = agg_df[(agg_df['release_year'] >= start_year) & (agg_df['release_year'] <= end_year)]

    fig, (mean_plot, median_plot) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))

    # Mean plot
    mean_plot.set_xlabel('Release Year')
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
    legend_params: list[tuple] = [
        (1993, 'green', '--', 'Magic First Comes Out (1993)'),
        (2019, 'red', '--', 'Fire Design Principle Implemented (2019)'),
        (2021, 'blue', '--', 'Modern Horizons 2 Released (2021)'),
    ]

    for year, color, linestyle, label in legend_params:
        mean_plot.axvline(year - 0.5, color=color, linestyle=linestyle, label=label)
        median_plot.axvline(year - 0.5, color=color, linestyle=linestyle, label=label)
        handles.append(plt.Line2D([], [], color=color, linestyle=linestyle, label=label))

    mean_plot.legend(handles=handles, loc='upper center', bbox_to_anchor=[0.55, 0.95])
    median_plot.legend(handles=handles, loc='upper center', bbox_to_anchor=[0.55, 0.95])

    caption: str = "The set a card is associated with is drawn from the set where it has the cheapest price, as per the daily listings."
    # fig.figtext(x=0, y=-0.1, s=caption, wrap=True, horizontalalignment='left')
    fig.tight_layout()
    fig.savefig(os.path.join(c.IMAGE_DIRECTORY, 'average_card_price_over_time.png'))
    fig.show()

def plot_stacked_set_counts(card_counts_df: pd.DataFrame, set_year_df: pd.DataFrame, format: str, relative: bool = True):
    """
    Function which creates a stacked histogram for the prevalence of card usage based on the
    number of times cards from the set get used based on the set release year.

    :param card_counts_df: Dataframe with the card counts.
    :param set_year_df:    Dataframe with the set and release year.
    :param format:         String for the format file (for saving file).
    :param relative:       Boolean flag for if the histogram should be in relative measures (%).
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
        title: str = '% of Total Cards Played by Set and Year'
        ylabel: str = '% of Total Cards Played' 
        file_name: str = '{format}_stacked_set_relative.png'
    else:
        title: str = 'Card Play Counts by Set and Year'
        ylabel: str = 'Total Card Count' 
        file_name: str = '{format}_stacked_set_counts.png'

    # Creating a DataFrame with 'set_name' as a column
    sets = pd.DataFrame({'set_name': grouped_data.columns})

    # Merging with 'set_year_df' to get the release year
    sets = pd.merge(sets, set_year_df, on='set_name', how='left')

    # Plotting the data as a stacked bar chart
    plt.style.use('seaborn-white')
    plt.figure(figsize=(12, 8))  # Adjust the figure size as needed

    grouped_data.reset_index(inplace=True)
    grouped_data['release_year'] = grouped_data['release_year'].astype('int32')

    grouped_data.plot(kind='bar', x='release_year', stacked=True, legend=None)

    min_year = grouped_data['release_year'].min()

    # Add additional context to plot
    handles: list = []
    legend_params: list[tuple] = [
        (1993, 'green', '--', 'Magic First Comes Out (1993)'),
        (2019, 'red', '--', 'Fire Design Principle Implemented (2019)'),
        (2021, 'blue', '--', 'Modern Horizons 2 Released (2021)'),
    ]

    for year, color, linestyle, label in legend_params:
        # 0.5 is a slight spacer before the bar
        plt.axvline(year - min_year - 0.5, color=color, linestyle=linestyle)
        handles.append(plt.Line2D([], [], color=color, linestyle=linestyle, label=label))

    plt.legend(handles=handles)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.xlim(1991 - min_year, 2025 - min_year)
    plt.ylabel(ylabel)
    plt.tick_params(axis='x', rotation=45)

    plt.tight_layout()    
    plt.savefig(os.path.join(c.IMAGE_DIRECTORY, file_name))
    plt.show()
    plt.rcParams.update(plt.rcParamsDefault)

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

def plot_outlier_distribution(outliers: pd.DataFrame):
    # Generate a histogram of outlier release years
    plt.style.use('seaborn-white')
    plt.figure(figsize=(10, 6))
    plt.hist(outliers['release_year'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    plt.title('Histogram of Outlier Release Years')
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