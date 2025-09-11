"""Plotting utilities for the Fresca catalog."""
import geopandas as gpd
from copy import deepcopy
from typing import List

from .catalog import Catalog
from .utils import build_agg_table

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def plot_map(
    catalog: Catalog,
    metric: str = "mean",
    time_bin: str = "D",
    log: bool = False
) -> None:
    """Maps datasets in the catalog.

    Parameters
    ----------
    catalog : Catalog
        The catalog containing the datasets to map.
    metric : str
        The metric to plot. Either 'mean' or 'sum'. Defaults to 'mean'.
    time_bin : str
        The time bin to aggregate data by. Defaults to 'D' (daily). Uses pandas offset alias strings.
    log : bool
        Whether or not to log normalize counts. Defaults to False.
    """
    base_cols= ['date', 'station', 'event_n', 'geometry']

    if not hasattr(catalog, 'agg_table'):
        catalog.agg_table = build_agg_table(catalog)
    
    agg_table = deepcopy(catalog.agg_table)
    
    num_cols = list(agg_table.select_dtypes(include="number").columns)

    if time_bin != "D":
        agg_table = agg_table.set_index('date')
        agg_table = agg_table.groupby(['station', 'geometry']).resample(time_bin)[num_cols].sum().fillna(0).reset_index()
        var_cols = agg_table.columns.difference(base_cols)
        if metric == 'mean':
            agg_table[var_cols] = (agg_table[var_cols] > 0).astype(int)
        agg_table['event_n'] = 1

    # Divide all non-base columns by the value in the event_n column and then drop the event_n column
    for col in agg_table.columns:
        if col not in base_cols and agg_table[col].dtype in ['int64', 'float64']:
            agg_table[col] = agg_table[col] / agg_table['event_n']
    
    agg_table.drop(columns=['event_n'], inplace=True)
    num_cols.remove('event_n')
    
    if metric == 'mean':
        agg_table = agg_table.groupby(['station', 'geometry'])[num_cols].mean().reset_index()
        agg_table[metric] = agg_table[num_cols].mean(axis=1)
    elif metric == 'sum':
        agg_table = agg_table.groupby(['station', 'geometry'])[num_cols].sum().reset_index()
        agg_table[metric] = agg_table[num_cols].sum(axis=1)
    agg_table = gpd.GeoDataFrame(agg_table)
    agg_table = agg_table.set_crs(epsg=4326)
    agg_table = agg_table[['station', 'geometry', metric]]

    plot = agg_table.hvplot.points(
        c=metric,
        cmap='viridis',
        logz=log,
        geo=True,
        tiles='OSM',
        hover_cols=['station', metric],
        width=800,
        height=600
    )
    from IPython.display import display
    display(plot)

def plot_grid(
    catalog: Catalog,
    stations: List[str] = None,
    metric: str = 'mean',
    time_bin: str = "D",
    log: bool = False
) -> None:
    """Maps datasets in the catalog.

    Parameters
    ----------
    catalog : Catalog
        The catalog containing the datasets to map.
    stations : list
        The stations to plot. If None, plots all stations. Defaults to None.
    metric : str
        The metric to plot. Either 'mean' or 'sum'. Defaults to 'mean'.
    time_bin : str
        The time bin to aggregate data by. Defaults to 'D' (daily). Uses pandas offset alias strings.
    log : bool
        Whether or not to log normalize counts. Defaults to False.
    """
    if log:
        norm = LogNorm()
    else:
        norm = None
    
    base_cols= ['date', 'station', 'event_n']

    if not hasattr(catalog, 'agg_table'):
        catalog.agg_table = build_agg_table(catalog)
    
    agg_table = deepcopy(catalog.agg_table)
    agg_table.drop(columns=['geometry'], inplace=True)

    num_cols = list(agg_table.select_dtypes(include="number").columns)
    
    if stations:
        if isinstance(stations, str):
            stations = [stations]
        agg_table = agg_table[agg_table['station'].isin(stations)]
    
    if time_bin != "D":
        agg_table = agg_table.set_index('date')
        agg_table = agg_table.groupby('station').resample(time_bin)[num_cols].sum().fillna(0).reset_index()
        var_cols = agg_table.columns.difference(base_cols)
        if metric == 'mean':
            agg_table[var_cols] = (agg_table[var_cols] > 0).astype(int)
        agg_table['event_n'] = 1

    # Divide all non-base columns by the value in the event_n column and then drop the event_n column
    for col in agg_table.columns:
        if col not in base_cols and agg_table[col].dtype in ['int64', 'float64']:
            agg_table[col] = agg_table[col] / agg_table['event_n']
    agg_table.drop(columns=['event_n'], inplace=True)
    num_cols.remove('event_n')

    if metric == 'mean':
        agg_table = agg_table.groupby('station')[num_cols].mean().reset_index()
    elif metric == 'sum':
        agg_table = agg_table.groupby('station')[num_cols].sum().reset_index()

    agg_table = agg_table.set_index('station')

    fig_height = len(agg_table) * 0.5
    fig_width = len(agg_table.columns) * 0.5
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(fig_width, fig_height))
    plt.yticks(ticks=range(agg_table.shape[0]), labels=agg_table.index, rotation=90)
    sns.heatmap(agg_table, cmap='viridis', norm=norm)
    plt.show()

def plot_timeseries(
    catalog: Catalog,
    stations: List[str] = None,
    metric: str = 'mean',
    time_bin: str = "D",
    log: bool = False
) -> None:
    """Maps datasets in the catalog.

    Parameters
    ----------
    catalog : Catalog
        The catalog containing the datasets to map.
    stations : list
        The stations to plot. If None, plots all stations. Defaults to None.
    metric : str
        The metric to plot. Either 'mean' or 'sum'. Defaults to 'mean'.
    time_bin : str
        The time bin to aggregate data by. Defaults to 'D' (daily). Uses pandas offset alias strings.
    log : bool
        Whether or not to log normalize counts. Defaults to False.
    """
    base_cols= ['date', 'station', 'event_n']

    if not hasattr(catalog, 'agg_table'):
        catalog.agg_table = build_agg_table(catalog)
    
    agg_table = deepcopy(catalog.agg_table)
    agg_table.drop(columns=['geometry'], inplace=True)

    num_cols = list(agg_table.select_dtypes(include="number").columns)

    if stations:
        if isinstance(stations, str):
            stations = [stations]
        agg_table = agg_table[agg_table['station'].isin(stations)]
    else:
        stations = agg_table['station'].unique().tolist()
    
    if time_bin != "D":
        agg_table = agg_table.set_index('date')
        agg_table = agg_table.groupby('station').resample(time_bin)[num_cols].sum().fillna(0).reset_index()
        var_cols = agg_table.columns.difference(base_cols)
        if metric == 'mean':
            agg_table[var_cols] = (agg_table[var_cols] > 0).astype(int)
        agg_table['event_n'] = 1

    # Divide all non-base columns by the value in the event_n column and then drop the event_n column
    for col in agg_table.columns:
        if col not in base_cols and agg_table[col].dtype in ['int64', 'float64']:
            agg_table[col] = agg_table[col] / agg_table['event_n']

    agg_table.drop(columns=['event_n'], inplace=True)
    num_cols.remove('event_n')

    if len(stations) > 1:
        if metric == 'mean':
            agg_table[metric] = agg_table[num_cols].mean(axis=1)
            agg_table[metric] /= len(stations)
        elif metric == 'sum':
            agg_table[metric] = agg_table[num_cols].max(axis=1)  # max bc we don't want to double count visits
        agg_table = agg_table[['date', 'station', metric]].pivot(index='date', columns='station')
        agg_table.columns = agg_table.columns.droplevel(0)
        agg_table.columns.name = None

    else:
        agg_table = agg_table.drop(columns=['station'])
        agg_table = agg_table.set_index('date')
        if metric == 'mean':
            for col in agg_table.columns:
                agg_table[col] = agg_table[col] / len(agg_table.columns)

    ax = agg_table.plot(kind='bar', stacked=True, logy=log, width=1.0)
    xticks = range(0, len(agg_table), max(1, len(agg_table)//10))  # adjust tick frequency
    ax.set_xticks(xticks)
    ax.set_xticklabels(agg_table.index.strftime('%Y-%m')[xticks], rotation=45)
    plt.tight_layout()
    plt.show()
