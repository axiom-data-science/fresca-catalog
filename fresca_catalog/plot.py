import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
from copy import deepcopy
from typing import List

from .catalog import Catalog

import seaborn as sns
from matplotlib.colors import LogNorm

def build_agg_table(catalog: Catalog) -> pd.DataFrame:

    base_cols = ['date', 'cruise_id', 'station', 'event_n', 'latitude', 'longitude']
    entry_names = list(catalog.entries.keys())
    dfs = []
    for entry_name in entry_names:
        df = catalog[entry_name].read()

        cols_to_keep = deepcopy(base_cols)

        # Handle CSV lat/lon columns
        if all(c in catalog[entry_name].metadata for c in ['lon_col', 'lat_col']):
            lon_col = catalog[entry_name].metadata['lon_col']
            lat_col = catalog[entry_name].metadata['lat_col']
        # Handle ERDDAP lat/lon columns (which have the unit appended)
        else:
            lon_col = next(c for c in df.columns if 'longitude' in c)
            lat_col = next(c for c in df.columns if 'latitude' in c)
        
        # Rename lat/lon columns to standard names
        df = df.rename(columns={lon_col: 'longitude', lat_col: 'latitude'})
        lon_col = 'longitude'
        lat_col = 'latitude'

        # Add something here that makes this bit optional since
        # it won't always exist (it's a product of `filter_catalog`)
        for v in catalog.metadata['variables']:
            if v in df.columns:
                cols_to_keep.append(v)
        cols_to_keep = list(set(cols_to_keep))
        df = df[cols_to_keep]

        df = df.reset_index(drop=True)
        dfs.append(df)

    df = pd.concat(dfs).fillna(0)

    df['date'] = pd.to_datetime(df['date'])

    if 'time_range' in catalog.metadata:
        start, end = tuple(pd.to_datetime(d).tz_localize(None) for d in catalog.metadata['time_range'])
    else:
        start, end = df['date'].min(), df['date'].max()

    df = df[df['date'].between(start, end)]

    var_cols = df.columns.difference(base_cols)
    df[var_cols] = (df[var_cols] > 0).astype(int)

    # Collapse points down to per-station centroid
    centroids = (
        df.groupby('station')[[lon_col, lat_col]]
        .mean()
        .apply(lambda row: Point(row[lon_col], row[lat_col]), axis=1)
    )
    df.drop(columns=[lon_col, lat_col], inplace=True)

    # Aggregate duplicate station visits per date
    df = df.groupby(['station', 'date']).sum(numeric_only=True).reset_index()

    date_range = pd.date_range(start, end)
    stations = df['station'].unique()

    full_index = pd.MultiIndex.from_product([stations, date_range], names=['station', 'date'])
    df = df.set_index(['station', 'date'])
    df = df.reindex(full_index, fill_value=0).reset_index()
    df['event_n'] = 1
        
    df['geometry'] = df['station'].map(centroids)
    gdf = gpd.GeoDataFrame(df)
    gdf = gdf.set_crs(epsg=4326)
    gdf = gdf[gdf.geometry.notnull()].copy()
    gdf = gdf[gdf.geometry.apply(lambda g: g.is_valid if g else False)]

    return gdf


def plot_map(
    catalog: Catalog,
    metric: str = "mean",
    time_bin: str = "D",
    log: bool = False
):
    """Maps datasets in the catalog.

    Parameters
    ----------
    catalog : Catalog
        The catalog containing the datasets to map.
    variables : list
        The variables map.
    log : bool
        Whether or not to log normalize counts. Defaults to False.
    hex : bool
        Whether or not to aggregate observations into hexes. Defaults to True.
    """
    base_cols= ['date', 'station', 'event_n', 'geometry']

    if not hasattr(catalog, 'agg_table'):
        catalog.agg_table = build_agg_table(catalog)
    
    agg_table = deepcopy(catalog.agg_table)

    if time_bin != "D":
        agg_table = agg_table.set_index('date')
        agg_table = agg_table.groupby(['station', 'geometry']).resample(time_bin).sum(numeric_only=True).fillna(0).reset_index()
        var_cols = agg_table.columns.difference(base_cols)
        if metric == 'mean':
            agg_table[var_cols] = (agg_table[var_cols] > 0).astype(int)
        agg_table['event_n'] = 1

    # Divide all non-base columns by the value in the event_n column and then drop the event_n column
    for col in agg_table.columns:
        if col not in base_cols and agg_table[col].dtype in ['int64', 'float64']:
            agg_table[col] = agg_table[col] / agg_table['event_n']
    agg_table.drop(columns=['event_n'], inplace=True)
    if metric == 'mean':
        agg_table = agg_table.groupby(['station', 'geometry']).mean(numeric_only=True).reset_index()
        agg_table[metric] = agg_table.mean(numeric_only=True, axis=1)
    elif metric == 'sum':
        agg_table = agg_table.groupby(['station', 'geometry']).sum(numeric_only=True).reset_index()
        agg_table[metric] = agg_table.sum(numeric_only=True, axis=1)
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
    metric: str = 'mean',
    stations: List[str] = None,
    log: bool = False,
    time_bin: str = "D"
):
    """Maps datasets in the catalog.

    Parameters
    ----------
    catalog : Catalog
        The catalog containing the datasets to map.
    variables : list
        The variables map.
    log : bool
        Whether or not to log normalize counts. Defaults to False.
    hex : bool
        Whether or not to aggregate observations into hexes. Defaults to True.
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
    
    if stations:
        if isinstance(stations, str):
            stations = [stations]
        agg_table = agg_table[agg_table['station'].isin(stations)]
    
    if time_bin != "D":
        agg_table = agg_table.set_index('date')
        agg_table = agg_table.groupby('station').resample(time_bin).sum(numeric_only=True).fillna(0).reset_index()
        var_cols = agg_table.columns.difference(base_cols)
        if metric == 'mean':
            agg_table[var_cols] = (agg_table[var_cols] > 0).astype(int)
        agg_table['event_n'] = 1

    # Divide all non-base columns by the value in the event_n column and then drop the event_n column
    for col in agg_table.columns:
        if col not in base_cols and agg_table[col].dtype in ['int64', 'float64']:
            agg_table[col] = agg_table[col] / agg_table['event_n']
    agg_table.drop(columns=['event_n'], inplace=True)

    if metric == 'mean':
        agg_table = agg_table.groupby('station').mean(numeric_only=True).reset_index()
    elif metric == 'sum':
        agg_table = agg_table.groupby('station').sum(numeric_only=True).reset_index()

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
    metric: str = 'mean',
    stations: List[str] = None,
    variables: List[str] = None,
    log: bool = False,
    time_bin: str = "D"
):
    """Maps datasets in the catalog.

    Parameters
    ----------
    catalog : Catalog
        The catalog containing the datasets to map.
    variables : list
        The variables map.
    log : bool
        Whether or not to log normalize counts. Defaults to False.
    hex : bool
        Whether or not to aggregate observations into hexes. Defaults to True.
    """
    base_cols= ['date', 'station', 'event_n']

    if not hasattr(catalog, 'agg_table'):
        catalog.agg_table = build_agg_table(catalog)
    
    agg_table = deepcopy(catalog.agg_table)
    agg_table.drop(columns=['geometry'], inplace=True)

    if stations:
        if isinstance(stations, str):
            stations = [stations]
        agg_table = agg_table[agg_table['station'].isin(stations)]
    else:
        stations = agg_table['station'].unique().tolist()
    
    if time_bin != "D":
        agg_table = agg_table.set_index('date')
        agg_table = agg_table.groupby('station').resample(time_bin).sum(numeric_only=True).fillna(0).reset_index()
        var_cols = agg_table.columns.difference(base_cols)
        if metric == 'mean':
            agg_table[var_cols] = (agg_table[var_cols] > 0).astype(int)
        agg_table['event_n'] = 1

    # Divide all non-base columns by the value in the event_n column and then drop the event_n column
    for col in agg_table.columns:
        if col not in base_cols and agg_table[col].dtype in ['int64', 'float64']:
            agg_table[col] = agg_table[col] / agg_table['event_n']
    agg_table.drop(columns=['event_n'], inplace=True)

    if len(stations) > 1:
        if metric == 'mean':
            agg_table[metric] = agg_table.mean(numeric_only=True, axis=1)
            agg_table[metric] /= len(stations)
        elif metric == 'sum':
            agg_table[metric] = agg_table.max(numeric_only=True, axis=1)  # max bc we don't want to double count visits
        agg_table = agg_table[['date', 'station', metric]].pivot(index='date', columns='station')
        agg_table.columns = agg_table.columns.droplevel(0)
        agg_table.columns.name = None

    else:
        agg_table = agg_table.drop(columns=['station'])
        agg_table = agg_table.set_index('date')
        if metric == 'mean':
            for col in agg_table.columns:
                agg_table[col] = agg_table[col] / len(agg_table.columns)

    import matplotlib.pyplot as plt
    ax = agg_table.plot(kind='bar', stacked=True, logy=log, width=1.0)
    xticks = range(0, len(agg_table), max(1, len(agg_table)//10))  # adjust tick frequency
    ax.set_xticks(xticks)
    ax.set_xticklabels(agg_table.index.strftime('%Y-%m')[xticks], rotation=45)
    plt.tight_layout()
    plt.show()
