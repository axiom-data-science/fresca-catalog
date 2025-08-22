"""Summary utilities for the Fresca catalog."""
import pandas as pd
from pandas import DataFrame
from IPython.display import display

from fresca_catalog.catalog import Catalog

def summarize_catalog(catalog: Catalog) -> None:
    """Summarizes the datasets in the catalog.
    
    Parameters
    ----------
    catalog : Catalog
        The catalog to summarize.
    """
    data = {
        'dataset': [],
        'date_start': [],
        'date_end': [],
        'lon_min': [],
        'lon_max': [],
        'lat_min': [],
        'lat_max': [],
        'total_stations': [],
        'total_casts': [],
        'mean_station_casts': [],
        'median_station_casts': []
    }
    for dataset_name in catalog.entries.keys():
        dataset = catalog[dataset_name]
        data['dataset'].append(dataset_name)
        data['date_start'].append(dataset.metadata['minTime'].split('T')[0])
        data['date_end'].append(dataset.metadata['maxTime'].split('T')[0])
        data['lon_min'].append(dataset.metadata['minLongitude'])
        data['lon_max'].append(dataset.metadata['maxLongitude'])
        data['lat_min'].append(dataset.metadata['minLatitude'])
        data['lat_max'].append(dataset.metadata['maxLatitude'])
        data['total_stations'].append(len(dataset.read()['station'].unique()))
        data['total_casts'].append(len(dataset.read()[['date', 'station']].drop_duplicates()))
        data['mean_station_casts'].append(round(dataset.read()[['date', 'station']].groupby('station').count().mean().item(), 1))
        data['median_station_casts'].append(round(dataset.read()[['date', 'station']].groupby('station').count().median().item(), 1))
    df = DataFrame(data)
    display(df)

def summarize_entry_variables(catalog: Catalog, entry_name: str) -> None:
    """Summarizes the variables in a catalog entry.
    
    Parameters
    ----------
    catalog : Catalog
        The catalog containing the entry.
    entry_name : str
        The name of the entry to summarize.
    """
    data = {
        'variable': [],
        'date_start': [],
        'date_end': [],
        'lon_min': [],
        'lon_max': [],
        'lat_min': [],
        'lat_max': [],
        'total_stations': [],
        'total_casts': [],
        'mean_station_casts': [],
        'median_station_casts': []
    }
    
    # Handle CSV lat/lon columns
    if all(c in catalog[entry_name].metadata for c in ['lon_col', 'lat_col']):
        lon_col = catalog[entry_name].metadata['lon_col']
        lat_col = catalog[entry_name].metadata['lat_col']
    # Handle ERDDAP lat/lon columns (which have the unit appended)
    else:
        lon_col = next(c for c in df.columns if 'longitude' in c)
        lat_col = next(c for c in df.columns if 'latitude' in c)

    base_cols = ['date', 'cruise_id', 'station', 'event_n', lon_col, lat_col]

    df = catalog[entry_name].read()
    
    variables = [v for v in catalog[entry_name].metadata['variables'] if v not in base_cols]
    for var in variables:
        data['variable'].append(var)
        var_df = df[base_cols + [var]][df[var] > 0]
        data['date_start'].append(var_df['date'].min())
        data['date_end'].append(var_df['date'].max())
        data['lon_min'].append(var_df[lon_col].min())
        data['lon_max'].append(var_df[lon_col].max())
        data['lat_min'].append(var_df[lat_col].min())
        data['lat_max'].append(var_df[lat_col].max())
        data['total_stations'].append(len(var_df['station'].unique()))
        data['total_casts'].append(len(var_df[['date', 'station']].drop_duplicates()))
        data['mean_station_casts'].append(round(var_df[['date', 'station']].groupby('station').count().mean().item(), 1))
        data['median_station_casts'].append(round(var_df[['date', 'station']].groupby('station').count().median().item(), 1))

    display(DataFrame(data))

def summarize_entry_stations(catalog: Catalog, entry_name: str, show_all: bool = False) -> None:
    """Summarizes the stations in a catalog entry.
    
    Parameters
    ----------
    catalog : Catalog
        The catalog containing the entry.
    entry_name : str
        The name of the entry to summarize.
    show_all : bool, optional
        Whether or not to show all rows in the output, by default False.
    """
    data = {
        'station': [],
        'date_start': [],
        'date_end': [],
        'lon': [],
        'lat': [],
        'total_casts': [],
        'max_depths': [],
        'median_depths': []
    }
    
    # Handle CSV lat/lon columns
    if all(c in catalog[entry_name].metadata for c in ['lon_col', 'lat_col']):
        lon_col = catalog[entry_name].metadata['lon_col']
        lat_col = catalog[entry_name].metadata['lat_col']
    # Handle ERDDAP lat/lon columns (which have the unit appended)
    else:
        lon_col = next(c for c in df.columns if 'longitude' in c)
        lat_col = next(c for c in df.columns if 'latitude' in c)

    df = catalog[entry_name].read()

    stations = df['station'].unique()
    for stn in stations:
        data['station'].append(stn)
        stn_df = df[df['station'] == stn]
        data['date_start'].append(stn_df['date'].min())
        data['date_end'].append(stn_df['date'].max())
        data['lon'].append(stn_df[lon_col].mean())
        data['lat'].append(stn_df[lat_col].mean())
        data['total_casts'].append(len(stn_df['date'].drop_duplicates()))
        data['max_depths'].append(stn_df['depth'].max())
        data['median_depths'].append(int(stn_df['depth'].median()))

        
    if show_all:
        with pd.option_context('display.max_rows', None):
            display(DataFrame(data))
    else:
        display(DataFrame(data))
