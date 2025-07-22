"""Builds a full Intake catalog from a base catalog file."""
import argparse
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Union
import yaml

import panel as pn

import contextily as ctx
import geopandas as gpd
import h3
import hvplot.pandas
from intake import Catalog
from intake.readers import CSV, PandasCSV
from intake_erddap import ERDDAPCatalogReader, TableDAPReader
from IPython.display import display, clear_output
import ipywidgets as w
import pandas as pd
import seaborn as sns
from matplotlib.colors import LogNorm
from shapely.geometry import box, Polygon, Point
from rapidfuzz import fuzz, process

import holoviews as hv

hv.extension('bokeh')

pn.extension()

def get_erddap_dataset_variables(server_url: str, dataset_id: str) -> List[str]:
    """Gets the list of variables in an ERDDAP dataset.
    
    Parameters
    ----------
    server_url : str
        The URL of the ERDDAP server
    dataset_id : str
        The ID of the dataset on the ERDDAP server
    
    Returns
    -------
    List[str]
        The list of variables in the dataset
    """
    dataset_reader = TableDAPReader(server_url, dataset_id)
    dataset_metadata = dataset_reader._get_dataset_metadata(server_url, dataset_id)
    variables = list(dataset_metadata['variables'].keys())
    return variables

def build_erddap_dataset_entry(server_url: str, dataset_id: str) -> ERDDAPCatalogReader:
    """Builds an ERDDAP dataset entry.

    Parameters
    ----------
    server_url : str
        The URL of the ERDDAP server
    dataset_id : str
        The ID of the dataset on the ERDDAP server

    Returns
    -------
    ERDDAPCatalogReader
        The ERDDAP dataset catalog entry
    """
    variables = get_erddap_dataset_variables(server_url, dataset_id)
    entry = ERDDAPCatalogReader(server_url, search_for=[dataset_id]).read()[dataset_id]
    entry.metadata['variables'] = variables
    return entry

def extract_csv_metadata(url: str, time_col: str, lat_col: str, lon_col: str) -> dict:
    """Extracts metadata from a CSV file.

    Parameters
    ----------
    url : str
        The URL of the CSV file
    time_col : str
        The name of the column containing the time values
    lat_col : str
        The name of the column containing the latitude values
    lon_col : str
        The name of the column containing the longitude values
    
    Returns
    -------
    dict
        The metadata extracted from the CSV file
    """
    df = pd.read_csv(url, parse_dates=[time_col])
    df[time_col] = pd.to_datetime(df[time_col], format='ISO8601')
    metadata = {
        'minTime': df[time_col].min().isoformat() + 'Z',
        'maxTime': df[time_col].max().isoformat() + 'Z',
        'minLatitude': float(df[lat_col].min()),
        'maxLatitude': float(df[lat_col].max()),
        'minLongitude': float(df[lon_col].min()),
        'maxLongitude': float(df[lon_col].max()),
        'variables': list(df.columns),
        'time_col': time_col,
        'lat_col': lat_col,
        'lon_col': lon_col
    }
    return metadata

def build_csv_entry(url: str, time_col: str, lat_col: str, lon_col: str) -> PandasCSV:
    """Builds a CSV entry.

    Parameters
    ----------
    url : str
        The URL of the CSV file
    time_col : str
        The name of the column containing the time values
    lat_col : str
        The name of the column containing the latitude values
    lon_col : str
        The name of the column containing the longitude values
    
    Returns
    -------
    PandasCSV
        The CSV catalog entry
    """
    metadata = extract_csv_metadata(url, time_col, lat_col, lon_col)
    data = CSV(url=url)
    entry = PandasCSV(data, metadata=metadata)
    return entry

def add_entry_to_catalog(catalog: Catalog, entry: Union[ERDDAPCatalogReader, PandasCSV], entry_name: str) -> Catalog:
    """Adds an entry to an intake catalog.

    Parameters
    ----------

    catalog : Catalog
        The intake catalog to which the entry will be added
    entry : Union[ERDDAPCatalogReader, PandasCSV]
        The entry to be added to the catalog
    entry_name : str
        The name of the entry in the catalog

    Returns
    -------
    Catalog
        The catalog with the entry added
    """
    catalog[entry_name] = entry
    catalog.give_name(entry_name, entry_name)
    return catalog

def build_catalog(base_catalog_path: str, full_catalog_path: str) -> None:
    """Builds an intake catalog from a base catalog file.

    Parameters
    ----------
    base_catalog_path : str
        The path to the base catalog file
    full_catalog_path : str
        The path to the full catalog file
    """
    with open(base_catalog_path, 'r') as f:
        datasets = yaml.safe_load(f)
    catalog = Catalog()
    for dataset in datasets:
        if dataset['type'] == 'csv':
            entry_name = Path(dataset['url']).stem
            entry = build_csv_entry(dataset['url'], dataset['time_col'], dataset['lat_col'], dataset['lon_col'])
        elif dataset['type'] == 'erddap':
            entry_name = dataset['dataset_id']
            entry = build_erddap_dataset_entry(dataset['server_url'], dataset['dataset_id'])
        catalog = add_entry_to_catalog(catalog, entry, entry_name)
    catalog.to_yaml_file(full_catalog_path)

def get_all_catalog_variables(catalog: Catalog) -> set:
    """Gets all variables from the catalog.

    Parameters
    ----------
    catalog : Catalog
        The catalog to search.
    
    Returns
    -------
    set
        A set of all variables in the catalog.
    """
    all_variables = set()
    for entry_name in catalog.entries.keys():
        all_variables.update(set(catalog[entry_name].metadata['variables']))
    return all_variables

def search_catalog_variables(catalog, query=None, case_insensitive=True, fuzzy=False) -> List[str]:
    """Searches the catalog for variables.

    Parameters
    ----------
    catalog : Catalog
        The catalog to search.
    query : str, optional
        The query to search for. If not provided, all variables are returned.
    case_insensitive : bool, optional
        Whether to perform a case-insensitive search. Default is True.
    fuzzy : bool, optional
        Whether to perform a fuzzy search. Default is False.
    
    Returns
    -------
    list
        A list of matching variables.
    """
    all_variables = get_all_catalog_variables(catalog)

    if not query:
        return sorted(all_variables, key=str.lower)
    
    if fuzzy:
        processor = None
        if case_insensitive:
            query = query.lower()
            processor = lambda x: x.lower()
        results = process.extract(query, all_variables, scorer=fuzz.WRatio, score_cutoff=70, processor=processor)
        matching_variables = [variable for variable, _, _ in results]
    else:
        if case_insensitive:
            matching_variables = [v for v in all_variables if query.lower() in v.lower()]
        else:
            matching_variables = [v for v in all_variables if query in v]

    return sorted(matching_variables, key=str.lower)

def to_dt(s: str) -> datetime:
    """Converts a string to a datetime object.

    Parameters
    ----------
    s : str
        The string to convert.
    
    Returns
    -------
    datetime
        The datetime object.
    """
    return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ")

def get_full_time_range(catalog):
    min_time, max_time = None, None
    for entry_name in catalog.entries.keys():
        entry = catalog[entry_name]
        entry_min_time = to_dt(entry.metadata['minTime'])
        if not min_time or entry_min_time < min_time:
            min_time = entry_min_time
        entry_max_time = to_dt(entry.metadata['maxTime'])
        if not max_time or entry_max_time > max_time:
            max_time = entry_max_time
    return min_time, max_time

def filter_catalog(
        catalog: Catalog,
        entry_names: List[str] = None,
        variables: List[str] = None,
        time_range: Tuple[str, str] = None,
        bbox: Tuple[float, float, float, float] = None
    ) -> Catalog:
    """Filters a catalog based on the provided criteria.

    Parameters
    ----------
    catalog : Catalog
        The catalog to filter.
    entries : list, optional
        The entries to be used for filtering.
    variables : list, optional
        The variables to be used for filtering.
    time_range : list, optional
        The time range to be used for filtering, formatted as [min_time, max_time].
    bbox : list, optional
        The bounding box to be used for filtering, formatted as [min_lon, min_lat, max_lon, max_lat].

    Returns
    -------
    Catalog
        The filtered catalog.
    """
    filtered_catalog = Catalog()
    filtered_catalog.metadata['time_range'] = time_range

    if not entry_names:
        entry_names = catalog.entries.keys()
    
    entry_names = set(entry_names)

    if variables and entry_names:
        matching_entry_names = set()
        matching_variables = set()
        for entry_name in entry_names:
            entry_variables = catalog[entry_name].metadata['variables']
            common_variables = set(variables).intersection(set(entry_variables))
            if common_variables:
                matching_entry_names.add(entry_name)
                matching_variables.update(common_variables)
        entry_names.intersection_update(matching_entry_names)
    else:
        matching_variables = set(get_all_catalog_variables(catalog))
    
    filtered_catalog.metadata['variables'] = list(matching_variables)

    if time_range and entry_names:
        matching_entry_names = set()
        filter_min_time, filter_max_time = tuple(to_dt(t) for t in time_range)
        for entry_name in entry_names:
            entry_min_time = to_dt(catalog[entry_name].metadata['minTime'])
            entry_max_time = to_dt(catalog[entry_name].metadata['maxTime'])
            if filter_min_time <= entry_max_time and entry_min_time <= filter_max_time:
                matching_entry_names.add(entry_name)
        entry_names.intersection_update(matching_entry_names)

    if bbox and entry_names:
        matching_entry_names = set()
        filter_bbox = box(*bbox)
        for entry_name in entry_names:
            entry_bbox = box(
                catalog[entry_name].metadata['minLongitude'],
                catalog[entry_name].metadata['minLatitude'],
                catalog[entry_name].metadata['maxLongitude'],
                catalog[entry_name].metadata['maxLatitude']
            )
            if filter_bbox.intersects(entry_bbox):
                matching_entry_names.add(entry_name)
        entry_names.intersection_update(matching_entry_names)

    for entry_name in entry_names:
        add_entry_to_catalog(filtered_catalog, catalog[entry_name], entry_name)
    
    return filtered_catalog

def build_entries_selector(catalog):
    sel  = w.SelectMultiple(options=list(catalog.entries.keys()))
    apply = w.Button(description="Apply")
    out  = w.Output()
    box  = w.VBox([sel, apply, out])

    def _apply(_):
        entry_names = list(sel.value)
        box.result = filter_catalog(catalog, entry_names=entry_names)
        sel.close(); apply.close()
        with out:
            clear_output()
            print(f"Entries selected: {', '.join(entry_names)}")

    apply.on_click(_apply)
    display(box)
    return box

def build_variables_selector(catalog):
    all_variables = sorted(get_all_catalog_variables(catalog), key=str.lower)
    sel  = w.SelectMultiple(options=list(all_variables))
    apply = w.Button(description="Apply")
    out  = w.Output()
    box  = w.VBox([sel, apply, out])

    def _apply(_):
        variables = list(sel.value)
        box.result = filter_catalog(catalog, variables=variables)
        sel.close(); apply.close()
        with out:
            clear_output()
            print(f"Variables selected: {', '.join(variables)}")

    apply.on_click(_apply)
    display(box)
    return box

DAY_MS = 24 * 60 * 60 * 1000

import ipywidgets as w
from IPython.display import display, clear_output
from datetime import datetime, timedelta

def build_time_range_selector(catalog):
    start_dt, end_dt = get_full_time_range(catalog)

    # one entry per calendar day
    n_days  = (end_dt.date() - start_dt.date()).days
    dates   = [start_dt + timedelta(days=i) for i in range(n_days + 1)]

    slider = w.SelectionRangeSlider(
        options       = dates,                    # discrete stops
        index         = (0, len(dates) - 1),      # full span pre‑selected
        description   = "Date range",
        continuous_update = False,                # reduce noise
        layout        = {'width': '500px'}
    )

    apply  = w.Button(description="Apply")
    out    = w.Output()
    box    = w.VBox([slider, apply, out])         # mirror variable selector

    def _apply(_):
        start, end = slider.value
        time_range = (start.isoformat() + "Z", end.isoformat() + "Z")
        box.result = filter_catalog(catalog, time_range=time_range)
        slider.close(); apply.close()
        with out:
            clear_output()
            print(f"Time range selected: {start:%Y‑%m‑%d} to {end:%Y‑%m‑%d}")

    apply.on_click(_apply)
    display(box)
    return box


def build_agg_table( catalog: Catalog) -> pd.DataFrame:

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
        agg_table[var_cols] = (agg_table[var_cols] > 0).astype(int)
        agg_table['event_n'] = 1

    # Divide all non-base columns by the value in the event_n column and then drop the event_n column
    for col in agg_table.columns:
        if col not in base_cols and agg_table[col].dtype in ['int64', 'float64']:
            agg_table[col] = agg_table[col] / agg_table['event_n']
    agg_table.drop(columns=['event_n'], inplace=True)

    agg_table = agg_table.groupby(['station', 'geometry']).mean(numeric_only=True).reset_index()
    agg_table['completeness'] = agg_table.mean(numeric_only=True, axis=1)
    agg_table = gpd.GeoDataFrame(agg_table)
    agg_table = agg_table.set_crs(epsg=4326)
    agg_table = agg_table[['station', 'geometry', 'completeness']]

    plot = agg_table.hvplot.points(
        c='completeness',
        cmap='viridis',
        logz=log,
        geo=True,
        tiles='OSM',
        hover_cols=['station', 'completeness'],
        width=800,
        height=600
    )
    from IPython.display import display
    display(plot)

def plot_grid(
    catalog: Catalog,
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
        catalog.agg_table = build_agg_table(catalog, variables)
    
    agg_table = deepcopy(catalog.agg_table)
    agg_table.drop(columns=['geometry'], inplace=True)
    
    if stations:
        if isinstance(stations, str):
            stations = [stations]
        agg_table = agg_table[agg_table['station'].isin(stations)]
    

    if stations:
        if isinstance(stations, str):
            stations = [stations]
        agg_table = agg_table[agg_table['station'].isin(stations)]
    

    if stations:
        if isinstance(stations, str):
            stations = [stations]
        agg_table = agg_table[agg_table['station'].isin(stations)]
    
    if time_bin != "D":
        agg_table = agg_table.set_index('date')
        agg_table = agg_table.groupby('station').resample(time_bin).sum(numeric_only=True).fillna(0).reset_index()
        var_cols = agg_table.columns.difference(base_cols)
        agg_table[var_cols] = (agg_table[var_cols] > 0).astype(int)
        agg_table['event_n'] = 1

    # Divide all non-base columns by the value in the event_n column and then drop the event_n column
    for col in agg_table.columns:
        if col not in base_cols and agg_table[col].dtype in ['int64', 'float64']:
            agg_table[col] = agg_table[col] / agg_table['event_n']
    agg_table.drop(columns=['event_n'], inplace=True)

    agg_table = agg_table.groupby('station').mean(numeric_only=True).reset_index()
    agg_table = agg_table.set_index('station')
    import matplotlib.pyplot as plt
    # plt.figure(figsize=(10, 60))
    plt.yticks(ticks=range(agg_table.shape[0]), labels=agg_table.index, rotation=0)
    sns.heatmap(agg_table, cmap='viridis', norm=norm)
    plt.show()

def plot_timeseries(
    catalog: Catalog,
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
        catalog.agg_table = build_agg_table(catalog, variables)
    
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
        agg_table[var_cols] = (agg_table[var_cols] > 0).astype(int)
        agg_table['event_n'] = 1

    # Divide all non-base columns by the value in the event_n column and then drop the event_n column
    for col in agg_table.columns:
        if col not in base_cols and agg_table[col].dtype in ['int64', 'float64']:
            agg_table[col] = agg_table[col] / agg_table['event_n']
    agg_table.drop(columns=['event_n'], inplace=True)

    if len(stations) > 1:
        agg_table['completeness'] = agg_table.mean(numeric_only=True, axis=1)
        agg_table['completeness'] /= len(stations)
        agg_table = agg_table[['date', 'station', 'completeness']].pivot(index='date', columns='station')
        agg_table.columns = agg_table.columns.droplevel(0)
        agg_table.columns.name = None

    else:
        agg_table = agg_table.drop(columns=['station'])
        agg_table = agg_table.set_index('date')
        for col in agg_table.columns:
            agg_table[col] = agg_table[col] / len(agg_table.columns)

    import matplotlib.pyplot as plt
    ax = agg_table.plot(kind='bar', stacked=True, logy=log, width=1.0)
    xticks = range(0, len(agg_table), max(1, len(agg_table)//10))  # adjust tick frequency
    ax.set_xticks(xticks)
    ax.set_xticklabels(agg_table.index.strftime('%Y-%m')[xticks], rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Manage the FRESCA catalog.')
    subparsers = parser.add_subparsers(dest='command', help='Subcommands')

    build_parser = subparsers.add_parser('build', help='Build the catalog from a base catalog file')
    build_parser.add_argument('input_file', type=str, help='The input base catalog, a YAML file')
    build_parser.add_argument('output_file', type=str, help='The output full catalog, a YAML file')

    args = parser.parse_args()

    if args.command == 'build':
        build_catalog(args.input_file, args.output_file)
    else:
        parser.print_help()
