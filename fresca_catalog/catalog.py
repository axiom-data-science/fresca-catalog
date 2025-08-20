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

from pandas import DataFrame

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
