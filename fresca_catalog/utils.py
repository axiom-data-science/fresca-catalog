"""Utility functions for working with FRESCA catalogs."""
from copy import deepcopy
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from .catalog import Catalog, get_all_catalog_variables

def build_agg_table(catalog: Catalog) -> pd.DataFrame:
    """Builds an aggregated table from all entries in the catalog.

    Parameters
    ----------
    catalog : Catalog
        The catalog containing the datasets to aggregate.

    Returns
    -------
    pd.DataFrame
        The aggregated table.
    """
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

        if 'variables' not in catalog.metadata:
            catalog.metadata['variables'] = get_all_catalog_variables(catalog)

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
    import numpy as np
    var_cols = [c for c in var_cols if np.issubdtype(df[c].dtype, np.number)]
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

def merge_and_save(
        counts_url: str,
        stations_url: str,
        counts_key: str,
        stations_key: str,
        out_csv: str
    ) -> None:
    """
    Merge two dataframes (counts and stations) to attach station latitude and longitude from the stations CSV.

    Parameters
    ----------
    counts_url : str
        URL or filepath to the counts CSV.
    stations_url : str
        URL or filepath to the stations CSV containing latitude and longitude.
    counts_key : str
        Column in counts dataframe to join on.
    stations_key : str
        Column in stations dataframe to join on.
    out_csv : str
        Path to save merged CSV.
    """
    counts = pd.read_csv(counts_url)
    stations = pd.read_csv(stations_url)
    merged = counts.merge(stations, left_on=counts_key, right_on=stations_key, how="inner")
    merged = merged.drop(columns=['new_key', stations_key])
    merged["date"] = merged["date"].str.split(" ").str[0]
    merged.to_csv(out_csv, index=False)

def stations_from_bbox(catalog, bbox):
    """Return station names from catalog that fall within bbox.
    
    Parameters
    ----------
    catalog : Catalog
        The catalog with an agg_table GeoDataFrame.
    bbox : list[float]
        Bounding box [xmin, ymin, xmax, ymax].
    
    Returns
    -------
    list[str]
        Station names inside the bounding box.
    """
    if not hasattr(catalog, 'agg_table'):
        catalog.agg_table = build_agg_table(catalog)
    
    agg = catalog.agg_table
    xmin, ymin, xmax, ymax = bbox
    within = agg.cx[xmin:xmax, ymin:ymax]
    return within['station'].unique().tolist()
