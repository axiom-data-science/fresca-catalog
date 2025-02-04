# fresca-catalog

An Intake-based catalog of FRESCA datasets and supporting tools

The purpose of `fresca-catalog` is to provide an interface to the FRESCA dataset inventory that facilitates exploration and discovery. To do this, `fresca-catalog` makes use of the open-source package [Intake](https://intake.readthedocs.io/en/latest/index.html) to represent the FRESCA datasets as a unified, machine-readable catalog. `fresca-catalog` also defines custom functionality for querying and filtering the catalog, as well as for curating the catalog as it grows.

## Installation

In order to use `fresca-catalog`, you'll need to install either [Anaconda or Miniconda](https://docs.anaconda.com/getting-started/) to handle environment management.

Once one of those are installed, the next step is to clone this repository:

```sh
git clone git@github.com:axiom-data-science/fresca-catalog.git
```

Now navigate into the resulting `fresca-catalog` directory, and create the Conda environment for this project with:

```sh
conda env update
```

This could take a while. After it finishes running (hopefully successfully) you can activate the `fresca` environment with:

```sh
conda activate fresca
```

Now all of `fresca-catalog`'s dependencies are installed and accessible, and you can get started running code in JupyterLab by running:

```sh
jupyter lab
```

And if you prefer another IDE to JupyterLab, like Visual Studio Code, then by all means use that instead.

## Usage

`fresca-catalog` addresses two primary use cases:

1. Loading and interactively exploring the FRESCA Intake catalog via a Jupyter notebook
2. Rebuilding the FRESCA Intake catalog in order to add new dataset entries or update existing entries

`fresca-catalog` contains three key components to enable these use cases:

- `full_catalog.yml`: The Intake catalog itself, in its on-disk YAML representation
- `fresca_catalog.py`: The logic required for building and interacting with that Intake catalog
- `base_catalog.yml`: The minimal "seed" catalog required to programmatically build the "enriched" `full_catalog.yml`

We'll explore these two use cases and the components listed above in the following sections.

### Exploring the catalog

One benefit of Intake is that its catalogs can easily be serialized to disk as YAML files. This makes them portable and easy to version. In `fresca-catalog` we store the full FRESCA Intake catalog as `full_catalog.yml`. Below, we'll cover the basics of querying and filtering the full catalog, but for details and to follow along, please open `example.ipynb`.

Loading the FRESCA Intake catalog is as simple as:

```python
import intake
catalog = intake.open_catalog('full_catalog.yml')
```

`catalog` is just a plain vanilla Intake catalog and as such, supports anything described in the [official Intake documentation](https://intake.readthedocs.io/en/latest/index.html). But in order to make use of `catalog` for querying and filtering, we've defined a couple helper functions in `fresca_catalog.py`. These include:

- `search_catalog_variables`
- `filter_catalog`

Details on the usage of these functions can be found in the docstrings in `fresca_catalog.py`, or by calling `help()` on either of them, for example `help(search_catalog_variables)`:

```
search_catalog_variables(
    catalog,
    query=None,
    case_insensitive=True,
    fuzzy=False
) -> List[str]
    Searches the catalog for variables.

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
```

The purpose of this pair of functions is to assist in narrowing down the search to a set of datasets that will comprise a sub-catalog which can then be used for more serious scientific analysis. `search_catalog_variables` is intended to help you find all relevant variable names across the catalog. `filter_catalog` can then be used to create a sub-catalog by passing in a select list of variables, as well as temporal and spatial bounds.

Say this searching and filtering results in a new catalog called `my_catalog`. Rather than needing to rebuild it everytime you want to explore it, you can simply save it to disk as a YAML file in the same way we do below with `full_catalog.yml`. For example:

```python
my_catalog.to_yaml_file('my_catalog.yml')
```

For more details on how to inpspect specific datset entries and read their linked data, please see `example.ipynb`

### Rebuilding the catalog

Another benefit of Intake is the ease with which you can programmatically build a new catalog from minimal input. In cases where we have a new dataset entry to add to the catalog, or we need to update an existing dataset entry, we'll want to take advantage of this functionality.

The key ingredient driving this programmatic generation is `base_catalog.yml`. For example, a `base_catalog.yml` with two entries could look like:

```yaml
- dataset_id: CRCP_Carbonate_Chemistry_Atlantic
  server_url: https://www.ncei.noaa.gov/erddap
  type: erddap
- lat_col: lat_dec
  lon_col: lon_dec
  time_col: datetime
  type: csv
  url: https://files.axds.co/tmp/SFER_data.csv
```

`base_catalog.yml` currently support two different types of entries: `csv` types and `erddap` types. Because ERDDAP servers provide metadata in addition to the data itself, the `erddap` type entry is very minimal, requiring just the `server_url`, the `dataset_id`, and the `type`. CSVs on the other hand embed less structured metadata, and thus require a more verbose entry, specifically the addition of `lat_col`, `lon_col_`, and `time_col` to support automated metadata extraction for those columns.

From this minimal `base_catalog.yml`, regenerating `full_catalog.yml` is as simple as running:

```sh
conda activate fresca
python fresca_catalog.py build base_catalog.yml full_catalog.yml
```

If you're more comfortable in a Jupyter notebook, the same thing can be accomplished by running a cell with the following:

```python
from fresca_catalog import build_catalog
build_catalog('base_catalog.yml', 'full_catalog.yml')
```

This should result in a modified `full_catalog.yml` being written to disk for you to explore. If the new `full_catalog.yml` contains new or updated dataset entries and represents the latest and greatest canonical version of the catalog that everyone should be using, then it should also be committed to the repository and pushed so others can access it.