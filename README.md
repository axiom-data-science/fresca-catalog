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

### Rebuilding the catalog

