from copy import deepcopy
from datetime import timedelta

from bokeh.models import BoxSelectTool
import geopandas as gpd
import holoviews as hv
from holoviews import streams
import ipywidgets as w
from IPython.display import display, clear_output
import panel as pn

from .catalog import (
    filter_catalog,
    get_full_time_range,
    get_all_catalog_variables,
    Catalog
)
from .plot import build_agg_table

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

def build_time_range_selector(catalog):
    start_dt, end_dt = get_full_time_range(catalog)

    # one entry per calendar day
    n_days  = (end_dt.date() - start_dt.date()).days
    dates   = [start_dt + timedelta(days=i) for i in range(n_days + 1)]

    slider = w.SelectionRangeSlider(
        options       = dates,
        index         = (0, len(dates) - 1),
        description   = "Date range",
        continuous_update = False,
        layout        = {'width': '500px'}
    )

    apply  = w.Button(description="Apply")
    out    = w.Output()
    box    = w.VBox([slider, apply, out])

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

import holoviews as hv
import hvplot.pandas
import panel as pn
from bokeh.models import BoxSelectTool
from copy import deepcopy
from IPython.display import display

def build_bbox_selector(catalog):
    if not hasattr(catalog, 'agg_table'):
        catalog.agg_table = build_agg_table(catalog)

    agg = deepcopy(catalog.agg_table)[['station','geometry']]
    agg = agg.groupby(['station','geometry']).sum().reset_index()
    agg = gpd.GeoDataFrame(agg)
    agg['lon'] = agg.geometry.x
    agg['lat'] = agg.geometry.y

    # Plot points with box_select
    points = agg.hvplot.points(
        x='lon', y='lat', geo=True,
        hover_cols=['station'],
        width=800, height=600,
        tools=['hover','reset','pan','wheel_zoom','box_select'],
        active_tools=['pan']
    )
    tiles = hv.element.tiles.OSM().opts(alpha=0.8, width=800, height=600)
    overlay = tiles * points

    # selection stream
    selection = streams.Selection1D(source=points)

    apply_btn = pn.widgets.Button(name="Apply", button_type="primary")
    status    = pn.pane.Markdown("")
    widgetbox = pn.Column(overlay, apply_btn, status)
    widgetbox.result = None

    def _apply(_):
        inds = selection.index
        if not inds:
            status.object = "⚠️ Use the **Box Select** tool (dashed square) to select stations."
            return
        selected = agg.iloc[inds]
        xmin, ymin, xmax, ymax = selected.total_bounds
        bbox = [xmin, ymin, xmax, ymax]
        widgetbox.result = bbox
        status.object = f"✅ bbox = {bbox}"
        apply_btn.disabled = True

    apply_btn.on_click(_apply)
    display(widgetbox)
    return widgetbox
