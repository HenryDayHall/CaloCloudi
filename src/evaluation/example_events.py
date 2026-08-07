"""Create a grid of plots comparing a reference event and a model-generated event.

The top row shows a randomly drawn event from the reference dataset, viewed
down the x, y, and z axes. The bottom row shows the output of the model for
the same conditioning and number of points per layer.

The user can specify the representation level:

- ``"data"``: raw data coordinates (as stored in the dataset).
- ``"physical"``: coordinates transformed to the detector frame.
- ``"cells"``: coordinates binned into detector cells (energy-weighted
  cell centres).
"""
import torch
import os
from typing import Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from .inference import (
    Sampler,
    points_per_layer_from_target,
    sample_to_physical,
    unshift_points,
    physical_to_cells,
)
from .plotting import center_arrow
from .summarise import target_to_physical
from ..data.read_write import n_events_in_part


def _project_points(points: np.ndarray, view: str) -> tuple:
    """Return (x_plot, y_plot, energies) for a given view axis.

    Parameters
    ----------
    points : np.ndarray
        Array of shape ``(n_points, 4)`` with columns (x, y, z, e).
    view : {"x", "y", "z"}
        Axis to look down.

    Returns
    -------
    x_plot : np.ndarray
        Horizontal coordinate for the scatter plot.
    y_plot : np.ndarray
        Vertical coordinate for the scatter plot.
    energies : np.ndarray
        Energy values (used for colour).
    """
    if view == "x":
        return points[:, 1], points[:, 2], points[:, 3]
    elif view == "y":
        return points[:, 0], points[:, 2], points[:, 3]
    elif view == "z":
        return points[:, 0], points[:, 1], points[:, 3]
    else:
        raise ValueError(f"Unknown view: {view}")


def _plot_event(
    ax: plt.Axes,
    points: np.ndarray,
    view: str,
    title: str,
    cmap: str = "viridis",
) -> None:
    """Scatter plot of one event projected onto a 2D plane.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    points : np.ndarray
        Array of shape ``(n_points, 4)`` with columns (x, y, z, e).
    view : {"x", "y", "z"}
        Axis to look down.
    title : str
        Title for the subplot.
    cmap : str, optional
        Matplotlib colormap name.
    """
    x_plot, y_plot, energies = _project_points(points, view)
    mask = energies > 0
    if mask.sum() == 0:
        ax.text(0.5, 0.5, "No active cells", transform=ax.transAxes, ha="center")
        ax.set_title(title)
        return
    lognorm_colour = mpl.colors.LogNorm(
        vmin=10**(-5), vmax=energies[mask].max()
    )
    sc = ax.scatter(
        x_plot[mask],
        y_plot[mask],
        c=energies[mask],
        norm=lognorm_colour,
        cmap=cmap,
        s=10,
    )
    ax.set_title(title)
    # ax.set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=ax, label="Energy")


def _draw_direction(ax, cond, view, level):
    # cond[0] is clearly energy
    # cond[3] is clearly direction of travel
    # so cond -> (e, x, y, z) in data
    # cond -> (e, z, x, y) in physical
    vector = np.copy(cond[0, 1:])
    if level != "data":
        vector = vector[[1, 2, 0]]

    if view == 'x':
        center_arrow(ax, vector[1], vector[2], frac=0.2, lw=4)
    elif view == 'y':
        center_arrow(ax, vector[0], vector[2], frac=0.2, lw=4)
    elif view == 'z':
        center_arrow(ax, vector[0], vector[1], frac=0.2, lw=4)


def plot_example_events(
    model_path: str,
    event_index: int | None = None,
    level: str = "data",
    data_part: str = "test",
    unshift: bool = True,
    output_path: Optional[str] = None,
) -> None:
    """Create a 3x2 grid comparing a reference event with a model-generated event.

    Parameters
    ----------
    config : dict
        Full configuration dictionary.
    model_path : str
        Path to the model checkpoint.
    event_index : int, optional
        Index of the event to display (default random).
    level : {"data", "physical", "cells"}, optional
        Representation level (default ``"data"``).
    data_part : str, optional
        Dataset partition to use (default ``"test"``).
    total_size : int, optional
        Number of events to load (default 1000).
    output_path : str or None, optional
        If given, save the figure to this path instead of showing it.
    """
    # ------------------------------------------------------------------
    # 1. Build sampler and fetch conditioning + target
    # ------------------------------------------------------------------
    config = Sampler.get_config_from_model_path(model_path)
    config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    n_events = sum(n_events_in_part(config, data_part))
    if event_index is None:
        event_index = int(np.random.randint(n_events))
    else:
        assert (
            0 <= event_index < n_events
        ), f"event_index {event_index} out of range, only {n_events} events"

    pick_events = [event_index]
    sampler = Sampler(config, model_path)
    cond_i, points_i, target_i = sampler.get_cond(
        data_part=data_part,
        pick_events=pick_events,
        return_target=True,
    )

    # ------------------------------------------------------------------
    # 2. Prepare reference event at the requested level
    # ------------------------------------------------------------------
    if level == "data":
        ref_points = target_i[0]  # (n_points, 4)
    elif level == "physical":
        phys_ref, layer_ids_ref = target_to_physical(target_i, config)
        if unshift:
            phys_ref = unshift_points(phys_ref, layer_ids_ref, cond_i, config)
        ref_points = phys_ref[0]  # (n_points, 4)
    elif level == "cells":
        phys_ref, layer_ids_ref = target_to_physical(target_i, config)
        if unshift:
            phys_ref = unshift_points(phys_ref, layer_ids_ref, cond_i, config)
        cells_ref = physical_to_cells(phys_ref, layer_ids_ref, config)
        # cells_ref shape: (1, n_cells, 4) – take first event
        ref_points = cells_ref[0]
    else:
        raise ValueError(f"Unknown level: {level}")

    # ------------------------------------------------------------------
    # 3. Generate model output for the same conditioning
    # ------------------------------------------------------------------
    # Compute points_per_layer from the reference target
    ppl = points_per_layer_from_target(target_i, config)  # (1, n_layers)

    # Sample model
    sample = sampler.sample(cond_i, points_i)  # (1, max_points, 4)

    if level == "data":
        # Model output is already in data coordinates (sample)
        model_points = sample[0]
    elif level == "physical":
        # Convert to physical
        phys_model, layer_ids_model = sample_to_physical(sample, ppl, config)
        if unshift:
            phys_model = unshift_points(phys_model, layer_ids_model, cond_i, config)
        model_points = phys_model[0]
    elif level == "cells":
        # Convert to physical
        phys_model, layer_ids_model = sample_to_physical(sample, ppl, config)
        if unshift:
            phys_model = unshift_points(phys_model, layer_ids_model, cond_i, config)
        cells_model = physical_to_cells(phys_model, layer_ids_model, config)
        model_points = cells_model[0]
    else:
        raise ValueError(f"Unknown level: {level}")

    # ------------------------------------------------------------------
    # 4. Create the 3x2 grid
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    views = ["x", "y", "z"]
    view_labels = ["View down x", "View down y", "View down z"]

    for col, (view, label) in enumerate(zip(views, view_labels)):
        # Top row: reference
        _plot_event(
            axes[0, col],
            ref_points,
            view,
            title=f"Reference – {label}",
        )
        # Bottom row: model
        _plot_event(
            axes[1, col],
            model_points,
            view,
            title=f"Model – {label}",
        )
        _draw_direction(axes[0, col], cond_i, view, level)
        _draw_direction(axes[1, col], cond_i, view, level)

    fig.suptitle(f"Event {event_index} – Level: {level}", fontsize=14)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150)
        print(f"Figure saved to {output_path}")
    else:
        plt.show()
