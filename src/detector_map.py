import numpy as np
import os


def load_muon_map(assets_dir=None):
    if assets_dir is None:
        this_dir = os.path.dirname(__file__)
        assets_dir = os.path.join(this_dir, "../assets/")

    data_dir = os.path.join(assets_dir, "muon_map")
    muon_map_X = np.load(data_dir + "/X.npy")
    muon_map_Y = np.load(data_dir + "/Y.npy")
    muon_map_Z = np.load(data_dir + "/Z.npy")
    muon_map_E = np.load(data_dir + "/E.npy")

    return muon_map_X, muon_map_Y, muon_map_Z, muon_map_E


def confine_to_box(config, X, Y, Z, E, detector_coords=True):
    """
    Remove hits that fall outside the box specified by the metadata.

    """
    if detector_coords:
        Xmin = config["data"]["Xmin_in_detector"]
        Xmax = config["data"]["Xmax_in_detector"]
        Ymin = config["detector"]["layer_bottom_pos"][0]
        Ymax = (
            config["detector"]["layer_bottom_pos"][-1]
            + config["detector"]["cell_thickness_hcal"]
        )
        Zmin = config["data"]["Zmin_in_detector"]
        Zmax = config["data"]["Zmax_in_detector"]
    else:
        Xmin = config["data"]["Xmin"]
        Xmax = config["data"]["Xmax"]
        Ymin = config["data"]["Ymin"]
        Ymax = config["data"]["Ymax"]
        Zmin = config["data"]["layer_bottom_pos"][0]
        Zmax = (
            config["data"]["layer_bottom_pos"][-1] + config["data"]["cell_thickness"]
        )

    inbox_idx = np.where(
        (X > Xmin) & (X < Xmax) & (Y > Ymin) & (Y < Ymax) & (Z > Zmin) & (Z < Zmax)
    )[0]

    X = X[inbox_idx]
    Z = Z[inbox_idx]
    Y = Y[inbox_idx]
    E = E[inbox_idx]
    return X, Y, Z, E


def _cell_edges(center, half_cell_size, offset, divisions):
    """
    Bin edges spanning one cell, split into ``divisions`` sub-cells.

    Parameters
    ----------
    center : float
        Position of the cell center.
    half_cell_size : float
        Half the width of a cell.
    offset : float
        Width of one sub-cell, i.e. ``cell_size / divisions``.
    divisions : int
        Number of sub-cells to split the cell into.

    Returns
    -------
    edges : list of float
        ``divisions + 1`` edges, from the bottom of the cell to the top.
    """
    bottom = center - half_cell_size
    return [bottom + offset * n for n in range(divisions)] + [center + half_cell_size]


def _drop_near_duplicate_edges(edges, tolerance=1e-3):
    """
    Remove edges that would create a bin narrower than ``tolerance``.

    Keeps the last edge unconditionally, so the outer extent is preserved.
    """
    kept = [
        edges[i] for i in range(len(edges) - 1)
        if abs(edges[i] - edges[i + 1]) > tolerance
    ]
    return kept + [edges[-1]]


def _cell_centers_in_row(unique_positions, half_cell_size):
    """
    Reduce measured positions along one axis to one center per cell.

    A position within about a cell of the previous one is the same cell seen
    again, so only the first of such a run is kept.
    """
    centers = [unique_positions[0]]
    for i in range(len(unique_positions) - 1):
        if abs(unique_positions[i] - unique_positions[i + 1]) > half_cell_size * 1.9:
            centers.append(unique_positions[i + 1])
    return centers


def detector_cell_thickness(config):
    layer_bottom_pos = config["detector"]["layer_bottom_pos"]
    cell_thickness_ecal = config["detector"]["cell_thickness_ecal"]
    cell_thickness_hcal = config["detector"]["cell_thickness_hcal"]
    hcal_start = config["detector"]["hcal_start"]

    cell_thickness = np.full(len(layer_bottom_pos), cell_thickness_ecal)
    cell_thickness[hcal_start:] = cell_thickness_hcal

    return cell_thickness


def create_map(config, confine=False):
    X, Y, Z, E = confine_to_box(config, *load_muon_map(), detector_coords=True)

    layer_bottom_pos = np.array(config["detector"]["layer_bottom_pos"])
    half_cell_size_global = config["detector"]["cell_size"] / 2

    cell_thickness = detector_cell_thickness(config)

    dm = config["data"]["divisions_per_cell"]
    offset = config["detector"]["cell_size"] / dm

    # Use the same bands as find_layers, so that a muon hit builds the cell
    # geometry of the layer that find_layers will later assign points to.
    # These are clamped, so layers cannot claim each other's hits.
    layer_floors, layer_ceilings = floors_ceilings(
        layer_bottom_pos,
        cell_thickness,
        percent_buffer=0.5,
    )

    layers = []
    for layer_n in range(len(layer_bottom_pos)):  # loop over layers
        # half open, to match find_layers
        idx = np.where(
            (Y >= layer_floors[layer_n]) & (Y < layer_ceilings[layer_n])
        )

        unique_X = np.unique(X[idx])
        unique_Z = np.unique(Z[idx])

        cell_centers_x = _cell_centers_in_row(unique_X, half_cell_size_global)

        # every cell is divided the same way, on both axes
        xedges = np.unique(np.concatenate([
            _cell_edges(center, half_cell_size_global, offset, dm)
            for center in cell_centers_x
        ]))
        zedges = np.unique(np.concatenate([
            _cell_edges(center, half_cell_size_global, offset, dm)
            for center in unique_Z
        ]))

        xedges = _drop_near_duplicate_edges(xedges)
        zedges = _drop_near_duplicate_edges(zedges)

        H, xedges, zedges = np.histogram2d(X[idx], Z[idx], bins=(xedges, zedges))
        layers.append({"xedges": xedges, "zedges": zedges, "grid": H})

    return layers, offset


def get_layer_centers(config, coordinates="data"):
    if coordinates == "data":
        layer_bottom_pos = config["data"]["layer_bottom_pos"]
        cell_thickness = config["data"]["cell_thickness"]
    elif coordinates == "detector":
        layer_bottom_pos = config["detector"]["layer_bottom_pos"]
        cell_thickness = detector_cell_thickness(config)

    layer_bottom_pos = np.array(layer_bottom_pos)
    layer_centers = layer_bottom_pos + cell_thickness / 2

    return layer_centers


def floors_ceilings(layer_bottom_pos, cell_thickness, percent_buffer=0.5):
    """
    Find top and bottom coordinates for the layers in the detector.

    Parameters
    ----------
    layer_bottom_pos : np.array
        Array of the bottom positions of the layers.
    cell_thickness : float or np.array
        Thickness of the cells in the detector, in the radial direction
    percent_buffer : float
        Percentage beyond the thickness of the cell to include hits
        in the layer. Won't extent the layer beyond the bottom of
        the next layer.
        (default is 0.5)

    Returns
    -------
    layer_floors : np.array
        Array of the bottom positions of the layers.
    layer_ceilings : np.array
        Array of the top positions of the layers.
    """
    # naive calculation of the layer floors and ceilings
    layer_floors = layer_bottom_pos - percent_buffer * cell_thickness
    layer_ceilings = layer_bottom_pos + (1 + percent_buffer) * cell_thickness
    # Unless the cells are thicker than the layers, (which they shouldn't be)
    # the true ceiling for each layer is the bottom of the layer plus the thickness
    true_ceilings = np.minimum(
        (layer_bottom_pos + cell_thickness)[:-1], layer_bottom_pos[1:]
    )
    # we dont' want any extention to cross the midpoint between the true
    # ceiling and the bottom of the next layer
    mid_points = 0.5 * (true_ceilings + layer_bottom_pos[1:])
    # now enforce not crossing those midpoints
    layer_floors[1:] = np.maximum(layer_floors[1:], mid_points)
    layer_ceilings[:-1] = np.minimum(layer_ceilings[:-1], mid_points)
    return layer_floors, layer_ceilings


def find_layers(config, points, coordinates="data"):
    if coordinates == "data":
        layer_bottom_pos = config["data"]["layer_bottom_pos"]
        cell_thickness = config["data"]["cell_thickness"]
        height = points[:, :, 2]
    elif coordinates == "detector":
        layer_bottom_pos = np.array(config["detector"]["layer_bottom_pos"])
        cell_thickness = detector_cell_thickness(config)

        height = points[:, :, 1]

    layer_bottom_pos = np.array(layer_bottom_pos)

    layer_floors, layer_ceilings = floors_ceilings(
        layer_bottom_pos,
        cell_thickness,
        percent_buffer=0.5,
    )
    real_points = points[:, :, 3] > 0
    point_layers = -np.ones(points.shape[:2], dtype=int)

    for i, (floor, ceiling) in enumerate(zip(layer_floors, layer_ceilings)):
        mask = (height >= floor) & (height < ceiling) & real_points
        point_layers[mask] = i

    return point_layers
