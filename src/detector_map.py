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


def confine_to_box(configs, X, Y, Z, E, detector_coords=True):
    """
    Remove hits that fall outside the box specified by the metadata.

    """
    if detector_coords:
        Xmin = configs["data"]["Xmin_in_detector"]
        Xmax = configs["data"]["Xmax_in_detector"]
        Ymin = configs["detector"]["layer_bottom_pos"][0]
        Ymax = (
            configs["detector"]["layer_bottom_pos"][-1]
            + configs["data"]["cell_thickness"]
        )
        Zmin = configs["data"]["Zmin_in_detector"]
        Zmax = configs["data"]["Zmax_in_detector"]
    else:
        Xmin = configs["data"]["Xmin"]
        Xmax = configs["data"]["Xmax"]
        Ymin = configs["data"]["Ymin"]
        Ymax = configs["data"]["Ymax"]
        Zmin = configs["data"]["layer_bottom_pos"][0]
        Zmax = (
            configs["data"]["layer_bottom_pos"][-1] + configs["data"]["cell_thickness"]
        )

    inbox_idx = np.where(
        (X > Xmin) & (X < Xmax) & (Y > Ymin) & (Y < Ymax) & (Z > Zmin) & (Z < Zmax)
    )[0]

    X = X[inbox_idx]
    Z = Z[inbox_idx]
    Y = Y[inbox_idx]
    E = E[inbox_idx]
    return X, Y, Z, E


def create_map(configs):
    X, Y, Z, E = confine_to_box(configs, *load_muon_map(), detector_coords=True)

    layer_bottom_pos = configs["detector"]["layer_bottom_pos"]
    half_cell_size_global = configs["detector"]["cell_size"] / 2
    cell_thickness_global = configs["detector"]["cell_thickness"]

    dm = configs["data"]["divisions_per_cell"]
    offset = configs["detector"]["cell_size"] / dm

    layers = []
    for layer_n in range(len(layer_bottom_pos)):  # loop over layers
        # layers are well seperated, so take a 0.5 buffer either side
        idx = np.where(
            (Y <= (layer_bottom_pos[layer_n] + cell_thickness_global * 1.5))
            & (Y >= layer_bottom_pos[layer_n] - cell_thickness_global / 2)
        )

        xedges = np.array([])
        zedges = np.array([])

        unique_X = np.unique(X[idx])
        unique_Z = np.unique(Z[idx])

        xedges = np.append(xedges, unique_X[0] - half_cell_size_global)
        xedges = np.append(xedges, unique_X[0] + half_cell_size_global)

        for i in range(len(unique_X) - 1):  # loop over X coordinate cell centers
            if abs(unique_X[i] - unique_X[i + 1]) > half_cell_size_global * 1.9:
                xedges = np.append(xedges, unique_X[i + 1] - half_cell_size_global)
                xedges = np.append(xedges, unique_X[i + 1] + half_cell_size_global)

                for of_m in range(dm):
                    xedges = np.append(
                        xedges, unique_X[i + 1] - half_cell_size_global + offset * of_m
                    )  # for higher granularity

        for z in unique_Z:  # loop over Z coordinate cell centers
            zedges = np.append(zedges, z - half_cell_size_global)
            zedges = np.append(zedges, z + half_cell_size_global)

            for of_m in range(dm):
                zedges = np.append(
                    zedges, z - half_cell_size_global + offset * of_m
                )  # for higher granularity

        zedges = np.unique(zedges)
        xedges = np.unique(xedges)

        xedges = [
            xedges[i]
            for i in range(len(xedges) - 1)
            if abs(xedges[i] - xedges[i + 1]) > 1e-3
        ] + [xedges[-1]]
        zedges = [
            zedges[i]
            for i in range(len(zedges) - 1)
            if abs(zedges[i] - zedges[i + 1]) > 1e-3
        ] + [zedges[-1]]

        H, xedges, zedges = np.histogram2d(X[idx], Z[idx], bins=(xedges, zedges))
        layers.append({"xedges": xedges, "zedges": zedges, "grid": H})

    return layers, offset


def floors_ceilings(layer_bottom_pos, cell_thickness, percent_buffer=0.5):
    """
    Find top and bottom coordinates for the layers in the detector.

    Parameters
    ----------
    layer_bottom_pos : np.array
        Array of the bottom positions of the layers.
    cell_thickness : float
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
        layer_bottom_pos = config["detector"]["layer_bottom_pos"]
        cell_thickness = config["detector"]["cell_thickness"]
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
