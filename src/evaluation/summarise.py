import os

from scipy.stats import wasserstein_distance_nd
import numpy as np

from . import inference
from ..data import read_write
from ..detector_map import find_layers, get_layer_centers


def emd(reference, predicted):
    """
    Earth movers distance between reference and predicted datasets.
    """
    n_events = reference.shape[0]
    distances = np.empty(n_events)
    for event_n in range(n_events):
        u_values = reference[event_n][:, :3]
        v_values = predicted[event_n][:, :3]
        u_weights = reference[event_n][:, 3]
        v_weights = predicted[event_n][:, 3]
        distances[event_n] = wasserstein_distance_nd(
            u_values, v_values, u_weights, v_weights
        )
    return distances


def pca(cells, energy_fraction=1.0):
    pass


def event_energy(cells):
    pass


def cell_energies(cells):
    pass


def radial_energy(cells, directions):
    pass


def layer_energies(cells):
    pass


def event_occupancies(cells):
    pass


def radial_occupancies(cells, directions):
    pass


def layer_occupancies(cells):
    pass


def target_to_physical(points, config):
    physical_points = np.zeros_like(points)
    point_layer_ids = find_layers(config, points, coordinates="data")
    layer_centers = get_layer_centers(config, coordinates="detector")
    mask = (points[:, :, 3] > 0) & (point_layer_ids >= 0)

    physical_points[mask, 2] = layer_centers[point_layer_ids[mask]]

    data_low_x = config["data"]["Xmin"]
    detector_low_z = config["data"]["Zmin_in_detector"]
    data_x_range = config["data"]["Xmax"] - data_low_x
    detector_z_range = config["data"]["Zmax_in_detector"] - detector_low_z
    shift_0 = detector_low_z - data_low_x
    scale_0 = detector_z_range / data_x_range
    physical_points[mask, 2] = (points[mask, 0] + shift_0) * scale_0

    data_low_y = config["data"]["Ymin"]
    detector_low_x = config["data"]["Xmin_in_detector"]
    data_y_range = config["data"]["Ymax"] - data_low_y
    detector_x_range = config["data"]["Xmax_in_detector"] - detector_low_x
    shift_1 = detector_low_x - data_low_y
    scale_1 = detector_x_range / data_y_range
    physical_points[mask, 0] = (points[mask, 2] + shift_1) * scale_1

    return physical_points, point_layer_ids


class Summary:
    def __init__(self, config, data_part="test", pick_events=None, total_size=10_000):
        self.config = config
        self._sampler = inference.Sampler.from_config(config)
        self.data_part = data_part
        self.pick_events = pick_events
        self.total_size = total_size
        reference_path = self.precalculated_reference_path()
        if os.path.exists(reference_path):
            self.reference = np.load(reference_path)
        else:
            self.reference = self.calculate_reference()
            np.savez(reference_path, **self.reference)

    def precalculated_reference_path(self):
        dataset_name = os.path.basename(self.config["data"]["dataset_path"])
        dataset_name = dataset_name.split(".")[0].split("{")[0]
        out_dir = self.config["output"]["path"]
        precalc_dir = os.path.join(out_dir, "precalculated_reference", dataset_name)
        os.makedirs(precalc_dir, exist_ok=True)
        file_name = f"pre{self.data_part}_Total{int(self.total_size)}"
        if self.pick_events is not None:
            picky = f"_Pick{self.pick_events}"
            file_name += "".join(p for p in picky if p.isalnum())
        else:
            file_name += "_NoPick"
        file_name += ".npz"
        path = os.path.join(precalc_dir, file_name)
        return path

    def fetch_reference(self):
        cond_columns = [
            self.config["data"][f"{name}_key"]
            for name in self.config["model"]["cond_features"]
        ]
        cond, target = read_write.read_raw_regaxes(
            self.config,
            part=self.data_part,
            pick_events=self.pick_events,
            total_size=self.total_size,
            per_event_cols=cond_columns,
        )
        physical_points, point_layer_ids = target_to_physical(target, self.config)
        physical_points = inference.unshift_points(
            physical_points, point_layer_ids, cond, self.config
        )
        cells = inference.physical_to_cells(
            physical_points, point_layer_ids, self.config
        )
        return cond, cells

    def calculate_reference(self):
        cond, cells = self.fetch_reference()
        return self.calculate_singulars(cond, cells)

    def calculate_singulars(self, cond, cells):
        singulars = {}
        singulars["cond"] = cond
        singulars["cells"] = cells
        singulars["pca"] = pca(cells)
        singulars["event_energy"] = event_energy(cells)
        singulars["cell_energies"] = cell_energies(cells)
        singulars["radial_energy"] = radial_energy(cells)
        singulars["layer_energies"] = layer_energies(cells)
        singulars["event_occupancies"] = event_occupancies(cells)
        singulars["radial_occupancies"] = radial_occupancies(cells)
        singulars["layer_occupancies"] = layer_occupancies(cells)
        return singulars, cells
