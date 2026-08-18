import torch
import os
import yaml
import numpy as np
import collections
from functools import lru_cache
from contextlib import contextmanager
from . import model_kind
from ..diffusion import Diffusion
from ..data import transforms, read_write
from ..data.dataset import pdgs_to_onehot
from ..detector_map import create_map, find_layers, get_layer_centers


@contextmanager
def evaluating(net):
    """
    Temporarily switch to evaluation mode.
    Attribution; Christoph Heindl
    https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
    """
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()


def get_pdg_col_number(config):
    cond_features = config["model"]["cond_features"]
    if "incident_pdg" not in cond_features:
        return None
    cond_columns = [config["data"][f"{name}_key"] for name in cond_features]
    col_lengths = read_write.get_per_event_length(config, cond_columns)
    col_lengths = [col_lengths[name] for name in cond_columns]
    pdg_pos = 0
    for name, length in zip(cond_features, col_lengths):
        if name == "incident_pdg":
            break
        pdg_pos += length
    return pdg_pos


def pdg_to_onehot_in_full_cond(config, cond):
    cond_features = config["model"]["cond_features"]
    if "incident_pdg" not in cond_features:
        return cond
    pdg_pos = get_pdg_col_number(config)
    incident_pdg = cond[:, pdg_pos]
    pdg_onehot_order = np.array(config["simulate_pdgs"])
    onehot = pdgs_to_onehot(pdg_onehot_order, incident_pdg)
    cond_before_pdg = cond[:, :pdg_pos]
    cond_after_pdg = cond[:, pdg_pos + 1 :]
    cond = np.concatenate([cond_before_pdg, onehot, cond_after_pdg], axis=1)
    return cond


class Sampler:
    """
    Note that when a model is given to the sampler it will be placed in eval mode
    """

    def __init__(self, config, model=None, distilled=None):
        """
        Parameters
        ----------
        config : dict
            Full configuration dictionary.
        model : str or Diffusion or None, optional
            A checkpoint path, an already built model, or nothing.
        distilled : bool or None, optional
            Whether to build the network as a consistency model.  ``None``,
            the default, works it out from the run that wrote the checkpoint.
            A teacher and a student state dict are interchangeable, so getting
            this wrong is silent: pass it explicitly only when the checkpoint
            has been moved away from its log directory.
        """
        self.config = config
        self.datatype = getattr(torch, config["training"]["dtype"])
        if isinstance(model, str):
            if distilled is None:
                distilled = model_kind.is_distilled(model)
            self.model = Diffusion(config, distillation=distilled)
            device = config["device"]
            self.model.load_state_dict(torch.load(model, map_location=device))
            # result = self.model.load_state_dict(torch.load(model, map_location=device), strict=False)
            # print("Missing keys:", result.missing_keys)
            # print("Unexpected keys:", result.unexpected_keys)
        else:
            self.model = model
            # an already built model carries its own flag
            self.distilled = bool(
                getattr(model, "distillation", distilled) if model is not None
                else distilled
            )
        if model is not None:
            self.model.to(config["device"], dtype=self.datatype)
        self.preprocess_conditioning = transforms.preprocessing(config, "conditioning")
        self.preprocess_features = transforms.preprocessing(config, "features")

        self.cond_columns = [
            config["data"][f"{name}_key"] for name in config["model"]["cond_features"]
        ]
        if self.config["data"]["n_points_key"] is not None:
            self.has_n_points = True
            self.per_event_cols = self.cond_columns + [config["data"]["n_points_key"]]
        else:
            self.has_n_points = False
            self.per_event_cols = self.cond_columns

    def update_model(self, model):
        self.model = model
        self.model.to(self.config["device"], dtype=self.datatype)

    def sample(self, cond, num_points):
        cond = torch.from_numpy(cond).to(self.config["device"], dtype=self.datatype)
        max_points = int(np.max(num_points))
        preprocessed_cond = self.preprocess_conditioning.forward(cond)
        with evaluating(self.model):
            with torch.no_grad():
                output = self.model.sample(preprocessed_cond, max_points)
        restored_output = self.preprocess_features.inverse(output)
        restored_output = restored_output.cpu().numpy()
        energies = restored_output[:, :, 3]
        energy_order = np.argsort(np.argsort(energies, axis=1))
        remove_from_event = max_points - num_points
        remove = energy_order < remove_from_event[:, None]
        restored_output[remove] = 0
        return restored_output

    def get_cond(
        self, data_part, pick_events=None, total_size=None, return_target=False
    ):
        if not isinstance(pick_events, collections.abc.Hashable):
            pick_events = tuple(int(i) for i in pick_events)
        if total_size is None and pick_events is None:
            total_size = -1
        return self._get_cond(data_part, pick_events, total_size, return_target)

    @lru_cache(maxsize=1)
    def _get_cond(
        self, data_part, pick_events=None, total_size=None, return_target=False
    ):
        if isinstance(pick_events, tuple):
            pick_events = list(pick_events)

        per_event, target = read_write.read_raw_regaxes(
            self.config,
            part=data_part,
            pick_events=pick_events,
            total_size=total_size,
            per_event_cols=self.per_event_cols,
        )
        if self.has_n_points:
            points = per_event[:, -1]
            cond = per_event[:, :-1]
        else:
            real = target[:, :, 3] > 0
            points = real.sum(1)
            cond = per_event
        if not return_target:
            target = None
        cond = pdg_to_onehot_in_full_cond(self.config, cond)
        return cond, points, target

    def sample_from_dataset(
        self, data_part, pick_events=None, total_size=None, return_target=False
    ):
        cond, points, target = self.get_cond(
            data_part, pick_events, total_size, return_target
        )
        sample = self.sample(cond, points)
        return cond, points, target, sample

    @staticmethod
    def get_config_from_model_path(model_path):
        log_dir = os.path.dirname(os.path.dirname(model_path))
        config = yaml.safe_load(open(os.path.join(log_dir, "config.yaml")))
        cuda_avaliable = torch.cuda.is_available()
        if not cuda_avaliable:
            print("CUDA not avaliable, using CPU")
            config["device"] = "cpu"
        return config

    @classmethod
    def from_model_path(cls, model_path, distilled=None):
        config = cls.get_config_from_model_path(model_path)
        sampler = cls(config, model_path, distilled=distilled)
        return sampler

    @property
    def num_sampling_steps(self):
        """Steps ``sample`` will take. A consistency model always takes one."""
        return 1 if self.distilled else self.config["num_steps"]


def points_per_layer_from_target(data_target, config, return_energy=False):
    point_layers = find_layers(config, data_target)
    real_points = data_target[:, :, 3] > 0
    n_layers = len(config["data"]["layer_bottom_pos"])
    points_per_layer = np.zeros((data_target.shape[0], n_layers), dtype=int)
    if return_energy:
        energy_per_layer = np.zeros((data_target.shape[0], n_layers), dtype=float)
    for i in range(n_layers):
        points_per_layer[:, i] = np.sum((point_layers == i) & real_points, axis=1)
        if return_energy:
            energy_per_layer[:, i] = np.sum(
                data_target[:, :, 3] * (point_layers == i), axis=1
            )
    if return_energy:
        return points_per_layer, energy_per_layer
    return points_per_layer


def sample_to_physical(points, points_per_layer, config):
    physical_points = np.zeros_like(points)
    point_layer_ids = -np.ones(points.shape[:2], dtype=int)

    total_points_requested = points_per_layer.sum(1)
    point_energy = points[:, :, 3]
    order_by_energy = np.argsort(np.argsort(point_energy, axis=1))
    num_to_remove = np.clip(points.shape[1] - total_points_requested, 0, None)
    remove_mask = order_by_energy < num_to_remove[:, None]

    physical_points[~remove_mask, 3] = points[~remove_mask, 3]

    beyond_detector = 10 * np.max(points[:, :, 2])
    points[:, :, 2][remove_mask] = beyond_detector

    layer_centers = get_layer_centers(config, coordinates="detector")
    points_by_height = np.argsort(np.argsort(points[:, :, 2], axis=1), axis=1)

    n_events, n_layers = points_per_layer.shape
    cumulative_points_per_layer = np.concatenate(
        (np.zeros((n_events, 1), dtype=int), np.cumsum(points_per_layer, axis=1)),
        axis=-1,
    ).astype(int)
    for layer in range(n_layers):
        layer_mask = (points_by_height >= cumulative_points_per_layer[:, [layer]]) & (
            points_by_height < cumulative_points_per_layer[:, [layer + 1]]
        )
        physical_points[layer_mask, 1] = layer_centers[layer]
        point_layer_ids[layer_mask] = layer

    data_low_x = config["data"]["Xmin"]
    detector_low_z = config["data"]["Zmin_in_detector"]
    data_x_range = config["data"]["Xmax"] - data_low_x
    detector_z_range = config["data"]["Zmax_in_detector"] - detector_low_z
    scale_0 = detector_z_range / data_x_range
    physical_points[~remove_mask, 2] = (
        points[~remove_mask, 0] - data_low_x
    ) * scale_0 + detector_low_z

    data_low_y = config["data"]["Ymin"]
    detector_low_x = config["data"]["Xmin_in_detector"]
    data_y_range = config["data"]["Ymax"] - data_low_y
    detector_x_range = config["data"]["Xmax_in_detector"] - detector_low_x
    scale_1 = detector_x_range / data_y_range
    physical_points[~remove_mask, 0] = (
        points[~remove_mask, 1] - data_low_y
    ) * scale_1 + detector_low_x

    physical_points[remove_mask] = 0

    return physical_points, point_layer_ids


def energy_corrections(physical_points, point_layer_ids, energy_per_layer):
    prior_energy_per_layer = np.zeros_like(energy_per_layer)
    for i in range(energy_per_layer.shape[1]):
        mask = point_layer_ids == i
        prior_energy_per_layer[:, i] = np.sum(physical_points[:, :, 3] * mask, axis=1)
    ratio = np.divide(
        energy_per_layer,
        prior_energy_per_layer,
        where=prior_energy_per_layer != 0,
        out=np.zeros_like(energy_per_layer),
    )
    ratio_per_point = np.take_along_axis(
        ratio, np.clip(point_layer_ids, 0, None), axis=1
    )
    physical_points[:, :, 3] *= ratio_per_point
    return physical_points


def unshift_points(physical_points, point_layer_ids, cond_data_coords, config):
    # defensive programming
    n_events = physical_points.shape[0]
    assert cond_data_coords.shape[0] == n_events
    assert point_layer_ids.shape[0] == n_events
    # cond -> (e, x, y, z) in data
    # cond -> (e, z, x, y) in physical
    direction_vectors = cond_data_coords[:, 1:4]
    normalised_direction_vectors = direction_vectors / np.linalg.norm(
        direction_vectors, axis=1, keepdims=True
    )
    layer_centers = get_layer_centers(config, coordinates="detector")
    layer_centers -= layer_centers[0]
    detetor_x = normalised_direction_vectors[:, 1]
    detetor_z = normalised_direction_vectors[:, 0]
    x_shift_per_layer = layer_centers[:, None] * detetor_x[None, :]
    z_shift_per_layer = layer_centers[:, None] * detetor_z[None, :]

    n_events = physical_points.shape[0]
    real_points = (physical_points[:, :, 3] > 0) & (point_layer_ids >= 0)

    x_shifts = x_shift_per_layer[point_layer_ids, np.arange(n_events)[:, None]]
    physical_points[real_points, 0] += x_shifts[real_points]

    z_shifts = z_shift_per_layer[point_layer_ids, np.arange(n_events)[:, None]]
    physical_points[real_points, 2] += z_shifts[real_points]

    physical_points[~real_points] = 0

    return physical_points


def physical_to_cells(physical_points, point_layer_ids, config):
    n_events = physical_points.shape[0]
    energy_mask = (physical_points[:, :, 3] > 0) & (point_layer_ids >= 0)
    layers, offset = create_map(config)
    x_bin_ids = -np.ones_like(point_layer_ids, dtype=int)
    z_bin_ids = -np.ones_like(point_layer_ids, dtype=int)
    layer_centers = get_layer_centers(config, coordinates="detector")

    event_numbers = np.tile(
        np.arange(n_events).reshape(-1, 1), (1, physical_points.shape[1])
    )

    flat_cell_ids = []
    cell_centers = []
    flat_event_numbers = []
    flat_energies = []
    cell_id_reached = 0
    for layer_n, layer in enumerate(layers):
        layer_mask = (point_layer_ids == layer_n) & energy_mask
        flat_event_numbers += [event_numbers[layer_mask]]
        flat_energies += [physical_points[layer_mask, 3]]

        flat_xs = physical_points[layer_mask, 0]  # n_points_in_layer
        xedges = np.sort(layer["xedges"])
        xcenters = 0.5 * (xedges[1:] + xedges[:-1])  # n_x_ids
        # add underflow and overflow
        xcenters = np.concatenate(([xedges[0] - 10], xcenters, [xedges[-1] + 10]))
        n_x_ids = len(xcenters)
        x_bin_ids[layer_mask] = np.digitize(flat_xs, xedges)

        flat_zs = physical_points[layer_mask, 2]  # n_points_in_layer
        zedges = np.sort(layer["zedges"])
        z_bin_ids[layer_mask] = np.digitize(flat_zs, zedges)
        zcenters = 0.5 * (zedges[1:] + zedges[:-1])  # n_z_ids
        # add underflow and overflow
        zcenters = np.concatenate(([zedges[0] - 10], zcenters, [zedges[-1] + 10]))
        n_z_ids = len(zcenters)

        # n_points_in_layer
        layer_ids = n_x_ids * z_bin_ids[layer_mask] + x_bin_ids[layer_mask]
        # n_x_ids * n_z_ids
        x_center_by_layer_id = np.tile(xcenters, n_z_ids)
        # n_x_ids * n_z_ids
        z_center_by_layer_id = np.repeat(zcenters, n_x_ids)
        y_center_for_layer = layer_centers[layer_n]

        # n_occupied_layer_ids
        occupided_layer_ids = np.sort(np.unique(layer_ids))
        n_occupied_layer_ids = len(occupided_layer_ids)
        use_cell_ids = cell_id_reached + np.arange(n_occupied_layer_ids)
        # n_points_in_layer
        cell_ids = use_cell_ids[np.searchsorted(occupided_layer_ids, layer_ids)]

        flat_cell_ids.append(cell_ids)
        cell_id_reached += n_occupied_layer_ids

        # n_occupied_layer_ids
        occupied_x_centers = x_center_by_layer_id[occupided_layer_ids]
        occupied_z_centers = z_center_by_layer_id[occupided_layer_ids]
        occupied_y_centers = np.repeat(y_center_for_layer, n_occupied_layer_ids)
        # n_occupied_layer_ids, 3
        cell_centers.append(
            np.array([occupied_x_centers, occupied_y_centers, occupied_z_centers]).T
        )

    # n_total_deposits
    flat_cell_ids = np.concatenate(flat_cell_ids)
    # n_total_deposits
    flat_event_numbers = np.concatenate(flat_event_numbers)
    # n_total_deposits
    flat_energies = np.concatenate(flat_energies)
    # n_cells
    cell_centers = np.concatenate(cell_centers)
    n_unique_cells = len(cell_centers)
    output = np.zeros((n_events, n_unique_cells, 4))
    output[:, :, :3] = cell_centers

    # n_total_deposits
    flat_global_id = flat_event_numbers * n_unique_cells + flat_cell_ids
    flat_cell_energy = np.bincount(
        flat_global_id, weights=flat_energies, minlength=n_unique_cells * n_events
    )
    output[:, :, 3] = flat_cell_energy.reshape(n_events, n_unique_cells)

    return output
