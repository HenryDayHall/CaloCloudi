import torch
import os
import yaml
import numpy as np
import collections
from functools import lru_cache
from ..diffusion import Diffusion
from ..data import transforms, read_write
from ..detector_map import find_layers


class Sampler:
    """
    Note that when a model is given to the sampler it will be placed in eval mode
    """
    def __init__(self, config, model, distilled=False):
        self.config = config
        self.datatype = getattr(torch, config["training"]["dtype"])
        if isinstance(model, str):
            self.model = Diffusion(config, distillation=distilled)
            self.model.load_state_dict(torch.load(model))
        else:
            self.model = model
        self.model.to(config["device"], dtype=self.datatype)
        self.model.eval().requires_grad_(False)
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
        self.model.eval().requires_grad_(False)

    def sample(self, cond, num_points):
        cond = torch.from_numpy(cond).to(self.config["device"], dtype=self.datatype)
        max_points = int(np.max(num_points))
        output = self.model.sample(
            self.preprocess_conditioning.forward(cond), max_points
        )
        output = self.preprocess_features.inverse(output)
        output = output.cpu().numpy()
        energies = output[:, :, 3]
        energy_order = np.argsort(energies, axis=1)
        remove_from_event = max_points - num_points
        remove = energy_order < remove_from_event[:, None]
        output[remove] = 0
        return output

    def get_cond(
        self, data_part, pick_events=None, total_size=None, return_target=False
    ):
        if not isinstance(pick_events, collections.abc.Hashable):
            pick_events = (int(i) for i in pick_events)
        if total_size is None and pick_events is None:
            total_size = -1
        return self._get_cond(data_part, pick_events, total_size, return_target)

    @lru_cache(maxsize=1)
    def _get_cond(
        self, data_part, pick_events=None, total_size=None, return_target=False
    ):
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
        return cond, points, target

    def sample_from_dataset(
        self, data_part, pick_events=None, total_size=None, return_target=False
    ):
        cond, points, target = self.get_cond(
            data_part, pick_events, total_size, return_target
        )
        sample = self.sample(cond, points)
        return cond, points, target, sample

    @classmethod
    def from_model_path(cls, model_path):
        log_dir = os.path.dirname(os.path.dirname(model_path))
        config = yaml.safe_load(open(os.path.join(log_dir, "config.yaml")))
        sampler = cls(config, model_path)
        return sampler


def sample_to_physical(points, config):
    physical_points = np.zeros_like(points)
    point_layers = -np.ones(points.shape[:2], dtype=int)

    data_low_x = config["data"]["Xmin"]
    detector_low_z = config["data"]["Zmin_in_detector"]
    data_x_range = config["data"]["Xmax"] - data_low_x
    detector_z_range = config["data"]["Zmax_in_detector"] - detector_low_z
    shift_0 = detector_low_z - data_low_x,
    scale_0 = detector_z_range / data_x_range

    data_low_y = config["data"]["Ymin"]
    detector_low_x = config["data"]["Xmin_in_detector"]
    data_y_range = config["data"]["Ymax"] - data_low_y
    detector_x_range = config["data"]["Xmax_in_detector"] - detector_low_x
    shift_1 = detector_low_x - data_low_y,
    scale_1 = detector_x_range / data_y_range

    data_layers = config["data"]["layer_bottom_pos"]

    pass


def sample_to_cells(physical_points, config):
    pass
