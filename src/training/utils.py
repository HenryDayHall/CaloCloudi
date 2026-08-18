import time
import os
import numpy as np
import yaml
import torch
import k_diffusion

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
from ..data.dataset import from_config as dataset_from_config
from ..evaluation.model_kind import RUN_INFO_NAME, DISTILLED_ROLES


class Logger:
    per_step_log = [
        "loss",
        "ema_decay",
        "grad_norm",
        "n_events",
        "n_updates",
        "epoch_number",
        "time",
    ]

    def __init__(
        self,
        config_path,
        existing_log_dir=None,
        chatty=True,
        run_type=None,
        run_info=None,
    ):
        self.text = ""
        self.chatty = chatty

        self.add_text(f"Loading config from {config_path}")
        self.config_path = config_path
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.values = {name: [] for name in self.per_step_log}

        self.validation_functions_dict = {}
        self.validation_values = {"n_events": []}

        if existing_log_dir is None:
            self.log_dir = self.setup_dir()
        else:
            self.log_dir = existing_log_dir
            self._load()
        self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        if run_type is not None:
            self.write_run_info(run_type, **(run_info or {}))

    def write_run_info(self, run_type, overwrite=False, **extra):
        """Record what kind of run this is, beside the config.

        A teacher checkpoint and a distilled student checkpoint are
        indistinguishable once written -- same keys, same shapes -- so the
        evaluation code cannot work out how to sample one without being told.
        This is where it is told.

        Existing files are left alone unless ``overwrite``, so resuming a run
        never rewrites its own history, and resuming an older run backfills
        the marker it never had.
        """
        path = os.path.join(self.log_dir, RUN_INFO_NAME)
        if os.path.exists(path) and not overwrite:
            return path
        info = {
            "run_type": run_type,
            "distilled_roles": (
                list(DISTILLED_ROLES) if run_type == "student" else []
            ),
            "written": time.strftime("%Y-%m-%d_%H-%M-%S"),
        }
        info.update(extra)
        with open(path, "w") as f:
            yaml.dump(info, f)
        self.add_text(f"Run info written to {path}")
        return path

    def add_validation_function(self, name, function):
        self.add_text(f"Adding validation function for {name}")
        self.validation_functions_dict[name] = function
        if name not in self.validation_values:
            self.validation_values[name] = []

    def setup_dir(self):
        datestamp = time.strftime("%Y_%m_%d__%H_%M_%S")
        log_dir = os.path.join(self.config["output_path"], "logs", datestamp)
        os.makedirs(log_dir, exist_ok=False)
        with open(os.path.join(log_dir, "config.yaml"), "w") as f:
            yaml.dump(self.config, f)
        return log_dir

    def add_step(self, **kwargs):
        for name in self.per_step_log:
            given = kwargs.get(name, 0.0)
            if isinstance(given, torch.Tensor):
                given = given.detach().cpu().item()
            self.values[name].append(given)

    def add_text(self, text):
        timestamp = time.strftime("%H-%M-%S")
        lines = text.split("\n")
        for line in lines:
            self.text += f"{timestamp}: {line}\n"
        if self.chatty:
            for line in lines:
                print(line)

    def do_validation(self, model):
        self.validation_values["n_events"].append(self.values["n_events"][-1])
        if self.chatty:
            # Don't actually log this, just print
            print(f"Have conf path {self.config_path} and "
                  f"dataset path {self.config['data']['dataset_path']}")
        for name, function in self.validation_functions_dict.items():
            self.add_text(f"Running validation for {name}")
            self.validation_values[name].append(function(model))

    def checkpoint_model(self, model, ema_model, **kwargs):
        timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{timestamp}_model.pt")
        torch.save(model.state_dict(), checkpoint_path)
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{timestamp}_ema_model.pt")
        torch.save(ema_model.state_dict(), checkpoint_path)
        for name in kwargs:
            path = os.path.join(self.checkpoint_dir, f"{timestamp}_{name}.pt")
            torch.save(kwargs[name].state_dict(), path)

    def save(self):
        for name in self.per_step_log:
            np.save(os.path.join(self.log_dir, name), self.values[name])
        for name in self.validation_values:
            np.save(
                os.path.join(self.log_dir, f"val_{name}.npy"),
                self.validation_values[name],
            )
        with open(os.path.join(self.log_dir, "logs.txt"), "w") as f:
            f.write(self.text)

    def _load(self):
        for name in self.per_step_log:
            self.values[name] = np.load(
                os.path.join(self.log_dir, name + ".npy")
            ).tolist()
        with open(os.path.join(self.log_dir, "config.yaml"), "r") as f:
            self.config = yaml.safe_load(f)
        with open(os.path.join(self.log_dir, "logs.txt"), "r") as f:
            self.text = f.read()

    @classmethod
    def from_model_path(cls, model_path, validation_functions_dict=None, run_type=None):
        log_dir = os.path.dirname(os.path.dirname(model_path))
        config_path = os.path.join(log_dir, "config.yaml")
        logger = cls(
            config_path,
            existing_log_dir=log_dir,
            run_type=run_type,
        )
        if validation_functions_dict is not None:
            for name, function in validation_functions_dict.items():
                logger.add_validation_function(name, function)
            for name in logger.validation_values:
                logger.validation_values[name] = np.load(
                    os.path.join(logger.log_dir, f"val_{name}")
                )
        return logger


def get_dataloader(config):
    dataset = dataset_from_config(config)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=config["training"]["shuffle_data"],
        num_workers=config["training"]["num_workers"],
    )
    return dataloader


def get_optimiser(config, model):
    name = config["training"]["optimiser"]
    kwargs = config["training"]["optimiser_kwargs"]
    if name == "Adam":
        optimiser = torch.optim.Adam(model.parameters(), **kwargs)
    elif name == "AdamW":
        optimiser = torch.optim.AdamW(model.parameters(), **kwargs)
    else:
        raise NotImplementedError
    return optimiser


def get_sample_density(config):
    sigma_data = config["training"]["sigma_data"]
    sigma_sample_density = config["training"]["sigma_sample_density"]
    sample_density = k_diffusion.config.make_sample_density(
        dict(
            sigma_data=sigma_data,
            sigma_sample_density=sigma_sample_density,
        )
    )
    return sample_density


def get_scheduler(config, optimiser, start_epoch):
    end_epoch = config["training"]["end_epoch"]
    start_lr = config["training"]["start_lr_multiplier"]
    end_lr = config["training"]["end_lr_multiplier"]

    def lr_func(epoch):
        if epoch <= start_epoch:
            return 1.0
        elif epoch <= end_epoch:
            total = end_epoch - start_epoch
            delta = epoch - start_epoch
            frac = delta / total
            return (1 - frac) * 1.0 + frac * (end_lr / start_lr)
        else:
            return end_lr / start_lr

    return LambdaLR(optimiser, lr_lambda=lr_func)
