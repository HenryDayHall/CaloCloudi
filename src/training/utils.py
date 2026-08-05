import time
import os
import numpy as np
import yaml
import torch


class Logger:
    per_step_log = ["loss", "n_events", "n_updates", "epoch_number", "time"]

    def __init__(self, configs, existing_log_dir=None):
        self.configs = configs
        self.values = {name: [] for name in self.per_step_log}

        if existing_log_dir is None:
            self.log_dir = self.setup_dir()
        else:
            self.log_dir = existing_log_dir
            self.checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
            self._load()

    def setup_dir(self):
        datestamp = time.strftime("%Y_%m_%d__%H_%M_%S")
        log_dir = os.path.join(self.configs["output_path"], "logs", datestamp)
        assert not os.path.exists(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        checkpoint_dir = os.path.join(log_dir, "checkpoints")
        with open(os.path.join(log_dir, "configs.yaml"), "w") as f:
            yaml.dump(self.configs, f)
        return log_dir, checkpoint_dir

    def add_step(self, loss, n_events, n_updates, epoch_number, time):
        self.values["loss"].append(loss)
        self.values["n_events"].append(n_events)
        self.values["n_updates"].append(n_updates)
        self.values["epoch_number"].append(epoch_number)
        self.values["time"].append(time)

    def do_validation(self):
        pass

    def checkpoint_model(self, model, **kwargs):
        timestamp = time.strftime("%Y_%m_%d__%H_%M_%S")
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{timestamp}_model.pt")
        torch.save(model.state_dict(), checkpoint_path)
        for name in kwargs:
            path = os.path.join(self.checkpoint_dir, f"{timestamp}_{name}.pt")
            torch.save(kwargs[name].state_dict(), path)

    def save(self):
        for name in self.per_step_log:
            np.save(self.values[name], os.path.join(self.log_dir, name))

    def _load(self):
        for name in self.per_step_log:
            self.values[name] = np.load(os.path.join(self.log_dir, name))
        with open(os.path.join(self.log_dir, "configs.yaml"), "r") as f:
            self.configs = yaml.safe_load(f)

    @classmethod
    def from_model_path(cls, model_path):
        log_dir = os.path.dirname(os.path.dirname(model_path))
        configs = yaml.safe_load(open(os.path.join(log_dir, "configs.yaml")))
        logger = cls(configs, existing_log_dir=log_dir)
        return logger
