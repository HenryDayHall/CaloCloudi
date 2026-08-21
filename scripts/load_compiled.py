"""
Load a compiled :class:`~src.evaluation.compile.CompiledDiffusion` and run it on
CPU only, given the compiled object and its config on disk.

The compiled object is loaded without importing the ``diffusion`` module, so
this script keeps its imports minimal.

Usage
-----
    python -m src.evaluation.load_compiled path/to/compiled_model.pt \\
        path/to/config.yaml
"""

import argparse

import torch
import yaml


def load_config(config_path):
    """Read the config and force CPU inference.

    Parameters
    ----------
    config_path : str
        Path to the ``config.yaml`` the compiled model was built with.

    Returns
    -------
    dict
        The config with ``device`` set to ``"cpu"``.
    """
    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    config["device"] = "cpu"
    return config


def load_compiled(compiled_path):
    """Load the compiled object onto the CPU.

    Parameters
    ----------
    compiled_path : str
        Path to the saved compiled object.

    Returns
    -------
    CompiledDiffusion
        The reloaded compiled model, on the CPU.
    """
    compiled = torch.load(
        compiled_path, map_location="cpu", weights_only=False
    )
    compiled.device = "cpu"
    return compiled


def dummy_inputs(config, batch_size):
    """Build zeroed inputs sized from the config.

    Parameters
    ----------
    config : dict
    batch_size : int
        Number of events to run.

    Returns
    -------
    (torch.Tensor, torch.Tensor, torch.Tensor)
        Conditioning, points per layer and energy per layer, all on the CPU.
    """
    dtype = getattr(torch, config["training"]["dtype"])
    cond_dim = config["model"]["cond_dim"]
    n_layers = len(config["data"]["layer_bottom_pos"])
    conditioning = torch.zeros((batch_size, cond_dim), dtype=dtype)
    points_per_layer = torch.zeros((batch_size, n_layers), dtype=torch.long)
    energy_per_layer = torch.zeros((batch_size, n_layers), dtype=dtype)
    return conditioning, points_per_layer, energy_per_layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("compiled_path", help="path to compiled_model.pt")
    parser.add_argument("config_path", help="path to config.yaml")
    parser.add_argument(
        "--batch-size", type=int, default=4, help="number of events to run"
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    compiled = load_compiled(args.compiled_path)
    print(f"Loaded compiled model from {args.compiled_path} on CPU")

    conditioning, points_per_layer, energy_per_layer = dummy_inputs(
        config, args.batch_size
    )
    output = compiled.infer(conditioning, points_per_layer, energy_per_layer)
    print(f"Output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
