"""
Build a :class:`~src.evaluation.compile.CompiledDiffusion` from a distilled
model, round trip it through disk, then run it on a batch of conditioning
pulled from the dataset named in the model's config.

Usage
-----
    python -m src.evaluation.run_compiled /path/to/distilled/checkpoint.pt
"""

import argparse

import torch

from src.evaluation.compile import CompiledDiffusion
from src.evaluation.inference import Sampler, points_per_layer_from_target


def build_and_roundtrip(model_path, max_points, save_path):
    """Compile a model, save it, and load it back.

    Parameters
    ----------
    model_path : str
        Distilled checkpoint to compile.
    max_points : int
        Points generated per event; fixed at compile time.
    save_path : str
        Where to write the compiled object.

    Returns
    -------
    (CompiledDiffusion, dict)
        The reloaded compiled model and the config it was built from.
    """
    config = Sampler.get_config_from_model_path(model_path)
    compiled = CompiledDiffusion(config, model_path, max_points)
    torch.save(compiled, save_path)
    reloaded = torch.load(save_path, map_location=config["device"], weights_only=False)
    return reloaded, config


def get_conditioning(config, model_path, data_part, total_size):
    """Pull conditioning and per layer targets from the dataset.

    Parameters
    ----------
    config : dict
    model_path : str
        Used to build a :class:`Sampler` for reading the dataset.
    data_part : str
        Dataset partition, e.g. ``"test"``.
    total_size : int
        Number of events to read.

    Returns
    -------
    (torch.Tensor, torch.Tensor, torch.Tensor)
        Conditioning, points per layer and energy per layer, as tensors on the
        configured device.
    """
    sampler = Sampler.from_model_path(model_path)
    cond, _points, target = sampler.get_cond(
        data_part, total_size=total_size, return_target=True
    )
    points_per_layer, energy_per_layer = points_per_layer_from_target(
        target, config, return_energy=True
    )
    device = config["device"]
    conditioning = torch.from_numpy(cond).to(device)
    points_per_layer = torch.from_numpy(points_per_layer).to(device)
    energy_per_layer = torch.from_numpy(energy_per_layer).to(device)
    return conditioning, points_per_layer, energy_per_layer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="path to a distilled checkpoint")
    parser.add_argument(
        "--max-points",
        type=int,
        default=6000,
        help="points generated per event (fixed at compile time)",
    )
    parser.add_argument(
        "--save-path",
        default="compiled_model.pt",
        help="where to save the compiled object",
    )
    parser.add_argument("--data-part", default="test", help="dataset partition")
    parser.add_argument(
        "--total-size", type=int, default=32, help="number of events to run"
    )
    args = parser.parse_args()

    compiled, config = build_and_roundtrip(
        args.model_path, args.max_points, args.save_path
    )
    print(f"Saved and reloaded compiled model from {args.save_path}")

    conditioning, points_per_layer, energy_per_layer = get_conditioning(
        config, args.model_path, args.data_part, args.total_size
    )
    print(f"Running on {conditioning.shape[0]} events")

    output = compiled.infer(conditioning, points_per_layer, energy_per_layer)
    print(f"Output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
