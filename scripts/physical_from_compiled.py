"""Physically rescaled showers from a compiled distilled checkpoint.

Takes a TorchScript module written by ``standalone_distilled.py export`` and
the per-layer truth from an external count model (pcFM, or
``inference.points_per_layer_from_target``), and runs the same post-processing
chain as ``scripts/generate_test_set_pcfm.py``:

    compiled module  ->  sample_to_physical  ->  energy_corrections
                     ->  unshift_points      [->  physical_to_cells]

Unlike ``standalone_distilled.py`` this one *does* import the project: the
geometry lives in ``src.evaluation.inference`` and there is no reason to
duplicate it.  Only the network is compiled; the rescaling stays in numpy.

    python physical_from_compiled.py \
        --compiled distilled.ts.pt --config config/default.yaml \
        --inputs pcfm_events.npz --out showers.npz
"""

import argparse

import numpy as np
import torch
import yaml

from src.evaluation import inference


def sample_compiled(module, cond, num_points, feature_dim, device="cpu",
                    dtype=torch.float32, batch_size=256, generator=None):
    """``Sampler.sample`` for a traced module: raw cond in, data coords out."""
    max_points = int(np.max(num_points))
    out = np.zeros((len(cond), max_points, feature_dim), dtype=np.float32)
    for start in range(0, len(cond), batch_size):
        stop = min(start + batch_size, len(cond))
        cond_t = torch.as_tensor(cond[start:stop], device=device, dtype=dtype)
        noise = torch.randn(
            stop - start, max_points, feature_dim,
            device=device, dtype=dtype, generator=generator,
        )
        keep = torch.as_tensor(
            np.asarray(num_points[start:stop], dtype=np.int64), device=device
        )
        with torch.no_grad():
            out[start:stop] = module(cond_t, noise, keep).cpu().numpy()
    return out


def physical_from_compiled(module, config, cond, points_per_layer,
                           energy_per_layer, with_cells=False, **sample_kwargs):
    """Sample, place in the detector, then rescale energy to the truth.

    ``cond`` is raw (one-hot already applied) and in data coordinates, since
    ``unshift_points`` reads the incident direction back out of it.
    """
    cond = np.asarray(cond)
    points_per_layer = np.asarray(points_per_layer).astype(int)
    energy_per_layer = np.asarray(energy_per_layer)

    n_layers = len(config["data"]["layer_bottom_pos"])
    if points_per_layer.shape[1] != n_layers:
        raise ValueError(
            f"points_per_layer has {points_per_layer.shape[1]} layers, "
            f"config has {n_layers}"
        )
    if energy_per_layer.shape != points_per_layer.shape:
        raise ValueError("energy_per_layer and points_per_layer must match in shape")

    # the count model decides the event size; sample_to_physical then hands
    # each point to a layer by height, cheapest to lowest
    num_points = points_per_layer.sum(1)
    sample = sample_compiled(
        module, cond, num_points, config["model"]["feature_dim"], **sample_kwargs
    )

    physical, layer_ids = inference.sample_to_physical(sample, points_per_layer, config)
    physical = inference.energy_corrections(physical, layer_ids, energy_per_layer)
    physical = inference.unshift_points(physical, layer_ids, cond, config)

    cells = (
        inference.physical_to_cells(physical, layer_ids, config) if with_cells else None
    )
    return physical, layer_ids, cells


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--compiled", required=True, help="a .ts.pt from `export`")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--inputs", required=True,
        help="npz with cond, points_per_layer, energy_per_layer",
    )
    parser.add_argument("--out", required=True, help="npz to write")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--cells", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as handle:
        config = yaml.safe_load(handle)
    config["device"] = args.device

    inputs = np.load(args.inputs)
    module = torch.jit.load(args.compiled, map_location=args.device).eval()

    generator = None
    if args.seed is not None:
        generator = torch.Generator(device=args.device).manual_seed(args.seed)

    physical, layer_ids, cells = physical_from_compiled(
        module,
        config,
        inputs["cond"],
        inputs["points_per_layer"],
        inputs["energy_per_layer"],
        with_cells=args.cells,
        device=args.device,
        dtype=getattr(torch, config["training"]["dtype"]),
        batch_size=args.batch_size,
        generator=generator,
    )

    saved = {"physical_points": physical, "point_layer_ids": layer_ids}
    if cells is not None:
        saved["cells"] = cells
    np.savez_compressed(args.out, **saved)
    print(f"wrote {args.out}: {physical.shape[0]} events, {physical.shape[1]} points")


if __name__ == "__main__":
    main()
