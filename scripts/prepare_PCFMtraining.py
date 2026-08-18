import argparse
import yaml
from pathlib import Path

import h5py
import numpy as np
from src.data import read_write
from src.evaluation.inference import points_per_layer_from_target


def parse_arguments(args: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert numpy arrays to showerdata-compatible HDF5 format"
    )
    parser.add_argument(
        "config",
        type=Path,
        help="Path to CaloClouds_diffusion config file "
        + "describing the data and diffusion model",
    )
    parser.add_argument(
        "output_file",
        type=Path,
        help="Path to the output HDF5 file",
    )
    parser.add_argument(
        "--data-part",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Data partition to read from, must be 'test', 'val' or 'train'",
    )
    return parser.parse_args(args)


def main(args: list[str] | None = None) -> None:
    parsed_args = parse_arguments(args)

    with open(parsed_args.config, "r") as f:
        config = yaml.safe_load(f)

    needed_per_event = ["incident_energy", "incident_direction", "incident_pdg"]
    on_disk = [config["data"][f"{name}_key"] for name in needed_per_event]
    per_event, points = read_write.read_raw_regaxes(
        config,
        part=parsed_args.data_part,
        total_size=-1,
        per_event_cols=on_disk,
    )

    condE = per_event[:, 0]
    direction = per_event[:, 1:4]
    pdg = per_event[:, 4]
    points, energy = points_per_layer_from_target(points, config, return_energy=True)

    n_showers, n_layers = energy.shape
    print(f"Loaded {n_showers} showers, {n_layers} layers")

    # fixed fields
    pdg_label_order = np.array(config["simulate_pdgs"])
    sorter = np.argsort(pdg_label_order)
    labels = sorter[np.searchsorted(pdg_label_order, pdg)]

    print(f"PDGs: {pdg_label_order}")

    with h5py.File(parsed_args.output_file, "w") as f:
        f.create_dataset("directions", data=direction)
        f.create_dataset("energies", data=condE)
        f.create_dataset("labels", data=labels)
        f.create_dataset("num_points", data=points)
        f.create_dataset("energy_per_layer", data=energy)
        f.attrs["pdg"] = pdg_label_order
        f.attrs["label_list"] = np.arange(len(pdg_label_order))

    print(f"Saved to {parsed_args.output_file}")


if __name__ == "__main__":
    main()
