import argparse
import yaml
from pathlib import Path

import h5py
import numpy as np
from src.data import read_write


def pick_gun_position(pxyz, calorimeter_surface_y=1804.7):
    distance_to_surface = calorimeter_surface_y / pxyz[:, 1]
    gun_pos = distance_to_surface[:, None] * pxyz
    return gun_pos


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
        default="test",
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
    global_direction = direction[:, [1, 2, 0]]
    gun_pos = pick_gun_position(global_direction)
    pdg = per_event[:, 4]
    assert np.all(pdg == 22)

    with h5py.File(parsed_args.output_file, "w") as f:
        f.create_dataset("input_p_global", data=global_direction)
        f.create_dataset("energy", data=condE)
        f.create_dataset("input_gun_position", data=gun_pos)

    print(f"Saved to {parsed_args.output_file}")


if __name__ == "__main__":
    main()
