"""Shared fixtures for the CaloCloudi test suite."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


try:  # pragma: no cover
    import showerdata  # noqa: F401
except ImportError:  # pragma: no cover
    sys.modules["showerdata"] = MagicMock()

import h5py  # noqa: E402

N_FEATURES = 4  # x, y, z, e


def make_padded_file(
    path,
    *,
    n_events=6,
    n_points=5,
    roll_axis=False,
    front_padded=False,
    energy_ndim=1,
    n_points_ndim=1,
    points_key="events",
    seed=0,
):
    rng = np.random.default_rng(seed)

    events = rng.normal(size=(n_events, n_points, N_FEATURES))
    n_real = rng.integers(1, n_points + 1, size=n_events)
    energies = np.zeros((n_events, n_points))
    for i, n in enumerate(n_real):
        values = rng.uniform(0.1, 1.0, size=n)
        if front_padded:
            energies[i, n_points - n :] = values
        else:
            energies[i, :n] = values
    events[:, :, 3] = energies

    on_disk = events.transpose(0, 2, 1) if roll_axis else events

    incident_energy = rng.uniform(10.0, 100.0, size=n_events)
    direction = rng.normal(size=(n_events, 3))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    energy_ds = incident_energy[:, None] if energy_ndim == 2 else incident_energy
    n_points_ds = n_real[:, None] if n_points_ndim == 2 else n_real

    with h5py.File(path, "w") as handle:
        handle.create_dataset(points_key, data=on_disk)
        handle.create_dataset("energy", data=energy_ds)
        handle.create_dataset("n_points", data=n_points_ds)
        handle.create_dataset("p_norm_local", data=direction)

    return {
        "events": events,
        "energy": incident_energy,
        "n_points": n_real,
        "p_norm_local": direction,
    }


def make_empty_padded_file(path, points_key="events"):
    with h5py.File(path, "w") as handle:
        handle.create_dataset(points_key, data=np.zeros((0, 0, 0)))
        handle.create_dataset("energy", data=np.zeros(0))
    return path


def base_config(dataset_path, *, fmt="padded", roll_axis=False, padding="back"):
    return {
        "device": "cpu",
        "preprocessing": {
            "conditioning": None,
            "features": [
                [
                    "Affine",
                    {
                        "inverse_scale": ["data", "Xstd"],
                        "neg_shift": ["data", "Xmean"],
                    },
                ]
            ],
        },
        "model": {
            "cond_features": ["incident_energy", "incident_direction"],
            "cond_dim": 4,
            "feature_dim": 4,
        },
        "data": {
            "dataset_path": dataset_path,
            "test_range_start": 0,
            "test_range_end": 1,
            "val_range_start": 1,
            "val_range_end": 2,
            "train_range_start": 0,
            "train_range_end": 3,
            "format": fmt,
            "roll_axis": roll_axis,
            "padding": padding,
            "points_key": "events",
            "incident_pdg_key": None,
            "incident_energy_key": "energy",
            "incident_direction_key": "p_norm_local",
            "n_points_key": "n_points",
            "Xmin": -250.0,
            "Xmax": 250.0,
            "Ymin": -250.0,
            "Ymax": 250.0,
            "Xmin_in_detector": -250.0,
            "Xmax_in_detector": 250.0,
            "Zmin_in_detector": -250.0,
            "Zmax_in_detector": 250.0,
            "Xmean": 0.0,
            "Xstd": 2.0,
            "orientation": "hdf5:xyz==local:xyz",
            "cell_thickness": 1.0,
            "cell_size": 5.0,
            "divisions_per_cell": 5,
            "layer_bottom_pos": [0.0, 1.0, 2.0],
        },
        "detector": {
            "orientation": "hdf5:xyz==global:zxy",
            # layers 0 and 1 are ecal, layer 2 is hcal, so the two thicknesses
            # and hcal_start are all exercised by a three layer stack
            "cell_thickness_ecal": 0.5,
            "cell_thickness_hcal": 2.0,
            "hcal_start": 2,
            "cell_size": 5.0,
            "layer_bottom_pos": [1811.0, 1814.0, 1823.0],
        },
        "training": {
            "dtype": "float32",
            "batch_size": 2,
            "max_points_per_event": 6000,
            "retain_quantized": True,
        },
    }


@pytest.fixture(autouse=True)
def clear_read_write_caches():
    from src.data import read_write

    cached = (
        read_write.get_possible_files,
        read_write.get_files,
        read_write.get_n_events,
    )
    for fn in cached:
        fn.cache_clear()
    yield
    for fn in cached:
        fn.cache_clear()


@pytest.fixture
def padded_dataset(tmp_path):
    def _build(n_events=(6, 6, 6), n_points=5, roll_axis=False, fmt="padded", **kwargs):
        pattern = str(tmp_path / "data_{}.h5")
        per_file = []
        for i, n in enumerate(n_events):
            per_file.append(
                make_padded_file(
                    pattern.format(i),
                    n_events=n,
                    n_points=n_points if isinstance(n_points, int) else n_points[i],
                    roll_axis=roll_axis,
                    seed=i,
                    **kwargs,
                )
            )
        config = base_config(pattern, fmt=fmt, roll_axis=roll_axis)
        config["data"]["train_range_end"] = len(n_events)
        truth = {
            key: np.concatenate([f[key] for f in per_file], axis=0)
            for key in ("energy", "n_points", "p_norm_local")
        }
        if len({f["events"].shape[1] for f in per_file}) == 1:
            truth["events"] = np.concatenate([f["events"] for f in per_file], axis=0)
        truth["per_file"] = per_file
        return config, truth

    return _build
