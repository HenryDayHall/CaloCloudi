import sys
from src.evaluation import summarise, inference
import os
import numpy as np
import h5py

model_path = sys.argv[1]
config = inference.Sampler.get_config_from_model_path(model_path)
n_events = 1000  # how many showers to summarise
print(f"Summarising {n_events} events")


def get_from_showerdata(config, pcfm_path, n_events):
    with h5py.File(pcfm_path, "r") as pcfm:
        max_events = pcfm["observables/num_points_per_layer"].shape[0]
        print(f"Max events: {max_events}")
        points_per_layer = pcfm["observables/num_points_per_layer"][:n_events]
        energy_per_layer = pcfm["observables/energy_per_layer"][:n_events]
        # Getting the points_per_layer and energy_per_layer from the file
        # conditioning for the same events
        keys_on_disk = {
            "incident_energy": "energies",
            "incident_direction": "directions",
            "incident_pdg": "pdg",
        }
        cond = [
            pcfm[keys_on_disk[k]][:n_events] for k in config["model"]["cond_features"]
        ]
        cond = np.concatenate(cond, axis=1)
        cond = inference.pdg_to_onehot_in_full_cond(config, cond)
    return cond, points_per_layer, energy_per_layer


def get_from_basic(config, pcfm_path, n_events):
    with h5py.File(pcfm_path, "r") as pcfm:
        max_events = pcfm["num_points"].shape[0]
        print(f"Max events: {max_events}")
        points_per_layer = pcfm["num_points"][:n_events]
        energy_per_layer = pcfm["energy_per_layer"][:n_events]
        # Getting the points_per_layer and energy_per_layer from the file
        # conditioning for the same events
        keys_on_disk = {
            "incident_energy": "energy",
            "incident_direction": "directions",
            "incident_pdg": "labels",
        }
        cond = [
            pcfm[keys_on_disk[k]][:n_events] for k in config["model"]["cond_features"]
        ]
        if "incident_pdg" in config["model"]["cond_features"]:
            simulated_pdgs = np.array(config["simulate_pdgs"])
            index = config["model"]["cond_features"].index("incident_pdg")
            cond[index] = simulated_pdgs[cond[index]][:, None]
            # Annoying energy scale missmatch
            energy_per_layer *= 10**(-3)
        cond = np.concatenate(cond, axis=1)
        cond = inference.pdg_to_onehot_in_full_cond(config, cond)
    return cond, points_per_layer, energy_per_layer


if len(sys.argv) > 2:
    print(
        "Will read the cond, "
        f"points_per_layer and energy_per_layer from {sys.argv[2]}"
    )
    # Reading in the data file from pcFM
    pcfm_path = sys.argv[2]
    try:
        cond, points_per_layer, energy_per_layer = get_from_showerdata(
            config, pcfm_path, n_events
        )
    except KeyError:
        cond, points_per_layer, energy_per_layer = get_from_basic(
            config, pcfm_path, n_events
        )
    assert cond.shape[0] >= n_events
    model_kwargs = {
        "cond": cond,
        "points_per_layer": points_per_layer,
        "energy_per_layer": energy_per_layer,
        "rescale_energy": True,
    }
else:  # will take from test set
    print("Sampling from test set")
    model_kwargs = {"rescale_energy": True}

summarise.complete_model(model_path, n_events, force=True, **model_kwargs)
