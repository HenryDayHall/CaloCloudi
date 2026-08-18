import sys
from src.evaluation import summarise, inference
import os
import numpy as np
import h5py

model_path = sys.argv[1]
config = inference.Sampler.get_config_from_model_path(model_path)
n_events = 1000  # how many showers to summarise
print(f"Summarising {n_events} events")

if len(sys.argv) > 2:
    print(
        "Will read the cond, "
        f"points_per_layer and energy_per_layer from {sys.argv[2]}"
    )
    # Reading in the data file from pcFM
    pcfm_path = sys.argv[2]
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

summarise.complete_model(model_path, n_events, **model_kwargs)
