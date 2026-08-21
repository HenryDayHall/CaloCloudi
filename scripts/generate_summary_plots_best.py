import os
import sys
import glob
import yaml
import numpy as np
import h5py
from src.evaluation import summarise, example_events, inference
from src.training.best_checkpoint import find_best_checkpoint


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


n_events = 1000

# this is photons only....
external_cond = {
        "photons_only": "/home/dayhallh/training/CC_ExpSpec/PointCountFM_private/results/20260819_154640_CaloClouds_photonsOnly/new_samples.h5",
        "EM": "/home/dayhallh/training/CC_ExpSpec/PointCountFM_private/results/20260820_182856_CaloClouds_EM/new_samples.h5"
        }

if len(sys.argv) > 1:
    log_dir = sys.argv[1]
else:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_conf_path = os.path.join(this_dir, "../config/default.yaml")
    default_conf = yaml.safe_load(open(default_conf_path))
    log_dir = os.path.join(default_conf["output_path"], "logs")


def search():
    to_do = []
    checkpoints_glob = os.path.join(log_dir, "*/checkpoints/")
    checkpoint_locations = glob.glob(checkpoints_glob)
    for ckpt in checkpoint_locations:
        folder = os.path.dirname(ckpt)
        if os.path.exists(os.path.join(folder, "no_summaries")):
            continue
        model_location = find_best_checkpoint(folder)
        summary_location = model_location[:-3] + "_summary.npz"
        if not os.path.exists(summary_location):
            to_do.append(model_location)

    print(f"Found {len(to_do)} models to process")
    return to_do


while True:
    to_do = search()
    if not to_do:
        print("No more models to process")
        break
    model_path = to_do.pop()
    print(f"Processing {model_path}")
    config = inference.Sampler.get_config_from_model_path(model_path)
    # raw model
    summarise.complete_model(model_path, n_events, force=True)
    # truth corrected points and energy
    summarise.complete_model(model_path, n_events, rescale_energy=True, force=True)
    example_events.plot_and_save(model_path, [100])
    folder_path = os.path.dirname(os.path.dirname(model_path))
    # external cond
    if "Padded_photon" in config["data"]["dataset_path"]:
        external = external_cond["photons_only"]
    elif "Padded_" in config["data"]["dataset_path"]:
        external = external_cond["EM"]
    else:
        external = None
    if external is not None:
        cond, points_per_layer, energy_per_layer = get_from_basic(
            config, external, n_events
        )
        model_kwargs = {
            "cond": cond,
            "points_per_layer": points_per_layer,
            "energy_per_layer": energy_per_layer,
            "rescale_energy": True,
        }
        summarise.complete_model(model_path, n_events, force=True, **model_kwargs)
    with open(os.path.join(folder_path, "last_best_seen.txt"), "w") as f:
        f.write(model_path)
    print("Done")
