import sys
import os
import numpy as np
from src.evaluation import inference
from src.data.read_write import event_idxs_in_part
import h5py

model_path = sys.argv[1]
model_name = os.path.basename(model_path)
model_dir = os.path.dirname(model_path)
output_path = os.path.join(model_dir, model_name.split(".")[0] + "_testset_wEnergy.npy")

sampler = inference.Sampler.from_model_path(model_path)
n_events = 100
print(f"Sampling {n_events} events")

if len(sys.argv) > 2:
    print("Will read the cond, "
          f"points_per_layer and energy_per_layer from {sys.argv[2]}")
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
            pcfm[keys_on_disk[k]][:n_events]
            for k in sampler.config["model"]["cond_features"]
        ]
        cond = np.concatenate(cond, axis=1)
        cond = inference.pdg_to_onehot_in_full_cond(sampler.config, cond)
    # Using pcFM's counts,
    # summing the points across the layers to get the per event total
    sample = sampler.sample(cond, points_per_layer.sum(1))
else:  # will take from test set
    print("Sampling from test set")
    test_file_sizes, _ = event_idxs_in_part(sampler.config, "test")
    total_test_size = np.sum(test_file_sizes)
    print(f"Total test size: {total_test_size}")
    cond, points, target, sample = sampler.sample_from_dataset(
        "test", return_target=True, total_size=n_events
    )
    print(f"Sample shape: {sample.shape}")
    del points
    print("Getting points per layer from target")
    points_per_layer, energy_per_layer = inference.points_per_layer_from_target(
        target, sampler.config, return_energy=True
    )
    # don't need the target anymore
    del target
    print(f"Max points per layer: {np.max(points_per_layer)}")


print(f"Sample shape: {sample.shape}")
np.save(output_path, sample)

# don't need this anymore as we get the points per layer from the dataset

print("Converting to physical points")
physical_points, point_layer_ids = inference.sample_to_physical(
    sample, points_per_layer, sampler.config
)

# Rescaling the energy per layer to match the information the pcFM gives
energies = physical_points[:, :, 3].copy()
diffusion_sum = np.zeros_like(energy_per_layer)

for layer in range(energy_per_layer.shape[1]):
    mask = point_layer_ids == layer
    diffusion_sum[:, layer] = (energies * mask).sum(axis=1)

ratio = np.divide(
    energy_per_layer,
    diffusion_sum,
    out=np.zeros_like(energy_per_layer),
    where=diffusion_sum != 0,
)
scale_per_point = np.take_along_axis(ratio, np.clip(point_layer_ids, 0, None), axis=1)
scale_per_point[point_layer_ids < 0] = 1.0

physical_points[:, :, 3] = energies * scale_per_point


print("Unshifting points")
physical_points = inference.unshift_points(
    physical_points, point_layer_ids, cond, sampler.config
)
np.save(output_path.replace(".npy", "_physical.npy"), physical_points)
del cond

print("Converting to cells")
cells = inference.physical_to_cells(physical_points, point_layer_ids, sampler.config)
np.save(output_path.replace(".npy", "_cells.npy"), cells)
print(f"Cell shape: {cells.shape}")

print("Done")
