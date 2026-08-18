import sys
from src.evaluation import summarise, inference
import os
import numpy as np
import h5py

##Goal: read pcFM -> sample the diffusion model -> convert to physical points -> rescale energy to pcFM's per layer predictions -> bucket into detector cells

model_path = sys.argv[1]

#config and sampler
sampler = inference.Sampler.from_model_path(model_path)
config = sampler.config

#Getting the layer count from the config file
n_layers = len(config["data"]["layer_bottom_pos"])
print(f"Model expects {n_layers} layers")

n_events = 1000 #how many showers to summarise
print(f"Summarising {n_events} events")

#Reading in the file from pcFM
pcfm = h5py.File("/home/baggjemm/PointCountFM_private/data/gen1_photons.h5", "r")

points_per_layer = pcfm["observables/num_points_per_layer"][:n_events, :n_layers]
energy_per_layer = pcfm["observables/energy_per_layer"][:n_events, :n_layers]
cond = np.concatenate([pcfm["energies"][:], pcfm["directions"][:]], axis=1)[:n_events]  # (n_events, 4): (e, d_x, d_y, d_z)
pcfm.close()

batch_size = 32
n_batches = int(np.ceil(n_events / batch_size))
cells_list = []

for b in range(n_batches):
    start = b * batch_size
    end = min((b + 1) * batch_size, n_events)

    cond_b = cond[start:end]
    ppl_b = points_per_layer[start:end]
    epl_b = energy_per_layer[start:end]

    sample = sampler.sample(cond_b, ppl_b.sum(1))
    physical_points, point_layer_ids = inference.sample_to_physical(sample, ppl_b, config)
    del sample

    # energy rescale — same as before, but on the batch slices
    energies = physical_points[:, :, 3].copy()
    diffusion_sum = np.zeros_like(epl_b)
    for layer in range(epl_b.shape[1]):
        mask = point_layer_ids == layer
        diffusion_sum[:, layer] = (energies * mask).sum(axis=1)
    ratio = np.divide(epl_b, diffusion_sum, out=np.zeros_like(epl_b), where=diffusion_sum != 0)
    scale_per_point = np.take_along_axis(ratio, np.clip(point_layer_ids, 0, None), axis=1)
    scale_per_point[point_layer_ids < 0] = 1.0
    physical_points[:, :, 3] = energies * scale_per_point

    physical_points = inference.unshift_points(physical_points, point_layer_ids, cond_b, config)
    cells_b = inference.physical_to_cells(physical_points, point_layer_ids, config)
    cells_list.append(cells_b)
    del physical_points, point_layer_ids
    print(f"batch {b+1}/{n_batches} done, cells {cells_b.shape}")

# stack batches -> one array. pad to common width first, since physical_to_cells
# pads each batch to its own max, so batches have different widths in dim 1
max_cells = max(c.shape[1] for c in cells_list)
cells = np.concatenate(
    [np.pad(c, ((0, 0), (0, max_cells - c.shape[1]), (0, 0))) for c in cells_list],
    axis=0,
)
print(f"Cell shape: {cells.shape}")

output_path = summarise.ModelSummary.get_output_path_from_model_path(model_path)
summary = summarise.ModelSummary(config, sample_cells=cells, cond=cond, data_part="test", total_size=n_events, output_path=output_path)
del summary

# runs fine at 1k events, cant do 10k events
summary = summarise.ReferenceSummary.from_model_path(
        model_path, data_part="test", total_size=n_events
)
