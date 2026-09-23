import sys
import yaml
import h5py
import numpy as np
from src.evaluation.inference import physical_to_cells
from src.evaluation.summarise import ModelSummary
from src.detector_map import find_layers

calorimeter_surface_y = 1804.7
def pick_gun_position(pxyz, calorimeter_surface_y=calorimeter_surface_y):
    distance_to_surface = calorimeter_surface_y / pxyz[:, 1]
    gun_pos = distance_to_surface[:, None] * pxyz
    return gun_pos

cond_file = "config/padded_photons.yaml"
data_file = "/data/dust/user/dayhallh/data/CaloClouds_diffusion/CC3_showers.h5"
#cond_file = sys.argv[1]
#data_file = sys.argv[2]
if len(sys.argv) > 3:
    output_file = sys.argv[3]
else:
    output_file = data_file[:-3] + "_external_cond_rescaled_energy_summary.h5"

with open(cond_file, "r") as f:
    config = yaml.safe_load(f)

print(f"Loading data from {data_file}")
#n_events = 1000
n_events = 56256
with h5py.File(data_file, "r") as f:
    cc3_showers = np.array(f["caloclouds3_showers"][:n_events])
    energy = np.array(f["energy"])[:n_events, None]
    direction = np.array(f["input_p_norm_global"][:n_events])

gun_pos = pick_gun_position(direction)
# convert direction to local, as it appears that way in cond
direction = direction[:, [2, 0, 1]]
print("Concatenating energy and direction")
cond = np.concatenate((energy, direction), axis=1)

mask = cc3_showers[:, :, 3] > 0
print(cc3_showers[mask].mean(0))
cc3_showers = cc3_showers[:, :, [1, 2, 0, 3]]
print(cc3_showers[mask].mean(0))
Xmean, Ymean = -0.0074305227, -0.21205868

cc3_showers[:, :, 0] -= gun_pos[:, None, 0]
cc3_showers[:, :, 0] -= Ymean
cc3_showers[:, :, 2] -= gun_pos[:, None, 2]
cc3_showers[:, :, 2] -= Xmean

print(cc3_showers[mask].mean(0))
print("Finding layers")
point_layer_ids = find_layers(config, cc3_showers, coordinates="detector")

print("Converting to cells")
cells = physical_to_cells(cc3_showers, point_layer_ids, config)

print("making summary")
summary = ModelSummary(config, sample_cells=cells, cond=cond, output_path=output_file)
print("Done")
