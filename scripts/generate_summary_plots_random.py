import os
import sys
import glob
import yaml
import numpy as np
from src.evaluation import summarise, example_events

n_events = 1000

if len(sys.argv) > 1:
    log_dir = sys.argv[1]
else:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    default_conf_path = os.path.join(this_dir, "../config/default.yaml")
    default_conf = yaml.safe_load(open(default_conf_path))
    log_dir = default_conf["output_path"]

to_do = []
date_stamp = []
optimiser_locations = glob.glob(os.path.join(log_dir, "*/checkpoints/*_optimiser.pt"))
for opt in optimiser_locations:
    model_location = opt[: -len("_optimiser.pt")] + "_model.pt"
    summary_location = model_location[:-3] + "_summary.npz"
    if os.path.exists(model_location) and not os.path.exists(summary_location):
        to_do.append(model_location)
        base_name = os.path.basename(model_location)
        date_stamp.append("_".join(base_name.split("_")[:2]))

print(f"Found {len(to_do)} models to process")
reverse_date_order = np.argsort(date_stamp)
# random_order = np.random.permutation(len(to_do))

for j, i in enumerate(reverse_date_order):
    model_path = to_do[i]
    print(f"Processing {model_path}")
    summarise.complete_model(model_path, n_events)
    example_events.plot_and_save(model_path, [100])
    print(f"Done {j}/{len(to_do)}")
