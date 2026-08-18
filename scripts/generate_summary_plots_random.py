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
    log_dir = os.path.join(default_conf["output_path"], "logs")


def search():
    to_do = []
    date_stamp = []
    optimiser_glob = os.path.join(log_dir, "*/checkpoints/*_optimiser.pt")
    optimiser_locations = glob.glob(optimiser_glob)
    for opt in optimiser_locations:
        folder = os.path.dirname(os.path.dirname(opt))
        if os.path.exists(os.path.join(folder, "no_summaries")):
            continue
        model_location = opt[: -len("_optimiser.pt")] + "_model.pt"
        summary_location = model_location[:-3] + "_summary.npz"
        if os.path.exists(model_location) and not os.path.exists(summary_location):
            to_do.append(model_location)
            base_name = os.path.basename(model_location)
            date_stamp.append("_".join(base_name.split("_")[:2]))

    print(f"Found {len(to_do)} models to process")
    return to_do, date_stamp


while True:
    to_do, date_stamp = search()
    if not to_do:
        print("No more models to process")
        break
    reverse_date_order = np.argsort(date_stamp)
    # random_order = np.random.permutation(len(to_do))
    model_path = to_do.pop(reverse_date_order[-1])
    model_date = date_stamp.pop(reverse_date_order[-1])
    print(f"Processing {model_path}")
    summarise.complete_model(model_path, n_events)
    example_events.plot_and_save(model_path, [100])
    print("Done")
