import os
import sys
import glob
import yaml
import numpy as np
from src.evaluation import summarise, example_events
from src.training.best_checkpoint import find_best_checkpoint

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
    summarise.complete_model(model_path, n_events)
    summarise.complete_model(model_path, n_events, rescale_energy=True)
    example_events.plot_and_save(model_path, [100])
    folder_path = os.path.dirname(os.path.dirname(model_path))
    with open(os.path.join(folder_path, "last_best_seen.txt"), "w") as f:
        f.write(model_path)
    print("Done")
