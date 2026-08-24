from src.evaluation.timing import run_time
import sys
import os
import torch

allocated_time = int(sys.argv[1])
config_path = sys.argv[2]
n_events = 1000
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Timing {config_path}, with {n_events} events on {device}")
if os.path.exists(sys.argv[3]):
    compiled_path = sys.argv[3]
    print(f"Using compiled path {compiled_path}")
    version = "compiled"
else:
    compiled_path = None
    version = sys.argv[3]
batch_size = 16
runs_per_cond = 10
print(f"batch_size: {batch_size}, runs_per_cond: {runs_per_cond}, version: {version}")

run_time(allocated_time, config_path, n_events, device, compiled_path, version, batch_size, runs_per_cond)
