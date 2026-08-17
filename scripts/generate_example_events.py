import sys
from src.evaluation.example_events import plot_and_save

model_path = sys.argv[1]

if len(sys.argv) > 2:
    event_indices = [int(i) for i in sys.argv[2:]]
else:
    event_indices = [0, 10, 100, 1000]

plot_and_save(model_path, event_indices)
