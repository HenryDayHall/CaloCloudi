import sys
from src.evaluation import summarise

model_path = sys.argv[1]

n_events = 1000

print(f"Summarising {n_events} events")
summarise.complete_model(model_path, n_events, force=True)
