import sys
from src.evaluation import summarise, inference

model_path = sys.argv[1]

n_events = 1000
print(f"Summarising {n_events} events")
# is batched and can do 1k events
summary = summarise.ModelSummary.from_model_path(
    model_path, data_part="test", total_size=n_events
)

del summary

ema_model_path = model_path.replace("_model.pt", "_ema_model.pt")
# is batched and can do 1k events
summary = summarise.ModelSummary.from_model_path(
    ema_model_path, data_part="test", total_size=n_events
)

del summary

# runs fine at 1k events, cant do 10k events
summary = summarise.ReferenceSummary.from_model_path(
        model_path, data_part="test", total_size=n_events
)
