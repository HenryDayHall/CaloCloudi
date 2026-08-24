import sys
from src.evaluation import summarise

model_path = sys.argv[1]

n_events = 1000

summarise.ReferenceSummary.from_model_path(
    model_path, data_part="test", total_size=n_events
)
