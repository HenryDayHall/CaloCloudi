import argparse
from src.evaluation.example_events import plot_example_events

parser = argparse.ArgumentParser(
    description="Plot example events from reference and model."
)
parser.add_argument("model_path", type=str, help="Path to the model checkpoint.")
parser.add_argument(
    "--event_index",
    type=int,
    default=None,
    help="Index of the event to display (default random).",
)
parser.add_argument(
    "--level",
    type=str,
    default="data",
    choices=["data", "physical", "cells"],
    help="Representation level (default 'data').",
)
parser.add_argument(
    "--data_part",
    type=str,
    default="test",
    help="Dataset partition (default 'test').",
)
parser.add_argument(
    "--total_size",
    type=int,
    default=1000,
    help="Number of events to load (default 1000).",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path to save the figure (default: show interactively).",
)
args = parser.parse_args()


plot_example_events(
    model_path=args.model_path,
    event_index=args.event_index,
    level=args.level,
    data_part=args.data_part,
    output_path=args.output,
)
