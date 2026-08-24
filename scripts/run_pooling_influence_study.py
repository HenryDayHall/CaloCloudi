"""Measure which branch the pooling stack listens to, against the conditioning.

Usage
-----
    python -m scripts.run_pooling_influence_study <checkpoint.pt> [options]

Writes ``<checkpoint>_pooling_influence.npz`` and a set of figures beside the
checkpoint.  See ``src/evaluation/pooling_influence.py`` for what the arms mean
and ``docs/pooling_influence_study.md`` for how to read the output.
"""
import argparse
import os

import matplotlib

from src.evaluation import pooling_influence


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path", help="a .pt checkpoint in a run's checkpoints dir")
    parser.add_argument("--data-part", default="test", choices=["train", "val", "test"])
    parser.add_argument("--n-events", type=int, default=256,
                        help="events read for the conditioning and the point pool")
    parser.add_argument("--n-probe-points", type=int, default=2000,
                        help="points in the shared probe used by the sweeps")
    parser.add_argument("--n-event-points", type=int, default=256,
                        help="points per event in the observational arm")
    parser.add_argument("--n-grid", type=int, default=9,
                        help="steps per swept conditioning feature")
    parser.add_argument("--n-sigma", type=int, default=6,
                        help="noise levels, taken from the Karras schedule")
    parser.add_argument("--n-placebo", type=int, default=20,
                        help="matched random-direction draws for the null; 0 to skip")
    parser.add_argument("--features", nargs="*", default=None,
                        help="conditioning features to sweep (default: all)")
    parser.add_argument("--arms", nargs="*", default=["frozen", "coupled", "observational"],
                        choices=["frozen", "coupled", "observational"])
    parser.add_argument("--no-shapley", action="store_true",
                        help="skip the exact Shapley values (2**n_branches passes)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default=None, help="npz path (default: beside the checkpoint)")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    results = pooling_influence.run_study(
        args.model_path,
        data_part=args.data_part,
        n_events=args.n_events,
        n_probe_points=args.n_probe_points,
        n_event_points=args.n_event_points,
        features=args.features,
        n_grid=args.n_grid,
        n_sigma=args.n_sigma,
        arms=tuple(args.arms),
        with_shapley=not args.no_shapley,
        n_placebo=args.n_placebo,
        seed=args.seed,
    )

    output = args.output or pooling_influence.default_output_path(args.model_path)
    pooling_influence.save_study(results, output)
    print(f"wrote {output}")

    if args.no_plots:
        return

    stem = output[: -len(".npz")]
    for feature, entry in results["sweeps"].items():
        for arm in ("frozen", "coupled"):
            if arm not in entry:
                continue
            figure = pooling_influence.plot_sweep(results, feature, arm)
            path = f"{stem}_{feature}_{arm}.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            print(f"wrote {path}")
        arm = "frozen" if "frozen" in entry else "coupled"
        for name, maker in (
            ("sigma", pooling_influence.plot_sigma_heatmap),
            ("per_output", pooling_influence.plot_per_output),
        ):
            figure = maker(results, feature, arm)
            path = f"{stem}_{feature}_{name}.png"
            figure.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(figure)
            print(f"wrote {path}")
        offset = entry[arm]["offset_fraction"]
        print(f"  {feature}: context-only offset carries "
              f"{offset.min():.0%}-{offset.max():.0%} of the output")

    if "observational" in results:
        from src.evaluation.inference import Sampler

        config = Sampler.get_config_from_model_path(args.model_path)
        figure = pooling_influence.plot_observational(results, config)
        path = f"{stem}_observational.png"
        figure.savefig(path, dpi=150)
        plt.close(figure)
        print(f"wrote {path}")

    print(f"\nfigures are in {os.path.dirname(output)}")


if __name__ == "__main__":
    main()
