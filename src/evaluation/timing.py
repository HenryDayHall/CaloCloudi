"""
Functions to test the timing, both in python and after compilation
It tests only the timing of the diffusion itself,
not the timing of casting back to physical coordinates and cells.
Thus, this is insensitive to actual physics output,
so they don't need to use trained models.
"""
import numpy as np
import glob
import torch
import time
import gc
import os
import yaml
from ..data import read_write
from ..training.teacher import init_from_scratch as init_teacher
from ..training.student import init_from_scratch as init_student
from ..evaluation.inference import evaluating, pdg_to_onehot_in_full_cond


class TimingData:
    def __init__(self, config, tag, device, batch_size, version):
        self.config = config
        self.tag = tag
        data = {
            "tag": tag,
            "device": device,
            "batch_size": batch_size,
            "version": version,
        }
        self.config["timing"] = data

    def get_output_base(self):
        output_dir = self.config["output_path"]
        output_base_format = os.path.join(output_dir, f"times_{self.tag}_")
        i = 0
        n_zero_pad = 3
        padded_tail = str(i).zfill(n_zero_pad)
        while os.path.exists(output_base_format + padded_tail):
            i += 1
            padded_tail = str(i).zfill(n_zero_pad)
        output_base = output_base_format + padded_tail
        return output_base

    def save(self, cond, points, times):
        output_base = self.get_output_base()
        np.savez(output_base + ".npz", cond=cond, points=points, times=times)
        with open(output_base + ".yaml", "w") as f:
            yaml.dump(self.config, f)
        print(f"Saved to {output_base}*")


def get_cond_and_n_points(config, n_events):
    """
    While the cond itself shouldn't effect the timing,
    the number of points per layer does
    as it changes the max points to be drawn.
    Take these from a dataset path specified by config.
    """
    # Build the list of per‑event column names (conditioning + optional n_points)
    cond_features = config["model"]["cond_features"]
    per_event_cols = [config["data"][f"{name}_key"] for name in cond_features]
    n_points_key = config["data"].get("n_points_key")
    if n_points_key is not None:
        per_event_cols_with_npoints = per_event_cols + [n_points_key]
    else:
        per_event_cols_with_npoints = per_event_cols

    # Read the raw data
    per_event, target = read_write.read_raw_regaxes(
        config,
        part="test",
        total_size=n_events,
        per_event_cols=per_event_cols_with_npoints,
    )

    # Separate conditioning from the number of points (if present)
    if n_points_key is not None:
        points = per_event[:, -1]
        cond = per_event[:, :-1]
    else:
        real = target[:, :, 3] > 0
        points = real.sum(1)
        cond = per_event

    cond = pdg_to_onehot_in_full_cond(config, cond)

    return cond, points


def make_restrictions():
    """
    Restrict to 1 cpu thread, etc.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def make_fake_teacher(config_path, out_path="temp.pt"):
    setup = init_teacher(config_path)
    model = setup["model"]
    torch.save(model, out_path)
    return out_path


def time_python(
    time_allocated,
    config_path,
    version,
    cond,
    n_points,
    device,
    batch_size=16,
    runs_per_cond=10,
    n_warmup_runs=2,
):
    """
    Initialise the undistilled teacher or the distilled student,
    and time the inference on the given cond and points per layer.
    Timing should include the inference of the diffusion,
    not the conversion of the diffusion to physical coordinates and cells.
    """
    end_by = time.time() + time_allocated
    if version == "teacher":
        setup = init_teacher(config_path)
    elif version == "student":
        fake_teacher_path = make_fake_teacher(config_path)
        setup = init_student(config_path, fake_teacher_path)
    else:
        raise ValueError(f"Unknown version {version}, should be teacher or student")
    cond = torch.from_numpy(cond).to(device, dtype=torch.float32)
    preprocess_conditioning = setup["preprocess_conditioning"]
    processed_cond = preprocess_conditioning.forward(cond)
    model = setup["model"]

    # clean up memory
    del cond, preprocess_conditioning, setup
    gc.collect()

    n_points = torch.from_numpy(n_points).to(device, dtype=torch.int)
    times = np.zeros((n_points.shape[0], runs_per_cond))
    with evaluating(model):
        with torch.no_grad():
            for i, max_points in enumerate(n_points):
                print(f"{i/n_points.shape[0]:.2%}", end="\r")
                event_cond = processed_cond[[i]].repeat(batch_size, 1)
                for j in range(n_warmup_runs):
                    model.sample(event_cond, max_points)
                for j in range(runs_per_cond):
                    start = time.time()
                    model.sample(event_cond, max_points)
                    end = time.time()
                    times[i, j] = end - start
                if end > end_by:
                    break
    times = times[: i + 1]
    return times


def time_compiled_student(
    time_allocated,
    compiled_path,
    cond,
    n_points,
    device,
    batch_size=16,
    runs_per_cond=10,
    n_warmup_runs=2,
):
    """
    It's easier to give a compiled path, rather than create the compiled object.
    Then time the inference over the compiled object
    """
    end_by = time.time() + time_allocated
    module = torch.jit.load(compiled_path, map_location=device).eval()
    cond = torch.from_numpy(cond).to(device, dtype=torch.float32)
    n_points = torch.from_numpy(n_points).to(device, dtype=torch.int)

    n_points = n_points.to(device, dtype=torch.int)
    times = np.zeros((n_points.shape[0], runs_per_cond))
    with torch.no_grad():
        for i, max_points in enumerate(n_points):
            print(f"{i/n_points.shape[0]:.2%}", end="\r")
            event_cond = cond[[i]].repeat(batch_size, 1)
            noise = torch.randn(
                batch_size, max_points, 4, device=device, dtype=torch.float32
            )
            keep = n_points[[i]].repeat(batch_size, 1)
            for j in range(n_warmup_runs):
                module(event_cond, noise, keep)
            for j in range(runs_per_cond):
                start = time.time()
                module(event_cond, noise, keep)
                end = time.time()
                times[i, j] = end - start
            if end > end_by:
                break
    times = times[: i + 1]
    return times


def get_tag(config_path):
    run_dir = os.path.dirname(config_path)
    pottential_tag = glob.glob(os.path.join(run_dir, "_a_*"))
    if len(pottential_tag) > 0:
        return os.path.basename(pottential_tag[0])[3:]
    elif os.path.basename(config_path) != "config.yaml":
        return os.path.basename(config_path)[:-len(".yaml")]
    else:
        return run_dir.rstrip("/").split("/")[-1]


def run_time(
    time_allocated,
    config_path,
    n_events,
    device,
    compiled_path=None,
    version="student",
    batch_size=16,
    runs_per_cond=10,
    n_warmup_runs=2,
):
    time_allocated -= 5 * 60  # buffer for messing around with saves and loads
    tag = get_tag(config_path)
    if compiled_path is not None:
        assert version == "compiled"
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    time_data = TimingData(config, tag, device, batch_size, version)
    cond, points = get_cond_and_n_points(config, n_events)
    make_restrictions()
    if compiled_path is not None:
        times = time_compiled_student(
            time_allocated, compiled_path, cond, points, device
        )
    else:
        times = time_python(time_allocated, config_path, version, cond, points, device)
    print(f"Managed to time {len(times)}/{len(cond)}")
    time_data.save(cond[: len(times)], points[: len(times)], times)
