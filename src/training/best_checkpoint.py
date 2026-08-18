import numpy as np
import os
import time

from ..evaluation.model_kind import (
    checkpoint_role,
    checkpoint_timestamp,
    log_dir_from_model_path,
)

# --------------------------------------------------------------------------- #
# picking the best checkpoint of a run
#
# Nothing on disk ties a checkpoint to a validation.  ``checkpoint_model``
# names its files with the wall clock, ``do_validation`` records ``n_events``,
# and the two are written next to each other in the training loop but never
# cross referenced.  The bridge is the per step ``time`` log: a checkpoint's
# timestamp gives a wall clock, the nearest logged step gives the ``n_events``
# reached by then, and ``val_n_events`` gives the validation taken at that
# point in the run.  Checkpoints are written immediately before or after a
# validation with no training in between, so those two ``n_events`` usually
# match exactly; ``gap`` in the returned records says by how much they did not.
# --------------------------------------------------------------------------- #
CHECKPOINT_TIME_FORMAT = "%Y-%m-%d_%H-%M-%S"


def log_dir_from_path(path):
    """The log directory of a run, from any of the usual ways of naming it.

    Accepts the log directory itself, its ``checkpoints`` subdirectory, or the
    path of a checkpoint inside that.
    """
    path = os.path.normpath(path)
    if os.path.isdir(path):
        if os.path.isdir(os.path.join(path, "checkpoints")):
            return path
        if os.path.basename(path) == "checkpoints":
            return os.path.dirname(path)
        raise ValueError(
            f"{path} is a directory with no checkpoints in it; expected a log "
            "directory, its checkpoints directory, or a checkpoint file"
        )
    return log_dir_from_model_path(path)


def _checkpoints_with_role(checkpoint_dir, role):
    """``(timestamp, wall clock, path)`` for each checkpoint of one role.

    Sorted by time, and quietly skipping anything not named
    ``<timestamp>_<role>.pt`` with a timestamp this module can read.
    """
    if not os.path.isdir(checkpoint_dir):
        raise FileNotFoundError(f"No checkpoints directory at {checkpoint_dir}")
    found = []
    for name in os.listdir(checkpoint_dir):
        if not name.endswith(".pt") or checkpoint_role(name) != role:
            continue
        stamp = checkpoint_timestamp(name)
        if stamp is None:
            continue
        try:
            # strftime wrote it in local time, so mktime reads it back; the
            # hour repeated by a DST change is the one case this can misplace
            when = time.mktime(time.strptime(stamp, CHECKPOINT_TIME_FORMAT))
        except ValueError:
            continue
        found.append((stamp, when, os.path.join(checkpoint_dir, name)))
    return sorted(found, key=lambda item: item[1])


def _step_history(log_dir):
    """The per step ``time`` and ``n_events`` logs, as arrays."""
    step_time = np.load(os.path.join(log_dir, "time.npy"))
    step_n_events = np.load(os.path.join(log_dir, "n_events.npy"))
    length = min(len(step_time), len(step_n_events))
    if length == 0:
        raise ValueError(f"No training steps recorded in {log_dir}")
    return step_time[:length], step_n_events[:length]


def _n_events_at(step_time, step_n_events, when):
    """How far the run had got at a given wall clock time."""
    index = int(np.searchsorted(step_time, when))
    if index >= len(step_time):
        index = len(step_time) - 1
    elif index > 0 and (when - step_time[index - 1]) <= (step_time[index] - when):
        index -= 1
    return float(step_n_events[index])


def _validation_curve(log_dir, metric):
    """``(n_events, value)`` per validation, one number per validation.

    ``ValidationChecker.loss`` returns one number per validation batch, so the
    saved array is usually two dimensional and is averaged here.
    """
    values_path = os.path.join(log_dir, f"val_{metric}.npy")
    if not os.path.exists(values_path):
        available = sorted(
            name[len("val_") : -len(".npy")]
            for name in os.listdir(log_dir)
            if name.startswith("val_") and name.endswith(".npy")
        )
        raise FileNotFoundError(
            f"No {os.path.basename(values_path)} in {log_dir}; "
            f"validation values on disk: {available}"
        )
    values = np.load(values_path, allow_pickle=True)
    n_events = np.load(os.path.join(log_dir, "val_n_events.npy"))
    if values.dtype == object:
        # ragged, from validations run over different numbers of batches
        values = np.array([np.mean(item) for item in values], dtype=float)
    else:
        values = np.asarray(values, dtype=float)
        if values.ndim > 1:
            values = values.reshape(len(values), -1).mean(axis=1)
    # do_validation appends n_events before it runs the functions, so a run
    # killed mid validation leaves one more of those than of the values
    length = min(len(values), len(n_events))
    if length == 0:
        raise ValueError(f"No {metric} validations recorded in {log_dir}")
    return np.asarray(n_events[:length], dtype=float), values[:length]


def checkpoint_scores(path, role="model", metric="loss", tolerance=None):
    """Every checkpoint of one role in a run, with its validation value.

    Parameters
    ----------
    path : str
        A log directory, a checkpoints directory, or a checkpoint file; any of
        them identifies the run.
    role : str, optional
        Which of the checkpointed roles to score, ``"model"`` by default.
        ``"ema_model"`` is the one to evaluate a teacher run with, and is the
        target network of a student run.
    metric : str, optional
        The validation function whose values to read, i.e. the ``val_<metric>``
        files.  ``"loss"`` is the only one both training scripts register.
    tolerance : float or None, optional
        Drop checkpoints whose nearest validation is more than this many events
        away.  ``None``, the default, keeps them all however poor the match.

    Returns
    -------
    list of dict
        One record per checkpoint, in time order, with keys ``path``,
        ``timestamp``, ``n_events``, ``validation_n_events``, ``value`` and
        ``gap`` (the distance in events between the last two).
    """
    log_dir = log_dir_from_path(path)
    checkpoints = _checkpoints_with_role(os.path.join(log_dir, "checkpoints"), role)
    if not checkpoints:
        raise FileNotFoundError(
            f"No <timestamp>_{role}.pt checkpoints in {log_dir}/checkpoints"
        )
    val_n_events, val_values = _validation_curve(log_dir, metric)
    step_time, step_n_events = _step_history(log_dir)

    scored = []
    for stamp, when, checkpoint_path in checkpoints:
        n_events = _n_events_at(step_time, step_n_events, when)
        gaps = np.abs(val_n_events - n_events)
        # a validation triggered by the interval and one triggered by the
        # checkpoint land on the same n_events whenever the two intervals
        # divide; the later of them is the one written next to the file
        index = int(np.flatnonzero(gaps == gaps.min())[-1])
        if tolerance is not None and gaps[index] > tolerance:
            continue
        scored.append(
            {
                "path": checkpoint_path,
                "timestamp": stamp,
                "n_events": n_events,
                "validation_n_events": float(val_n_events[index]),
                "value": float(val_values[index]),
                "gap": float(gaps[index]),
            }
        )
    return scored


def find_best_checkpoint(
    path, role="model", metric="loss", tolerance=None, with_score=False
):
    """The checkpoint of a run with the lowest validation loss.

    Parameters
    ----------
    path : str
        A log directory, a checkpoints directory, or a checkpoint file from the
        run to search.  Passing one checkpoint asks about the run it came from,
        not about that file.
    role, metric, tolerance
        As for ``checkpoint_scores``.
    with_score : bool, optional
        Also return the record ``checkpoint_scores`` built for the winner, which
        carries the validation value chosen and how well it matched.

    Returns
    -------
    str, or (str, dict) when ``with_score``
        The path of the best checkpoint file.

    Notes
    -----
    Only checkpoints that were written are candidates, so this is the best of
    the run's saved states rather than the best moment of the run: with a long
    ``checkpoint_time_interval_min`` the lowest validation the log records may
    have happened between two checkpoints and no file holds those weights.
    """
    scored = checkpoint_scores(path, role=role, metric=metric, tolerance=tolerance)
    usable = [record for record in scored if np.isfinite(record["value"])]
    if not usable:
        raise ValueError(
            f"No checkpoint in {log_dir_from_path(path)} could be matched to a "
            f"finite {metric} validation"
        )
    best = min(usable, key=lambda record: record["value"])
    if with_score:
        return best["path"], best
    return best["path"]
