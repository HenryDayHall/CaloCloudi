"""Tell a teacher checkpoint apart from a distilled student checkpoint.

A distilled model is not a different network.  ``Diffusion.__init__`` uses its
``distillation`` argument only to choose which preconditioning the ``Denoiser``
applies (``get_scalings_for_boundary_condition`` rather than ``get_scalings``)
and which branch ``Diffusion.sample`` takes (one call at ``sigma_max`` rather
than a Karras schedule).  ``Denoiser.__init__`` keeps ``distillation`` and
``sigma_min`` as plain attributes, and registers only ``sigma_data`` as a
buffer, so a teacher state dict and a student state dict have exactly the same
keys and the same shapes.  ``training/student.py`` relies on that:
``init_from_scratch`` loads a teacher state dict straight into a distilled
student with a strict ``load_state_dict``.

Two consequences, and they are the reason this module exists:

* the kind of a checkpoint cannot be recovered from the weights, and
* loading a student with ``distillation=False`` does not raise.  It quietly
  runs a consistency model through a multi step ODE solver and returns
  nonsense, which is the failure the evaluation scripts have been hitting.

The kind therefore has to come from the run that wrote the file.  Three
sources are tried, in descending order of trustworthiness:

1. ``run_info.yaml`` in the log directory, written by ``training.utils.Logger``
   when a run starts.  Authoritative, but only present on runs started after
   that logging was added.
2. The sibling checkpoints written beside it.  A student run checkpoints a
   ``teacher_model``; a teacher run checkpoints a ``scheduler`` and an
   ``ema_sched``.  This works on runs made before ``run_info.yaml`` existed,
   which is what makes the existing checkpoints usable.
3. The text ``training/student.py`` writes into ``logs.txt``.

If none of them answers, ``run_type`` warns and falls back to ``"teacher"``,
which is what every evaluation script silently assumed before this module.
"""

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import yaml


RUN_INFO_NAME = "run_info.yaml"

TEACHER = "teacher"
STUDENT = "student"

# Checkpoint roles, longest suffix first so "..._ema_model.pt" and
# "..._teacher_model.pt" are not both read as "..._model.pt".
ROLES = ("teacher_model", "ema_model", "model")

# In a student run these two roles hold the consistency (distilled) network.
# The third, "teacher_model", is the frozen teacher and is not distilled.
DISTILLED_ROLES = ("model", "ema_model")

# Checkpoint roles only one kind of run ever writes.  A student run
# checkpoints model/ema_model/teacher_model/optimiser; a teacher run
# checkpoints model/ema_model/ema_sched/optimiser/scheduler.
STUDENT_MARKERS = ("teacher_model",)
TEACHER_MARKERS = ("scheduler", "ema_sched")

# The line training/student.py writes into logs.txt, and nothing else does.
STUDENT_LOG_PHRASE = "Teacher model loaded from"


@dataclass
class ModelKind:
    """What a checkpoint is, and how it should be sampled.

    Attributes
    ----------
    model_path : str
        The checkpoint this describes.
    run_type : str
        ``"teacher"`` or ``"student"``, the kind of run that wrote the file.
    role : str or None
        ``"model"``, ``"ema_model"`` or ``"teacher_model"``, taken from the
        file name.  ``None`` when the name follows no known pattern.
    distilled : bool
        What to pass as ``Diffusion(config, distillation=...)``.
    source : str
        Where the answer came from, for logging.
    confident : bool
        False when nothing on disk answered and the teacher default was used.
    """

    model_path: str
    run_type: str
    role: Optional[str]
    distilled: bool
    source: str
    confident: bool = True
    extra: dict = field(default_factory=dict)

    def num_sampling_steps(self, config=None):
        """Steps ``Diffusion.sample`` will take, or None if unknown.

        A distilled model always takes one; a teacher takes ``num_steps`` from
        its config.
        """
        if self.distilled:
            return 1
        if config is None:
            return None
        return config.get("num_steps")

    def one_line(self, config=None):
        """A single line fit for printing at the top of a script."""
        name = os.path.basename(self.model_path)
        if self.distilled:
            how = "distilled student, sampled in one step"
        else:
            steps = self.num_sampling_steps(config)
            sampler = None if config is None else config.get("sampler")
            how = "teacher"
            if steps is not None:
                how += f", sampled in {steps} steps"
            if sampler is not None:
                how += f" with {sampler}"
        flag = "" if self.confident else " -- GUESSED"
        return f"{name}: {how} ({self.source}){flag}"


def log_dir_from_model_path(model_path):
    """``<log_dir>/checkpoints/<file>.pt`` -> ``<log_dir>``.

    The same two ``dirname`` calls ``Sampler.get_config_from_model_path`` and
    ``Logger.from_model_path`` make, kept in one place.
    """
    return os.path.dirname(os.path.dirname(model_path))


def checkpoint_role(model_path):
    """The role in ``<timestamp>_<role>.pt``, or None if unrecognised."""
    stem = os.path.splitext(os.path.basename(model_path))[0]
    for role in ROLES:
        if stem == role or stem.endswith("_" + role):
            return role
    return None


def checkpoint_timestamp(model_path):
    """The ``<timestamp>`` half of ``<timestamp>_<role>.pt``, or None."""
    role = checkpoint_role(model_path)
    if role is None:
        return None
    stem = os.path.splitext(os.path.basename(model_path))[0]
    if stem == role:
        return None
    return stem[: -(len(role) + 1)]


def read_run_info(model_path):
    """The contents of ``run_info.yaml`` for this checkpoint, or None."""
    path = os.path.join(log_dir_from_model_path(model_path), RUN_INFO_NAME)
    if not os.path.exists(path):
        return None
    with open(path, "r") as handle:
        info = yaml.safe_load(handle)
    if not isinstance(info, dict):
        return None
    return info


def _from_run_info(model_path):
    info = read_run_info(model_path)
    if info is None:
        return None
    run_type = info.get("run_type")
    if run_type in (TEACHER, STUDENT):
        return run_type, RUN_INFO_NAME, info
    return None


def _sibling_roles(model_path):
    """Role suffixes of the ``.pt`` files written beside this checkpoint.

    Restricted to the same timestamp when the name allows it, so a directory
    holding several checkpoints is read one checkpoint at a time.
    """
    checkpoint_dir = os.path.dirname(model_path)
    if not os.path.isdir(checkpoint_dir):
        return set()
    stems = [
        os.path.splitext(name)[0]
        for name in os.listdir(checkpoint_dir)
        if name.endswith(".pt")
    ]
    timestamp = checkpoint_timestamp(model_path)
    if timestamp is not None:
        same_run = [stem for stem in stems if stem.startswith(timestamp + "_")]
        if same_run:
            stems = same_run
    roles = set()
    for marker in STUDENT_MARKERS + TEACHER_MARKERS + ROLES:
        if any(stem == marker or stem.endswith("_" + marker) for stem in stems):
            roles.add(marker)
    return roles


def _from_siblings(model_path):
    roles = _sibling_roles(model_path)
    student = any(marker in roles for marker in STUDENT_MARKERS)
    teacher = any(marker in roles for marker in TEACHER_MARKERS)
    if student and not teacher:
        return STUDENT, "the checkpoints written beside it", None
    if teacher and not student:
        return TEACHER, "the checkpoints written beside it", None
    return None


def _from_logs(model_path):
    path = os.path.join(log_dir_from_model_path(model_path), "logs.txt")
    if not os.path.exists(path):
        return None
    with open(path, "r") as handle:
        text = handle.read()
    if STUDENT_LOG_PHRASE in text:
        return STUDENT, "logs.txt", None
    return None


def run_type(model_path, default=TEACHER, warn=True):
    """``"teacher"`` or ``"student"``, from whichever source answers first."""
    found = _run_type_with_source(model_path, default=default, warn=warn)
    return found[0]


def _run_type_with_source(model_path, default=TEACHER, warn=True):
    """(run_type, source, run_info or None, confident)."""
    for lookup in (_from_run_info, _from_siblings, _from_logs):
        found = lookup(model_path)
        if found is not None:
            return found[0], found[1], found[2], True
    if warn:
        warnings.warn(
            f"Cannot tell whether {model_path} came from a teacher or a "
            f"student run: no {RUN_INFO_NAME}, no telltale sibling "
            "checkpoints and nothing in logs.txt. Assuming "
            f"{default!r}. Pass distilled=True/False explicitly if that is "
            "wrong -- a student run through the teacher sampler returns "
            "nonsense rather than failing.",
            RuntimeWarning,
            stacklevel=3,
        )
    return default, "assumed, nothing on disk said", None, False


def describe(model_path, distilled=None, warn=True):
    """Everything known about a checkpoint, without opening it.

    Parameters
    ----------
    model_path : str
        Path to a ``.pt`` checkpoint inside a run's ``checkpoints`` directory.
    distilled : bool or None, optional
        Skip detection and take this as the answer.  For checkpoints whose run
        directory has been split up, or to force a re-run of an old result.
    warn : bool, optional
        Warn when nothing on disk answers (default True).

    Returns
    -------
    ModelKind
    """
    role = checkpoint_role(model_path)
    if distilled is not None:
        return ModelKind(
            model_path=model_path,
            run_type=STUDENT if distilled else TEACHER,
            role=role,
            distilled=bool(distilled),
            source="given by the caller",
        )

    kind, source, info, confident = _run_type_with_source(model_path, warn=warn)

    # A run may name its own distilled roles; fall back to the convention that
    # a student run distils everything except the teacher it was given.
    distilled_roles = None
    if info is not None:
        distilled_roles = info.get("distilled_roles")
    if distilled_roles is None:
        distilled_roles = DISTILLED_ROLES if kind == STUDENT else ()

    if role is None and warn:
        warnings.warn(
            f"{os.path.basename(model_path)} is not named "
            "<timestamp>_<role>.pt, so its role in the run is unknown; "
            "treating it as not distilled.",
            RuntimeWarning,
            stacklevel=2,
        )

    return ModelKind(
        model_path=model_path,
        run_type=kind,
        role=role,
        distilled=role in distilled_roles,
        source=source,
        confident=confident,
        extra=info or {},
    )


def is_distilled(model_path, warn=True):
    """True when this checkpoint needs ``Diffusion(config, distillation=True)``."""
    return describe(model_path, warn=warn).distilled
