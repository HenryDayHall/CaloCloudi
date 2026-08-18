"""Tests for ``src/evaluation/model_kind.py``.

No torch and no real checkpoints: the module never opens a ``.pt`` file, it
only reads names and two small text files, so the fixtures write empty files.

The point being pinned down is that a checkpoint's kind comes from its run and
nothing else.  A teacher state dict and a student state dict are
interchangeable -- ``training/student.py`` loads one into the other with a
strict ``load_state_dict`` -- so if these three sources disagree with reality
there is nothing further to fall back on.
"""

import os
import warnings

import pytest

yaml = pytest.importorskip("yaml")

from src.evaluation import model_kind  # noqa: E402


TIMESTAMP = "2026-08-18_16-40-00"

# what each training script hands to Logger.checkpoint_model
TEACHER_ROLES = ["model", "ema_model", "ema_sched", "optimiser", "scheduler"]
STUDENT_ROLES = ["model", "ema_model", "teacher_model", "optimiser"]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def run_dir(tmp_path):
    """Factory: writes a log directory that looks like a finished run.

    Returns the path of the ``<timestamp>_model.pt`` checkpoint in it.
    """

    def _build(
        roles=None,
        run_info=None,
        logs="",
        timestamp=TIMESTAMP,
        name="run",
    ):
        log_dir = tmp_path / name
        checkpoints = log_dir / "checkpoints"
        checkpoints.mkdir(parents=True)
        (log_dir / "config.yaml").write_text("device: cpu\n")
        if logs:
            (log_dir / "logs.txt").write_text(logs)
        if run_info is not None:
            (log_dir / model_kind.RUN_INFO_NAME).write_text(yaml.dump(run_info))
        for role in roles or []:
            (checkpoints / f"{timestamp}_{role}.pt").write_bytes(b"")
        return str(checkpoints / f"{timestamp}_model.pt")

    return _build


def sibling(model_path, role):
    """The path of another role from the same checkpoint."""
    directory = os.path.dirname(model_path)
    return os.path.join(directory, f"{TIMESTAMP}_{role}.pt")


# --------------------------------------------------------------------------- #
# name parsing
# --------------------------------------------------------------------------- #
class TestCheckpointRole:
    @pytest.mark.parametrize(
        "name, expected",
        [
            (f"{TIMESTAMP}_model.pt", "model"),
            (f"{TIMESTAMP}_ema_model.pt", "ema_model"),
            (f"{TIMESTAMP}_teacher_model.pt", "teacher_model"),
        ],
    )
    def test_the_longest_matching_role_wins(self, name, expected):
        """``..._ema_model.pt`` also ends in ``_model``; the order in ``ROLES``
        is what stops it being read as the online model."""
        assert model_kind.checkpoint_role(name) == expected

    def test_an_unknown_name_has_no_role(self):
        assert model_kind.checkpoint_role("weights.pt") is None

    def test_the_timestamp_comes_back_without_the_role(self):
        assert (
            model_kind.checkpoint_timestamp(f"{TIMESTAMP}_ema_model.pt") == TIMESTAMP
        )

    def test_a_full_path_is_fine(self):
        path = f"/logs/2026/checkpoints/{TIMESTAMP}_teacher_model.pt"

        assert model_kind.checkpoint_role(path) == "teacher_model"


# --------------------------------------------------------------------------- #
# run_info.yaml
# --------------------------------------------------------------------------- #
class TestRunInfo:
    def test_a_student_run_info_is_believed(self, run_dir):
        path = run_dir(roles=STUDENT_ROLES, run_info={"run_type": "student"})

        kind = model_kind.describe(path)

        assert kind.run_type == "student"
        assert kind.distilled is True
        assert kind.source == model_kind.RUN_INFO_NAME

    def test_a_teacher_run_info_is_believed(self, run_dir):
        path = run_dir(roles=TEACHER_ROLES, run_info={"run_type": "teacher"})

        assert model_kind.describe(path).distilled is False

    def test_run_info_beats_the_siblings(self, run_dir):
        """A directory of teacher-shaped checkpoints that says it is a student
        run is taken at its word: the file was written deliberately."""
        path = run_dir(roles=TEACHER_ROLES, run_info={"run_type": "student"})

        assert model_kind.describe(path).distilled is True

    def test_a_run_may_name_its_own_distilled_roles(self, run_dir):
        path = run_dir(
            roles=STUDENT_ROLES,
            run_info={"run_type": "student", "distilled_roles": ["ema_model"]},
        )

        assert model_kind.describe(path).distilled is False
        assert model_kind.describe(sibling(path, "ema_model")).distilled is True

    def test_junk_in_run_info_falls_through(self, run_dir):
        path = run_dir(roles=STUDENT_ROLES, run_info={"run_type": "banana"})

        kind = model_kind.describe(path)

        assert kind.run_type == "student"  # from the siblings instead
        assert kind.source != model_kind.RUN_INFO_NAME


# --------------------------------------------------------------------------- #
# sibling checkpoints, the retrospective route
# --------------------------------------------------------------------------- #
class TestSiblings:
    def test_a_teacher_model_beside_it_means_a_student_run(self, run_dir):
        path = run_dir(roles=STUDENT_ROLES)

        kind = model_kind.describe(path)

        assert kind.run_type == "student"
        assert kind.distilled is True
        assert kind.confident is True

    def test_a_scheduler_beside_it_means_a_teacher_run(self, run_dir):
        path = run_dir(roles=TEACHER_ROLES)

        assert model_kind.describe(path).run_type == "teacher"

    def test_the_teacher_inside_a_student_run_is_not_distilled(self, run_dir):
        """``teacher_model.pt`` is the frozen score function the student was
        trained against, so it needs the multi step sampler even though every
        other checkpoint in the directory is a consistency model."""
        path = run_dir(roles=STUDENT_ROLES)

        kind = model_kind.describe(sibling(path, "teacher_model"))

        assert kind.run_type == "student"
        assert kind.distilled is False

    def test_the_target_model_of_a_student_run_is_distilled(self, run_dir):
        path = run_dir(roles=STUDENT_ROLES)

        assert model_kind.describe(sibling(path, "ema_model")).distilled is True

    def test_only_the_matching_timestamp_is_read(self, run_dir):
        """Older checkpoints in the same directory are from the same run, but
        restricting to one timestamp keeps a half written directory honest."""
        path = run_dir(roles=STUDENT_ROLES)
        directory = os.path.dirname(path)
        for role in TEACHER_ROLES:
            open(os.path.join(directory, f"2020-01-01_00-00-00_{role}.pt"), "wb").close()

        assert model_kind.describe(path).run_type == "student"


# --------------------------------------------------------------------------- #
# logs.txt, the last resort
# --------------------------------------------------------------------------- #
class TestLogs:
    def test_the_student_log_line_is_enough(self, run_dir):
        path = run_dir(
            roles=["model", "ema_model"],
            logs="16-40-00: Teacher model loaded from /somewhere/model.pt\n",
        )

        kind = model_kind.describe(path)

        assert kind.run_type == "student"
        assert kind.source == "logs.txt"

    def test_a_teacher_log_says_nothing_and_falls_back(self, run_dir):
        path = run_dir(
            roles=["model", "ema_model"],
            logs="16-40-00: Training from scratch > creating initial checkpoint\n",
        )

        with pytest.warns(RuntimeWarning, match="Cannot tell"):
            kind = model_kind.describe(path)

        assert kind.run_type == "teacher"
        assert kind.confident is False


# --------------------------------------------------------------------------- #
# giving up, and overriding
# --------------------------------------------------------------------------- #
class TestFallback:
    def test_nothing_on_disk_warns_and_assumes_a_teacher(self, run_dir):
        path = run_dir(roles=["model", "ema_model"])

        with pytest.warns(RuntimeWarning, match="Cannot tell"):
            kind = model_kind.describe(path)

        assert kind.distilled is False
        assert kind.confident is False

    def test_the_warning_can_be_turned_off(self, run_dir):
        path = run_dir(roles=["model", "ema_model"])

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            model_kind.describe(path, warn=False)

        assert not [w for w in caught if issubclass(w.category, RuntimeWarning)]

    @pytest.mark.parametrize("distilled", [True, False])
    def test_an_explicit_answer_skips_detection_entirely(self, tmp_path, distilled):
        """No run directory at all: a checkpoint copied somewhere else can
        still be evaluated by saying what it is."""
        path = str(tmp_path / "somebody_elses_model.pt")

        kind = model_kind.describe(path, distilled=distilled)

        assert kind.distilled is distilled
        assert kind.source == "given by the caller"

    def test_an_explicit_answer_beats_the_run_directory(self, run_dir):
        path = run_dir(roles=STUDENT_ROLES, run_info={"run_type": "student"})

        assert model_kind.describe(path, distilled=False).distilled is False


# --------------------------------------------------------------------------- #
# the line the scripts print
# --------------------------------------------------------------------------- #
class TestOneLine:
    def test_a_student_says_one_step(self, run_dir):
        path = run_dir(roles=STUDENT_ROLES)

        line = model_kind.describe(path).one_line({"num_steps": 40})

        assert "one step" in line
        assert os.path.basename(path) in line

    def test_a_teacher_says_how_many_steps_and_which_sampler(self, run_dir):
        path = run_dir(roles=TEACHER_ROLES)

        line = model_kind.describe(path).one_line({"num_steps": 40, "sampler": "heun"})

        assert "40 steps" in line
        assert "heun" in line

    def test_a_guess_is_marked_as_one(self, run_dir):
        path = run_dir(roles=["model", "ema_model"])

        line = model_kind.describe(path, warn=False).one_line()

        assert "GUESSED" in line

    def test_num_sampling_steps_is_one_for_a_student(self, run_dir):
        path = run_dir(roles=STUDENT_ROLES)

        assert model_kind.describe(path).num_sampling_steps() == 1
