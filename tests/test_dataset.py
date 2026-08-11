"""Tests for ``src/data/dataset.py`` -- padded / padded_unordered only.

``ShowerDataDataset`` is out of scope, so it only appears in the ``from_config``
dispatch tests, where it is mocked.

The pure helpers on ``AbstractBase`` are exercised against a lightweight stub
that carries just the attributes they read; everything below that is built from
real HDF5 files written by ``conftest.make_padded_file``, because most of this
module is about the shapes h5py hands back and those are not worth faking.

Tests carrying a docstring record current behaviour that looks unintended.
"""

import numpy as np
import pytest
import h5py

pytest.importorskip("torch")

import torch  # noqa: E402

from src.data import dataset  # noqa: E402

from conftest import base_config, make_padded_file  # noqa: E402


CLASSES = ["PointCloudDataset", "PointCloudDatasetUnordered", "ShowerDataDataset"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def replace_key(path, name, data):
    """Swap one dataset inside an existing h5 file for a differently shaped one."""
    with h5py.File(path, "a") as handle:
        del handle[name]
        handle.create_dataset(name, data=data)


def drop_key(path, name):
    with h5py.File(path, "a") as handle:
        del handle[name]


def gather(per_file, rows, key):
    """Expected per-event values for ``rows`` of an ``index_list``."""
    return np.array([per_file[file_n][key][event_n] for _, file_n, event_n in rows])


class Stub(dataset.AbstractBase):
    """Just enough state for the methods ``AbstractBase`` defines on its own."""

    def __init__(self, bs=4, length=100, config=None, offset=1.0):
        self.bs = bs
        self._len = length
        self.config = config
        self.offset = offset


def as_indices(idxs, length):
    """Normalise a slice or an index array to a list of ints."""
    if isinstance(idxs, slice):
        return list(range(*idxs.indices(length)))
    return list(idxs)


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def padded(tmp_path):
    """Factory: writes N padded files, returns ``(config, per_file, pattern)``.

    ``per_file[i]`` is the truth dict for file ``i``, so ``(file_idx, event_idx)``
    pairs out of an ``index_list`` can be looked up directly.
    """

    def _build(
        n_events=(6,),
        *,
        n_points=5,
        fmt="padded",
        roll_axis=False,
        padding="back",
        **kwargs,
    ):
        pattern = str(tmp_path / "data_{}.h5")
        per_file = [
            make_padded_file(
                pattern.format(i),
                n_events=n,
                n_points=n_points,
                roll_axis=roll_axis,
                seed=i,
                **kwargs,
            )
            for i, n in enumerate(n_events)
        ]
        config = base_config(pattern, fmt=fmt, roll_axis=roll_axis, padding=padding)
        config["data"]["train_range_start"] = 0
        config["data"]["train_range_end"] = len(n_events)
        return config, per_file, pattern

    return _build


@pytest.fixture
def build():
    """Factory that builds a dataset and closes its file handles afterwards.

    ``_open_data_files`` deliberately never closes anything, which is fine for a
    training run but leaks handles across a test session.
    """
    made = []

    def _build(config, part="train"):
        built = dataset.from_config(config, dataset_part=part)
        made.append(built)
        return built

    yield _build

    for built in made:
        for handle in getattr(built, "open_files", []):
            try:
                handle.close()
            except Exception:  # pragma: no cover - teardown must not fail a test
                pass


# --------------------------------------------------------------------------- #
# AbstractBase.get_n_points
# --------------------------------------------------------------------------- #
class TestGetNPoints:
    def test_counts_positive_energies_in_a_batch(self):
        events = np.zeros((3, 5, 4))
        events[0, :2, 3] = 1.0
        events[1, :5, 3] = 1.0

        counts = dataset.AbstractBase.get_n_points(events)

        assert list(counts) == [2, 5, 0]

    def test_works_on_a_single_event(self):
        event = np.zeros((5, 4))
        event[:3, 3] = 1.0

        assert dataset.AbstractBase.get_n_points(event) == 3

    def test_axis_selects_a_different_column(self):
        events = np.zeros((2, 5, 4))
        events[:, :4, 0] = 1.0

        assert list(dataset.AbstractBase.get_n_points(events, axis=0)) == [4, 4]

    def test_zero_and_negative_do_not_count(self):
        events = np.zeros((1, 4, 4))
        events[0, :, 3] = [-1.0, 0.0, 1e-12, 2.0]

        assert dataset.AbstractBase.get_n_points(events) == 2

    def test_reachable_without_an_instance(self):
        assert dataset.PointCloudDataset.get_n_points(np.ones((1, 3, 4))) == 3

    def test_accepts_an_h5py_dataset(self, padded):
        """``_make_index_list`` hands it a live handle, not an array."""
        _, per_file, pattern = padded()

        with h5py.File(pattern.format(0), "r") as handle:
            counts = dataset.AbstractBase.get_n_points(handle["events"])

        assert list(counts) == list(per_file[0]["n_points"])


# --------------------------------------------------------------------------- #
# AbstractBase.choose_idxs
# --------------------------------------------------------------------------- #
class TestChooseIdxs:
    def test_returns_a_slice(self):
        assert isinstance(Stub().choose_idxs(0), slice)

    def test_near_the_start_the_window_begins_at_idx(self):
        assert Stub(bs=4, length=100).choose_idxs(2) == slice(2, 6)

    def test_in_the_middle_the_window_is_centred_on_idx(self):
        assert Stub(bs=4, length=100).choose_idxs(50) == slice(48, 52)

    def test_near_the_end_the_window_finishes_at_idx(self):
        assert Stub(bs=4, length=100).choose_idxs(97) == slice(93, 97)

    def test_idx_equal_to_the_batch_size_takes_the_trailing_branch(self):
        """``idx > bs`` and ``idx < bs`` are both false at ``idx == bs``."""
        assert Stub(bs=4, length=100).choose_idxs(4) == slice(0, 4)

    @pytest.mark.parametrize("idx", [0, 3, 4, 5, 50, 95, 96, 99])
    def test_even_batch_size_always_yields_a_full_batch(self, idx):
        length = 100
        chosen = as_indices(Stub(bs=4, length=length).choose_idxs(idx), length)

        assert len(chosen) == 4

    def test_odd_batch_size_loses_an_event_in_the_middle(self):
        """``int(bs / 2)`` either side spans ``bs - 1``, not ``bs``.

        With an odd ``batch_size`` the early and late batches hold ``bs`` events
        but every middle batch holds ``bs - 1``, so the batch size is not
        constant across an epoch.
        """
        stub = Stub(bs=5, length=100)

        assert len(as_indices(stub.choose_idxs(50), 100)) == 4
        assert len(as_indices(stub.choose_idxs(0), 100)) == 5

    def test_the_last_event_is_never_selected(self):
        """No branch reaches ``_len - 1`` once the dataset is longer than 2 * bs.

        The middle branch stops at ``idx + bs // 2`` for ``idx < _len - bs`` and
        the trailing branch is exclusive of ``idx``, so the final event is
        unreachable for every index.
        """
        length, bs = 100, 4
        stub = Stub(bs=bs, length=length)

        seen = set()
        for idx in range(length):
            seen.update(as_indices(stub.choose_idxs(idx), length))

        assert length - 1 not in seen
        assert seen == set(range(length - 1))

    def test_a_dataset_smaller_than_the_batch_runs_off_the_end(self):
        """The slice is not clipped, so short datasets give short batches."""
        stub = Stub(bs=4, length=3)

        assert stub.choose_idxs(0) == slice(0, 4)
        assert as_indices(stub.choose_idxs(2), 3) == [2]

    def test_consecutive_indices_overlap(self):
        stub = Stub(bs=4, length=100)

        first = set(as_indices(stub.choose_idxs(50), 100))
        second = set(as_indices(stub.choose_idxs(51), 100))

        assert first & second


# --------------------------------------------------------------------------- #
# AbstractBase.__len__
# --------------------------------------------------------------------------- #
class TestLen:
    def test_reports_the_cached_length(self):
        assert len(Stub(length=17)) == 17

    def test_counts_every_event_across_every_file(self, padded, build):
        config, per_file, _ = padded(n_events=(4, 6, 5))

        built = build(config)

        assert len(built) == 15
        assert len(built) == len(built.index_list)

    def test_only_the_requested_part_is_counted(self, padded, build):
        config, _, _ = padded(n_events=(4, 6, 5))

        # val_range is 1..2, i.e. the middle file only
        assert len(build(config, part="val")) == 6
        assert len(build(config, part="test")) == 4


# --------------------------------------------------------------------------- #
# AbstractBase.fuzz_parallel
# --------------------------------------------------------------------------- #
class TestFuzzParallel:
    @pytest.fixture
    def stub(self):
        return Stub(config=base_config("unused"), offset=1.0)

    def test_moves_x_and_y_only(self, stub):
        events = np.zeros((4, 6, 4))
        events[:, :, 2] = 3.0
        events[:, :, 3] = 1.0

        stub.fuzz_parallel(events)

        assert np.any(events[:, :, 0] != 0.0)
        assert np.any(events[:, :, 1] != 0.0)
        np.testing.assert_array_equal(events[:, :, 2], 3.0)
        np.testing.assert_array_equal(events[:, :, 3], 1.0)

    def test_offsets_stay_within_half_a_division(self, stub):
        events = np.zeros((4, 6, 4))

        stub.fuzz_parallel(events)

        assert np.all(np.abs(events[:, :, :2]) <= stub.offset * 0.5)

    def test_x_and_y_get_independent_offsets(self, stub):
        events = np.zeros((4, 6, 4))

        stub.fuzz_parallel(events)

        assert not np.allclose(events[:, :, 0], events[:, :, 1])

    def test_acts_in_place_and_returns_nothing(self, stub):
        events = np.zeros((2, 3, 4))

        assert stub.fuzz_parallel(events) is None
        assert np.any(events != 0.0)

    def test_shape_is_untouched(self, stub):
        events = np.zeros((2, 3, 4))

        stub.fuzz_parallel(events)

        assert events.shape == (2, 3, 4)

    def test_padding_is_fuzzed_too(self, stub):
        """Energy is never consulted, so zero-energy padding is displaced.

        Harmless while the trim keeps padding out of the batch, but it means the
        padded entries of a partially filled event no longer sit at the origin.
        """
        events = np.zeros((2, 3, 4))  # every point is padding

        stub.fuzz_parallel(events)

        assert np.any(events[:, :, 0] != 0.0)


# --------------------------------------------------------------------------- #
# AbstractBase.fuzz_perpendicular
# --------------------------------------------------------------------------- #
class TestFuzzPerpendicular:
    """``layer_bottom_pos`` is ``[0, 1, 2]`` and ``cell_thickness`` is 1.0, so
    the selection windows are ``[-0.5, 1)``, ``[1, 2)``, ``[2, 3.5)`` and the
    target bands are ``[0, 1)``, ``[1, 2)``, ``[2, 3)``.
    """

    @pytest.fixture
    def stub(self):
        return Stub(config=base_config("unused"), offset=1.0)

    def test_a_list_of_layer_positions_is_accepted(self, stub):
        """``layer_bottom_pos`` arrives from YAML as a list, not an array."""
        assert isinstance(stub.config["data"]["layer_bottom_pos"], list)

        stub.fuzz_perpendicular(np.zeros((1, 2, 4)))

    @pytest.mark.parametrize(
        "start, low, high",
        [(0.5, 0.0, 1.0), (1.5, 1.0, 2.0), (2.5, 2.0, 3.0)],
    )
    def test_points_are_redrawn_inside_their_own_layer(self, stub, start, low, high):
        events = np.zeros((1, 1, 4))
        events[0, 0, 2] = start
        events[0, 0, 3] = 1.0

        stub.fuzz_perpendicular(events)

        assert low <= events[0, 0, 2] < high

    @pytest.mark.parametrize("height", [-5.0, -0.6, 10.0])
    def test_points_outside_every_window_are_left_alone(self, stub, height):
        events = np.zeros((1, 1, 4))
        events[0, 0, 2] = height
        events[0, 0, 3] = 1.0

        stub.fuzz_perpendicular(events)

        assert events[0, 0, 2] == height

    def test_only_the_perpendicular_axis_moves(self, stub):
        events = np.zeros((1, 3, 4))
        events[0, :, 0] = 7.0
        events[0, :, 1] = 8.0
        events[0, :, 2] = [0.5, 1.5, 2.5]
        events[0, :, 3] = 1.0

        stub.fuzz_perpendicular(events)

        np.testing.assert_array_equal(events[0, :, 0], 7.0)
        np.testing.assert_array_equal(events[0, :, 1], 8.0)
        np.testing.assert_array_equal(events[0, :, 3], 1.0)

    def test_acts_in_place_and_returns_nothing(self, stub):
        events = np.zeros((1, 1, 4))
        events[0, 0, 2] = 0.5
        events[0, 0, 3] = 1.0

        assert stub.fuzz_perpendicular(events) is None
        assert events[0, 0, 2] != 0.5

    def test_padding_is_moved_as_well(self, stub):
        """``done`` marks zero-energy points but is never read again.

        It is set from the energy column and updated in the loop, yet the mask
        that does the moving does not consult it, so padding sitting inside a
        layer window is redrawn like a real hit.
        """
        events = np.zeros((1, 2, 4))
        events[0, :, 2] = [0.5, 1.5]
        events[0, :, 3] = 0.0  # all padding

        stub.fuzz_perpendicular(events)

        assert not np.array_equal(events[0, :, 2], [0.5, 1.5])


# --------------------------------------------------------------------------- #
# PointCloudDataset construction
# --------------------------------------------------------------------------- #
class TestConstruction:
    def test_it_is_a_torch_dataset(self, padded, build):
        config, _, _ = padded()

        assert isinstance(build(config), torch.utils.data.Dataset)

    def test_attributes_come_from_the_config(self, padded, build):
        config, _, _ = padded()

        built = build(config)

        assert built.bs == config["training"]["batch_size"]
        assert built.max_ds_seq_len == config["training"]["max_points_per_event"]
        assert built.retain_quantized is config["training"]["retain_quantized"]
        assert built.front_padded is False
        assert built._len == len(built.index_list)

    def test_offset_is_a_cell_divided_into_sub_cells(self, padded, build):
        config, _, _ = padded()
        config["data"]["cell_size"] = 10.0
        config["data"]["divisions_per_cell"] = 4

        assert build(config).offset == 2.5

    def test_cond_features_and_points_are_mapped_to_disk_names(self, padded, build):
        config, _, _ = padded()

        assert build(config).keys_to_include == {
            "incident_energy": "energy",
            "incident_direction": "p_norm_local",
            "points": "events",
        }

    def test_a_feature_without_a_key_entry_falls_back_to_its_own_name(
        self, padded, build
    ):
        config, _, _ = padded()
        config["model"]["cond_features"] = ["energy"]

        assert build(config).keys_to_include["energy"] == "energy"

    def test_energy_scale_is_available_without_instantiating(self):
        assert dataset.PointCloudDataset.energy_scale == 1000


# --------------------------------------------------------------------------- #
# PointCloudDataset._open_data_files
# --------------------------------------------------------------------------- #
class TestOpenDataFiles:
    def test_opens_one_handle_per_file_in_the_range(self, padded, build):
        config, _, _ = padded(n_events=(6, 6, 6))

        assert len(build(config).open_files) == 3

    @pytest.mark.parametrize("part, expected", [("train", 3), ("val", 1), ("test", 1)])
    def test_each_part_reads_its_own_range(self, padded, build, part, expected):
        config, _, _ = padded(n_events=(6, 6, 6))

        assert len(build(config, part=part).open_files) == expected

    def test_files_are_opened_read_only(self, padded, build):
        config, _, _ = padded()

        assert all(f.mode == "r" for f in build(config).open_files)

    def test_no_files_on_disk(self, tmp_path, build):
        config = base_config(str(tmp_path / "nothing_{}.h5"))

        with pytest.raises(FileNotFoundError):
            build(config)

    def test_an_empty_range_raises(self, padded, build):
        config, _, _ = padded(n_events=(6, 6))
        config["data"]["train_range_start"] = 1
        config["data"]["train_range_end"] = 1

        with pytest.raises(FileNotFoundError, match="No files found at"):
            build(config)

    def test_asking_for_more_files_than_exist(self, padded, build):
        config, _, _ = padded(n_events=(6,))
        config["data"]["train_range_end"] = 3

        with pytest.raises(AssertionError, match="only 1 available"):
            build(config)

    def test_an_unknown_part(self, padded, build):
        config, _, _ = padded()

        with pytest.raises(KeyError):
            build(config, part="validation")


# --------------------------------------------------------------------------- #
# PointCloudDataset._make_index_list
# --------------------------------------------------------------------------- #
class TestMakeIndexList:
    def test_shape_and_dtype(self, padded, build):
        config, _, _ = padded(n_events=(4, 6))

        index_list = build(config).index_list

        assert index_list.shape == (10, 3)
        assert index_list.dtype == int

    def test_sorted_by_point_count(self, padded, build):
        config, _, _ = padded(n_events=(6, 6, 6))

        counts = build(config).index_list[:, 0]

        assert np.all(np.diff(counts) >= 0)

    def test_every_event_appears_exactly_once(self, padded, build):
        config, _, _ = padded(n_events=(4, 6, 5))

        index_list = build(config).index_list

        pairs = {(int(f), int(e)) for _, f, e in index_list}
        expected = {(f, e) for f, n in enumerate((4, 6, 5)) for e in range(n)}
        assert pairs == expected
        assert len(index_list) == len(expected)

    def test_point_counts_match_the_file(self, padded, build):
        config, per_file, _ = padded(n_events=(4, 6))

        index_list = build(config).index_list

        for n_pts, file_n, event_n in index_list:
            assert n_pts == per_file[file_n]["n_points"][event_n]

    def test_ties_keep_file_then_event_order(self, padded, build):
        """``list.sort`` is stable, so equal counts stay in on-disk order."""
        config, _, pattern = padded(n_events=(3, 3))
        for i in range(2):
            replace_key(pattern.format(i), "n_points", np.ones(3, dtype=int))

        index_list = build(config).index_list

        assert [tuple(row[1:]) for row in index_list] == [
            (0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2),
        ]

    def test_the_stored_count_is_trusted_over_the_energies(self, padded, build):
        """Whatever ``n_points`` says on disk wins, right or wrong."""
        config, _, pattern = padded(n_events=(6,))
        replace_key(pattern.format(0), "n_points", np.zeros(6, dtype=int))

        assert set(build(config).index_list[:, 0]) == {0}

    def test_falls_back_to_counting_energies(self, padded, build):
        config, per_file, pattern = padded(n_events=(6,))
        drop_key(pattern.format(0), "n_points")

        counts = build(config).index_list[:, 0]

        np.testing.assert_array_equal(np.sort(counts), np.sort(per_file[0]["n_points"]))

    def test_the_fallback_handles_rolled_axes(self, padded, build):
        config, per_file, pattern = padded(n_events=(6,), roll_axis=True)
        drop_key(pattern.format(0), "n_points")

        counts = build(config).index_list[:, 0]

        np.testing.assert_array_equal(np.sort(counts), np.sort(per_file[0]["n_points"]))

    def test_the_n_points_key_setting_is_ignored(self, padded, build):
        """The lookup is the literal string ``"n_points"``.

        ``config["data"]["n_points_key"]`` is never consulted, so a dataset that
        stores its counts under any other name silently falls back to counting
        energies -- correct here, but slow, and wrong if the stored counts
        differ from the energy column.
        """
        config, per_file, pattern = padded(n_events=(6,))
        stored = per_file[0]["n_points"]
        drop_key(pattern.format(0), "n_points")
        with h5py.File(pattern.format(0), "a") as handle:
            handle.create_dataset("num_points", data=np.zeros_like(stored))
        config["data"]["n_points_key"] = "num_points"

        counts = build(config).index_list[:, 0]

        np.testing.assert_array_equal(np.sort(counts), np.sort(stored))

    def test_counts_are_capped_at_max_points_per_event(self, padded, build):
        config, _, _ = padded(n_events=(6,), n_points=5)
        config["training"]["max_points_per_event"] = 2

        assert build(config).index_list[:, 0].max() == 2

    @pytest.mark.xfail(
        reason=(
            "n_points stored as (n_events, 1) is not squeezed in "
            "_make_index_list, so np.array() on the ragged tuples fails"
        ),
        strict=False,
    )
    def test_a_trailing_axis_on_n_points(self, padded, build):
        config, per_file, _ = padded(n_events=(6,), n_points_ndim=2)

        assert len(build(config)) == len(per_file[0]["energy"])


# --------------------------------------------------------------------------- #
# PointCloudDataset._get_prior_event_axes
# --------------------------------------------------------------------------- #
class TestGetPriorEventAxes:
    def test_event_first_data_needs_no_padding(self, padded, build):
        config, _, _ = padded()

        assert build(config)._prior_event_axes == {
            "incident_energy": [],
            "incident_direction": [],
            "points": [],
        }

    def test_an_event_axis_further_in_is_padded_with_slices(self, padded, build):
        config, per_file, pattern = padded(n_events=(6,))
        replace_key(pattern.format(0), "energy", per_file[0]["energy"][None, :])
        replace_key(pattern.format(0), "p_norm_local", per_file[0]["p_norm_local"].T)

        axes = build(config)._prior_event_axes

        assert axes["incident_energy"] == [slice(None)]
        assert axes["incident_direction"] == [slice(None)]
        assert axes["points"] == []

    def test_transposed_columns_still_read_the_right_event(self, padded, build):
        config, per_file, pattern = padded(n_events=(6,))
        replace_key(pattern.format(0), "p_norm_local", per_file[0]["p_norm_local"].T)

        built = build(config)
        batch = built[0]

        rows = built.index_list[built.choose_idxs(0)]
        np.testing.assert_allclose(
            batch["incident_direction"], gather(per_file, rows, "p_norm_local")
        )

    def test_keys_configured_as_none_are_dropped(self, padded, build):
        config, _, _ = padded()
        config["model"]["cond_features"] = ["incident_pdg", "incident_energy"]

        built = build(config)

        assert "incident_pdg" in built.keys_to_include
        assert "incident_pdg" not in built._prior_event_axes

    def test_a_dropped_key_breaks_getitem(self, padded, build):
        """``keys_to_include`` keeps the ``None`` entry, ``_prior_event_axes``
        does not, so ``__getitem__`` looks up a key that was never stored."""
        config, _, _ = padded()
        config["model"]["cond_features"] = ["incident_pdg", "incident_energy"]
        built = build(config)

        with pytest.raises(KeyError, match="incident_pdg"):
            built[0]

    def test_a_column_with_no_matching_axis(self, padded, build):
        config, _, pattern = padded(n_events=(6,))
        replace_key(pattern.format(0), "energy", np.zeros(5))

        with pytest.raises(ValueError):
            build(config)

    def test_the_first_matching_axis_wins_even_when_wrong(self, padded, build):
        """``shape.index`` returns the first axis of that length.

        With three events and a ``(3, n_events)`` direction column both axes are
        length 3, the leading one is picked, and every event is handed one
        component of the direction repeated instead of its own vector. Nothing
        raises.
        """
        config, per_file, pattern = padded(n_events=(3,))
        transposed = per_file[0]["p_norm_local"].T
        replace_key(pattern.format(0), "p_norm_local", transposed)

        built = build(config)
        batch = built[0]

        rows = built.index_list[built.choose_idxs(0)]
        wrong = np.array([transposed[event_n] for _, _, event_n in rows])
        np.testing.assert_allclose(batch["incident_direction"], wrong)


# --------------------------------------------------------------------------- #
# PointCloudDataset._is_front_padded
# --------------------------------------------------------------------------- #
class TestIsFrontPadded:
    @pytest.mark.parametrize("padding, expected", [("front", True), ("back", False)])
    def test_reads_the_padding_setting(self, padded, build, padding, expected):
        config, _, _ = padded(padding=padding)

        assert build(config).front_padded is expected

    @pytest.mark.parametrize("padding", ["Front", "BACK", "none", "", None])
    def test_anything_else_raises(self, padded, build, padding):
        config, _, _ = padded()
        config["data"]["padding"] = padding

        with pytest.raises(ValueError, match="must be 'front' or 'back'"):
            build(config)

    def test_the_check_file_argument_is_ignored(self, padded, build):
        """It is never read -- the setting is taken from the config alone."""
        config, _, _ = padded(padding="front")
        built = build(config)

        assert built._is_front_padded(check_file=99) is True


# --------------------------------------------------------------------------- #
# PointCloudDataset._event_processing
# --------------------------------------------------------------------------- #
class TestEventProcessing:
    def test_trims_to_the_widest_event_in_the_batch(self, padded, build):
        config, _, _ = padded()
        built = build(config)
        events = np.zeros((3, 8, 4))
        events[0, :2, 3] = 1.0
        events[1, :5, 3] = 1.0

        assert built._event_processing(events).shape == (3, 5, 4)

    def test_the_trim_is_capped_at_max_points_per_event(self, padded, build):
        config, _, _ = padded()
        config["training"]["max_points_per_event"] = 3
        built = build(config)
        events = np.zeros((2, 8, 4))
        events[:, :6, 3] = 1.0

        assert built._event_processing(events).shape[1] == 3

    def test_back_padding_keeps_the_leading_columns(self, padded, build):
        config, _, _ = padded(padding="back")
        built = build(config)
        events = np.zeros((1, 5, 4))
        events[0, :2, 3] = 1.0
        events[0, :, 0] = [10, 11, 12, 13, 14]

        processed = built._event_processing(events)

        np.testing.assert_array_equal(processed[0, :, 0], [10, 11])

    def test_front_padding_keeps_the_trailing_columns(self, padded, build):
        config, _, _ = padded(padding="front")
        built = build(config)
        events = np.zeros((1, 5, 4))
        events[0, -2:, 3] = 1.0
        events[0, :, 0] = [10, 11, 12, 13, 14]

        processed = built._event_processing(events)

        np.testing.assert_array_equal(processed[0, :, 0], [13, 14])

    def test_the_padding_setting_is_trusted_not_verified(self, padded, build):
        """Declaring the wrong side silently returns nothing but padding.

        The trim length is the real point count, but it is taken from the end
        the config names, so a mismatch drops every hit without any warning.
        """
        config, _, _ = padded(padding="front")  # data is back padded
        built = build(config)
        events = np.zeros((1, 5, 4))
        events[0, :2, 3] = 1.0

        processed = built._event_processing(events)

        assert np.all(processed[:, :, 3] == 0.0)

    def test_orientation_permutes_the_position_columns(self, padded, build):
        config, _, _ = padded()
        config["data"]["orientation"] = "hdf5:xyz==local:zxy"
        built = build(config)
        events = np.zeros((1, 1, 4))
        events[0, 0] = [1.0, 2.0, 3.0, 1.0]

        processed = built._event_processing(events)

        # x -> z, y -> x, z -> y
        np.testing.assert_array_equal(processed[0, 0], [2.0, 3.0, 1.0, 1.0])

    def test_the_default_orientation_is_a_no_op(self, padded, build):
        config, _, _ = padded()
        built = build(config)
        events = np.zeros((1, 1, 4))
        events[0, 0] = [1.0, 2.0, 3.0, 1.0]

        processed = built._event_processing(events)

        np.testing.assert_array_equal(processed[0, 0], [1.0, 2.0, 3.0, 1.0])

    def test_rolled_axes_are_moved_back(self, padded, build):
        config, _, _ = padded(roll_axis=True)
        built = build(config)
        events = np.zeros((2, 4, 6))  # (n_events, n_features, n_points)
        events[:, 3, :3] = 1.0

        processed = built._event_processing(events)

        assert processed.shape == (2, 3, 4)

    def test_quantized_data_is_left_alone(self, padded, build, mocker):
        config, _, _ = padded()
        config["training"]["retain_quantized"] = True
        built = build(config)
        parallel = mocker.spy(built, "fuzz_parallel")
        perpendicular = mocker.spy(built, "fuzz_perpendicular")
        events = np.zeros((1, 2, 4))
        events[0, :, 3] = 1.0

        built._event_processing(events)

        parallel.assert_not_called()
        perpendicular.assert_not_called()

    def test_unquantized_data_is_fuzzed(self, padded, build, mocker):
        config, _, _ = padded()
        config["training"]["retain_quantized"] = False
        built = build(config)
        parallel = mocker.spy(built, "fuzz_parallel")
        perpendicular = mocker.spy(built, "fuzz_perpendicular")
        events = np.zeros((1, 2, 4))
        events[0, :, 3] = 1.0

        built._event_processing(events)

        parallel.assert_called_once()
        perpendicular.assert_called_once()

    def test_fuzzing_happens_after_the_trim(self, padded, build):
        config, _, _ = padded()
        config["training"]["retain_quantized"] = False
        built = build(config)
        events = np.zeros((1, 6, 4))
        events[0, :2, 3] = 1.0

        processed = built._event_processing(events)

        assert processed.shape[1] == 2
        assert np.any(processed[:, :, 0] != 0.0)


# --------------------------------------------------------------------------- #
# PointCloudDataset.__getitem__
# --------------------------------------------------------------------------- #
class TestGetItem:
    def test_the_batch_holds_the_configured_keys_plus_n_points(self, padded, build):
        config, _, _ = padded()

        assert set(build(config)[0]) == {
            "incident_energy",
            "incident_direction",
            "points",
            "n_points",
        }

    def test_shapes(self, padded, build):
        config, _, _ = padded()
        batch_size = config["training"]["batch_size"]

        batch = build(config)[0]

        assert batch["points"].shape[0] == batch_size
        assert batch["points"].shape[2] == 4
        assert batch["incident_energy"].shape == (batch_size, 1)
        assert batch["incident_direction"].shape == (batch_size, 3)
        assert batch["n_points"].shape == (batch_size, 1)

    def test_one_dimensional_columns_gain_a_trailing_axis(self, padded, build):
        config, _, _ = padded()

        assert build(config)[0]["incident_energy"].ndim == 2

    def test_a_column_stored_with_a_trailing_axis_is_unchanged(self, padded, build):
        config, _, _ = padded(energy_ndim=2)
        batch_size = config["training"]["batch_size"]

        assert build(config)[0]["incident_energy"].shape == (batch_size, 1)

    @pytest.mark.parametrize("idx", [0, 1, 5, 11])
    def test_per_event_values_match_the_file(self, padded, build, idx):
        config, per_file, _ = padded(n_events=(6, 6))
        built = build(config)

        batch = built[idx]

        rows = built.index_list[built.choose_idxs(idx)]
        np.testing.assert_allclose(
            batch["incident_energy"][:, 0], gather(per_file, rows, "energy")
        )
        np.testing.assert_allclose(
            batch["incident_direction"], gather(per_file, rows, "p_norm_local")
        )

    def test_points_match_the_file(self, padded, build):
        config, per_file, _ = padded(n_events=(6,))
        built = build(config)

        batch = built[0]

        rows = built.index_list[built.choose_idxs(0)]
        expected = gather(per_file, rows, "events")[:, : batch["points"].shape[1]]
        np.testing.assert_allclose(batch["points"], expected)

    def test_n_points_is_the_first_column_of_the_index_list(self, padded, build):
        config, _, _ = padded(n_events=(6,))
        built = build(config)

        batch = built[0]

        rows = built.index_list[built.choose_idxs(0)]
        np.testing.assert_array_equal(batch["n_points"][:, 0], rows[:, 0])

    def test_the_batch_width_is_the_largest_count_in_the_batch(self, padded, build):
        config, _, _ = padded(n_events=(6,))

        batch = build(config)[0]

        assert batch["points"].shape[1] == batch["n_points"].max()

    def test_events_arrive_in_point_count_order(self, padded, build):
        config, _, _ = padded(n_events=(8,))

        batch = build(config)[2]

        assert np.all(np.diff(batch["n_points"][:, 0]) >= 0)

    def test_rolled_axes_come_back_as_points_by_features(self, padded, build):
        config, per_file, _ = padded(n_events=(6,), roll_axis=True)
        built = build(config)

        batch = built[0]

        assert batch["points"].shape[2] == 4
        rows = built.index_list[built.choose_idxs(0)]
        expected = gather(per_file, rows, "events")[:, : batch["points"].shape[1]]
        np.testing.assert_allclose(batch["points"], expected)

    def test_a_custom_points_key_is_honoured(self, padded, build):
        config, _, _ = padded(points_key="showers")
        config["data"]["points_key"] = "showers"

        assert build(config)[0]["points"].shape[2] == 4

    def test_repeated_reads_agree_when_quantized(self, padded, build):
        config, _, _ = padded()
        built = build(config)

        first, second = built[0], built[0]

        for key in first:
            np.testing.assert_allclose(first[key], second[key])

    def test_events_are_drawn_from_every_file(self, padded, build):
        config, _, _ = padded(n_events=(4, 4, 4))
        built = build(config)

        seen = {int(f) for idx in range(len(built)) for _, f, _ in
                built.index_list[built.choose_idxs(idx)]}

        assert seen == {0, 1, 2}


# --------------------------------------------------------------------------- #
# PointCloudDatasetUnordered
# --------------------------------------------------------------------------- #
class TestPointCloudDatasetUnordered:
    @pytest.fixture
    def unordered(self, padded, build):
        """``(dataset, per_file)`` for a 20 event ``padded_unordered`` file."""
        config, per_file, _ = padded(n_events=(20,), fmt="padded_unordered")
        return build(config), per_file

    def test_it_only_overrides_the_selection(self):
        assert issubclass(
            dataset.PointCloudDatasetUnordered, dataset.PointCloudDataset
        )
        assert (
            dataset.PointCloudDatasetUnordered.__getitem__
            is dataset.PointCloudDataset.__getitem__
        )

    def test_returns_an_array_not_a_slice(self, unordered):
        selector, _ = unordered

        assert isinstance(selector.choose_idxs(0), np.ndarray)

    def test_the_same_index_gives_the_same_events(self, unordered):
        selector, _ = unordered

        assert np.array_equal(selector.choose_idxs(3), selector.choose_idxs(3))

    def test_different_indices_give_different_events(self, unordered):
        selector, _ = unordered

        selections = {tuple(selector.choose_idxs(i)) for i in range(5)}

        assert len(selections) > 1

    def test_no_event_is_drawn_twice(self, unordered):
        selector, _ = unordered

        idxs = selector.choose_idxs(0)

        assert len(set(idxs.tolist())) == len(idxs)

    def test_the_selection_stays_sorted(self, unordered):
        """h5py needs an increasing index array, and the batch inherits the
        ``index_list`` ordering, so batches are still point-count ordered."""
        selector, _ = unordered

        assert np.all(np.diff(selector.choose_idxs(7)) > 0)

    def test_the_batch_is_a_full_batch(self, unordered):
        selector, _ = unordered

        assert len(selector.choose_idxs(0)) == selector.bs

    def test_a_dataset_smaller_than_the_batch_returns_everything(self, padded, build):
        config, _, _ = padded(n_events=(3,), fmt="padded_unordered")
        config["training"]["batch_size"] = 10
        selector = build(config)

        idxs = selector.choose_idxs(0)

        assert sorted(idxs.tolist()) == [0, 1, 2]

    def test_the_seed_is_the_index_alone(self, unordered):
        """The draw depends only on the index and the length, not the data, so
        two datasets of the same length see identical batches."""
        selector, _ = unordered
        expected = np.random.default_rng(seed=4).choice(
            len(selector), selector.bs, replace=False
        )
        expected.sort()

        np.testing.assert_array_equal(selector.choose_idxs(4), expected)

    def test_the_batch_looks_like_an_ordered_one(self, unordered):
        selector, _ = unordered

        batch = selector[0]

        assert set(batch) == {
            "incident_energy",
            "incident_direction",
            "points",
            "n_points",
        }
        assert batch["incident_energy"].shape == (selector.bs, 1)
        assert batch["points"].shape[0] == selector.bs

    def test_values_still_match_the_file(self, unordered):
        selector, per_file = unordered

        batch = selector[1]

        rows = selector.index_list[selector.choose_idxs(1)]
        np.testing.assert_allclose(
            batch["incident_energy"][:, 0], gather(per_file, rows, "energy")
        )

    def test_every_event_is_reachable(self, unordered):
        """Unlike the ordered walk, the random draw does reach the last event."""
        selector, _ = unordered

        seen = set()
        for idx in range(200):
            seen.update(selector.choose_idxs(idx).tolist())

        assert seen == set(range(len(selector)))


# --------------------------------------------------------------------------- #
# from_config dispatch
# --------------------------------------------------------------------------- #
class TestFromConfigDispatch:
    @pytest.fixture
    def patched(self, mocker):
        return {name: mocker.patch(f"src.data.dataset.{name}") for name in CLASSES}

    @pytest.mark.parametrize(
        "fmt, expected",
        [
            ("padded", "PointCloudDataset"),
            ("padded_unordered", "PointCloudDatasetUnordered"),
            ("showerdata", "ShowerDataDataset"),
        ],
    )
    def test_each_format_builds_its_own_class(self, patched, fmt, expected):
        config = {"data": {"format": fmt}}

        result = dataset.from_config(config, dataset_part="val")

        patched[expected].assert_called_once_with(config, dataset_part="val")
        assert result is patched[expected].return_value
        for name, cls in patched.items():
            if name != expected:
                cls.assert_not_called()

    def test_default_part_is_train(self, patched):
        config = {"data": {"format": "padded"}}

        dataset.from_config(config)

        patched["PointCloudDataset"].assert_called_once_with(
            config, dataset_part="train"
        )

    def test_unknown_format(self, patched):
        with pytest.raises(NotImplementedError):
            dataset.from_config({"data": {"format": "parquet"}})

    def test_missing_format_key(self, patched):
        with pytest.raises(KeyError):
            dataset.from_config({"data": {}})


# --------------------------------------------------------------------------- #
# from_config on real files
# --------------------------------------------------------------------------- #
class TestFromConfigEndToEnd:
    @pytest.mark.parametrize(
        "fmt, expected",
        [
            ("padded", dataset.PointCloudDataset),
            ("padded_unordered", dataset.PointCloudDatasetUnordered),
        ],
    )
    def test_builds_a_usable_dataset(self, padded, build, fmt, expected):
        config, per_file, _ = padded(fmt=fmt)

        built = build(config)

        assert isinstance(built, expected)
        assert len(built) == len(per_file[0]["energy"])
        assert set(built[0]) == {
            "incident_energy",
            "incident_direction",
            "points",
            "n_points",
        }

    @pytest.mark.parametrize("part", ["train", "val", "test"])
    def test_every_part_can_be_built(self, padded, build, part):
        config, _, _ = padded(n_events=(6, 6, 6))

        assert len(build(config, part=part)) > 0
