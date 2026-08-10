"""Tests for ``src/data/read_write.py`` -- padded / padded_unordered only.

The file-discovery helpers are ``lru_cache``d; ``conftest.clear_read_write_caches``
resets them around every test.
"""

import os

import numpy as np
import pytest
import h5py

from src.data import read_write

from conftest import base_config, make_empty_padded_file, make_padded_file


# --------------------------------------------------------------------------- #
# get_possible_files
# --------------------------------------------------------------------------- #
class TestGetPossibleFiles:
    def test_expands_a_single_placeholder(self, tmp_path):
        for i in range(3):
            (tmp_path / f"data_{i}.h5").touch()
        (tmp_path / "notes.txt").touch()

        found = read_write.get_possible_files(str(tmp_path / "data_{}.h5"))

        assert [os.path.basename(f) for f in found] == [
            "data_0.h5",
            "data_1.h5",
            "data_2.h5",
        ]

    def test_expands_several_placeholders(self, tmp_path):
        (tmp_path / "d_1_a.h5").touch()
        (tmp_path / "d_2_b.h5").touch()
        (tmp_path / "d_2.h5").touch()

        found = read_write.get_possible_files(str(tmp_path / "d_{}_{}.h5"))

        assert len(found) == 2

    def test_path_without_a_placeholder(self, tmp_path):
        target = tmp_path / "only.h5"
        target.touch()

        assert read_write.get_possible_files(str(target)) == [str(target)]

    def test_ordering_is_lexicographic_not_numeric(self, tmp_path):
        for i in (0, 1, 2, 10, 11):
            (tmp_path / f"data_{i}.h5").touch()

        found = read_write.get_possible_files(str(tmp_path / "data_{}.h5"))

        names = [os.path.basename(f) for f in found]
        assert names == [
            "data_0.h5",
            "data_1.h5",
            "data_10.h5",
            "data_11.h5",
            "data_2.h5",
        ], (
            "sorted() is lexicographic -- with >=10 files the train/val/test "
            "ranges do not line up with the numbers in the file names"
        )

    def test_no_matches_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No files found with pattern"):
            read_write.get_possible_files(str(tmp_path / "missing_{}.h5"))

    def test_result_is_cached(self, tmp_path):
        (tmp_path / "data_0.h5").touch()
        pattern = str(tmp_path / "data_{}.h5")
        first = read_write.get_possible_files(pattern)

        (tmp_path / "data_1.h5").touch()
        second = read_write.get_possible_files(pattern)

        assert first is second, "lru_cache means new files on disk are not picked up"


# --------------------------------------------------------------------------- #
# get_files
# --------------------------------------------------------------------------- #
class TestGetFiles:
    @pytest.fixture
    def pattern(self, tmp_path):
        for i in range(5):
            (tmp_path / f"data_{i}.h5").touch()
        return str(tmp_path / "data_{}.h5")

    def test_slices_the_requested_range(self, pattern):
        found = read_write.get_files(pattern, 1, 3)
        assert [os.path.basename(f) for f in found] == ["data_1.h5", "data_2.h5"]

    def test_end_is_exclusive(self, pattern):
        assert len(read_write.get_files(pattern, 0, 5)) == 5

    def test_asking_for_more_files_than_exist_raises(self, pattern):
        with pytest.raises(AssertionError, match="only 5 available"):
            read_write.get_files(pattern, 0, 6)

    def test_empty_range(self, pattern):
        assert read_write.get_files(pattern, 3, 3) == []


# --------------------------------------------------------------------------- #
# get_n_events
# --------------------------------------------------------------------------- #
class TestGetNEvents:
    def test_single_file_returns_a_scalar(self, tmp_path):
        make_padded_file(tmp_path / "data_0.h5", n_events=7)

        n = read_write.get_n_events(
            str(tmp_path / "data_{}.h5"), 0, 1, "padded", "events"
        )

        assert np.ndim(n) == 0
        assert n == 7

    def test_several_files_return_one_count_each(self, tmp_path):
        for i, n in enumerate([4, 6, 5]):
            make_padded_file(tmp_path / f"data_{i}.h5", n_events=n)

        n = read_write.get_n_events(
            str(tmp_path / "data_{}.h5"), 0, 3, "padded", "events"
        )

        assert list(n) == [4, 6, 5]

    def test_rolled_axes_still_report_the_event_count(self, tmp_path):
        # on disk this is (n_events, n_features, n_points); the count is taken
        # from shape[-3], which is the event axis either way
        make_padded_file(tmp_path / "data_0.h5", n_events=7, roll_axis=True)

        n = read_write.get_n_events(
            str(tmp_path / "data_{}.h5"), 0, 1, "padded", "events"
        )

        assert n == 7

    def test_honours_the_points_key(self, tmp_path):
        make_padded_file(tmp_path / "data_0.h5", n_events=7, points_key="showers")

        n = read_write.get_n_events(
            str(tmp_path / "data_{}.h5"), 0, 1, "padded", "showers"
        )

        assert n == 7

    def test_only_the_requested_range_is_opened(self, tmp_path):
        for i, n in enumerate([4, 6, 5]):
            make_padded_file(tmp_path / f"data_{i}.h5", n_events=n)

        n = read_write.get_n_events(
            str(tmp_path / "data_{}.h5"), 1, 3, "padded", "events"
        )

        assert list(n) == [6, 5]

    def test_zero_sized_files_are_skipped(self, tmp_path):
        make_empty_padded_file(tmp_path / "data_0.h5")
        make_padded_file(tmp_path / "data_1.h5", n_events=5)

        n = read_write.get_n_events(
            str(tmp_path / "data_{}.h5"), 0, 2, "padded", "events"
        )

        # Only one non-empty file, so the "< 2" branch collapses the list to a
        # scalar even though two files were requested.  read_raw_regaxes then
        # does ``n_events[i]`` on it -- worth knowing about before it bites.
        assert np.ndim(n) == 0
        assert n == 5


# --------------------------------------------------------------------------- #
# n_events_in_part
# --------------------------------------------------------------------------- #
class TestNEventsInPart:
    def test_forwards_the_right_config_entries(self, mocker, tmp_path):
        get_n_events = mocker.patch(
            "src.data.read_write.get_n_events", return_value=42
        )
        config = base_config(str(tmp_path / "data_{}.h5"))

        result = read_write.n_events_in_part(config, "val")

        assert result == 42
        get_n_events.assert_called_once_with(
            config["data"]["dataset_path"],
            config["data"]["val_range_start"],
            config["data"]["val_range_end"],
            config["data"]["format"],
            config["data"]["points_key"],
        )

    @pytest.mark.parametrize("part", ["train", "val", "test"])
    def test_each_part_reads_its_own_range(self, tmp_path, part):
        for i, n in enumerate([4, 6, 5]):
            make_padded_file(tmp_path / f"data_{i}.h5", n_events=n)
        config = base_config(str(tmp_path / "data_{}.h5"))
        config["data"]["train_range_end"] = 3

        expected = {"train": 15, "val": 6, "test": 4}[part]
        assert np.sum(read_write.n_events_in_part(config, part)) == expected

    def test_unknown_part(self, tmp_path):
        config = base_config(str(tmp_path / "data_{}.h5"))
        with pytest.raises(KeyError):
            read_write.n_events_in_part(config, "validation")


# --------------------------------------------------------------------------- #
# _validate_orientations
# --------------------------------------------------------------------------- #
class TestValidateOrientations:
    def test_local_only_returns_a_string(self):
        assert read_write._validate_orientations("hdf5:xyz==local:zxy") == "zxy"

    def test_local_and_global_return_a_pair(self):
        local, glob = read_write._validate_orientations(
            "hdf5:xyz==local:xyz", "hdf5:xyz==global:zxy"
        )
        assert (local, glob) == ("xyz", "zxy")

    @pytest.mark.parametrize(
        "orientation",
        ["xyz", "hdf5:xyz==global:xyz", "hdf5:zyx==local:xyz", ""],
    )
    def test_bad_local_prefix(self, orientation):
        with pytest.raises(NotImplementedError):
            read_write._validate_orientations(orientation)

    def test_bad_global_prefix(self):
        with pytest.raises(NotImplementedError):
            read_write._validate_orientations(
                "hdf5:xyz==local:xyz", "hdf5:xyz==local:zxy"
            )

    def test_the_suffix_itself_is_not_validated(self):
        """Only the prefix is checked -- a nonsense suffix survives here and
        blows up later inside ``events_to_local``.  Documented, not endorsed."""
        assert read_write._validate_orientations("hdf5:xyz==local:abc") == "abc"


# --------------------------------------------------------------------------- #
# events_to_local
# --------------------------------------------------------------------------- #
class TestEventsToLocal:
    @staticmethod
    def sentinel_events():
        """One event, one point, with each column tagged by its index."""
        return np.array([[[0.0, 1.0, 2.0, 99.0]]])

    def test_identity_orientation_changes_nothing(self):
        events = self.sentinel_events()
        read_write.events_to_local(events, "hdf5:xyz==local:xyz")
        np.testing.assert_array_equal(events, self.sentinel_events())

    def test_permutation_moves_columns_as_documented(self):
        events = self.sentinel_events()

        read_write.events_to_local(events, "hdf5:xyz==local:zxy")

        # column_target = [2, 0, 1]; new[target[k]] = old[k]
        np.testing.assert_array_equal(events[0, 0], [1.0, 2.0, 0.0, 99.0])

    def test_energy_column_is_never_touched(self):
        events = self.sentinel_events()
        read_write.events_to_local(events, "hdf5:xyz==local:yzx")
        assert events[0, 0, 3] == 99.0

    def test_modifies_in_place_and_returns_none(self):
        events = self.sentinel_events()
        assert read_write.events_to_local(events, "hdf5:xyz==local:zxy") is None
        assert not np.array_equal(events, self.sentinel_events())

    @pytest.mark.parametrize("shape", [(0, 5, 4), (3, 0, 4)])
    def test_zero_sized_arrays_short_circuit(self, shape):
        events = np.zeros(shape)
        read_write.events_to_local(events, "hdf5:xyz==local:zxy")
        assert events.shape == shape

    def test_applying_it_twice_is_not_generally_a_no_op(self):
        """The docstring says re-applying reverts the swap; that only holds for
        a transposition, not for a 3-cycle like ``zxy``."""
        events = self.sentinel_events()
        read_write.events_to_local(events, "hdf5:xyz==local:zxy")
        read_write.events_to_local(events, "hdf5:xyz==local:zxy")
        assert not np.array_equal(events, self.sentinel_events())

        events = self.sentinel_events()
        read_write.events_to_local(events, "hdf5:xyz==local:yxz")
        read_write.events_to_local(events, "hdf5:xyz==local:yxz")
        np.testing.assert_array_equal(events, self.sentinel_events())

    def test_invalid_orientation(self):
        with pytest.raises(NotImplementedError):
            read_write.events_to_local(self.sentinel_events(), "nonsense")


# --------------------------------------------------------------------------- #
# local_to_global / global_to_local
# --------------------------------------------------------------------------- #
LOCAL = "hdf5:xyz==local:xyz"
GLOBAL = "hdf5:xyz==global:zxy"


class TestLocalGlobal:
    @staticmethod
    def events():
        return np.array([[[0.0, 1.0, 2.0, 99.0]]])

    def test_round_trip_returns_the_original(self):
        original = self.events()

        as_global = read_write.local_to_global(original, LOCAL, GLOBAL)
        back = read_write.global_to_local(as_global, LOCAL, GLOBAL)

        np.testing.assert_allclose(back, original)

    def test_round_trip_the_other_way_round(self):
        original = self.events()

        as_local = read_write.global_to_local(original, LOCAL, GLOBAL)
        back = read_write.local_to_global(as_local, LOCAL, GLOBAL)

        np.testing.assert_allclose(back, original)

    @pytest.mark.parametrize(
        "function", [read_write.local_to_global, read_write.global_to_local]
    )
    def test_does_not_act_in_place(self, function):
        original = self.events()
        untouched = original.copy()

        result = function(original, LOCAL, GLOBAL)

        np.testing.assert_array_equal(original, untouched)
        assert result is not original

    @pytest.mark.parametrize(
        "function", [read_write.local_to_global, read_write.global_to_local]
    )
    def test_energy_survives(self, function):
        assert function(self.events(), LOCAL, GLOBAL)[0, 0, 3] == 99.0

    @pytest.mark.parametrize(
        "function", [read_write.local_to_global, read_write.global_to_local]
    )
    def test_matching_orientations_are_a_no_op(self, function):
        result = function(self.events(), LOCAL, "hdf5:xyz==global:xyz")
        np.testing.assert_array_equal(result, self.events())

    @pytest.mark.parametrize(
        "function", [read_write.local_to_global, read_write.global_to_local]
    )
    def test_empty_input_is_returned_unchanged(self, function):
        empty = np.zeros((0, 4))
        assert function(empty, LOCAL, GLOBAL) is empty


# --------------------------------------------------------------------------- #
# read_raw_regaxes
# --------------------------------------------------------------------------- #
class TestReadRawRegaxes:
    def test_reads_every_event_in_regular_axis_order(self, padded_dataset):
        config, truth = padded_dataset(n_events=(4, 6, 5))

        per_event, events = read_write.read_raw_regaxes(
            config, part="train", pick_events=slice(None)
        )

        assert events.shape == (15, 5, 4)
        np.testing.assert_allclose(events, truth["events"])
        np.testing.assert_allclose(per_event, truth["energy"])

    def test_rolled_axes_are_unrolled(self, padded_dataset):
        config, truth = padded_dataset(n_events=(4, 6, 5), roll_axis=True)

        _, events = read_write.read_raw_regaxes(
            config, part="train", pick_events=slice(None)
        )

        assert events.shape == (15, 5, 4)
        np.testing.assert_allclose(events, truth["events"])

    def test_single_column_per_event_is_squeezed_to_1d(self, padded_dataset):
        config, truth = padded_dataset(n_events=(6,))

        per_event, _ = read_write.read_raw_regaxes(
            config, part="train", pick_events=slice(None), per_event_cols=["energy"]
        )

        assert per_event.shape == (6,)
        np.testing.assert_allclose(per_event, truth["energy"])

    def test_several_per_event_columns_are_stacked_in_order(self, padded_dataset):
        config, truth = padded_dataset(n_events=(6,))

        per_event, _ = read_write.read_raw_regaxes(
            config,
            part="train",
            pick_events=slice(None),
            per_event_cols=["energy", "p_norm_local"],
        )

        assert per_event.shape == (6, 4)
        np.testing.assert_allclose(per_event[:, 0], truth["energy"])
        np.testing.assert_allclose(per_event[:, 1:], truth["p_norm_local"])

    @pytest.mark.parametrize("ndim", [1, 2])
    def test_per_event_columns_work_with_or_without_a_trailing_axis(
        self, padded_dataset, ndim
    ):
        config, truth = padded_dataset(n_events=(6,), energy_ndim=ndim)

        per_event, _ = read_write.read_raw_regaxes(
            config, part="train", pick_events=slice(None), per_event_cols=["energy"]
        )

        assert per_event.shape == (6,)
        np.testing.assert_allclose(per_event, truth["energy"])

    def test_default_per_event_cols_is_energy(self, padded_dataset):
        config, truth = padded_dataset(n_events=(6,))

        per_event, _ = read_write.read_raw_regaxes(
            config, part="train", pick_events=slice(None)
        )

        np.testing.assert_allclose(per_event, truth["energy"])

    def test_pick_events_selects_exactly_those_events(self, padded_dataset):
        config, truth = padded_dataset(n_events=(4, 6, 5))
        wanted = [0, 3, 4, 9, 14]

        per_event, events = read_write.read_raw_regaxes(
            config, part="train", pick_events=wanted
        )

        np.testing.assert_allclose(per_event, truth["energy"][wanted])
        np.testing.assert_allclose(events, truth["events"][wanted])

    def test_pick_events_crossing_a_file_boundary(self, padded_dataset):
        config, truth = padded_dataset(n_events=(4, 6, 5))
        wanted = [3, 4]  # last of file 0, first of file 1

        per_event, _ = read_write.read_raw_regaxes(
            config, part="train", pick_events=wanted
        )

        np.testing.assert_allclose(per_event, truth["energy"][wanted])

    def test_total_size_selects_evenly_spaced_events(self, padded_dataset):
        config, truth = padded_dataset(n_events=(4, 6, 5))

        per_event, _ = read_write.read_raw_regaxes(
            config, part="train", total_size=5
        )

        expected = np.linspace(0, 14, 5).astype(int)
        np.testing.assert_allclose(per_event, truth["energy"][expected])

    def test_total_size_minus_one_reads_everything(self, padded_dataset):
        config, truth = padded_dataset(n_events=(4, 6, 5))

        per_event, events = read_write.read_raw_regaxes(
            config, part="train", total_size=-1
        )

        assert len(per_event) == 15
        np.testing.assert_allclose(events, truth["events"])

    def test_total_size_is_capped_at_the_dataset_size(self, padded_dataset):
        config, _ = padded_dataset(n_events=(4,))

        per_event, _ = read_write.read_raw_regaxes(
            config, part="train", total_size=1000
        )

        assert len(per_event) == 4

    def test_total_size_zero_returns_empty_arrays(self, padded_dataset):
        config, _ = padded_dataset(n_events=(4,))

        per_event, events = read_write.read_raw_regaxes(
            config, part="train", total_size=0
        )

        assert len(per_event) == 0 and len(events) == 0

    def test_out_of_range_pick_events(self, padded_dataset):
        config, _ = padded_dataset(n_events=(4,))

        with pytest.raises(AssertionError, match="out of range"):
            read_write.read_raw_regaxes(config, part="train", pick_events=[0, 99])

    def test_files_of_different_widths_are_padded_at_the_back(self, padded_dataset):
        config, truth = padded_dataset(n_events=(3, 3), n_points=(4, 7))

        _, events = read_write.read_raw_regaxes(
            config, part="train", pick_events=slice(None)
        )

        assert events.shape == (6, 7, 4)
        np.testing.assert_allclose(events[:3, :4], truth["per_file"][0]["events"])
        np.testing.assert_allclose(events[:3, 4:], 0.0)

    def test_orientation_is_applied_on_the_way_out(self, padded_dataset):
        config, truth = padded_dataset(n_events=(3,))
        config["data"]["orientation"] = "hdf5:xyz==local:zxy"

        _, events = read_write.read_raw_regaxes(
            config, part="train", pick_events=slice(None)
        )

        expected = truth["events"].copy()
        expected[..., [2, 0, 1]] = truth["events"][..., [0, 1, 2]]
        np.testing.assert_allclose(events, expected)

    def test_selection_confined_to_one_of_several_files(self, padded_dataset):
        """Every event comes from file 0, so files 1 and 2 are handed an empty
        index array.  If this fails inside h5py, ``_read_padded`` needs to skip
        files with no selected events rather than slicing them with ``[]``.
        """
        config, truth = padded_dataset(n_events=(4, 6, 5))

        per_event, events = read_write.read_raw_regaxes(
            config, part="train", pick_events=[0, 1]
        )

        assert len(per_event) == 2
        np.testing.assert_allclose(events, truth["events"][[0, 1]])


# --------------------------------------------------------------------------- #
# _read_padded -- the per-event axis sniffing deserves direct coverage
# --------------------------------------------------------------------------- #
class TestReadPaddedPerEventLayouts:
    @pytest.fixture
    def written(self, tmp_path):
        path = tmp_path / "odd_layout.h5"
        n_events, n_points = 6, 5
        events = np.arange(n_events * n_points * 4, dtype=float).reshape(
            n_events, n_points, 4
        )
        events[..., 3] = 1.0
        with h5py.File(path, "w") as handle:
            handle.create_dataset("events", data=events)
            handle.create_dataset("leading_one", data=np.arange(n_events)[None, :])
            handle.create_dataset(
                "leading_three", data=np.arange(3 * n_events).reshape(3, n_events)
            )
        return str(path), events

    @pytest.mark.parametrize(
        "key, width", [("leading_one", 1), ("leading_three", 3)]
    )
    def test_event_axis_is_found_even_when_it_is_not_first(
        self, tmp_path, written, key, width
    ):
        path, _ = written
        config = base_config(str(tmp_path / "unused_{}.h5"))
        indices = np.array([1, 2, 3, 4])

        per_event, events = read_write._read_padded(
            config, [path], [indices], [key]
        )

        assert per_event[0].shape == (4, width)
        assert events[0].shape == (4, 5, 4)


# --------------------------------------------------------------------------- #
# check_regaxes
# --------------------------------------------------------------------------- #
class TestCheckRegaxes:
    def test_accepts_well_shaped_data(self):
        read_write.check_regaxes(np.zeros(5), np.zeros((5, 10, 4)))

    def test_accepts_zero_events(self):
        read_write.check_regaxes(np.zeros(0), np.zeros((0, 10, 4)))

    @pytest.mark.parametrize(
        "energy, events, message",
        [
            (np.zeros((5, 1)), np.zeros((5, 10, 4)), "1D"),
            (np.zeros(5), np.zeros((5, 4)), "3D"),
            (np.zeros(5), np.zeros((5, 10, 4, 1)), "3D"),
            (np.zeros(5), np.zeros((6, 10, 4)), "do not match"),
            (np.zeros(5), np.zeros((5, 10, 3)), "4 coordinates"),
        ],
        ids=["energy_2d", "events_2d", "events_4d", "count_mismatch", "wrong_width"],
    )
    def test_rejects_bad_shapes(self, energy, events, message):
        with pytest.raises(ValueError, match=message):
            read_write.check_regaxes(energy, events)


# --------------------------------------------------------------------------- #
# write_raw_regaxes
# --------------------------------------------------------------------------- #
class TestWriteRawRegaxes:
    def test_round_trips_through_h5(self, tmp_path):
        energy = np.arange(5, dtype=float)
        events = np.arange(5 * 3 * 4, dtype=float).reshape(5, 3, 4)
        destination = tmp_path / "out.h5"

        read_write.write_raw_regaxes(str(destination), energy, events)

        with h5py.File(destination, "r") as handle:
            assert set(handle) == {"energy", "events"}
            np.testing.assert_allclose(handle["energy"][:], energy)
            np.testing.assert_allclose(handle["events"][:], events)

    def test_output_is_readable_by_read_raw_regaxes(self, tmp_path):
        energy = np.arange(1.0, 6.0)
        events = np.zeros((5, 3, 4))
        events[..., 3] = 1.0
        read_write.write_raw_regaxes(str(tmp_path / "out_0.h5"), energy, events)

        config = base_config(str(tmp_path / "out_{}.h5"), roll_axis=False)
        config["data"]["train_range_end"] = 1

        per_event, read_back = read_write.read_raw_regaxes(
            config, part="train", pick_events=slice(None)
        )

        np.testing.assert_allclose(per_event, energy)
        np.testing.assert_allclose(read_back, events)

    def test_validation_runs_before_the_file_is_created(self, tmp_path):
        destination = tmp_path / "out.h5"

        with pytest.raises(ValueError):
            read_write.write_raw_regaxes(
                str(destination), np.zeros((5, 1)), np.zeros((5, 3, 4))
            )

        assert not destination.exists()

    def test_existing_file_is_overwritten(self, tmp_path):
        destination = tmp_path / "out.h5"
        with h5py.File(destination, "w") as handle:
            handle.create_dataset("stale", data=np.zeros(3))

        read_write.write_raw_regaxes(
            str(destination), np.zeros(2), np.zeros((2, 3, 4))
        )

        with h5py.File(destination, "r") as handle:
            assert "stale" not in handle
