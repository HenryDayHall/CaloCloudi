"""Tests for ``src.data.dataset.from_config``.

The dispatch itself is tested with mocks; a small set of end-to-end tests then
builds real ``padded`` / ``padded_unordered`` datasets from temporary h5 files,
because dispatching to a class that cannot be constructed is not much use.
"""

import numpy as np
import pytest

pytest.importorskip("torch")

from src.data import dataset  # noqa: E402

from conftest import base_config, make_padded_file  # noqa: E402


CLASSES = ["PointCloudDataset", "PointCloudDatasetUnordered", "ShowerDataDataset"]


# --------------------------------------------------------------------------- #
# dispatch
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
# end to end on real files
# --------------------------------------------------------------------------- #
@pytest.fixture
def padded_config(tmp_path):
    def _build(fmt="padded", roll_axis=False, n_events=6, n_points=5, **kwargs):
        truth = make_padded_file(
            tmp_path / "data_0.h5",
            n_events=n_events,
            n_points=n_points,
            roll_axis=roll_axis,
            **kwargs,
        )
        config = base_config(
            str(tmp_path / "data_{}.h5"), fmt=fmt, roll_axis=roll_axis
        )
        config["data"]["train_range_start"] = 0
        config["data"]["train_range_end"] = 1
        return config, truth

    return _build


class TestFromConfigEndToEnd:
    @pytest.mark.parametrize(
        "fmt, expected",
        [
            ("padded", dataset.PointCloudDataset),
            ("padded_unordered", dataset.PointCloudDatasetUnordered),
        ],
    )
    def test_builds_a_usable_dataset(self, padded_config, fmt, expected):
        config, truth = padded_config(fmt=fmt)

        built = dataset.from_config(config, dataset_part="train")

        assert isinstance(built, expected)
        assert len(built) == len(truth["energy"])

    def test_batch_has_the_configured_keys(self, padded_config):
        config, _ = padded_config()

        batch = dataset.from_config(config, dataset_part="train")[0]

        assert set(batch) == {
            "incident_energy",
            "incident_direction",
            "points",
            "n_points",
        }

    def test_batch_shapes(self, padded_config):
        config, _ = padded_config()
        batch_size = config["training"]["batch_size"]

        batch = dataset.from_config(config, dataset_part="train")[0]

        assert batch["points"].shape[0] == batch_size
        assert batch["points"].shape[2] == 4
        assert batch["incident_energy"].shape == (batch_size, 1)
        assert batch["incident_direction"].shape == (batch_size, 3)
        assert batch["n_points"].shape == (batch_size, 1)

    def test_padding_is_trimmed_to_the_widest_event_in_the_batch(self, padded_config):
        config, _ = padded_config()

        batch = dataset.from_config(config, dataset_part="train")[0]

        assert batch["points"].shape[1] == batch["n_points"].max()

    def test_rolled_axes_come_back_as_points_by_features(self, padded_config):
        config, _ = padded_config(roll_axis=True)

        batch = dataset.from_config(config, dataset_part="train")[0]

        assert batch["points"].shape[2] == 4

    def test_events_are_ordered_by_point_count(self, padded_config):
        config, _ = padded_config()

        built = dataset.from_config(config, dataset_part="train")

        counts = built.index_list[:, 0]
        assert np.all(np.diff(counts) >= 0)

    def test_unordered_selection_is_reproducible(self, padded_config):
        config, _ = padded_config(fmt="padded_unordered", n_events=20)
        built = dataset.from_config(config, dataset_part="train")

        assert np.array_equal(built.choose_idxs(3), built.choose_idxs(3))

    def test_unordered_selection_varies_with_the_index(self, padded_config):
        config, _ = padded_config(fmt="padded_unordered", n_events=20)
        built = dataset.from_config(config, dataset_part="train")

        selections = {tuple(built.choose_idxs(i)) for i in range(5)}
        assert len(selections) > 1

    def test_unordered_selection_never_repeats_an_event(self, padded_config):
        config, _ = padded_config(fmt="padded_unordered", n_events=20)
        built = dataset.from_config(config, dataset_part="train")

        idxs = built.choose_idxs(0)
        assert len(set(idxs)) == len(idxs)
        assert np.all(np.diff(idxs) > 0), "must stay sorted for h5py fancy indexing"

    @pytest.mark.xfail(
        reason=(
            "n_points stored as (n_events, 1) is not squeezed in "
            "_make_index_list, so np.array() on the ragged tuples fails"
        ),
        strict=False,
    )
    def test_n_points_with_a_trailing_axis(self, padded_config):
        config, truth = padded_config(n_points_ndim=2)

        built = dataset.from_config(config, dataset_part="train")

        assert len(built) == len(truth["energy"])

    def test_missing_files(self, tmp_path):
        config = base_config(str(tmp_path / "nothing_{}.h5"))

        with pytest.raises(FileNotFoundError):
            dataset.from_config(config, dataset_part="train")
