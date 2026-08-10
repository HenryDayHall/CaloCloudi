"""Tests for ``src/detector_map.py``.

The geometry helpers are pure functions of numpy arrays, so most of these are
plain value tests.  ``load_muon_map`` touches disk and ``create_map`` calls it,
so those two use ``tmp_path`` / ``mocker`` respectively.
"""

import os

import numpy as np
import pytest

from src import detector_map


# --------------------------------------------------------------------------- #
# load_muon_map
# --------------------------------------------------------------------------- #
class TestLoadMuonMap:
    def test_returns_x_y_z_e_in_order(self, tmp_path):
        data_dir = tmp_path / "muon_map"
        data_dir.mkdir()
        expected = {
            "X": np.array([1.0, 2.0]),
            "Y": np.array([3.0, 4.0]),
            "Z": np.array([5.0, 6.0]),
            "E": np.array([7.0, 8.0]),
        }
        for name, array in expected.items():
            np.save(data_dir / f"{name}.npy", array)

        loaded = detector_map.load_muon_map(assets_dir=str(tmp_path))

        assert len(loaded) == 4
        for got, name in zip(loaded, ["X", "Y", "Z", "E"]):
            np.testing.assert_array_equal(got, expected[name])

    def test_trailing_slash_on_assets_dir_is_tolerated(self, tmp_path):
        data_dir = tmp_path / "muon_map"
        data_dir.mkdir()
        for name in "XYZE":
            np.save(data_dir / f"{name}.npy", np.zeros(1))

        assert detector_map.load_muon_map(assets_dir=str(tmp_path) + "/")

    def test_missing_assets_raise(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            detector_map.load_muon_map(assets_dir=str(tmp_path))

    def test_default_dir_is_assets_next_to_src(self, mocker):
        load = mocker.patch("src.detector_map.np.load", return_value=np.zeros(1))

        detector_map.load_muon_map()

        paths = [os.path.normpath(call.args[0]) for call in load.call_args_list]
        assert [os.path.basename(p) for p in paths] == [
            "X.npy",
            "Y.npy",
            "Z.npy",
            "E.npy",
        ]
        expected_dir = os.path.normpath(
            os.path.join(os.path.dirname(detector_map.__file__), "..", "assets", "muon_map")
        )
        assert all(os.path.dirname(p) == expected_dir for p in paths)


# --------------------------------------------------------------------------- #
# confine_to_box
# --------------------------------------------------------------------------- #
@pytest.fixture
def box_config():
    """Detector box is x(-10,10), y(100,111), z(-20,20).

    Data box is x(-1,1), y(-2,2), z(0,6).  The two are deliberately disjoint so
    a test can tell which branch ran.
    """
    return {
        "data": {
            "Xmin_in_detector": -10.0,
            "Xmax_in_detector": 10.0,
            "Zmin_in_detector": -20.0,
            "Zmax_in_detector": 20.0,
            "cell_thickness": 1.0,
            "Xmin": -1.0,
            "Xmax": 1.0,
            "Ymin": -2.0,
            "Ymax": 2.0,
            "layer_bottom_pos": np.array([0.0, 5.0]),
        },
        "detector": {"layer_bottom_pos": np.array([100.0, 110.0])},
    }


def _confine(configs, points, detector_coords=True):
    points = np.asarray(points, dtype=float)
    return detector_map.confine_to_box(
        configs, points[:, 0], points[:, 1], points[:, 2], points[:, 3],
        detector_coords=detector_coords,
    )


class TestConfineToBox:
    def test_keeps_interior_hits_and_preserves_pairing(self, box_config):
        # column 3 doubles as a label so we can check E is filtered in step
        points = [
            [0.0, 105.0, 0.0, 1.0],
            [50.0, 105.0, 0.0, 2.0],   # x too big
            [0.0, 105.0, 0.0, 3.0],
        ]
        X, Y, Z, E = _confine(box_config, points)

        np.testing.assert_array_equal(E, [1.0, 3.0])
        np.testing.assert_array_equal(X, [0.0, 0.0])
        np.testing.assert_array_equal(Y, [105.0, 105.0])
        np.testing.assert_array_equal(Z, [0.0, 0.0])

    @pytest.mark.parametrize(
        "point",
        [
            (-50.0, 105.0, 0.0),   # below Xmin
            (50.0, 105.0, 0.0),    # above Xmax
            (0.0, 50.0, 0.0),      # below Ymin
            (0.0, 500.0, 0.0),     # above Ymax
            (0.0, 105.0, -50.0),   # below Zmin
            (0.0, 105.0, 50.0),    # above Zmax
        ],
    )
    def test_drops_hits_outside_each_face(self, box_config, point):
        X, _, _, _ = _confine(box_config, [list(point) + [1.0]])
        assert len(X) == 0

    @pytest.mark.parametrize(
        "point",
        [
            (-10.0, 105.0, 0.0),
            (10.0, 105.0, 0.0),
            (0.0, 100.0, 0.0),     # == layer_bottom_pos[0]
            (0.0, 111.0, 0.0),     # == layer_bottom_pos[-1] + cell_thickness
            (0.0, 105.0, -20.0),
            (0.0, 105.0, 20.0),
        ],
    )
    def test_bounds_are_exclusive(self, box_config, point):
        """The comparisons are strict (``>``/``<``), so hits exactly on a face
        are dropped.  Pinned because it is easy to "fix" by accident."""
        X, _, _, _ = _confine(box_config, [list(point) + [1.0]])
        assert len(X) == 0

    def test_detector_flag_selects_the_other_key_set(self, box_config):
        # inside the detector box, outside the data box
        point = [[0.0, 105.0, 0.0, 1.0]]
        assert len(_confine(box_config, point, detector_coords=True)[0]) == 1
        assert len(_confine(box_config, point, detector_coords=False)[0]) == 0

        # inside the data box, outside the detector box
        point = [[0.0, 0.0, 3.0, 1.0]]
        assert len(_confine(box_config, point, detector_coords=True)[0]) == 0
        assert len(_confine(box_config, point, detector_coords=False)[0]) == 1

    def test_detector_y_ceiling_uses_the_data_cell_thickness(self, box_config):
        """Documents a cross-namespace read: in the ``detector_coords`` branch
        Ymax is ``detector.layer_bottom_pos[-1] + data.cell_thickness``, not
        ``detector.cell_thickness``.  Probably a bug -- flagged, not asserted
        as correct."""
        point = [[0.0, 115.0, 0.0, 1.0]]
        assert len(_confine(box_config, point)[0]) == 0

        box_config["data"]["cell_thickness"] = 20.0
        assert len(_confine(box_config, point)[0]) == 1

    def test_empty_input(self, box_config):
        empty = np.zeros(0)
        out = detector_map.confine_to_box(box_config, empty, empty, empty, empty)
        assert all(len(a) == 0 for a in out)


# --------------------------------------------------------------------------- #
# create_map
# --------------------------------------------------------------------------- #
@pytest.fixture
def map_config():
    return {
        "data": {
            "divisions_per_cell": 2,
            "cell_thickness": 1.0,
            "Xmin_in_detector": -10.0,
            "Xmax_in_detector": 10.0,
            "Zmin_in_detector": -10.0,
            "Zmax_in_detector": 10.0,
        },
        "detector": {
            "layer_bottom_pos": np.array([0.0, 10.0]),
            "cell_size": 2.0,
            "cell_thickness": 1.0,
        },
    }


@pytest.fixture
def tiny_muon_map():
    """A 3x3 grid of cell centres in each of two layers -- 18 hits total."""
    centres = [-2.0, 0.0, 2.0]
    xs, ys, zs = [], [], []
    for layer_y in (0.2, 10.2):
        for x in centres:
            for z in centres:
                xs.append(x)
                ys.append(layer_y)
                zs.append(z)
    return (
        np.array(xs),
        np.array(ys),
        np.array(zs),
        np.ones(len(xs)),
    )


class TestCreateMap:
    def test_one_entry_per_layer_and_offset(self, mocker, map_config, tiny_muon_map):
        mocker.patch("src.detector_map.load_muon_map", return_value=tiny_muon_map)

        layers, offset = detector_map.create_map(map_config)

        assert len(layers) == len(map_config["detector"]["layer_bottom_pos"])
        assert offset == pytest.approx(2.0 / 2)  # cell_size / divisions_per_cell

    def test_grid_shape_matches_its_own_edges(self, mocker, map_config, tiny_muon_map):
        mocker.patch("src.detector_map.load_muon_map", return_value=tiny_muon_map)

        layers, _ = detector_map.create_map(map_config)

        for layer in layers:
            assert set(layer) == {"xedges", "zedges", "grid"}
            assert layer["grid"].shape == (
                len(layer["xedges"]) - 1,
                len(layer["zedges"]) - 1,
            )
            for edges in (layer["xedges"], layer["zedges"]):
                assert np.all(np.diff(edges) > 0), "edges must be sorted and unique"

    def test_every_hit_lands_in_its_layer(self, mocker, map_config, tiny_muon_map):
        mocker.patch("src.detector_map.load_muon_map", return_value=tiny_muon_map)

        layers, _ = detector_map.create_map(map_config)

        assert [layer["grid"].sum() for layer in layers] == [9, 9]

    def test_out_of_box_hits_are_discarded_first(self, mocker, map_config, tiny_muon_map):
        X, Y, Z, E = tiny_muon_map
        stray = (
            np.append(X, 500.0),
            np.append(Y, 0.2),
            np.append(Z, 0.0),
            np.append(E, 1.0),
        )
        mocker.patch("src.detector_map.load_muon_map", return_value=stray)

        layers, _ = detector_map.create_map(map_config)

        assert [layer["grid"].sum() for layer in layers] == [9, 9]
        assert layers[0]["xedges"].max() < 100, "stray hit should not widen the grid"

    def test_confine_argument_is_currently_ignored(self, mocker, map_config, tiny_muon_map):
        """``create_map(configs, confine=...)`` never reads ``confine`` -- it
        always confines.  Pinned so a future signature change is deliberate."""
        mocker.patch("src.detector_map.load_muon_map", return_value=tiny_muon_map)

        with_confine, _ = detector_map.create_map(map_config, confine=True)
        without, _ = detector_map.create_map(map_config, confine=False)

        for a, b in zip(with_confine, without):
            np.testing.assert_array_equal(a["grid"], b["grid"])


# --------------------------------------------------------------------------- #
# get_layer_centers
# --------------------------------------------------------------------------- #
class TestGetLayerCenters:
    @pytest.fixture
    def config(self):
        return {
            "data": {"layer_bottom_pos": [0.0, 2.0, 4.0], "cell_thickness": 1.0},
            "detector": {"layer_bottom_pos": [100.0, 110.0], "cell_thickness": 0.5},
        }

    def test_data_coordinates(self, config):
        centers = detector_map.get_layer_centers(config, coordinates="data")
        np.testing.assert_allclose(centers, [0.5, 2.5, 4.5])

    def test_detector_coordinates(self, config):
        centers = detector_map.get_layer_centers(config, coordinates="detector")
        np.testing.assert_allclose(centers, [100.25, 110.25])

    def test_default_is_data(self, config):
        np.testing.assert_allclose(
            detector_map.get_layer_centers(config),
            detector_map.get_layer_centers(config, coordinates="data"),
        )

    def test_returns_array_without_mutating_the_config(self, config):
        before = list(config["data"]["layer_bottom_pos"])
        centers = detector_map.get_layer_centers(config)
        assert isinstance(centers, np.ndarray)
        assert config["data"]["layer_bottom_pos"] == before

    def test_unknown_coordinate_system(self, config):
        """Currently an ``UnboundLocalError`` from the un-taken if/elif.  A
        ``ValueError`` would be friendlier -- change this test when it is."""
        with pytest.raises(UnboundLocalError):
            detector_map.get_layer_centers(config, coordinates="galactic")


# --------------------------------------------------------------------------- #
# floors_ceilings
# --------------------------------------------------------------------------- #
class TestFloorsCeilings:
    WELL_SEPARATED = np.array([0.0, 10.0, 20.0])

    def test_zero_buffer_gives_exactly_the_cells(self):
        floors, ceilings = detector_map.floors_ceilings(
            self.WELL_SEPARATED, 1.0, percent_buffer=0.0
        )
        np.testing.assert_allclose(floors, self.WELL_SEPARATED)
        np.testing.assert_allclose(ceilings, self.WELL_SEPARATED + 1.0)

    def test_buffer_extends_symmetrically_when_there_is_room(self):
        floors, ceilings = detector_map.floors_ceilings(
            self.WELL_SEPARATED, 1.0, percent_buffer=0.5
        )
        np.testing.assert_allclose(floors, self.WELL_SEPARATED - 0.5)
        np.testing.assert_allclose(ceilings, self.WELL_SEPARATED + 1.5)

    def test_layers_never_overlap(self):
        # tight packing: gap between layers is smaller than the buffer wants
        bottoms = np.array([0.0, 1.2, 2.4, 3.6])
        floors, ceilings = detector_map.floors_ceilings(bottoms, 1.0, percent_buffer=5.0)
        assert np.all(ceilings[:-1] <= floors[1:])

    def test_floor_below_ceiling_everywhere(self):
        bottoms = np.array([0.0, 1.2, 2.4, 3.6])
        floors, ceilings = detector_map.floors_ceilings(bottoms, 1.0, percent_buffer=0.5)
        assert np.all(floors < ceilings)

    def test_every_cell_stays_inside_its_own_layer(self):
        bottoms = np.array([0.0, 1.2, 2.4])
        thickness = 1.0
        floors, ceilings = detector_map.floors_ceilings(bottoms, thickness, 0.5)
        assert np.all(floors <= bottoms)
        assert np.all(ceilings >= np.minimum(bottoms + thickness, np.append(bottoms[1:], np.inf)))

    def test_coverage_grows_monotonically_with_buffer(self):
        bottoms = np.array([0.0, 1.2, 2.4, 3.6])
        small = detector_map.floors_ceilings(bottoms, 1.0, percent_buffer=0.1)
        large = detector_map.floors_ceilings(bottoms, 1.0, percent_buffer=0.9)
        assert np.all(large[0] <= small[0])
        assert np.all(large[1] >= small[1])

    def test_single_layer(self):
        floors, ceilings = detector_map.floors_ceilings(np.array([5.0]), 2.0, 0.5)
        np.testing.assert_allclose(floors, [4.0])
        np.testing.assert_allclose(ceilings, [8.0])

    def test_output_length_matches_input(self):
        bottoms = np.arange(30.0)
        floors, ceilings = detector_map.floors_ceilings(bottoms, 1.0)
        assert floors.shape == ceilings.shape == bottoms.shape

    def test_input_is_not_mutated(self):
        bottoms = self.WELL_SEPARATED.copy()
        detector_map.floors_ceilings(bottoms, 1.0, 0.5)
        np.testing.assert_array_equal(bottoms, self.WELL_SEPARATED)

    def test_requires_an_ndarray_not_a_list(self):
        """A plain list raises ``TypeError`` on the first subtraction.

        This matters: ``dataset.AbstractBase.fuzz_perpendicular`` passes
        ``config["data"]["layer_bottom_pos"]`` straight through, and that comes
        out of YAML as a list.  Either this function should coerce, or the
        caller should.
        """
        with pytest.raises(TypeError):
            detector_map.floors_ceilings([0.0, 10.0], 1.0)


# --------------------------------------------------------------------------- #
# find_layers
# --------------------------------------------------------------------------- #
class TestFindLayers:
    @pytest.fixture
    def config(self):
        return {
            "data": {"layer_bottom_pos": [0.0, 10.0, 20.0], "cell_thickness": 1.0},
            "detector": {"layer_bottom_pos": [100.0, 110.0], "cell_thickness": 1.0},
        }

    @staticmethod
    def points(rows):
        """rows are (x, y, z, e) -> a single-event array of shape (1, n, 4)."""
        return np.array([rows], dtype=float)

    def test_data_coordinates_use_column_two(self, config):
        pts = self.points([[0, 0, 0.5, 1.0], [0, 0, 10.5, 1.0], [0, 0, 20.5, 1.0]])
        np.testing.assert_array_equal(
            detector_map.find_layers(config, pts, coordinates="data"), [[0, 1, 2]]
        )

    def test_detector_coordinates_use_column_one(self, config):
        # column 1 is in the detector layers, column 2 is not -- only one of the
        # two can be the height, so this pins which.
        pts = self.points([[0, 100.5, 999.0, 1.0], [0, 110.5, 999.0, 1.0]])
        np.testing.assert_array_equal(
            detector_map.find_layers(config, pts, coordinates="detector"), [[0, 1]]
        )

    def test_padding_points_are_minus_one(self, config):
        pts = self.points([[0, 0, 0.5, 0.0], [0, 0, 0.5, -1.0], [0, 0, 0.5, 1.0]])
        np.testing.assert_array_equal(
            detector_map.find_layers(config, pts), [[-1, -1, 0]]
        )

    def test_points_outside_all_layers_are_minus_one(self, config):
        pts = self.points([[0, 0, -5.0, 1.0], [0, 0, 5.0, 1.0], [0, 0, 500.0, 1.0]])
        np.testing.assert_array_equal(
            detector_map.find_layers(config, pts), [[-1, -1, -1]]
        )

    def test_floor_inclusive_ceiling_exclusive(self, config):
        floors, ceilings = detector_map.floors_ceilings(
            np.array(config["data"]["layer_bottom_pos"]),
            config["data"]["cell_thickness"],
            percent_buffer=0.5,
        )
        pts = self.points([[0, 0, floors[1], 1.0], [0, 0, ceilings[1], 1.0]])
        found = detector_map.find_layers(config, pts)
        assert found[0, 0] == 1
        assert found[0, 1] == -1

    def test_shape_and_dtype(self, config):
        pts = np.zeros((4, 7, 4))
        pts[..., 3] = 1.0
        found = detector_map.find_layers(config, pts)
        assert found.shape == (4, 7)
        assert np.issubdtype(found.dtype, np.integer)

    def test_uses_the_half_cell_buffer(self, config):
        """0.4 is outside the cell (0, 1) shifted down, but inside the buffered
        floor at -0.5, so it should still be assigned to layer 0."""
        pts = self.points([[0, 0, -0.4, 1.0]])
        assert detector_map.find_layers(config, pts)[0, 0] == 0

    def test_unknown_coordinate_system(self, config):
        pts = self.points([[0, 0, 0.5, 1.0]])
        with pytest.raises(UnboundLocalError):
            detector_map.find_layers(config, pts, coordinates="galactic")
