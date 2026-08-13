"""Tests for ``src/detector_map.py``.

The module is a mix of pure geometry (``floors_ceilings``, ``get_layer_centers``,
``find_layers``, ``detector_cell_thickness``), a simple filter
(``confine_to_box``), and one heavyweight function that turns a muon map into a
per-layer cell grid (``create_map``).

``create_map`` calls ``load_muon_map()`` with no arguments, so there is no way
to hand it a map; the tests patch ``load_muon_map`` and feed it a synthetic grid
of cell centres instead of shipping a fixture file.

Two coordinate conventions run through the module: in ``data`` coordinates the
layers stack along z with a single ``cell_thickness``, in ``detector``
coordinates they stack along y and the thickness is per layer -- ecal below
``hcal_start`` and hcal from it on. The fixture puts its first two layers in the
ecal and its last in the hcal, so a test that passes has read the right one.

Tests carrying a docstring record current behaviour that looks unintended.
"""

import os

import numpy as np
import pytest

from src import detector_map

from conftest import base_config


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def muon_grid(config, x_centres=(-10.0, -5.0, 0.0, 5.0, 10.0), z_centres=None,
              layer_indices=None):
    """A synthetic muon map: one hit at every (x, z) cell centre of every layer.

    Returned in the ``(X, Y, Z, E)`` order ``load_muon_map`` uses. Each hit sits
    at the middle of its own layer -- which depends on whether that layer is
    ecal or hcal -- so it falls inside both the confinement box and the
    per-layer band.
    """
    if z_centres is None:
        z_centres = x_centres
    bottoms = np.array(config["detector"]["layer_bottom_pos"], dtype=float)
    thickness = detector_map.detector_cell_thickness(config)
    if layer_indices is None:
        layer_indices = range(len(bottoms))

    xs, ys, zs = [], [], []
    for layer_n in layer_indices:
        height = bottoms[layer_n] + thickness[layer_n] / 2
        for x in x_centres:
            for z in z_centres:
                xs.append(x)
                zs.append(z)
                ys.append(height)

    xs = np.array(xs, dtype=float)
    return xs, np.array(ys, dtype=float), np.array(zs, dtype=float), np.ones_like(xs)


def one_point(height, energy=1.0, axis=2):
    """A ``(1, 1, 4)`` point array with its height on the requested axis."""
    points = np.zeros((1, 1, 4))
    points[0, 0, axis] = height
    points[0, 0, 3] = energy
    return points


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def config():
    """Mirrors ``config/default.yaml``: ecal/hcal thicknesses, no ``cell_thickness``.

    ``data`` layers at 0/1/2 with unit cells; ``detector`` layers at 1811/1814/1823
    with the last one in the hcal. The two sections deliberately disagree on
    every number, so a test that passes has read from the section it meant to.
    """
    return base_config("unused")


@pytest.fixture
def fake_muon_map(mocker):
    """Factory: replace ``load_muon_map`` with a fixed set of arrays."""

    def _patch(arrays):
        mocker.patch("src.detector_map.load_muon_map", return_value=arrays)
        return arrays

    return _patch


# --------------------------------------------------------------------------- #
# load_muon_map
# --------------------------------------------------------------------------- #
class TestLoadMuonMap:
    @pytest.fixture
    def assets(self, tmp_path):
        """An assets directory holding a ``muon_map`` with distinct arrays."""
        data_dir = tmp_path / "muon_map"
        data_dir.mkdir()
        contents = {"X": [1.0, 2.0], "Y": [3.0, 4.0], "Z": [5.0, 6.0], "E": [7.0, 8.0]}
        for name, values in contents.items():
            np.save(data_dir / f"{name}.npy", np.array(values))
        return tmp_path, contents

    def test_returns_x_y_z_e_in_that_order(self, assets):
        assets_dir, contents = assets

        X, Y, Z, E = detector_map.load_muon_map(str(assets_dir))

        np.testing.assert_array_equal(X, contents["X"])
        np.testing.assert_array_equal(Y, contents["Y"])
        np.testing.assert_array_equal(Z, contents["Z"])
        np.testing.assert_array_equal(E, contents["E"])

    def test_returns_arrays(self, assets):
        assets_dir, _ = assets

        loaded = detector_map.load_muon_map(str(assets_dir))

        assert all(isinstance(array, np.ndarray) for array in loaded)

    def test_a_trailing_separator_is_harmless(self, assets):
        assets_dir, contents = assets

        X, _, _, _ = detector_map.load_muon_map(str(assets_dir) + os.sep)

        np.testing.assert_array_equal(X, contents["X"])

    def test_the_default_directory_sits_beside_the_package(self, mocker):
        load = mocker.patch("src.detector_map.np.load", return_value=np.zeros(1))

        detector_map.load_muon_map()

        module_dir = os.path.dirname(detector_map.__file__)
        asked_for = [os.path.normpath(call.args[0]) for call in load.call_args_list]
        assert asked_for == [
            os.path.normpath(os.path.join(module_dir, "..", "assets", "muon_map", name))
            for name in ("X.npy", "Y.npy", "Z.npy", "E.npy")
        ]

    def test_a_missing_directory_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            detector_map.load_muon_map(str(tmp_path / "nowhere"))

    def test_a_partial_directory_raises(self, tmp_path):
        data_dir = tmp_path / "muon_map"
        data_dir.mkdir()
        np.save(data_dir / "X.npy", np.zeros(2))

        with pytest.raises(FileNotFoundError):
            detector_map.load_muon_map(str(tmp_path))


# --------------------------------------------------------------------------- #
# detector_cell_thickness
# --------------------------------------------------------------------------- #
class TestDetectorCellThickness:
    """The fixture has three layers with ``hcal_start`` of 2, so layers 0 and 1
    are ecal (0.5) and layer 2 is hcal (2.0).
    """

    def test_one_thickness_per_layer(self, config):
        thickness = detector_map.detector_cell_thickness(config)

        assert len(thickness) == len(config["detector"]["layer_bottom_pos"])

    def test_the_split_falls_at_hcal_start(self, config):
        thickness = detector_map.detector_cell_thickness(config)

        np.testing.assert_allclose(thickness, [0.5, 0.5, 2.0])

    def test_returns_an_array(self, config):
        assert isinstance(detector_map.detector_cell_thickness(config), np.ndarray)

    def test_hcal_start_of_zero_makes_every_layer_hcal(self, config):
        config["detector"]["hcal_start"] = 0

        thickness = detector_map.detector_cell_thickness(config)

        np.testing.assert_allclose(thickness, 2.0)

    @pytest.mark.parametrize("hcal_start", [3, 5, 99])
    def test_hcal_start_past_the_stack_makes_every_layer_ecal(self, config, hcal_start):
        """``config/default.yaml`` is this case: 30 layers and ``hcal_start`` of
        30, i.e. an ecal-only detector."""
        config["detector"]["hcal_start"] = hcal_start

        thickness = detector_map.detector_cell_thickness(config)

        np.testing.assert_allclose(thickness, 0.5)

    def test_a_negative_hcal_start_counts_from_the_end(self, config):
        config["detector"]["hcal_start"] = -1

        thickness = detector_map.detector_cell_thickness(config)

        np.testing.assert_allclose(thickness, [0.5, 0.5, 2.0])

    def test_the_config_is_left_alone(self, config):
        original = list(config["detector"]["layer_bottom_pos"])

        detector_map.detector_cell_thickness(config)

        assert config["detector"]["layer_bottom_pos"] == original

    def test_the_result_is_float_whatever_the_yaml_holds(self, config):
        """``np.full`` would otherwise take its dtype from the ecal value, so a
        thickness written without a decimal point in YAML would give an integer
        array and silently truncate the hcal one into it.
        """
        config["detector"]["cell_thickness_ecal"] = 1
        config["detector"]["cell_thickness_hcal"] = 0.5

        thickness = detector_map.detector_cell_thickness(config)

        assert thickness.dtype == float
        np.testing.assert_allclose(thickness, [1.0, 1.0, 0.5])

    def test_integer_thicknesses_are_still_exact(self, config):
        config["detector"]["cell_thickness_ecal"] = 1
        config["detector"]["cell_thickness_hcal"] = 3

        np.testing.assert_allclose(
            detector_map.detector_cell_thickness(config), [1.0, 1.0, 3.0]
        )


# --------------------------------------------------------------------------- #
# confine_to_box
# --------------------------------------------------------------------------- #
class TestConfineToBox:
    """The default box is X and Z from the ``*_in_detector`` entries and Y from
    the detector layer stack: (-250, 250), (1811, 1825), (-250, 250).
    """

    @staticmethod
    def confine(config, x, y, z, e=1.0, **kwargs):
        return detector_map.confine_to_box(
            config,
            np.atleast_1d(np.asarray(x, dtype=float)),
            np.atleast_1d(np.asarray(y, dtype=float)),
            np.atleast_1d(np.asarray(z, dtype=float)),
            np.atleast_1d(np.asarray(e, dtype=float)),
            **kwargs,
        )

    def test_a_hit_in_the_middle_survives(self, config):
        kept = self.confine(config, 0.0, 1815.0, 0.0)

        assert [len(array) for array in kept] == [1, 1, 1, 1]

    @pytest.mark.parametrize(
        "axis, value",
        [
            ("x", -260.0),
            ("x", 260.0),
            ("y", 1810.0),
            ("y", 1830.0),
            ("z", -260.0),
            ("z", 260.0),
        ],
    )
    def test_a_hit_beyond_any_face_is_dropped(self, config, axis, value):
        coords = {"x": 0.0, "y": 1815.0, "z": 0.0}
        coords[axis] = value

        kept = self.confine(config, **coords)

        assert len(kept[0]) == 0

    @pytest.mark.parametrize("edge", [-250.0, 250.0])
    def test_the_faces_themselves_are_excluded(self, config, edge):
        """The comparisons are strict, so a hit exactly on a face is dropped."""
        kept = self.confine(config, edge, 1815.0, 0.0)

        assert len(kept[0]) == 0

    def test_returns_x_y_z_e_in_that_order(self, config):
        X, Y, Z, E = self.confine(config, 1.0, 1815.0, 2.0, 3.0)

        assert (X[0], Y[0], Z[0], E[0]) == (1.0, 1815.0, 2.0, 3.0)

    def test_every_array_is_filtered_together(self, config):
        X = np.array([0.0, 999.0, 5.0])
        Y = np.full(3, 1815.0)
        Z = np.zeros(3)
        E = np.array([10.0, 20.0, 30.0])

        kept = detector_map.confine_to_box(config, X, Y, Z, E)

        np.testing.assert_array_equal(kept[0], [0.0, 5.0])
        np.testing.assert_array_equal(kept[3], [10.0, 30.0])

    def test_energy_does_not_affect_selection(self, config):
        kept = self.confine(config, 0.0, 1815.0, 0.0, e=-99.0)

        assert len(kept[0]) == 1

    def test_the_x_and_z_bounds_come_from_the_in_detector_entries(self, config):
        config["data"]["Xmax_in_detector"] = 5.0
        config["data"]["Zmax_in_detector"] = 5.0
        config["data"]["Xmax"] = 500.0  # the plain data entries must be ignored
        config["data"]["Ymax"] = 500.0

        assert len(self.confine(config, 10.0, 1815.0, 0.0)[0]) == 0
        assert len(self.confine(config, 0.0, 1815.0, 10.0)[0]) == 0

    def test_the_y_bounds_come_from_the_detector_layer_stack(self, config):
        floor = config["detector"]["layer_bottom_pos"][0]
        ceiling = (
            config["detector"]["layer_bottom_pos"][-1]
            + config["detector"]["cell_thickness_hcal"]
        )

        assert len(self.confine(config, 0.0, floor + 0.01, 0.0)[0]) == 1
        assert len(self.confine(config, 0.0, ceiling - 0.01, 0.0)[0]) == 1
        assert len(self.confine(config, 0.0, ceiling + 0.01, 0.0)[0]) == 0

    def test_the_y_ceiling_uses_the_hcal_cell_thickness(self, config):
        """Y is the layer axis in this branch, so both ends come from the
        ``detector`` section even though X and Z come from ``data``."""
        config["detector"]["cell_thickness_hcal"] = 2.0  # ceiling at 1825.0
        config["data"]["cell_thickness"] = 100.0  # would put it far higher

        assert len(self.confine(config, 0.0, 1826.0, 0.0)[0]) == 0

    def test_the_y_ceiling_uses_the_last_layers_own_thickness(self, config):
        """Not ``cell_thickness_hcal`` unconditionally: for an ecal-only stack --
        ``config/default.yaml``, whose ``hcal_start`` equals its layer count --
        the last layer is ecal and the box has to stop at the ecal thickness,
        or it reaches above the real top of the detector.
        """
        config["detector"]["hcal_start"] = 99  # every layer is ecal now
        thickness = detector_map.detector_cell_thickness(config)
        true_top = config["detector"]["layer_bottom_pos"][-1] + thickness[-1]

        assert len(self.confine(config, 0.0, true_top - 0.01, 0.0)[0]) == 1
        assert len(self.confine(config, 0.0, true_top + 0.01, 0.0)[0]) == 0

    def test_the_y_ceiling_follows_a_mixed_stack(self, config):
        """With the fixture's hcal last layer the ceiling is the hcal thickness
        above it, which is the same number the layer's own band starts from."""
        thickness = detector_map.detector_cell_thickness(config)
        assert thickness[-1] == config["detector"]["cell_thickness_hcal"]

        true_top = config["detector"]["layer_bottom_pos"][-1] + thickness[-1]

        assert len(self.confine(config, 0.0, true_top - 0.01, 0.0)[0]) == 1
        assert len(self.confine(config, 0.0, true_top + 0.01, 0.0)[0]) == 0

    def test_the_ceiling_at_the_hcal_start_boundary(self, config):
        """``config/default.yaml`` has exactly ``hcal_start`` layers, so its
        last layer is the final ecal one. The test is ``len(layers) >
        hcal_start``: one notch looser and an ecal-only detector would be given
        an hcal-sized box.
        """
        n_layers = len(config["detector"]["layer_bottom_pos"])
        config["detector"]["hcal_start"] = n_layers  # no hcal layers at all
        thickness = detector_map.detector_cell_thickness(config)
        assert thickness[-1] == config["detector"]["cell_thickness_ecal"]

        true_top = config["detector"]["layer_bottom_pos"][-1] + thickness[-1]

        assert len(self.confine(config, 0.0, true_top - 0.01, 0.0)[0]) == 1
        assert len(self.confine(config, 0.0, true_top + 0.01, 0.0)[0]) == 0

    @pytest.mark.parametrize("hcal_start", [-1, 0, 1, 2, 3, 4, 99])
    def test_the_ceiling_tracks_the_last_layers_thickness_either_way(
        self, config, hcal_start
    ):
        """Whatever ``hcal_start`` says, the ceiling is the top of the last
        layer -- the same number ``detector_cell_thickness(config)[-1]`` gives.
        """
        config["detector"]["hcal_start"] = hcal_start
        thickness = detector_map.detector_cell_thickness(config)
        true_top = config["detector"]["layer_bottom_pos"][-1] + thickness[-1]

        assert len(self.confine(config, 0.0, true_top - 0.01, 0.0)[0]) == 1
        assert len(self.confine(config, 0.0, true_top + 0.01, 0.0)[0]) == 0

    def test_every_layer_centre_falls_inside_the_box(self, config):
        """``physical_to_cells`` gives each cell its layer's centre as a y
        coordinate, and the muon hits that build those cells have to survive
        confinement first, so a ceiling below the top layer's centre would
        leave the last layer with no geometry at all.
        """
        centres = detector_map.get_layer_centers(config, coordinates="detector")

        kept = detector_map.confine_to_box(
            config,
            np.zeros_like(centres),
            centres,
            np.zeros_like(centres),
            np.ones_like(centres),
        )

        assert len(kept[0]) == len(centres)

    def test_detector_coords_is_the_default(self, config):
        explicit = self.confine(config, 0.0, 1815.0, 0.0, detector_coords=True)
        default = self.confine(config, 0.0, 1815.0, 0.0)

        np.testing.assert_array_equal(default[1], explicit[1])

    def test_the_data_branch_uses_a_different_box(self, config):
        """In data coordinates z is the layer axis and y is a transverse one, so
        the roles of the two are swapped relative to the detector branch."""
        # data box: x, y in (-250, 250); z in (0, 3)
        assert len(self.confine(config, 0.0, 0.0, 1.5, detector_coords=False)[0]) == 1
        assert len(self.confine(config, 0.0, 0.0, 3.5, detector_coords=False)[0]) == 0
        outside_y = self.confine(config, 0.0, 1815.0, 1.5, detector_coords=False)
        assert len(outside_y[0]) == 0

    def test_the_data_branch_z_ceiling_uses_the_data_cell_thickness(self, config):
        config["data"]["layer_bottom_pos"] = [0.0, 1.0, 2.0]
        config["data"]["cell_thickness"] = 5.0  # ceiling at 7.0

        assert len(self.confine(config, 0.0, 0.0, 6.0, detector_coords=False)[0]) == 1

    def test_the_data_branch_z_floor_is_the_first_layer_bottom(self, config):
        """Not ``Ymin``: the transverse bounds do not apply to the layer axis,
        and below the first layer there is no detector to hit."""
        assert config["data"]["layer_bottom_pos"][0] == 0.0
        assert config["data"]["Ymin"] == -250.0

        assert len(self.confine(config, 0.0, 0.0, -1.0, detector_coords=False)[0]) == 0
        assert len(self.confine(config, 0.0, 0.0, 1.0, detector_coords=False)[0]) == 1

    def test_empty_input_gives_empty_output(self, config):
        empty = np.array([])

        kept = detector_map.confine_to_box(config, empty, empty, empty, empty)

        assert [len(array) for array in kept] == [0, 0, 0, 0]

    def test_everything_outside_gives_empty_output(self, config):
        kept = self.confine(config, 9e9, 9e9, 9e9)

        assert [len(array) for array in kept] == [0, 0, 0, 0]

    def test_the_inputs_are_left_alone(self, config):
        X = np.array([0.0, 999.0])
        original = X.copy()

        detector_map.confine_to_box(
            config, X, np.full(2, 1815.0), np.zeros(2), np.ones(2)
        )

        np.testing.assert_array_equal(X, original)

    def test_two_dimensional_input_is_silently_wrong(self, config):
        """``np.where(...)[0]`` keeps only the first axis of the match.

        For a flat hit list that is the hit index, but a 2-D array is indexed by
        row, so rows containing any in-box hit are returned whole and repeated
        once per hit. Nothing raises, and the output is longer than the input.
        """
        X = np.array([[0.0, 999.0], [0.0, 0.0]])
        Y = np.full((2, 2), 1815.0)
        Z = np.zeros((2, 2))
        E = np.arange(4.0).reshape(2, 2)

        kept = detector_map.confine_to_box(config, X, Y, Z, E)

        assert kept[0].shape == (3, 2)
        np.testing.assert_array_equal(kept[3], [[0.0, 1.0], [2.0, 3.0], [2.0, 3.0]])


# --------------------------------------------------------------------------- #
# floors_ceilings
# --------------------------------------------------------------------------- #
class TestFloorsCeilings:
    def test_a_half_cell_buffer_either_side(self, config):
        """Layers at 0, 1, 2 with unit cells: the buffer would reach into the
        neighbouring layer, so it is clamped to the midpoint between them."""
        positions = np.array(config["data"]["layer_bottom_pos"])

        floors, ceilings = detector_map.floors_ceilings(positions, 1.0)

        np.testing.assert_allclose(floors, [-0.5, 1.0, 2.0])
        np.testing.assert_allclose(ceilings, [1.0, 2.0, 3.5])

    def test_well_separated_layers_keep_the_whole_buffer(self, config):
        positions = np.array(config["detector"]["layer_bottom_pos"])
        thickness = detector_map.detector_cell_thickness(config)

        floors, ceilings = detector_map.floors_ceilings(positions, thickness)

        np.testing.assert_allclose(floors, positions - 0.5 * thickness)
        np.testing.assert_allclose(ceilings, positions + 1.5 * thickness)

    def test_the_default_buffer_is_half_a_cell(self, config):
        positions = np.array(config["detector"]["layer_bottom_pos"])

        default = detector_map.floors_ceilings(positions.copy(), 0.5)
        explicit = detector_map.floors_ceilings(positions.copy(), 0.5, 0.5)

        np.testing.assert_allclose(default[0], explicit[0])
        np.testing.assert_allclose(default[1], explicit[1])

    def test_a_per_layer_thickness_is_accepted(self, config):
        """``detector_cell_thickness`` hands it an array, one entry per layer."""
        positions = np.array(config["detector"]["layer_bottom_pos"])
        thickness = detector_map.detector_cell_thickness(config)

        floors, ceilings = detector_map.floors_ceilings(positions, thickness)

        np.testing.assert_allclose(floors, [1810.75, 1813.75, 1822.0])
        np.testing.assert_allclose(ceilings, [1811.75, 1814.75, 1826.0])

    def test_each_band_is_scaled_by_its_own_thickness(self, config):
        positions = np.array(config["detector"]["layer_bottom_pos"])
        thickness = detector_map.detector_cell_thickness(config)

        floors, ceilings = detector_map.floors_ceilings(positions, thickness)

        # unclamped here, so each band spans twice its own cell thickness
        np.testing.assert_allclose(ceilings - floors, 2 * thickness)

    def test_a_uniform_array_matches_the_scalar(self, config):
        positions = np.array(config["detector"]["layer_bottom_pos"])

        scalar = detector_map.floors_ceilings(positions.copy(), 0.5)
        array = detector_map.floors_ceilings(positions.copy(), np.full(3, 0.5))

        np.testing.assert_allclose(scalar[0], array[0])
        np.testing.assert_allclose(scalar[1], array[1])

    def test_layers_never_overlap_across_a_thickness_change(self):
        """The clamp works elementwise, so a thin layer next to a thick one
        still gives disjoint bands."""
        positions = np.array([0.0, 1.0, 3.0, 9.0])
        thickness = np.array([0.5, 0.5, 3.0, 3.0])

        floors, ceilings = detector_map.floors_ceilings(positions, thickness)

        assert np.all(ceilings[:-1] <= floors[1:])

    def test_no_buffer_gives_exactly_the_cells(self):
        positions = np.array([0.0, 10.0, 20.0])

        floors, ceilings = detector_map.floors_ceilings(
            positions, 1.0, percent_buffer=0
        )

        np.testing.assert_allclose(floors, [0.0, 10.0, 20.0])
        np.testing.assert_allclose(ceilings, [1.0, 11.0, 21.0])

    def test_ceilings_are_above_their_floors(self):
        positions = np.array([0.0, 3.0, 6.0, 9.0])

        floors, ceilings = detector_map.floors_ceilings(positions, 1.0)

        assert np.all(ceilings > floors)

    @pytest.mark.parametrize(
        "positions, thickness",
        [
            ([0.0, 1.0, 2.0], 1.0),
            ([0.0, 10.0, 20.0], 1.0),
            ([0.0, 1.0], 3.0),
            ([1811.0, 1814.0, 1823.0], 0.5250244140625),
        ],
    )
    def test_layers_never_overlap(self, positions, thickness):
        """Each ceiling is clamped to the midpoint and the next floor is raised
        to it, so a hit can never fall inside two layers."""
        floors, ceilings = detector_map.floors_ceilings(
            np.array(positions), thickness
        )

        assert np.all(ceilings[:-1] <= floors[1:])

    def test_layers_never_overlap_for_random_geometries(self):
        rng = np.random.default_rng(0)

        for _ in range(100):
            positions = np.sort(rng.uniform(0, 100, 6))
            thickness = rng.uniform(0.1, 40.0)

            floors, ceilings = detector_map.floors_ceilings(positions, thickness)

            assert np.all(ceilings[:-1] <= floors[1:] + 1e-12)

    def test_cells_thicker_than_the_gaps_give_touching_bands(self):
        """With no room between layers the clamp collapses the buffer entirely
        and the bands meet at the next layer bottom."""
        floors, ceilings = detector_map.floors_ceilings(np.array([0.0, 1.0]), 3.0)

        np.testing.assert_allclose(floors, [-1.5, 1.0])
        np.testing.assert_allclose(ceilings, [1.0, 5.5])

    def test_the_outer_edges_are_never_clamped(self):
        """Only interior boundaries are trimmed, so the stack reaches half a
        cell below the first layer and one and a half above the last."""
        positions = np.array([0.0, 10.0, 20.0])

        floors, ceilings = detector_map.floors_ceilings(positions, 2.0)

        assert floors[0] == -1.0
        assert ceilings[-1] == 23.0

    def test_a_single_layer_is_returned_unclamped(self):
        floors, ceilings = detector_map.floors_ceilings(np.array([10.0]), 2.0)

        np.testing.assert_allclose(floors, [9.0])
        np.testing.assert_allclose(ceilings, [13.0])

    def test_an_empty_stack_gives_empty_bands(self):
        floors, ceilings = detector_map.floors_ceilings(np.array([]), 1.0)

        assert len(floors) == 0 and len(ceilings) == 0

    def test_the_input_positions_are_left_alone(self):
        positions = np.array([0.0, 1.0, 2.0])
        original = positions.copy()

        detector_map.floors_ceilings(positions, 1.0)

        np.testing.assert_array_equal(positions, original)

    def test_integer_positions_are_promoted(self):
        floors, _ = detector_map.floors_ceilings(np.array([0, 10, 20]), 1.0)

        np.testing.assert_allclose(floors, [-0.5, 9.5, 19.5])

    def test_a_plain_list_raises(self):
        """It needs an array, but ``layer_bottom_pos`` arrives from YAML as a
        list. ``find_layers`` and ``get_layer_centers`` wrap it first; anything
        calling this directly has to do the same.
        """
        with pytest.raises(TypeError):
            detector_map.floors_ceilings([0.0, 1.0, 2.0], 1.0)


# --------------------------------------------------------------------------- #
# get_layer_centers
# --------------------------------------------------------------------------- #
class TestGetLayerCenters:
    """The ``data`` branch has one thickness for the whole stack; the
    ``detector`` branch takes one per layer from ``detector_cell_thickness``.
    """

    def test_data_coordinates(self, config):
        centres = detector_map.get_layer_centers(config)

        np.testing.assert_allclose(centres, [0.5, 1.5, 2.5])

    def test_detector_coordinates(self, config):
        centres = detector_map.get_layer_centers(config, coordinates="detector")

        np.testing.assert_allclose(centres, [1811.25, 1814.25, 1824.0])

    def test_data_is_the_default(self, config):
        np.testing.assert_allclose(
            detector_map.get_layer_centers(config),
            detector_map.get_layer_centers(config, coordinates="data"),
        )

    def test_one_centre_per_layer(self, config):
        for coordinates in ("data", "detector"):
            centres = detector_map.get_layer_centers(config, coordinates=coordinates)

            assert len(centres) == len(config[coordinates]["layer_bottom_pos"])

    def test_a_centre_sits_half_a_cell_above_the_bottom(self, config):
        bottoms = np.array(config["data"]["layer_bottom_pos"])
        thickness = config["data"]["cell_thickness"]

        centres = detector_map.get_layer_centers(config)

        np.testing.assert_allclose(centres - bottoms, thickness / 2)

    def test_each_detector_centre_uses_its_own_thickness(self, config):
        bottoms = np.array(config["detector"]["layer_bottom_pos"])
        thickness = detector_map.detector_cell_thickness(config)

        centres = detector_map.get_layer_centers(config, coordinates="detector")

        np.testing.assert_allclose(centres - bottoms, thickness / 2)

    def test_an_hcal_centre_is_not_the_ecal_one(self, config):
        """The whole point of the split: layer 2 is hcal, so its centre sits
        1.0 above its bottom, not the 0.25 an ecal layer would give.
        """
        bottoms = config["detector"]["layer_bottom_pos"]
        ecal = config["detector"]["cell_thickness_ecal"]

        centres = detector_map.get_layer_centers(config, coordinates="detector")

        assert centres[2] == bottoms[2] + config["detector"]["cell_thickness_hcal"] / 2
        assert not np.isclose(centres[2], bottoms[2] + ecal / 2)

    def test_a_centre_lies_inside_its_own_layer_band(self, config):
        """``physical_to_cells`` gives every cell its layer's centre as a y
        coordinate, and ``find_layers`` has to put that y back in the same
        layer, so the centres have to sit inside the bands.
        """
        centres = detector_map.get_layer_centers(config, coordinates="detector")
        points = np.zeros((1, len(centres), 4))
        points[0, :, 1] = centres
        points[0, :, 3] = 1.0

        assigned = detector_map.find_layers(config, points, "detector")

        np.testing.assert_array_equal(assigned[0], np.arange(len(centres)))

    def test_the_removed_cell_thickness_key_is_not_needed(self, config):
        """No shipped config defines ``detector.cell_thickness`` any more."""
        assert "cell_thickness" not in config["detector"]

        detector_map.get_layer_centers(config, coordinates="detector")

    def test_a_list_of_positions_is_accepted(self, config):
        assert isinstance(config["detector"]["layer_bottom_pos"], list)

        assert isinstance(
            detector_map.get_layer_centers(config, coordinates="detector"), np.ndarray
        )

    @pytest.mark.parametrize("coordinates", ["data", "detector"])
    def test_the_result_is_safe_to_modify(self, config, coordinates):
        """``inference.unshift_points`` does ``layer_centers -= centers[0]`` in
        place, so this must not hand back a view onto the config."""
        original = list(config[coordinates]["layer_bottom_pos"])

        centres = detector_map.get_layer_centers(config, coordinates=coordinates)
        centres -= centres[0]

        assert config[coordinates]["layer_bottom_pos"] == original

    def test_an_unknown_coordinate_system(self, config):
        """Neither branch runs and neither raises, so the failure surfaces as an
        unbound local rather than a message naming the bad argument."""
        with pytest.raises(UnboundLocalError):
            detector_map.get_layer_centers(config, coordinates="global")


# --------------------------------------------------------------------------- #
# find_layers
# --------------------------------------------------------------------------- #
class TestFindLayers:
    @pytest.mark.parametrize(
        "height, expected",
        [(-0.5, 0), (0.0, 0), (0.99, 0), (1.0, 1), (2.0, 2), (3.49, 2)],
    )
    def test_a_hit_is_labelled_with_its_layer(self, config, height, expected):
        layers = detector_map.find_layers(config, one_point(height))

        assert layers[0, 0] == expected

    @pytest.mark.parametrize("height", [-0.51, 3.5, 100.0])
    def test_a_hit_outside_every_band_is_unlabelled(self, config, height):
        layers = detector_map.find_layers(config, one_point(height))

        assert layers[0, 0] == -1

    def test_the_floor_is_inclusive_and_the_ceiling_is_not(self, config):
        assert detector_map.find_layers(config, one_point(1.0))[0, 0] == 1
        assert detector_map.find_layers(config, one_point(2.0))[0, 0] == 2

    def test_the_bands_are_floors_ceilings_with_a_half_cell_buffer(self, config):
        floors, ceilings = detector_map.floors_ceilings(
            np.array(config["data"]["layer_bottom_pos"]),
            config["data"]["cell_thickness"],
            percent_buffer=0.5,
        )
        heights = np.concatenate([floors, ceilings - 1e-9])
        points = np.zeros((1, len(heights), 4))
        points[0, :, 2] = heights
        points[0, :, 3] = 1.0

        layers = detector_map.find_layers(config, points)

        assert layers[0].tolist() == [0, 1, 2, 0, 1, 2]

    @pytest.mark.parametrize("energy", [0.0, -1.0])
    def test_padding_is_unlabelled_even_inside_a_band(self, config, energy):
        layers = detector_map.find_layers(config, one_point(0.5, energy=energy))

        assert layers[0, 0] == -1

    def test_data_coordinates_read_the_z_column(self, config):
        points = one_point(0.5, axis=2)

        assert detector_map.find_layers(config, points)[0, 0] == 0

    def test_detector_coordinates_read_the_y_column(self, config):
        points = one_point(1814.25, axis=1)

        assert detector_map.find_layers(config, points, "detector")[0, 0] == 1

    @pytest.mark.parametrize(
        "height, expected",
        [(1810.75, 0), (1811.74, 0), (1811.75, -1), (1813.75, 1), (1822.0, 2),
         (1825.99, 2), (1826.0, -1)],
    )
    def test_the_detector_bands_use_the_per_layer_thickness(
        self, config, height, expected
    ):
        """Layer 2 is hcal, so its band is four units wide against the ecal
        layers' one."""
        layers = detector_map.find_layers(config, one_point(height, axis=1), "detector")

        assert layers[0, 0] == expected

    def test_an_hcal_hit_needs_the_hcal_thickness_to_be_found(self, config):
        """At 1825 the point is inside layer 2 only because that layer is hcal;
        with the ecal thickness the band would stop at 1823.75.
        """
        point = one_point(1825.0, axis=1)
        assert detector_map.find_layers(config, point, "detector")[0, 0] == 2

        config["detector"]["cell_thickness_hcal"] = config["detector"][
            "cell_thickness_ecal"
        ]
        assert detector_map.find_layers(config, point, "detector")[0, 0] == -1

    def test_the_data_branch_keeps_its_single_thickness(self, config):
        """Only the detector side is split; ``data.cell_thickness`` is a scalar
        and ``data`` has no ecal/hcal keys at all."""
        assert "cell_thickness_ecal" not in config["data"]

        assert detector_map.find_layers(config, one_point(0.5))[0, 0] == 0

    def test_data_is_the_default(self, config):
        points = one_point(0.5, axis=2)

        np.testing.assert_array_equal(
            detector_map.find_layers(config, points),
            detector_map.find_layers(config, points, coordinates="data"),
        )

    def test_the_wrong_coordinate_system_fails_quietly(self, config):
        """A detector-coordinate shower read as data coordinates has z near 0,
        which lands in the first data layer, so the answer looks plausible."""
        points = one_point(1814.25, axis=1)

        assert detector_map.find_layers(config, points, "data")[0, 0] == 0

    def test_the_bands_match_the_ones_create_map_uses(self, config, mocker):
        """``physical_to_cells`` indexes ``create_map``'s layer list with ids
        from here, so the two have to partition y identically."""
        mocker.patch(
            "src.detector_map.load_muon_map",
            return_value=muon_grid(config, x_centres=(0.0,), z_centres=(0.0,)),
        )
        heights = np.array([1811.25, 1814.25, 1824.0, 1825.5, 1900.0])
        points = np.zeros((1, len(heights), 4))
        points[0, :, 1] = heights
        points[0, :, 3] = 1.0

        layers, _ = detector_map.create_map(config)
        assigned = detector_map.find_layers(config, points, "detector")[0]

        assert assigned.tolist() == [0, 1, 2, 2, -1]
        assert len(layers) == len(config["detector"]["layer_bottom_pos"])

    def test_shape_and_dtype(self, config):
        points = np.zeros((3, 7, 4))
        points[:, :, 3] = 1.0

        layers = detector_map.find_layers(config, points)

        assert layers.shape == (3, 7)
        assert np.issubdtype(layers.dtype, np.integer)

    def test_each_event_is_labelled_independently(self, config):
        points = np.zeros((2, 3, 4))
        points[:, :, 2] = 0.5
        points[0, :, 3] = [1.0, 0.0, -1.0]
        points[1, :, 3] = 1.0

        layers = detector_map.find_layers(config, points)

        assert layers[0].tolist() == [0, -1, -1]
        assert layers[1].tolist() == [0, 0, 0]

    def test_an_all_padding_event(self, config):
        points = np.zeros((1, 4, 4))
        points[0, :, 2] = 0.5

        assert detector_map.find_layers(config, points).tolist() == [[-1, -1, -1, -1]]

    def test_the_points_are_left_alone(self, config):
        points = np.zeros((2, 3, 4))
        points[:, :, 2] = 0.5
        points[:, :, 3] = 1.0
        original = points.copy()

        detector_map.find_layers(config, points)

        np.testing.assert_array_equal(points, original)

    def test_a_list_of_positions_is_accepted(self, config):
        assert isinstance(config["data"]["layer_bottom_pos"], list)

        assert detector_map.find_layers(config, one_point(0.5))[0, 0] == 0

    def test_an_unknown_coordinate_system(self, config):
        with pytest.raises(UnboundLocalError):
            detector_map.find_layers(config, one_point(0.5), coordinates="global")


# --------------------------------------------------------------------------- #
# create_map
# --------------------------------------------------------------------------- #
class TestCreateMap:
    def test_returns_a_layer_list_and_an_offset(self, config, fake_muon_map):
        fake_muon_map(muon_grid(config))

        layers, offset = detector_map.create_map(config)

        assert len(layers) == len(config["detector"]["layer_bottom_pos"])
        assert all(set(layer) == {"xedges", "zedges", "grid"} for layer in layers)
        assert np.isscalar(offset)

    def test_the_offset_is_a_cell_split_into_sub_cells(self, config, fake_muon_map):
        """``cell_size`` is read from ``detector`` but ``divisions_per_cell``
        from ``data``, so the two sections have to agree about the cell."""
        config["detector"]["cell_size"] = 8.0
        config["data"]["divisions_per_cell"] = 4
        fake_muon_map(muon_grid(config))

        _, offset = detector_map.create_map(config)

        assert offset == 2.0

    def test_the_grid_matches_its_edges(self, config, fake_muon_map):
        fake_muon_map(muon_grid(config))

        layers, _ = detector_map.create_map(config)

        for layer in layers:
            assert layer["grid"].shape == (
                len(layer["xedges"]) - 1,
                len(layer["zedges"]) - 1,
            )

    def test_the_edges_are_strictly_increasing(self, config, fake_muon_map):
        fake_muon_map(muon_grid(config))

        layers, _ = detector_map.create_map(config)

        for layer in layers:
            assert np.all(np.diff(layer["xedges"]) > 0)
            assert np.all(np.diff(layer["zedges"]) > 0)

    def test_the_grid_counts_the_hits_in_that_layer(self, config, fake_muon_map):
        centres = (-10.0, -5.0, 0.0, 5.0, 10.0)
        fake_muon_map(muon_grid(config, x_centres=centres))

        layers, _ = detector_map.create_map(config)

        for layer in layers:
            assert layer["grid"].sum() == len(centres) ** 2

    def test_each_hit_lands_in_its_own_cell(self, config, fake_muon_map):
        centres = (-10.0, -5.0, 0.0, 5.0, 10.0)
        fake_muon_map(muon_grid(config, x_centres=centres))

        layers, _ = detector_map.create_map(config)

        grid = layers[0]["grid"]
        assert (grid > 0).sum() == len(centres) ** 2
        assert grid.max() == 1

    def test_only_hits_from_that_layer_are_counted(self, config, fake_muon_map):
        """The Y bands are disjoint, so a layer sees only its own hits even
        though the map holds every layer at once."""
        bottoms = config["detector"]["layer_bottom_pos"]
        thickness = detector_map.detector_cell_thickness(config)
        X, Y, Z, E = muon_grid(config, x_centres=(0.0,), z_centres=(0.0,))
        # a second hit for layer 0 only
        fake_muon_map(
            (
                np.append(X, 0.0),
                np.append(Y, bottoms[0] + thickness[0] / 2),
                np.append(Z, 0.0),
                np.append(E, 1.0),
            )
        )

        layers, _ = detector_map.create_map(config)

        assert [layer["grid"].sum() for layer in layers] == [2.0, 1.0, 1.0]

    def test_each_layer_uses_its_own_thickness(self, config, fake_muon_map):
        """Layer 2 is hcal, so its band reaches hits an ecal-width band would
        miss. Placing a hit high in that band lands it in layer 2 only.
        """
        bottoms = config["detector"]["layer_bottom_pos"]
        thickness = detector_map.detector_cell_thickness(config)
        X, Y, Z, E = muon_grid(config, x_centres=(0.0,), z_centres=(0.0,))
        # above an ecal-width band (which would stop at 1823.75) but still
        # inside the confinement box, whose ceiling is 1825.0
        high_in_the_hcal = bottoms[2] + 0.75 * thickness[2]
        fake_muon_map(
            (
                np.append(X, 0.0),
                np.append(Y, high_in_the_hcal),
                np.append(Z, 0.0),
                np.append(E, 1.0),
            )
        )

        layers, _ = detector_map.create_map(config)

        assert [layer["grid"].sum() for layer in layers] == [1.0, 1.0, 2.0]

    def test_the_confinement_box_clips_the_top_of_the_last_band(
        self, config, fake_muon_map
    ):
        """``confine_to_box`` stops at ``last bottom + cell_thickness_hcal``
        while the last band runs to ``last bottom + 1.5 * thickness``, so hits
        in the half cell between them are assigned that layer by
        ``find_layers`` but never contribute a cell to the map. Only the buffer
        above the topmost layer is affected, where there is no detector anyway.
        """
        bottoms = config["detector"]["layer_bottom_pos"]
        thickness = detector_map.detector_cell_thickness(config)
        above_the_box = bottoms[-1] + 1.25 * thickness[-1]

        X, Y, Z, E = muon_grid(config, x_centres=(0.0,), z_centres=(0.0,))
        fake_muon_map(
            (
                np.append(X, 0.0),
                np.append(Y, above_the_box),
                np.append(Z, 0.0),
                np.append(E, 1.0),
            )
        )

        layers, _ = detector_map.create_map(config)

        assert layers[-1]["grid"].sum() == 1.0  # the extra hit never arrives
        point = one_point(above_the_box, axis=1)
        assert detector_map.find_layers(config, point, "detector")[0, 0] == 2

    def test_the_confine_argument_is_never_read(self, config, fake_muon_map):
        """It is accepted and then ignored; confinement always happens, in
        detector coordinates, whatever is passed."""
        fake_muon_map(muon_grid(config))

        without, offset_without = detector_map.create_map(config, confine=False)
        with_it, offset_with = detector_map.create_map(config, confine=True)

        assert offset_without == offset_with
        for left, right in zip(without, with_it):
            np.testing.assert_array_equal(left["grid"], right["grid"])
            np.testing.assert_array_equal(left["xedges"], right["xedges"])

    def test_hit_energies_are_ignored(self, config, fake_muon_map):
        """The map is a cell geometry, not a deposit map: the histogram is
        unweighted and the ``E`` array is loaded, confined and dropped."""
        X, Y, Z, E = muon_grid(config)
        fake_muon_map((X, Y, Z, E))
        expected, _ = detector_map.create_map(config)

        fake_muon_map((X, Y, Z, E * 1000.0))
        actual, _ = detector_map.create_map(config)

        for left, right in zip(expected, actual):
            np.testing.assert_array_equal(left["grid"], right["grid"])

    def test_hits_outside_the_box_never_reach_the_map(self, config, fake_muon_map):
        X, Y, Z, E = muon_grid(config, x_centres=(0.0,), z_centres=(0.0,))
        fake_muon_map(
            (
                np.append(X, 9999.0),  # far outside the x bounds
                np.append(Y, Y[0]),
                np.append(Z, 0.0),
                np.append(E, 1.0),
            )
        )

        layers, _ = detector_map.create_map(config)

        assert layers[0]["grid"].sum() == 1.0

    def test_one_division_per_cell_gives_whole_cells(self, config, fake_muon_map):
        config["data"]["divisions_per_cell"] = 1
        fake_muon_map(muon_grid(config))

        layers, offset = detector_map.create_map(config)

        assert offset == config["detector"]["cell_size"]
        np.testing.assert_allclose(
            np.diff(layers[0]["zedges"]), config["detector"]["cell_size"]
        )

    def test_z_cells_are_split_into_sub_cells(self, config, fake_muon_map):
        fake_muon_map(muon_grid(config))
        expected = (
            config["detector"]["cell_size"] / config["data"]["divisions_per_cell"]
        )

        layers, _ = detector_map.create_map(config)

        np.testing.assert_allclose(np.diff(layers[0]["zedges"]), expected)

    def test_every_x_cell_is_split_into_sub_cells(self, config, fake_muon_map):
        """Including the lowest one: no cell is a special case."""
        fake_muon_map(muon_grid(config))
        expected = (
            config["detector"]["cell_size"] / config["data"]["divisions_per_cell"]
        )

        layers, _ = detector_map.create_map(config)

        np.testing.assert_allclose(np.diff(layers[0]["xedges"]), expected)

    def test_the_two_axes_are_divided_the_same_way(self, config, fake_muon_map):
        fake_muon_map(muon_grid(config))

        layers, _ = detector_map.create_map(config)

        for layer in layers:
            np.testing.assert_allclose(
                np.diff(layer["xedges"]).min(), np.diff(layer["zedges"]).min()
            )

    def test_a_single_x_column_is_split_like_any_other(self, config, fake_muon_map):
        fake_muon_map(muon_grid(config, x_centres=(0.0,), z_centres=(0.0,)))
        divisions = config["data"]["divisions_per_cell"]

        layers, _ = detector_map.create_map(config)

        assert len(layers[0]["xedges"]) - 1 == divisions
        assert len(layers[0]["zedges"]) - 1 == divisions

    def test_x_columns_closer_than_a_cell_are_merged(self, config, fake_muon_map):
        """The gap test skips any x within about one cell of the previous one,
        which is how repeated measurements of the same cell get folded together.
        """
        half = config["detector"]["cell_size"] / 2
        fake_muon_map(muon_grid(config, x_centres=(0.0, half), z_centres=(0.0,)))

        layers, _ = detector_map.create_map(config)

        xedges = layers[0]["xedges"]
        assert np.isclose(xedges[-1] - xedges[0], config["detector"]["cell_size"])
        assert len(xedges) - 1 == config["data"]["divisions_per_cell"]

    def test_near_duplicate_edges_are_collapsed(self, config, fake_muon_map):
        """Edges within 1e-3 of each other are dropped, so two x columns that
        are almost but not quite a whole cell apart cannot produce a sliver bin.
        """
        fake_muon_map(
            muon_grid(config, x_centres=(0.0, 5.0000005), z_centres=(0.0,))
        )

        layers, _ = detector_map.create_map(config)

        assert np.diff(layers[0]["xedges"]).min() > 1e-3

    def test_a_layer_with_no_hits_raises(self, config, fake_muon_map):
        """``unique_X[0]`` is taken before anything checks the window found
        hits, so a detector layer the muon map does not cover fails here rather
        than producing an empty grid.
        """
        fake_muon_map(muon_grid(config, layer_indices=[0]))

        with pytest.raises(IndexError):
            detector_map.create_map(config)

    def test_closely_spaced_layers_do_not_double_count(self, config, fake_muon_map):
        """Layers less than two thicknesses apart have overlapping unclamped
        buffers. The bands are clamped, so a hit in the overlap belongs to one
        layer only, and the map holds exactly as many hits as were supplied.
        """
        config["detector"]["layer_bottom_pos"] = [0.0, 0.6]
        config["detector"]["cell_thickness_ecal"] = 0.5
        config["detector"]["hcal_start"] = 2  # both layers ecal
        # 0.5 falls in the overlap of the unclamped windows [-0.25, 0.75] and
        # [0.35, 1.35]; the other two keep each layer non-empty
        heights = np.array([0.25, 0.5, 0.85])
        flat = np.zeros_like(heights)
        fake_muon_map((flat, heights, flat.copy(), np.ones_like(heights)))

        layers, _ = detector_map.create_map(config)

        assert sum(layer["grid"].sum() for layer in layers) == len(heights)

    def test_a_hit_lands_in_the_layer_find_layers_would_give_it(
        self, config, fake_muon_map
    ):
        """The two functions have to agree: ``physical_to_cells`` looks up the
        grid of the layer ``find_layers`` assigned the point to.
        """
        config["detector"]["layer_bottom_pos"] = [0.0, 0.6]
        config["detector"]["cell_thickness_ecal"] = 0.5
        config["detector"]["hcal_start"] = 2  # both layers ecal
        heights = np.array([0.25, 0.5, 0.85])
        flat = np.zeros_like(heights)
        fake_muon_map((flat, heights, flat.copy(), np.ones_like(heights)))

        layers, _ = detector_map.create_map(config)

        points = np.zeros((1, len(heights), 4))
        points[0, :, 1] = heights
        points[0, :, 3] = 1.0
        assigned = detector_map.find_layers(config, points, "detector")[0]
        for layer_n, layer in enumerate(layers):
            assert layer["grid"].sum() == (assigned == layer_n).sum()

    def test_the_window_excludes_its_top_edge(self, config, fake_muon_map):
        """``find_layers`` closes its band with ``<`` at the same number, so a
        hit exactly there must not be given a cell no point can be assigned to.
        """
        bottoms = config["detector"]["layer_bottom_pos"]
        thickness = detector_map.detector_cell_thickness(config)
        top = bottoms[0] + 1.5 * thickness[0]

        X, Y, Z, E = muon_grid(config, x_centres=(0.0,), z_centres=(0.0,))
        fake_muon_map(
            (
                np.append(X, 0.0),
                np.append(Y, top),
                np.append(Z, 0.0),
                np.append(E, 1.0),
            )
        )

        layers, _ = detector_map.create_map(config)

        assert layers[0]["grid"].sum() == 1.0
        unassigned = detector_map.find_layers(
            config, one_point(top, axis=1), "detector"
        )
        assert unassigned[0, 0] == -1

    def test_the_edges_come_back_as_arrays(self, config, fake_muon_map):
        fake_muon_map(muon_grid(config))

        layers, _ = detector_map.create_map(config)

        assert isinstance(layers[0]["xedges"], np.ndarray)
        assert isinstance(layers[0]["zedges"], np.ndarray)
        assert isinstance(layers[0]["grid"], np.ndarray)
