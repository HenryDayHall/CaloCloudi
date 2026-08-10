"""Tests for ``src/data/transforms.py``.

Covers the config-plumbing helpers (``compose``, ``_resolve_value``,
``leaf_like``, ``fetch_values``, ``preprocessing``) rather than the individual
``Transformation`` subclasses, which are borrowed from PointCountFM.
"""

import math

import pytest

torch = pytest.importorskip("torch")

from src.data import transforms  # noqa: E402
from conftest import REPO_ROOT  # noqa: E402


CONFIGS = {
    "data": {
        "Xstd": 2.0,
        "Xmean": 1.0,
        "layer_bottom_pos": [0.0, 1.0, 2.0],
    },
    "training": {"batch_size": 32},
}


# --------------------------------------------------------------------------- #
# compose
# --------------------------------------------------------------------------- #
class TestCompose:
    def test_none_gives_an_identity(self):
        result = transforms.compose(None)

        assert isinstance(result, transforms.Sequence)
        assert len(result.sub_modules) == 1
        assert isinstance(result.sub_modules[0], transforms.Identity)

    def test_empty_list_gives_an_empty_sequence(self):
        result = transforms.compose([])

        x = torch.arange(4.0)
        torch.testing.assert_close(result.forward(x), x)

    def test_name_only_entry_uses_default_arguments(self):
        result = transforms.compose([["Affine"]])

        x = torch.tensor([3.0])
        torch.testing.assert_close(result.forward(x), x)  # scale 1, shift 0

    def test_explicit_none_arguments_are_the_same_as_omitting_them(self):
        torch.testing.assert_close(
            transforms.compose([["Affine", None]]).forward(torch.tensor([3.0])),
            transforms.compose([["Affine"]]).forward(torch.tensor([3.0])),
        )

    def test_dict_arguments_are_keywords(self):
        result = transforms.compose([["Affine", {"scale": 2.0, "shift": 1.0}]])

        torch.testing.assert_close(result.forward(torch.tensor([3.0])), torch.tensor([8.0]))

    def test_list_arguments_are_positional(self):
        result = transforms.compose([["Affine", [2.0, 1.0]]])

        torch.testing.assert_close(result.forward(torch.tensor([3.0])), torch.tensor([8.0]))

    def test_transformations_are_applied_in_order(self):
        result = transforms.compose(
            [["Affine", {"scale": 2.0, "shift": 1.0}], ["Log", {"alpha": 0.0}]]
        )

        # log(2 * (1 + 1)), not 2 * (log(1) + 1)
        torch.testing.assert_close(
            result.forward(torch.tensor([1.0])), torch.tensor([math.log(4.0)])
        )

    def test_inverse_unwinds_in_the_opposite_order(self):
        result = transforms.compose(
            [["Affine", {"scale": 2.0, "shift": 1.0}], ["Log", {"alpha": 0.0}]]
        )
        x = torch.tensor([1.5, 3.0])

        torch.testing.assert_close(result.inverse(result.forward(x.clone())), x)

    @pytest.mark.parametrize("name", ["NotATransformation", "torch", "nn", ""])
    def test_unknown_names_are_rejected(self, name):
        with pytest.raises(ValueError, match="Invalid transformation"):
            transforms.compose([[name]])

    @pytest.mark.parametrize("name", ["Transformation", "Sequence", "compose"])
    def test_structural_names_are_rejected(self, name):
        """These are exported but are not usable as leaf transformations."""
        with pytest.raises(ValueError, match="Invalid transformation"):
            transforms.compose([[name]])

    @pytest.mark.parametrize("argument", ["2.0", 2.0, ("scale", 2.0)])
    def test_arguments_must_be_a_list_or_a_dict(self, argument):
        with pytest.raises(ValueError, match="must be a list or a dict"):
            transforms.compose([["Affine", argument]])

    def test_nested_partial_from_a_config_shaped_spec(self):
        spec = [
            [
                "Partial",
                {
                    "split_indices": [1],
                    "components": [
                        [["Affine", {"scale": 10.0}]],
                        None,
                    ],
                },
            ]
        ]

        result = transforms.compose(spec)
        out = result.forward(torch.tensor([[2.0, 3.0]]))

        torch.testing.assert_close(out, torch.tensor([[20.0, 3.0]]))


# --------------------------------------------------------------------------- #
# _resolve_value
# --------------------------------------------------------------------------- #
class TestResolveValue:
    @pytest.mark.parametrize("value", [1, 2.5, None, True])
    def test_non_iterables_pass_straight_through(self, value):
        assert transforms._resolve_value(CONFIGS, value) is value

    def test_a_valid_key_path_is_looked_up(self):
        assert transforms._resolve_value(CONFIGS, ["data", "Xstd"]) == 2.0

    def test_an_unknown_key_path_is_returned_unchanged(self):
        value = ["data", "not_a_key"]
        assert transforms._resolve_value(CONFIGS, value) == value

    def test_a_bare_string_is_returned_unchanged(self):
        assert transforms._resolve_value(CONFIGS, "data") == "data"

    def test_lists_are_resolved_to_whatever_they_point_at(self):
        assert transforms._resolve_value(CONFIGS, ["data", "layer_bottom_pos"]) == [
            0.0,
            1.0,
            2.0,
        ]

    def test_an_empty_path_resolves_to_the_whole_config(self):
        """Only ``KeyError`` is caught, and an empty loop body never raises, so
        ``[]`` silently returns the entire config dict."""
        assert transforms._resolve_value(CONFIGS, []) is CONFIGS

    def test_a_path_that_runs_off_the_end_of_the_tree_raises(self):
        """``configs["data"]["Xstd"]`` is a float, so indexing it again is a
        ``TypeError``, which the ``except KeyError`` does not catch."""
        with pytest.raises(TypeError):
            transforms._resolve_value(CONFIGS, ["data", "Xstd", "deeper"])


# --------------------------------------------------------------------------- #
# leaf_like
# --------------------------------------------------------------------------- #
class TestLeafLike:
    @pytest.mark.parametrize(
        "value", [1, 2.5, None, "a string", ["data", "Xstd"], []]
    )
    def test_leaves(self, value):
        assert transforms.leaf_like(value) is True

    @pytest.mark.parametrize(
        "value", [{"a": 1}, [{"a": 1}], [1, {"a": 1}], (1, 2)]
    )
    def test_non_leaves(self, value):
        assert transforms.leaf_like(value) is False

    def test_a_list_of_key_paths_also_counts_as_a_leaf(self):
        """Nested string lists satisfy ``all(leaf_like(...))`` and are then
        handed to ``_resolve_value``, which cannot use them as dict keys.  Do
        not write configs shaped like this."""
        assert transforms.leaf_like([["data", "Xstd"], ["data", "Xmean"]]) is True


# --------------------------------------------------------------------------- #
# fetch_values
# --------------------------------------------------------------------------- #
class TestFetchValues:
    def test_resolves_leaves_inside_a_dict(self):
        result = transforms.fetch_values(
            CONFIGS, {"scale": ["data", "Xstd"], "shift": 0.0}
        )
        assert result == {"scale": 2.0, "shift": 0.0}

    def test_resolves_leaves_inside_nested_lists(self):
        spec = [["Affine", {"inverse_scale": ["data", "Xstd"]}], ["Identity"]]

        result = transforms.fetch_values(CONFIGS, spec)

        assert result == [["Affine", {"inverse_scale": 2.0}], ["Identity"]]

    def test_dictionary_keys_are_left_alone(self):
        result = transforms.fetch_values(CONFIGS, {"data": 1})
        assert result == {"data": 1}

    def test_unresolvable_values_survive(self):
        spec = [["Log", {"alpha": 0.001}]]
        assert transforms.fetch_values(CONFIGS, spec) == spec

    def test_does_not_mutate_the_input(self):
        spec = [["Affine", {"inverse_scale": ["data", "Xstd"]}]]
        transforms.fetch_values(CONFIGS, spec)
        assert spec == [["Affine", {"inverse_scale": ["data", "Xstd"]}]]

    def test_tuples_are_not_descended_into(self):
        spec = (["data", "Xstd"],)
        assert transforms.fetch_values(CONFIGS, spec) is spec


# --------------------------------------------------------------------------- #
# preprocessing
# --------------------------------------------------------------------------- #
class TestPreprocessing:
    @pytest.fixture
    def configs(self):
        return {
            "data": {"Xstd": 2.0, "Xmean": 1.0},
            "preprocessing": {
                "features": [
                    [
                        "Affine",
                        {
                            "inverse_scale": ["data", "Xstd"],
                            "neg_shift": ["data", "Xmean"],
                        },
                    ]
                ],
                "conditioning": None,
            },
        }

    def test_config_references_are_resolved_before_construction(self, configs):
        result = transforms.preprocessing(configs, "features")

        # inverse_scale 2 -> a = 0.5, neg_shift 1 -> b = -1
        torch.testing.assert_close(
            result.forward(torch.tensor([5.0])), torch.tensor([2.0])
        )

    def test_returns_a_sequence(self, configs):
        assert isinstance(transforms.preprocessing(configs, "features"), transforms.Sequence)

    def test_round_trips(self, configs):
        result = transforms.preprocessing(configs, "features")
        x = torch.tensor([1.0, 5.0, -3.0])

        torch.testing.assert_close(result.inverse(result.forward(x.clone())), x)

    def test_a_null_part_becomes_the_identity(self, configs):
        result = transforms.preprocessing(configs, "conditioning")

        x = torch.tensor([1.0, 2.0])
        torch.testing.assert_close(result.forward(x), x)

    def test_unknown_part(self, configs):
        with pytest.raises(KeyError):
            transforms.preprocessing(configs, "not_a_part")


# --------------------------------------------------------------------------- #
# Smoke test against the configs actually shipped in the repo
# --------------------------------------------------------------------------- #
CONFIG_DIR = REPO_ROOT / "config"
CONFIG_FILES = sorted(CONFIG_DIR.glob("*.yaml")) if CONFIG_DIR.is_dir() else []


@pytest.mark.skipif(not CONFIG_FILES, reason="no config/*.yaml in the repo")
@pytest.mark.parametrize("path", CONFIG_FILES, ids=lambda p: p.name)
class TestShippedConfigs:
    @staticmethod
    def load(path):
        yaml = pytest.importorskip("yaml")
        with open(path) as handle:
            return yaml.safe_load(handle)

    def test_features_transform_round_trips(self, path):
        configs = self.load(path)
        transform = transforms.preprocessing(configs, "features")

        events = torch.rand(2, 5, 4) * 10 - 5
        events[..., 3] = torch.rand(2, 5) * 0.5 + 0.01  # energies must be positive

        with torch.no_grad():
            restored = transform.inverse(transform.forward(events.clone()))

        torch.testing.assert_close(restored, events, rtol=1e-3, atol=1e-3)

    def test_conditioning_transform_round_trips(self, path):
        configs = self.load(path)
        transform = transforms.preprocessing(configs, "conditioning")

        cond = torch.rand(2, configs["model"]["cond_dim"])
        cond[:, 0] = torch.rand(2) * 100 + 1  # incident energy

        with torch.no_grad():
            restored = transform.inverse(transform.forward(cond.clone()))

        torch.testing.assert_close(restored, cond, rtol=1e-3, atol=1e-3)
