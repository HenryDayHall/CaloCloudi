"""Tests for ``src/data/transforms.py``.

Two halves: the individual ``Transformation`` subclasses and the
config-plumbing helpers on top of them (``compose``, ``_resolve_value``,
``leaf_like``, ``fetch_values``, ``preprocessing``).

Everything except ``Partial`` is upstream PointCountFM code, so those tests
mostly pin down what it already does rather than assert what it ought to do.
``Partial`` has been reworked here, and its tests are specifications: it
validates its arguments, leaves its input alone, and behaves like a normal
module tree.
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
# Transformation (the base class)
# --------------------------------------------------------------------------- #
class TestTransformation:
    def test_is_a_module(self):
        assert isinstance(transforms.Transformation(), torch.nn.Module)

    @pytest.mark.parametrize("method", ["forward", "inverse", "fit"])
    def test_the_base_class_transforms_nothing(self, method):
        """``fit`` too: it defers to ``forward``, which is not implemented."""
        with pytest.raises(NotImplementedError):
            getattr(transforms.Transformation(), method)(torch.tensor([1.0]))

    def test_fit_defers_to_forward(self):
        """Stateless subclasses inherit ``fit`` and get ``forward`` for free."""
        x = torch.tensor([3.0])
        torch.testing.assert_close(transforms.Affine(scale=2.0).fit(x), torch.tensor([6.0]))

    def test_fit_ignores_the_mask_unless_a_subclass_overrides_it(self):
        x = torch.tensor([1.0, 2.0])
        mask = torch.tensor([True, False])

        torch.testing.assert_close(
            transforms.Affine(scale=2.0).fit(x, mask), torch.tensor([2.0, 4.0])
        )

    def test_calling_the_module_runs_forward(self):
        torch.testing.assert_close(
            transforms.Affine(scale=2.0)(torch.tensor([1.0])), torch.tensor([2.0])
        )


# --------------------------------------------------------------------------- #
# Sequence
# --------------------------------------------------------------------------- #
class TestSequence:
    def test_forward_applies_left_to_right(self):
        result = transforms.Sequence(
            [transforms.Affine(scale=2.0), transforms.Affine(shift=1.0)]
        )

        # 2 * 3 = 6, then 6 + 1 = 7
        torch.testing.assert_close(result.forward(torch.tensor([3.0])), torch.tensor([7.0]))

    def test_inverse_applies_right_to_left(self):
        result = transforms.Sequence(
            [transforms.Affine(scale=2.0), transforms.Affine(shift=1.0)]
        )

        torch.testing.assert_close(result.inverse(torch.tensor([7.0])), torch.tensor([3.0]))

    def test_an_empty_sequence_is_the_identity(self):
        result = transforms.Sequence([])

        x = torch.arange(3.0)
        torch.testing.assert_close(result.forward(x), x)
        torch.testing.assert_close(result.inverse(x), x)

    def test_sequences_nest(self):
        inner = transforms.Sequence([transforms.Affine(scale=2.0)])
        outer = transforms.Sequence([inner, transforms.Affine(scale=3.0)])

        torch.testing.assert_close(outer.forward(torch.tensor([1.0])), torch.tensor([6.0]))

    def test_fit_is_threaded_through_every_sub_module(self):
        scaler = transforms.StandardScaler((1, 1))
        result = transforms.Sequence([transforms.Affine(scale=2.0), scaler])

        out = result.fit(torch.tensor([[1.0], [2.0], [3.0]]))

        # the scaler sees 2, 4, 6 -- not 1, 2, 3
        torch.testing.assert_close(scaler.mean, torch.tensor([[4.0]]))
        torch.testing.assert_close(scaler.std, torch.tensor([[2.0]]))
        torch.testing.assert_close(out, torch.tensor([[-1.0], [0.0], [1.0]]))

    def test_fit_forwards_the_mask(self):
        scaler = transforms.StandardScaler((1, 1))
        result = transforms.Sequence([scaler])
        x = torch.tensor([[1.0], [2.0], [1000.0]])

        result.fit(x, torch.tensor([[True], [True], [False]]))

        torch.testing.assert_close(scaler.mean, torch.tensor([[1.5]]))

    def test_sub_modules_are_registered(self):
        """Unlike :class:`Partial`, a ``Sequence`` really is a module tree, so
        buffers survive ``state_dict`` and ``.to()``."""
        result = transforms.Sequence([transforms.StandardScaler((1, 1))])

        assert list(result.state_dict()) == ["sub_modules.0.mean", "sub_modules.0.std"]

    def test_a_nested_partial_is_registered_too(self):
        result = transforms.Sequence(
            [transforms.Partial([1], [transforms.StandardScaler((1, 1)), None])]
        )

        assert list(result.state_dict()) == [
            "sub_modules.0.components.0.mean",
            "sub_modules.0.components.0.std",
        ]

    def test_forward_does_not_check_the_sub_module_type(self):
        """Only ``fit`` and ``inverse`` guard against non-``Transformation``
        modules; ``forward`` will happily run anything with a ``forward``."""
        result = transforms.Sequence([torch.nn.Identity()])

        torch.testing.assert_close(result.forward(torch.tensor([1.0])), torch.tensor([1.0]))

    @pytest.mark.parametrize("method", ["fit", "inverse"])
    def test_non_transformation_sub_modules_are_rejected(self, method):
        result = transforms.Sequence([torch.nn.Identity()])

        with pytest.raises(ValueError, match="must be of type Transformation"):
            getattr(result, method)(torch.tensor([1.0]))

    def test_non_modules_are_rejected_at_construction(self):
        with pytest.raises(TypeError):
            transforms.Sequence([lambda x: x])


# --------------------------------------------------------------------------- #
# Partial
# --------------------------------------------------------------------------- #
class TestPartial:
    def test_each_slice_gets_its_own_component(self):
        result = transforms.Partial(
            [1], [transforms.Affine(scale=10.0), transforms.Affine(scale=100.0)]
        )

        torch.testing.assert_close(
            result.forward(torch.tensor([[2.0, 3.0]])), torch.tensor([[20.0, 300.0]])
        )

    def test_a_null_component_leaves_its_slice_alone(self):
        result = transforms.Partial([1], [transforms.Affine(scale=10.0), None])

        torch.testing.assert_close(
            result.forward(torch.tensor([[2.0, 3.0]])), torch.tensor([[20.0, 3.0]])
        )

    def test_components_may_be_specs_instead_of_instances(self):
        result = transforms.Partial([1], [[["Affine", {"scale": 10.0}]], None])

        torch.testing.assert_close(
            result.forward(torch.tensor([[2.0, 3.0]])), torch.tensor([[20.0, 3.0]])
        )

    def test_a_transformation_instance_is_used_as_given(self):
        affine = transforms.Affine(scale=10.0)
        result = transforms.Partial([1], [affine, None])

        assert result.components[0] is affine

    def test_a_module_that_is_not_a_transformation_is_rejected(self):
        with pytest.raises(TypeError, match="must be Transformations"):
            transforms.Partial([1], [torch.nn.Identity(), None])

    def test_n_split_indices_give_n_plus_one_slices(self):
        result = transforms.Partial([1, 2, 3], [None, None, None, None])

        assert len(result.components) == 4

    def test_no_split_indices_means_one_slice_covering_everything(self):
        result = transforms.Partial([], [transforms.Affine(scale=10.0)])

        torch.testing.assert_close(
            result.forward(torch.tensor([[1.0, 2.0]])), torch.tensor([[10.0, 20.0]])
        )

    def test_the_last_slice_runs_to_the_end(self):
        result = transforms.Partial([1], [None, transforms.Affine(scale=10.0)])

        torch.testing.assert_close(
            result.forward(torch.tensor([[1.0, 2.0, 3.0]])),
            torch.tensor([[1.0, 20.0, 30.0]]),
        )

    def test_the_shape_is_preserved(self):
        result = transforms.Partial([1, 2], [None, None, None])

        assert result.forward(torch.ones(2, 5, 4)).shape == (2, 5, 4)

    def test_splits_the_last_axis_by_default(self):
        result = transforms.Partial([1], [transforms.Affine(scale=10.0), None])

        out = result.forward(torch.ones(2, 3, 2))

        torch.testing.assert_close(out[..., 0], torch.full((2, 3), 10.0))
        torch.testing.assert_close(out[..., 1], torch.ones(2, 3))

    def test_a_negative_axis_counts_from_the_back(self):
        result = transforms.Partial([1], [transforms.Affine(scale=10.0), None], axis=-2)

        out = result.forward(torch.ones(2, 3, 4))

        torch.testing.assert_close(out[:, 0], torch.full((2, 4), 10.0))
        torch.testing.assert_close(out[:, 1:], torch.ones(2, 2, 4))

    def test_a_positive_axis_counts_from_the_front(self):
        result = transforms.Partial([1], [transforms.Affine(scale=10.0), None], axis=0)

        torch.testing.assert_close(
            result.forward(torch.tensor([[1.0, 1.0], [2.0, 2.0]])),
            torch.tensor([[10.0, 10.0], [2.0, 2.0]]),
        )

    def test_forward_leaves_its_argument_alone(self):
        result = transforms.Partial([1], [transforms.Affine(scale=10.0), None])
        x = torch.tensor([[2.0, 3.0]])

        out = result.forward(x)

        assert out is not x
        torch.testing.assert_close(x, torch.tensor([[2.0, 3.0]]))
        torch.testing.assert_close(out, torch.tensor([[20.0, 3.0]]))

    def test_inverse_leaves_its_argument_alone(self):
        result = transforms.Partial([1], [transforms.Affine(scale=10.0), None])
        x = torch.tensor([[20.0, 3.0]])

        out = result.inverse(x)

        assert out is not x
        torch.testing.assert_close(x, torch.tensor([[20.0, 3.0]]))
        torch.testing.assert_close(out, torch.tensor([[2.0, 3.0]]))

    def test_round_trips(self):
        result = transforms.Partial(
            [1, 2], [transforms.Affine(scale=10.0), transforms.Log(alpha=0.0), None]
        )
        x = torch.tensor([[2.0, 3.0, 4.0]])

        torch.testing.assert_close(result.inverse(result.forward(x)), x)

    def test_partials_nest(self):
        inner = transforms.Partial([1], [transforms.Affine(scale=10.0), None])
        result = transforms.Partial([2], [inner, transforms.Affine(scale=100.0)])

        torch.testing.assert_close(
            result.forward(torch.tensor([[1.0, 1.0, 1.0]])),
            torch.tensor([[10.0, 1.0, 100.0]]),
        )

    def test_gradients_flow_through_every_slice(self):
        result = transforms.Partial(
            [1], [transforms.Affine(scale=10.0), transforms.Affine(scale=100.0)]
        )
        x = torch.ones(1, 2, requires_grad=True)

        result.forward(x).sum().backward()

        torch.testing.assert_close(x.grad, torch.tensor([[10.0, 100.0]]))

    @pytest.mark.parametrize("components", [[None], [None, None, None]])
    def test_the_number_of_components_must_match_the_number_of_slices(self, components):
        with pytest.raises(ValueError, match="components were given"):
            transforms.Partial([1], components)

    @pytest.mark.parametrize("split_indices", [[0], [-1], [1, 1], [2, 1]])
    def test_split_indices_must_be_positive_and_increasing(self, split_indices):
        """Anything else would describe an empty slice."""
        with pytest.raises(ValueError, match="split_indices must be"):
            transforms.Partial(split_indices, [None] * (len(split_indices) + 1))

    @pytest.mark.parametrize("split_indices", [[1.0], ["1"], [True]])
    def test_split_indices_must_be_integers(self, split_indices):
        with pytest.raises(TypeError, match="split_indices must be integers"):
            transforms.Partial(split_indices, [None, None])

    @pytest.mark.parametrize("method", ["forward", "inverse", "fit"])
    def test_an_input_too_small_for_the_splits_is_rejected(self, method):
        """Slicing past the end would silently give an empty last slice."""
        result = transforms.Partial([3], [None, None])

        with pytest.raises(ValueError, match="do not fit along axis"):
            getattr(result, method)(torch.ones(1, 2))

    @pytest.mark.parametrize("method", ["forward", "inverse", "fit"])
    def test_an_axis_the_input_does_not_have_is_rejected(self, method):
        result = transforms.Partial([1], [None, None], axis=3)

        with pytest.raises(ValueError, match="out of range"):
            getattr(result, method)(torch.ones(2, 2))

    def test_components_are_registered_as_sub_modules(self):
        """A ``ModuleList``, so nested buffers survive ``state_dict`` and
        ``.to()``."""
        result = transforms.Partial([1], [transforms.StandardScaler((1, 1)), None])

        assert list(result.state_dict()) == ["components.0.mean", "components.0.std"]

    def test_dtype_conversion_reaches_the_components(self):
        result = transforms.Partial([1], [transforms.StandardScaler((1, 1)), None])

        result.double()

        assert result.components[0].mean.dtype == torch.float64

    def test_fit_reaches_the_components(self):
        scaler = transforms.StandardScaler((1, 1))
        result = transforms.Partial([1], [scaler, None])

        out = result.fit(torch.tensor([[1.0, 7.0], [5.0, 7.0]]))

        torch.testing.assert_close(scaler.mean, torch.tensor([[3.0]]))
        torch.testing.assert_close(scaler.std, torch.tensor([[2.0 * 2**0.5]]))
        torch.testing.assert_close(out[:, 1], torch.tensor([7.0, 7.0]))

    def test_fit_passes_each_component_the_matching_slice_of_the_mask(self):
        scaler = transforms.StandardScaler((1, 1))
        result = transforms.Partial([1], [scaler, None])
        x = torch.tensor([[1.0, 0.0], [2.0, 0.0], [1000.0, 0.0]])

        result.fit(x, torch.tensor([[True, False], [True, False], [False, False]]))

        torch.testing.assert_close(scaler.mean, torch.tensor([[1.5]]))

    def test_an_integer_input_is_promoted_not_truncated(self):
        result = transforms.Partial([1], [transforms.Affine(scale=0.5), None])

        out = result.forward(torch.tensor([[3, 4]]))

        assert out.dtype == torch.get_default_dtype()
        torch.testing.assert_close(out, torch.tensor([[1.5, 4.0]]))


# --------------------------------------------------------------------------- #
# Identity
# --------------------------------------------------------------------------- #
class TestIdentity:
    @pytest.mark.parametrize("method", ["forward", "inverse", "fit"])
    def test_hands_back_the_very_same_tensor(self, method):
        x = torch.arange(4.0)
        assert getattr(transforms.Identity(), method)(x) is x


# --------------------------------------------------------------------------- #
# Log
# --------------------------------------------------------------------------- #
class TestLog:
    def test_defaults(self):
        result = transforms.Log()

        assert result.alpha == 1e-6
        assert result.log_base == pytest.approx(1.0)  # natural log

    def test_forward_is_a_shifted_log(self):
        result = transforms.Log(alpha=1.0)

        torch.testing.assert_close(
            result.forward(torch.tensor([0.0, math.e - 1])), torch.tensor([0.0, 1.0])
        )

    def test_the_base_can_be_changed(self):
        result = transforms.Log(alpha=0.0, base=10.0)

        torch.testing.assert_close(
            result.forward(torch.tensor([1.0, 100.0])), torch.tensor([0.0, 2.0])
        )

    @pytest.mark.parametrize("base", [math.e, 2.0, 10.0])
    def test_round_trips(self, base):
        result = transforms.Log(alpha=1e-3, base=base)
        x = torch.tensor([0.0, 0.5, 7.0])

        torch.testing.assert_close(result.inverse(result.forward(x)), x)

    def test_alpha_keeps_zero_finite(self):
        torch.testing.assert_close(
            transforms.Log(alpha=1.0).forward(torch.tensor([0.0])), torch.tensor([0.0])
        )

    def test_without_alpha_zero_and_negatives_blow_up(self):
        out = transforms.Log(alpha=0.0).forward(torch.tensor([0.0, -1.0]))

        assert out[0] == -float("inf")
        assert torch.isnan(out[1])

    def test_the_inverse_can_never_return_less_than_minus_alpha(self):
        """``exp`` is positive, so the reconstruction is bounded below.  With
        the default ``alpha`` that floor is essentially zero, which is what
        keeps reconstructed energies non-negative."""
        out = transforms.Log(alpha=0.5).inverse(torch.tensor([-100.0]))

        torch.testing.assert_close(out, torch.tensor([-0.5]))

    def test_base_one_divides_by_zero(self):
        """``log(1) == 0``, and nothing checks for it -- do not write
        ``base: 1`` in a config."""
        out = transforms.Log(alpha=0.0, base=1.0).forward(torch.tensor([2.0]))

        assert out.isinf().all()


# --------------------------------------------------------------------------- #
# LogIt
# --------------------------------------------------------------------------- #
class TestLogIt:
    @pytest.mark.parametrize("alpha", [0.0, 1e-6, 0.1])
    def test_a_half_maps_to_zero(self, alpha):
        torch.testing.assert_close(
            transforms.LogIt(alpha=alpha).forward(torch.tensor([0.5])),
            torch.tensor([0.0]),
        )

    def test_alpha_squeezes_the_unit_interval_so_the_ends_stay_finite(self):
        result = transforms.LogIt(alpha=0.1)

        # 0 -> 0.1, 1 -> 0.9
        torch.testing.assert_close(
            result.forward(torch.tensor([0.0, 1.0])),
            torch.tensor([math.log(1 / 9), math.log(9.0)]),
        )

    def test_without_alpha_the_ends_are_infinite(self):
        out = transforms.LogIt(alpha=0.0).forward(torch.tensor([0.0, 1.0]))

        assert out[0] == -float("inf")
        assert out[1] == float("inf")

    def test_is_increasing(self):
        out = transforms.LogIt(alpha=1e-3).forward(torch.tensor([0.1, 0.4, 0.6, 0.9]))

        assert (out.diff() > 0).all()

    @pytest.mark.parametrize("alpha", [0.0, 1e-6, 0.1])
    def test_round_trips(self, alpha):
        result = transforms.LogIt(alpha=alpha)
        x = torch.tensor([0.1, 0.5, 0.9])

        torch.testing.assert_close(result.inverse(result.forward(x)), x)

    def test_the_inverse_of_zero_is_a_half(self):
        torch.testing.assert_close(
            transforms.LogIt(alpha=0.25).inverse(torch.tensor([0.0])),
            torch.tensor([0.5]),
        )

    def test_inputs_outside_the_unit_interval_give_nan(self):
        out = transforms.LogIt(alpha=0.1).forward(torch.tensor([2.0, -1.0]))

        assert out.isnan().all()

    def test_the_inverse_can_leave_the_unit_interval(self):
        """``sigmoid`` lands in ``(0, 1)``, but undoing the ``alpha`` squeeze
        then stretches that back out to ``(-a, 1 + a)``."""
        out = transforms.LogIt(alpha=0.1).inverse(torch.tensor([-100.0, 100.0]))

        torch.testing.assert_close(out, torch.tensor([-0.125, 1.125]))


# --------------------------------------------------------------------------- #
# Affine
# --------------------------------------------------------------------------- #
class TestAffine:
    def test_defaults_to_the_identity(self):
        x = torch.tensor([3.0])
        torch.testing.assert_close(transforms.Affine().forward(x), x)

    def test_the_shift_happens_before_the_scale(self):
        """``a * (x + b)``, not ``a * x + b``."""
        result = transforms.Affine(scale=2.0, shift=1.0)

        torch.testing.assert_close(result.forward(torch.tensor([3.0])), torch.tensor([8.0]))

    def test_inverse_scale_is_one_over_scale(self):
        assert transforms.Affine(inverse_scale=4.0).a == pytest.approx(0.25)

    def test_neg_shift_is_minus_shift(self):
        assert transforms.Affine(neg_shift=1.0).b == pytest.approx(-1.0)

    def test_standardising_form_matches_the_config_idiom(self):
        """``inverse_scale``/``neg_shift`` is how the configs write
        ``(x - mean) / std``."""
        result = transforms.Affine(inverse_scale=2.0, neg_shift=1.0)

        torch.testing.assert_close(result.forward(torch.tensor([5.0])), torch.tensor([2.0]))

    def test_inverse_scale_wins_over_scale(self):
        assert transforms.Affine(scale=2.0, inverse_scale=4.0).a == pytest.approx(0.25)

    def test_neg_shift_wins_over_shift(self):
        assert transforms.Affine(shift=5.0, neg_shift=1.0).b == pytest.approx(-1.0)

    def test_round_trips(self):
        result = transforms.Affine(scale=2.5, shift=-1.5)
        x = torch.tensor([-3.0, 0.0, 4.0])

        torch.testing.assert_close(result.inverse(result.forward(x)), x)

    def test_a_zero_inverse_scale_raises_at_construction(self):
        with pytest.raises(ZeroDivisionError):
            transforms.Affine(inverse_scale=0.0)

    def test_a_zero_scale_is_not_invertible(self):
        result = transforms.Affine(scale=0.0)

        torch.testing.assert_close(result.forward(torch.tensor([5.0])), torch.tensor([0.0]))
        assert result.inverse(torch.tensor([1.0])).isinf().all()

    def test_integer_inputs_are_promoted(self):
        out = transforms.Affine(scale=0.5).forward(torch.tensor([3, 5]))

        assert out.dtype == torch.get_default_dtype()
        torch.testing.assert_close(out, torch.tensor([1.5, 2.5]))

    def test_broadcasts_over_any_shape(self):
        result = transforms.Affine(scale=2.0)
        x = torch.ones(2, 3, 4)

        torch.testing.assert_close(result.forward(x), 2 * x)


# --------------------------------------------------------------------------- #
# Clamp
# --------------------------------------------------------------------------- #
class TestClamp:
    def test_defaults_to_the_unit_interval(self):
        result = transforms.Clamp()

        torch.testing.assert_close(
            result.forward(torch.tensor([-1.0, 0.5, 2.0])),
            torch.tensor([0.0, 0.5, 1.0]),
        )

    def test_the_bounds_can_be_changed(self):
        result = transforms.Clamp(min=-1.0, max=1.0)

        torch.testing.assert_close(
            result.forward(torch.tensor([-5.0, 0.0, 5.0])),
            torch.tensor([-1.0, 0.0, 1.0]),
        )

    def test_values_inside_the_range_are_untouched(self):
        x = torch.tensor([0.1, 0.5, 0.9])
        torch.testing.assert_close(transforms.Clamp().forward(x), x)

    def test_the_inverse_is_a_no_op_not_an_inverse(self):
        """Clamping loses information, so ``inverse`` just passes values
        through -- a ``Sequence`` containing a ``Clamp`` does not round trip
        for anything that was actually clamped."""
        result = transforms.Clamp()
        x = torch.tensor([-1.0, 2.0])

        torch.testing.assert_close(result.inverse(result.forward(x)), torch.tensor([0.0, 1.0]))
        torch.testing.assert_close(result.inverse(x), x)

    def test_an_inverted_range_collapses_everything_onto_max(self):
        """``min > max`` is not rejected; ``torch.clamp`` resolves it in favour
        of ``max``."""
        result = transforms.Clamp(min=1.0, max=0.0)

        torch.testing.assert_close(
            result.forward(torch.tensor([-5.0, 0.5, 5.0])), torch.zeros(3)
        )

    def test_fit_clamps_like_forward(self):
        torch.testing.assert_close(
            transforms.Clamp().fit(torch.tensor([5.0])), torch.tensor([1.0])
        )


# --------------------------------------------------------------------------- #
# StandardScaler
# --------------------------------------------------------------------------- #
class TestStandardScaler:
    def test_starts_as_the_identity(self):
        result = transforms.StandardScaler((1, 4))
        x = torch.tensor([[1.0, 2.0, 3.0, 4.0]])

        torch.testing.assert_close(result.forward(x), x)

    def test_mean_and_std_are_buffers_not_parameters(self):
        result = transforms.StandardScaler((1, 4))

        assert list(result.parameters()) == []
        assert sorted(result.state_dict()) == ["mean", "std"]

    def test_fit_reduces_over_the_axes_marked_with_a_one(self):
        result = transforms.StandardScaler((1, 4))
        x = torch.arange(20.0).reshape(5, 4)

        result.fit(x.clone())

        torch.testing.assert_close(result.mean, torch.tensor([[8.0, 9.0, 10.0, 11.0]]))
        assert result.mean.shape == (1, 4)

    def test_fit_uses_the_sample_standard_deviation(self):
        result = transforms.StandardScaler((1, 1))

        result.fit(torch.tensor([[1.0], [2.0], [3.0]]))

        torch.testing.assert_close(result.std, torch.tensor([[1.0]]))  # n - 1, not n

    def test_fit_returns_the_standardised_tensor(self):
        result = transforms.StandardScaler((1, 4))
        x = torch.arange(20.0).reshape(5, 4)

        out = result.fit(x.clone())

        torch.testing.assert_close(out.mean(dim=0), torch.zeros(4), atol=1e-6, rtol=0)
        torch.testing.assert_close(out.std(dim=0), torch.ones(4))
        torch.testing.assert_close(out, result.forward(x))

    def test_round_trips_after_fitting(self):
        result = transforms.StandardScaler((1, 4))
        x = torch.arange(20.0).reshape(5, 4)

        result.fit(x.clone())

        torch.testing.assert_close(result.inverse(result.forward(x)), x)

    def test_a_mask_excludes_entries_from_the_statistics(self):
        result = transforms.StandardScaler((1, 1))
        x = torch.tensor([[1.0], [2.0], [1000.0]])

        result.fit(x, torch.tensor([[True], [True], [False]]))

        torch.testing.assert_close(result.mean, torch.tensor([[1.5]]))
        torch.testing.assert_close(result.std, torch.tensor([[0.5**0.5]]))

    def test_a_constant_feature_gets_a_unit_std(self):
        result = transforms.StandardScaler((1, 2))
        x = torch.tensor([[3.0, 1.0], [3.0, 2.0], [3.0, 3.0]])

        out = result.fit(x.clone())

        torch.testing.assert_close(result.std[0, 0], torch.tensor(1.0))
        torch.testing.assert_close(out[:, 0], torch.zeros(3))

    def test_a_single_unmasked_entry_gives_a_nan_std(self):
        """The ``n - 1`` denominator is zero, and the ``std == 0`` guard does
        not catch the resulting nan -- everything downstream becomes nan."""
        result = transforms.StandardScaler((1, 1))
        x = torch.tensor([[1.0], [2.0], [3.0]])

        out = result.fit(x, torch.tensor([[True], [False], [False]]))

        assert result.std.isnan().all()
        assert out.isnan().all()

    def test_a_shape_without_a_one_reduces_over_everything(self):
        """``dims`` comes out empty and ``torch.sum(..., dim=())`` reduces all
        axes, so ``StandardScaler((4,))`` standardises globally rather than
        per feature."""
        result = transforms.StandardScaler((4,))
        x = torch.arange(20.0).reshape(5, 4)

        result.fit(x.clone())

        assert result.mean.shape == (1, 1)
        torch.testing.assert_close(result.mean, x.mean().reshape(1, 1))

    def test_fitting_can_change_the_buffer_shape_and_break_a_reload(self):
        """``fit`` overwrites the buffers with ``keepdim`` reductions, whose
        shape need not match the ``shape`` the scaler was built with."""
        fitted = transforms.StandardScaler((4,))
        fitted.fit(torch.arange(20.0).reshape(5, 4))

        with pytest.raises(RuntimeError, match="size mismatch"):
            transforms.StandardScaler((4,)).load_state_dict(fitted.state_dict())

    def test_a_shape_with_a_one_reloads_cleanly(self):
        fitted = transforms.StandardScaler((1, 4))
        fitted.fit(torch.arange(20.0).reshape(5, 4))

        fresh = transforms.StandardScaler((1, 4))
        fresh.load_state_dict(fitted.state_dict())

        torch.testing.assert_close(fresh.mean, fitted.mean)


# --------------------------------------------------------------------------- #
# Dequantize
# --------------------------------------------------------------------------- #
class TestDequantize:
    def test_adds_noise_from_the_unit_interval(self):
        torch.manual_seed(0)
        x = torch.zeros(1000)

        noise = transforms.Dequantize().forward(x)

        assert (noise >= 0).all()
        assert (noise < 1).all()

    def test_does_not_touch_the_input(self):
        x = torch.zeros(4)

        out = transforms.Dequantize().forward(x)

        assert out is not x
        torch.testing.assert_close(x, torch.zeros(4))

    def test_the_inverse_recovers_the_original_bin(self):
        torch.manual_seed(0)
        result = transforms.Dequantize()
        x = torch.tensor([0.0, 1.0, 7.0, 30.0])

        torch.testing.assert_close(result.inverse(result.forward(x)), x)

    def test_the_inverse_is_a_floor_so_it_only_undoes_whole_numbers(self):
        result = transforms.Dequantize()

        torch.testing.assert_close(
            result.inverse(torch.tensor([2.9, 2.0])), torch.tensor([2.0, 2.0])
        )

    def test_the_floor_rounds_negatives_away_from_zero(self):
        """Only meaningful for non-negative bin indices; a negative input comes
        back one bin lower than it went in."""
        result = transforms.Dequantize()

        torch.testing.assert_close(
            result.inverse(torch.tensor([-0.5, -1.2])), torch.tensor([-1.0, -2.0])
        )

    def test_is_not_deterministic(self):
        torch.manual_seed(0)
        result = transforms.Dequantize()
        x = torch.zeros(100)

        assert not torch.equal(result.forward(x), result.forward(x))

    def test_is_reproducible_under_a_seed(self):
        result = transforms.Dequantize()
        x = torch.zeros(100)

        torch.manual_seed(0)
        first = result.forward(x)
        torch.manual_seed(0)

        torch.testing.assert_close(result.forward(x), first)

    def test_fit_also_adds_noise(self):
        torch.manual_seed(0)

        out = transforms.Dequantize().fit(torch.zeros(10))

        assert (out > 0).all()


# --------------------------------------------------------------------------- #
# Round trips, once, for everything that claims to be invertible
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "spec",
    [
        [["Identity"]],
        [["Log", {"alpha": 1e-3}]],
        [["Log", {"alpha": 0.0, "base": 10.0}]],
        [["LogIt", {"alpha": 1e-3}]],
        [["Affine", {"scale": 3.0, "shift": -2.0}]],
        [["Affine", {"inverse_scale": 3.0, "neg_shift": 2.0}]],
        [["Log", {"alpha": 1e-3}], ["Affine", {"inverse_scale": 2.0}]],
        [["Partial", {"split_indices": [1], "components": [[["Log"]], None]}]],
    ],
    ids=lambda spec: "+".join(element[0] for element in spec),
)
def test_forward_then_inverse_is_the_identity(spec):
    result = transforms.compose(spec)
    x = torch.tensor([[0.1, 0.4, 0.6, 0.9]])

    torch.testing.assert_close(result.inverse(result.forward(x)), x)


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

    @pytest.mark.parametrize("name", ["Partial", "StandardScaler"])
    def test_transformations_with_required_arguments_need_them(self, name):
        """``compose([[name]])`` calls the constructor with nothing, so these
        two cannot be written name-only in a config."""
        with pytest.raises(TypeError):
            transforms.compose([[name]])

    @pytest.mark.parametrize(
        "name", ["Identity", "Log", "LogIt", "Affine", "Clamp", "Dequantize"]
    )
    def test_every_other_exported_name_composes_with_defaults(self, name):
        result = transforms.compose([[name]])

        assert isinstance(result.sub_modules[0], getattr(transforms, name))

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

    def test_the_features_transform_leaves_its_input_alone(self, path):
        configs = self.load(path)
        transform = transforms.preprocessing(configs, "features")

        events = torch.rand(2, 5, 4) + 0.01
        before = events.clone()

        with torch.no_grad():
            transform.forward(events)

        torch.testing.assert_close(events, before)

    def test_conditioning_transform_round_trips(self, path):
        configs = self.load(path)
        transform = transforms.preprocessing(configs, "conditioning")

        cond = torch.rand(2, configs["model"]["cond_dim"])
        cond[:, 0] = torch.rand(2) * 100 + 1  # incident energy

        with torch.no_grad():
            restored = transform.inverse(transform.forward(cond.clone()))

        torch.testing.assert_close(restored, cond, rtol=1e-3, atol=1e-3)
