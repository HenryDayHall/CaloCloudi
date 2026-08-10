"""Tests for ``src.diffusion.mean_flat``.

``src/diffusion.py`` imports ``k_diffusion`` at module level and subclasses out
of it, so it cannot be sensibly stubbed -- skip if it is not installed.

Everything here runs on the CPU with deliberately tiny hidden layers
(``diffusion_pointwise_hidden_l*``), so a full ``Diffusion`` model can be built
and even sampled from inside a unit test.
"""

import pytest

torch = pytest.importorskip("torch")
k_diffusion = pytest.importorskip("k_diffusion")

from src.diffusion import (  # noqa: E402
    ConcatSquashLinear,
    Denoiser,
    Diffusion,
    KLDloss,
    PointwiseNet_kDiffusion,
    mean_flat,
)


TIME_DIM = 64  # hard coded inside PointwiseNet_kDiffusion


def model_config(**overrides):
    """A ``config`` with the same shape as ``config/default.yaml``, but small.

    ``device`` is forced to cpu: ``Denoiser.__init__`` puts ``sigma_data``
    straight onto ``config["device"]``, so the shipped ``cuda`` default cannot
    be constructed on a test runner.
    """
    config = {
        "device": "cpu",
        "sampler": "euler",
        "num_steps": 3,
        "model": {
            "logarithmic_point_energy": True,
            "cond_dim": 4,
            "cond_features": ["incident_energy", "incident_direction"],
            "feature_dim": 4,
            "rho": 7.0,
            "s_churn": 0.0,
            "s_noise": 1.0,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "diffusion_pointwise_hidden_l1": 8,
            "diffusion_pointwise_hidden_l2": 8,
            "diffusion_pointwise_hidden_l3": 8,
            "diffusion_pointwise_hidden_l4": 8,
            "diffusion_pointwise_hidden_l5": 8,
        },
        "training": {
            "dtype": "float32",
            "sigma_data": 0.5,
            "diffusion_loss": "l2",
        },
    }
    for section, values in overrides.items():
        if isinstance(values, dict):
            config[section].update(values)
        else:
            config[section] = values
    return config


class ConstantInner(torch.nn.Module):
    """Stand-in for ``PointwiseNet_kDiffusion`` with a known, constant output.

    Records what it was called with so the preconditioning can be checked
    without also depending on the real network's weights.
    """

    def __init__(self, value=0.0):
        super().__init__()
        self.value = value
        self.calls = []

    def forward(self, x, sigma, **kwargs):
        self.calls.append({"x": x.detach().clone(), "sigma": sigma, "kwargs": kwargs})
        return torch.full_like(x, self.value)


def karras_scalings(sigma, sigma_data):
    """c_skip, c_out, c_in from Karras et al., written out independently.

    ``sigma`` is (B,), ``sigma_data`` is (4,); all three come back as (B, 1, 4)
    -- i.e. already unsqueezed the way ``Denoiser`` uses them.
    """
    s = sigma.reshape(-1, 1, 1)
    d = sigma_data.reshape(1, 1, -1)
    denom = s**2 + d**2
    return d**2 / denom, s * d / denom.sqrt(), 1 / denom.sqrt()


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(1234)


# --------------------------------------------------------------------------- #
# mean_flat
# --------------------------------------------------------------------------- #
class TestMeanFlat:
    @pytest.mark.parametrize("shape", [(3, 5), (3, 5, 4), (3, 5, 4, 2)])
    def test_reduces_everything_except_the_batch_axis(self, shape):
        x = torch.randn(shape)

        out = mean_flat(x)

        assert out.shape == (shape[0],)
        torch.testing.assert_close(out, x.reshape(shape[0], -1).mean(dim=1))

    def test_matches_a_hand_computed_value(self):
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[0.0, 0.0], [0.0, 8.0]]])
        torch.testing.assert_close(mean_flat(x), torch.tensor([2.5, 2.0]))

    def test_batch_of_one(self):
        x = torch.ones(1, 6, 4)
        torch.testing.assert_close(mean_flat(x), torch.tensor([1.0]))

    def test_each_row_is_reduced_independently(self):
        x = torch.stack([torch.full((4, 3), float(i)) for i in range(5)])
        torch.testing.assert_close(mean_flat(x), torch.arange(5, dtype=torch.float32))

    def test_does_not_modify_its_input(self):
        x = torch.randn(3, 5, 4)
        before = x.clone()
        mean_flat(x)
        torch.testing.assert_close(x, before)

    def test_preserves_dtype(self):
        x = torch.randn(3, 5, 4, dtype=torch.float64)
        assert mean_flat(x).dtype == torch.float64

    def test_gradient_is_spread_evenly(self):
        x = torch.zeros(2, 3, 4, requires_grad=True)

        mean_flat(x).sum().backward()

        expected = torch.full((2, 3, 4), 1.0 / 12)
        torch.testing.assert_close(x.grad, expected)

    def test_works_on_a_non_contiguous_tensor(self):
        x = torch.randn(3, 5, 4).transpose(1, 2)
        assert not x.is_contiguous()
        torch.testing.assert_close(mean_flat(x), x.reshape(3, -1).mean(dim=1))

    # NB: mean_flat on a 1-D tensor becomes ``tensor.mean(dim=[])``, whose
    # meaning changed across torch versions (reduce-all vs reduce-nothing).
    # The loss code only ever passes >=2-D tensors, so that case is left
    # untested rather than pinned to one torch release.


# --------------------------------------------------------------------------- #
# KLDloss
# --------------------------------------------------------------------------- #
class TestKLDloss:
    def test_a_standard_normal_costs_nothing(self):
        mu = torch.zeros(4, 8)
        logvar = torch.zeros(4, 8)

        torch.testing.assert_close(KLDloss()(mu, logvar), torch.tensor(0.0))

    def test_matches_a_hand_computed_value(self):
        # one element, mu=1, logvar=0 -> -0.5 * (1 + 0 - 1 - 1) = 0.5
        mu = torch.tensor([[1.0]])
        logvar = torch.tensor([[0.0]])

        torch.testing.assert_close(KLDloss()(mu, logvar), torch.tensor(0.5))

    def test_matches_the_closed_form(self):
        mu = torch.randn(3, 5)
        logvar = torch.randn(3, 5)

        expected = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / 3
        torch.testing.assert_close(KLDloss()(mu, logvar), expected)

    def test_returns_a_scalar(self):
        out = KLDloss()(torch.randn(3, 5), torch.randn(3, 5))

        assert out.shape == ()

    def test_is_non_negative(self):
        for _ in range(20):
            mu = torch.randn(4, 6)
            logvar = torch.randn(4, 6)
            assert KLDloss()(mu, logvar) >= 0

    def test_grows_with_the_distance_from_the_prior(self):
        logvar = torch.zeros(2, 3)
        near = KLDloss()(torch.full((2, 3), 0.5), logvar)
        far = KLDloss()(torch.full((2, 3), 5.0), logvar)

        assert far > near

    def test_repeating_the_batch_leaves_the_value_unchanged(self):
        """The sum is over *every* element but the divisor is only the batch."""
        mu = torch.randn(3, 5)
        logvar = torch.randn(3, 5)

        single = KLDloss()(mu, logvar)
        doubled = KLDloss()(mu.repeat(2, 1), logvar.repeat(2, 1))

        torch.testing.assert_close(single, doubled)

    def test_widening_the_feature_axis_does_scale_the_value(self):
        """Only dim 0 is normalised away, so the loss scales with feature count."""
        mu = torch.randn(3, 5)
        logvar = torch.randn(3, 5)

        narrow = KLDloss()(mu, logvar)
        wide = KLDloss()(mu.repeat(1, 2), logvar.repeat(1, 2))

        torch.testing.assert_close(wide, narrow * 2)

    def test_the_divisor_comes_from_logvar_not_mu(self):
        """``B = logvar.size(0)``; a broadcastable ``mu`` does not change it."""
        logvar = torch.zeros(4, 3)
        broadcast = KLDloss()(torch.ones(1, 3), logvar)
        expanded = KLDloss()(torch.ones(4, 3), logvar)

        torch.testing.assert_close(broadcast, expanded)
        torch.testing.assert_close(broadcast, torch.tensor(1.5))  # 4*3*0.5 / 4

    def test_gradients_reach_both_arguments(self):
        mu = torch.randn(2, 3, requires_grad=True)
        logvar = torch.randn(2, 3, requires_grad=True)

        KLDloss()(mu, logvar).backward()

        assert mu.grad is not None and logvar.grad is not None
        torch.testing.assert_close(mu.grad, mu.detach() / 2)  # d/dmu = mu / B

    def test_gradient_vanishes_at_the_prior(self):
        mu = torch.zeros(2, 3, requires_grad=True)
        logvar = torch.zeros(2, 3, requires_grad=True)

        KLDloss()(mu, logvar).backward()

        torch.testing.assert_close(mu.grad, torch.zeros(2, 3))
        torch.testing.assert_close(logvar.grad, torch.zeros(2, 3))

    def test_is_a_module_with_no_parameters(self):
        loss = KLDloss()

        assert isinstance(loss, torch.nn.Module)
        assert list(loss.parameters()) == []
        assert loss.state_dict() == {}

    def test_use_cuda_is_recorded_but_never_used(self):
        """``self.use_cuda`` is set in ``__init__`` and then read nowhere."""
        loss = KLDloss()

        assert loss.use_cuda == torch.cuda.is_available()

    def test_works_on_more_than_two_dimensions(self):
        mu = torch.randn(2, 3, 4)
        logvar = torch.randn(2, 3, 4)

        expected = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum() / 2
        torch.testing.assert_close(KLDloss()(mu, logvar), expected)

    def test_is_never_called_by_the_diffusion_model(self):
        """``Diffusion`` builds a ``KLDloss`` but no code path uses it."""
        model = Diffusion(model_config())

        assert isinstance(model.kld, KLDloss)


# --------------------------------------------------------------------------- #
# ConcatSquashLinear
# --------------------------------------------------------------------------- #
class TestConcatSquashLinear:
    def test_layer_shapes(self):
        layer = ConcatSquashLinear(3, 5, 7)

        assert layer._layer.weight.shape == (5, 3)
        assert layer._hyper_bias.weight.shape == (5, 7)
        assert layer._hyper_gate.weight.shape == (5, 7)

    def test_the_bias_projection_has_no_bias_of_its_own(self):
        layer = ConcatSquashLinear(3, 5, 7)

        assert layer._hyper_bias.bias is None
        assert layer._layer.bias is not None
        assert layer._hyper_gate.bias is not None

    def test_output_shape(self):
        layer = ConcatSquashLinear(3, 5, 7)

        out = layer(torch.randn(2, 4, 7), torch.randn(2, 4, 3))

        assert out.shape == (2, 4, 5)

    def test_context_broadcasts_over_the_point_axis(self):
        """The net passes ctx as (B, 1, C) against x as (B, N, D)."""
        layer = ConcatSquashLinear(3, 5, 7)

        out = layer(torch.randn(2, 1, 7), torch.randn(2, 6, 3))

        assert out.shape == (2, 6, 5)

    def test_the_context_argument_comes_first(self):
        """``forward(self, ctx, x)`` -- the reverse order is a size mismatch."""
        layer = ConcatSquashLinear(3, 5, 7)
        ctx = torch.randn(2, 1, 7)
        x = torch.randn(2, 6, 3)

        torch.testing.assert_close(layer(ctx, x), layer(ctx=ctx, x=x))
        with pytest.raises(RuntimeError):
            layer(x, ctx)

    def test_a_neutral_context_halves_the_inner_layer(self):
        """Zero weights and zero bias on the gate give sigmoid(0) = 1/2."""
        layer = ConcatSquashLinear(3, 5, 7)
        with torch.no_grad():
            layer._hyper_gate.weight.zero_()
            layer._hyper_gate.bias.zero_()
            layer._hyper_bias.weight.zero_()
        x = torch.randn(2, 6, 3)
        ctx = torch.randn(2, 1, 7)

        torch.testing.assert_close(layer(ctx, x), 0.5 * layer._layer(x))

    def test_an_open_gate_reduces_to_layer_plus_shift(self):
        layer = ConcatSquashLinear(3, 5, 7)
        with torch.no_grad():
            layer._hyper_gate.weight.zero_()
            layer._hyper_gate.bias.fill_(30.0)  # sigmoid(30) == 1 in float32
        x = torch.randn(2, 6, 3)
        ctx = torch.randn(2, 1, 7)

        expected = layer._layer(x) + layer._hyper_bias(ctx)
        torch.testing.assert_close(layer(ctx, x), expected)

    def test_a_closed_gate_leaves_only_the_shift(self):
        layer = ConcatSquashLinear(3, 5, 7)
        with torch.no_grad():
            layer._hyper_gate.weight.zero_()
            layer._hyper_gate.bias.fill_(-30.0)
        x = torch.randn(2, 6, 3)
        ctx = torch.randn(2, 1, 7)

        expected = layer._hyper_bias(ctx).expand(2, 6, 5)
        torch.testing.assert_close(layer(ctx, x), expected)

    def test_matches_the_formula(self):
        layer = ConcatSquashLinear(3, 5, 7)
        x = torch.randn(2, 6, 3)
        ctx = torch.randn(2, 1, 7)

        expected = (
            layer._layer(x) * torch.sigmoid(layer._hyper_gate(ctx))
            + layer._hyper_bias(ctx)
        )
        torch.testing.assert_close(layer(ctx, x), expected)

    def test_the_context_alone_can_move_the_output(self):
        layer = ConcatSquashLinear(3, 5, 7)
        x = torch.randn(2, 6, 3)

        first = layer(torch.zeros(2, 1, 7), x)
        second = layer(torch.ones(2, 1, 7), x)

        assert not torch.allclose(first, second)

    def test_each_point_is_transformed_independently(self):
        layer = ConcatSquashLinear(3, 5, 7)
        x = torch.randn(1, 6, 3)
        ctx = torch.randn(1, 1, 7)

        together = layer(ctx, x)
        apart = torch.cat([layer(ctx, x[:, i : i + 1]) for i in range(6)], dim=1)

        torch.testing.assert_close(together, apart)

    def test_all_three_sub_layers_receive_gradients(self):
        layer = ConcatSquashLinear(3, 5, 7)

        layer(torch.randn(2, 1, 7), torch.randn(2, 6, 3)).sum().backward()

        for name, parameter in layer.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name

    def test_a_mismatched_context_width_is_an_error(self):
        layer = ConcatSquashLinear(3, 5, 7)

        with pytest.raises(RuntimeError):
            layer(torch.randn(2, 1, 6), torch.randn(2, 6, 3))


# --------------------------------------------------------------------------- #
# PointwiseNet_kDiffusion
# --------------------------------------------------------------------------- #
class TestPointwiseNetKDiffusion:
    @pytest.fixture
    def net(self):
        return PointwiseNet_kDiffusion(model_config())

    def test_output_matches_the_input_shape(self, net):
        x = torch.randn(2, 6, 4)

        out = net(x, torch.rand(2) + 0.5, context=torch.randn(2, 4))

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_there_are_six_layers(self, net):
        assert len(net.layers) == 6

    def test_hidden_widths_come_from_the_config(self):
        config = model_config(
            model={
                "diffusion_pointwise_hidden_l1": 3,
                "diffusion_pointwise_hidden_l2": 5,
                "diffusion_pointwise_hidden_l3": 7,
                "diffusion_pointwise_hidden_l4": 9,
                "diffusion_pointwise_hidden_l5": 11,
            }
        )

        net = PointwiseNet_kDiffusion(config)

        widths = [layer._layer.out_features for layer in net.layers]
        assert widths == [3, 5, 7, 9, 11, 4]

    def test_default_widths_are_derived_from_the_first(self):
        """128 -> 256 -> 512 -> 256 -> 128, then back to feature_dim."""
        config = model_config()
        for key in list(config["model"]):
            if key.startswith("diffusion_pointwise_hidden"):
                del config["model"][key]

        net = PointwiseNet_kDiffusion(config)

        widths = [layer._layer.out_features for layer in net.layers]
        assert widths == [128, 256, 512, 256, 128, 4]

    def test_a_single_hidden_override_rescales_the_rest(self):
        config = model_config()
        for key in list(config["model"]):
            if key.startswith("diffusion_pointwise_hidden"):
                del config["model"][key]
        config["model"]["diffusion_pointwise_hidden_l1"] = 16

        net = PointwiseNet_kDiffusion(config)

        widths = [layer._layer.out_features for layer in net.layers]
        assert widths == [16, 32, 64, 32, 16, 4]

    def test_the_first_and_last_layers_use_feature_dim(self):
        net = PointwiseNet_kDiffusion(model_config(model={"feature_dim": 6}))

        assert net.layers[0]._layer.in_features == 6
        assert net.layers[-1]._layer.out_features == 6

    def test_every_layer_is_conditioned_on_context_plus_time(self):
        net = PointwiseNet_kDiffusion(model_config(model={"cond_dim": 5}))

        for layer in net.layers:
            assert layer._hyper_gate.in_features == 5 + TIME_DIM
            assert layer._hyper_bias.in_features == 5 + TIME_DIM

    def test_the_fourier_features_are_a_buffer_not_a_parameter(self):
        """The comment in the source promises these weights are not trained."""
        net = PointwiseNet_kDiffusion(model_config())

        parameters = dict(net.named_parameters())
        buffers = dict(net.named_buffers())
        assert "timestep_embed.0.weight" in buffers
        assert "timestep_embed.0.weight" not in parameters
        assert not buffers["timestep_embed.0.weight"].requires_grad

    def test_the_fourier_features_still_land_in_the_state_dict(self, net):
        """Buffers are checkpointed, so a reload reproduces the same embedding."""
        assert "timestep_embed.0.weight" in net.state_dict()

    def test_the_time_embedding_is_the_configured_width(self, net):
        assert net.timestep_embed[1].out_features == TIME_DIM

    def test_context_may_be_flat_or_already_unsqueezed(self, net):
        x = torch.randn(2, 6, 4)
        sigma = torch.rand(2) + 0.5
        context = torch.randn(2, 4)

        flat = net(x, sigma, context=context)
        unsqueezed = net(x, sigma, context=context.unsqueeze(1))

        torch.testing.assert_close(flat, unsqueezed)

    def test_sigma_may_be_flat_or_already_unsqueezed(self, net):
        x = torch.randn(2, 6, 4)
        sigma = torch.rand(2) + 0.5
        context = torch.randn(2, 4)

        torch.testing.assert_close(
            net(x, sigma, context=context), net(x, sigma.unsqueeze(1), context=context)
        )

    def test_points_are_processed_independently(self):
        """Nothing mixes the point axis, so the net is genuinely pointwise."""
        net = PointwiseNet_kDiffusion(model_config())
        x = torch.randn(1, 6, 4)
        sigma = torch.tensor([1.5])
        context = torch.randn(1, 4)

        together = net(x, sigma, context=context)
        apart = torch.cat(
            [net(x[:, i : i + 1], sigma, context=context) for i in range(6)], dim=1
        )

        torch.testing.assert_close(together, apart, rtol=1e-5, atol=1e-6)

    def test_permuting_points_permutes_the_output(self, net):
        x = torch.randn(1, 6, 4)
        sigma = torch.tensor([1.5])
        context = torch.randn(1, 4)
        order = torch.randperm(6)

        straight = net(x, sigma, context=context)[:, order]
        shuffled = net(x[:, order], sigma, context=context)

        torch.testing.assert_close(straight, shuffled, rtol=1e-5, atol=1e-6)

    def test_events_in_a_batch_do_not_leak_into_each_other(self, net):
        x = torch.randn(3, 6, 4)
        sigma = torch.rand(3) + 0.5
        context = torch.randn(3, 4)

        together = net(x, sigma, context=context)
        alone = net(x[1:2], sigma[1:2], context=context[1:2])

        torch.testing.assert_close(together[1:2], alone, rtol=1e-5, atol=1e-6)

    def test_the_output_depends_on_sigma(self, net):
        x = torch.randn(2, 6, 4)
        context = torch.randn(2, 4)

        low = net(x, torch.full((2,), 0.1), context=context)
        high = net(x, torch.full((2,), 10.0), context=context)

        assert not torch.allclose(low, high)

    def test_the_output_depends_on_the_context(self, net):
        x = torch.randn(2, 6, 4)
        sigma = torch.rand(2) + 0.5

        first = net(x, sigma, context=torch.zeros(2, 4))
        second = net(x, sigma, context=torch.ones(2, 4))

        assert not torch.allclose(first, second)

    def test_a_sigma_of_zero_produces_nans(self, net):
        """``c_noise = sigma.log()`` is -inf at 0, and cos(-inf) is nan."""
        out = net(torch.randn(2, 6, 4), torch.zeros(2), context=torch.randn(2, 4))

        assert torch.isnan(out).all()

    def test_a_negative_sigma_produces_nans(self, net):
        out = net(torch.randn(2, 6, 4), -torch.ones(2), context=torch.randn(2, 4))

        assert torch.isnan(out).all()

    def test_context_is_required_despite_defaulting_to_none(self, net):
        """The signature says ``context=None`` but the body calls ``.view`` on it."""
        with pytest.raises(AttributeError):
            net(torch.randn(2, 6, 4), torch.rand(2) + 0.5)

    def test_a_context_of_the_wrong_width_is_an_error(self, net):
        with pytest.raises(RuntimeError):
            net(torch.randn(2, 6, 4), torch.rand(2) + 0.5, context=torch.randn(2, 9))

    def test_gradients_reach_every_trainable_parameter(self, net):
        out = net(torch.randn(2, 6, 4), torch.rand(2) + 0.5, context=torch.randn(2, 4))

        out.sum().backward()

        for name, parameter in net.named_parameters():
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name

    def test_does_not_modify_its_input(self, net):
        x = torch.randn(2, 6, 4)
        before = x.clone()

        net(x, torch.rand(2) + 0.5, context=torch.randn(2, 4))

        torch.testing.assert_close(x, before)

    def test_is_deterministic(self, net):
        x = torch.randn(2, 6, 4)
        sigma = torch.rand(2) + 0.5
        context = torch.randn(2, 4)

        torch.testing.assert_close(
            net(x, sigma, context=context), net(x, sigma, context=context)
        )

    def test_a_single_point_event(self, net):
        out = net(torch.randn(1, 1, 4), torch.tensor([1.0]), context=torch.randn(1, 4))

        assert out.shape == (1, 1, 4)


# --------------------------------------------------------------------------- #
# Denoiser -- construction
# --------------------------------------------------------------------------- #
class TestDenoiserConstruction:
    def test_a_float_sigma_data_is_broadcast_to_four_features(self):
        denoiser = Denoiser(ConstantInner(), sigma_data=0.5, device="cpu")

        torch.testing.assert_close(denoiser.sigma_data, torch.full((4,), 0.5))

    def test_a_list_of_four_is_kept_as_given(self):
        denoiser = Denoiser(ConstantInner(), sigma_data=[1.0, 2.0, 3.0, 4.0], device="cpu")

        torch.testing.assert_close(
            denoiser.sigma_data, torch.tensor([1.0, 2.0, 3.0, 4.0])
        )

    @pytest.mark.parametrize("bad", [[0.5], [0.5] * 3, [0.5] * 5, []])
    def test_a_list_of_the_wrong_length_is_rejected(self, bad):
        with pytest.raises(ValueError, match="sigma_data"):
            Denoiser(ConstantInner(), sigma_data=bad, device="cpu")

    def test_an_integer_sigma_data_is_not_broadcast(self):
        """``isinstance(1, float)`` is False, so ``len(1)`` is reached instead.

        A YAML ``sigma_data: 1`` parses as an int and dies here with a TypeError
        rather than the ValueError the code clearly intends.
        """
        with pytest.raises(TypeError):
            Denoiser(ConstantInner(), sigma_data=1, device="cpu")

    def test_defaults(self):
        denoiser = Denoiser(ConstantInner(), device="cpu")

        assert denoiser.distillation is False
        assert denoiser.sigma_min == 0.002
        assert denoiser.diffusion_loss == "l2"

    def test_the_inner_model_is_registered_as_a_submodule(self):
        inner = torch.nn.Linear(4, 4)

        denoiser = Denoiser(inner, device="cpu")

        assert denoiser.inner_model is inner
        assert dict(denoiser.named_modules())["inner_model"] is inner

    def test_sigma_data_is_a_plain_attribute_not_a_buffer(self):
        """It is assigned directly, so it is not checkpointed and ``.to()``
        will not move or recast it along with the rest of the module."""
        denoiser = Denoiser(torch.nn.Linear(4, 4), sigma_data=0.5, device="cpu")

        assert "sigma_data" not in dict(denoiser.named_buffers())
        assert "sigma_data" not in denoiser.state_dict()

        denoiser.to(dtype=torch.float64)
        assert denoiser.sigma_data.dtype == torch.float32
        assert denoiser.inner_model.weight.dtype == torch.float64


# --------------------------------------------------------------------------- #
# Denoiser -- preconditioning scalings
# --------------------------------------------------------------------------- #
class TestDenoiserScalings:
    @pytest.fixture
    def denoiser(self):
        return Denoiser(ConstantInner(), sigma_data=[0.5, 1.0, 2.0, 4.0], device="cpu")

    def test_shapes(self, denoiser):
        c_skip, c_out, c_in = denoiser.get_scalings(torch.rand(3) + 0.1)

        for scaling in (c_skip, c_out, c_in):
            assert scaling.shape == (3, 4)

    def test_matches_the_karras_formulas(self, denoiser):
        sigma = torch.rand(3) + 0.1

        c_skip, c_out, c_in = denoiser.get_scalings(sigma)

        want_skip, want_out, want_in = karras_scalings(sigma, denoiser.sigma_data)
        torch.testing.assert_close(c_skip, want_skip.squeeze(1))
        torch.testing.assert_close(c_out, want_out.squeeze(1))
        torch.testing.assert_close(c_in, want_in.squeeze(1))

    def test_each_feature_gets_its_own_sigma_data(self, denoiser):
        c_skip, _, _ = denoiser.get_scalings(torch.ones(1))

        # c_skip = d^2 / (1 + d^2) for d in 0.5, 1, 2, 4
        expected = torch.tensor([[0.2, 0.5, 0.8, 16 / 17]])
        torch.testing.assert_close(c_skip, expected)

    def test_a_uniform_sigma_data_gives_uniform_scalings(self):
        denoiser = Denoiser(ConstantInner(), sigma_data=0.5, device="cpu")

        c_skip, c_out, c_in = denoiser.get_scalings(torch.rand(2) + 0.1)

        for scaling in (c_skip, c_out, c_in):
            assert (scaling == scaling[:, :1]).all()

    def test_at_tiny_sigma_the_skip_connection_dominates(self, denoiser):
        c_skip, c_out, c_in = denoiser.get_scalings(torch.full((1,), 1e-8))

        torch.testing.assert_close(c_skip, torch.ones(1, 4))
        torch.testing.assert_close(c_out, torch.zeros(1, 4))
        torch.testing.assert_close(c_in, 1 / denoiser.sigma_data.reshape(1, 4))

    def test_at_large_sigma_the_skip_connection_vanishes(self, denoiser):
        c_skip, _, c_in = denoiser.get_scalings(torch.full((1,), 1e6))

        torch.testing.assert_close(c_skip, torch.zeros(1, 4), atol=1e-9, rtol=0)
        torch.testing.assert_close(c_in, torch.full((1, 4), 1e-6), rtol=1e-4, atol=0)

    def test_the_scalings_keep_the_network_input_at_unit_variance(self, denoiser):
        """That is the point of c_in: (sigma^2 + sigma_data^2) * c_in^2 == 1."""
        sigma = torch.rand(5) + 0.1

        _, _, c_in = denoiser.get_scalings(sigma)

        variance = sigma.unsqueeze(1) ** 2 + denoiser.sigma_data**2
        torch.testing.assert_close(variance * c_in**2, torch.ones(5, 4))

    def test_an_already_unsqueezed_sigma_is_accepted(self, denoiser):
        """``forward`` passes sigma as (B, 1) while ``loss`` passes it as (B,)."""
        sigma = torch.rand(3) + 0.1

        flat = denoiser.get_scalings(sigma)
        column = denoiser.get_scalings(sigma.unsqueeze(1))

        for one, other in zip(flat, column):
            torch.testing.assert_close(one, other)

    def test_boundary_condition_scalings_are_exact_at_sigma_min(self, denoiser):
        """c_out is 0 and c_skip is 1 there -- the consistency-model boundary."""
        sigma = torch.full((2,), denoiser.sigma_min)

        c_skip, c_out, _ = denoiser.get_scalings_for_boundary_condition(sigma)

        torch.testing.assert_close(c_skip, torch.ones(2, 4))
        torch.testing.assert_close(c_out, torch.zeros(2, 4))

    def test_boundary_condition_shapes(self, denoiser):
        scalings = denoiser.get_scalings_for_boundary_condition(torch.rand(3) + 0.1)

        for scaling in scalings:
            assert scaling.shape == (3, 4)

    def test_boundary_condition_shares_c_in_with_the_plain_scalings(self, denoiser):
        sigma = torch.rand(3) + 0.1

        _, _, plain = denoiser.get_scalings(sigma)
        _, _, boundary = denoiser.get_scalings_for_boundary_condition(sigma)

        torch.testing.assert_close(plain, boundary)

    def test_boundary_condition_differs_from_the_plain_scalings(self, denoiser):
        sigma = torch.rand(3) + 0.1

        plain_skip, plain_out, _ = denoiser.get_scalings(sigma)
        bound_skip, bound_out, _ = denoiser.get_scalings_for_boundary_condition(sigma)

        assert not torch.allclose(plain_skip, bound_skip)
        assert not torch.allclose(plain_out, bound_out)


# --------------------------------------------------------------------------- #
# Denoiser -- forward
# --------------------------------------------------------------------------- #
class TestDenoiserForward:
    def test_output_shape(self):
        denoiser = Denoiser(ConstantInner(), sigma_data=0.5, device="cpu")
        x = torch.randn(2, 6, 4)

        out = denoiser(x, torch.rand(2) + 0.1)

        assert out.shape == x.shape

    def test_a_zero_output_network_leaves_only_the_skip_branch(self):
        denoiser = Denoiser(
            ConstantInner(0.0), sigma_data=[0.5, 1.0, 2.0, 4.0], device="cpu"
        )
        x = torch.randn(2, 6, 4)
        sigma = torch.rand(2) + 0.1

        out = denoiser(x, sigma)

        c_skip, _, _ = karras_scalings(sigma, denoiser.sigma_data)
        torch.testing.assert_close(out, x * c_skip)

    def test_matches_the_full_preconditioning_formula(self):
        inner = ConstantInner(3.0)
        denoiser = Denoiser(inner, sigma_data=[0.5, 1.0, 2.0, 4.0], device="cpu")
        x = torch.randn(2, 6, 4)
        sigma = torch.rand(2) + 0.1

        out = denoiser(x, sigma)

        c_skip, c_out, _ = karras_scalings(sigma, denoiser.sigma_data)
        torch.testing.assert_close(out, 3.0 * c_out + x * c_skip)

    def test_the_network_sees_the_rescaled_input(self):
        inner = ConstantInner()
        denoiser = Denoiser(inner, sigma_data=[0.5, 1.0, 2.0, 4.0], device="cpu")
        x = torch.randn(2, 6, 4)
        sigma = torch.rand(2) + 0.1

        denoiser(x, sigma)

        _, _, c_in = karras_scalings(sigma, denoiser.sigma_data)
        torch.testing.assert_close(inner.calls[0]["x"], x * c_in)

    def test_keyword_arguments_are_forwarded_to_the_network(self):
        inner = ConstantInner()
        denoiser = Denoiser(inner, device="cpu")
        context = torch.randn(2, 4)

        denoiser(torch.randn(2, 6, 4), torch.rand(2) + 0.1, context=context)

        assert inner.calls[0]["kwargs"]["context"] is context

    @pytest.mark.parametrize("sigma", [1.0, 1, 0.5])
    def test_a_scalar_sigma_is_expanded_to_the_batch(self, sigma):
        inner = ConstantInner()
        denoiser = Denoiser(inner, device="cpu")

        denoiser(torch.randn(3, 6, 4), sigma)

        assert inner.calls[0]["sigma"].shape == (3, 1)
        assert (inner.calls[0]["sigma"] == float(sigma)).all()

    def test_a_scalar_and_a_tensor_sigma_agree(self):
        denoiser = Denoiser(ConstantInner(2.0), sigma_data=0.5, device="cpu")
        x = torch.randn(3, 6, 4)

        scalar = denoiser(x, 1.5)
        tensor = denoiser(x, torch.full((3,), 1.5))

        torch.testing.assert_close(scalar, tensor)

    def test_a_per_event_sigma_is_applied_per_event(self):
        denoiser = Denoiser(ConstantInner(0.0), sigma_data=0.5, device="cpu")
        x = torch.randn(2, 6, 4)

        mixed = denoiser(x, torch.tensor([0.1, 10.0]))

        torch.testing.assert_close(mixed[0], denoiser(x[:1], 0.1)[0])
        torch.testing.assert_close(mixed[1], denoiser(x[1:], 10.0)[0])

    def test_the_distillation_flag_switches_the_scalings(self):
        x = torch.randn(2, 6, 4)
        sigma = torch.rand(2) + 0.1
        plain = Denoiser(ConstantInner(1.0), sigma_data=0.5, device="cpu")
        distilled = Denoiser(
            ConstantInner(1.0), sigma_data=0.5, device="cpu", distillation=True
        )

        assert not torch.allclose(plain(x, sigma), distilled(x, sigma))

    def test_the_boundary_condition_is_the_identity_at_sigma_min(self):
        """The defining property of a consistency model at t = sigma_min."""
        denoiser = Denoiser(
            ConstantInner(7.0), sigma_data=0.5, device="cpu", distillation=True,
            sigma_min=0.002,
        )
        x = torch.randn(2, 6, 4)

        out = denoiser(x, torch.full((2,), 0.002))

        torch.testing.assert_close(out, x)

    def test_a_real_network_round_trips_through_the_denoiser(self):
        config = model_config()
        denoiser = Denoiser(
            PointwiseNet_kDiffusion(config), sigma_data=0.5, device="cpu"
        )
        x = torch.randn(2, 6, 4)

        out = denoiser(x, torch.rand(2) + 0.1, context=torch.randn(2, 4))

        assert out.shape == x.shape
        assert torch.isfinite(out).all()

    def test_does_not_modify_its_input(self):
        denoiser = Denoiser(ConstantInner(1.0), device="cpu")
        x = torch.randn(2, 6, 4)
        before = x.clone()

        denoiser(x, torch.rand(2) + 0.1)

        torch.testing.assert_close(x, before)


# --------------------------------------------------------------------------- #
# Denoiser -- loss
# --------------------------------------------------------------------------- #
class TestDenoiserLoss:
    @pytest.fixture
    def denoiser(self):
        return Denoiser(
            ConstantInner(0.0), sigma_data=[0.5, 1.0, 2.0, 4.0], device="cpu"
        )

    def test_returns_one_number_per_event(self, denoiser):
        loss = denoiser.loss(
            torch.randn(3, 6, 4), torch.randn(3, 6, 4), torch.rand(3) + 0.1
        )

        assert loss.shape == (3,)

    def test_matches_the_hand_written_target(self, denoiser):
        x = torch.randn(3, 6, 4)
        noise = torch.randn(3, 6, 4)
        sigma = torch.rand(3) + 0.1

        loss = denoiser.loss(x.clone(), noise, sigma)

        c_skip, c_out, _ = karras_scalings(sigma, denoiser.sigma_data)
        noised = x + noise * sigma.reshape(-1, 1, 1)
        target = (x - c_skip * noised) / c_out
        torch.testing.assert_close(loss, target.pow(2).flatten(1).mean(1))

    def test_the_network_sees_the_rescaled_noised_input(self, denoiser):
        x = torch.randn(3, 6, 4)
        noise = torch.randn(3, 6, 4)
        sigma = torch.rand(3) + 0.1

        denoiser.loss(x.clone(), noise, sigma)

        _, _, c_in = karras_scalings(sigma, denoiser.sigma_data)
        noised = x + noise * sigma.reshape(-1, 1, 1)
        torch.testing.assert_close(denoiser.inner_model.calls[0]["x"], noised * c_in)

    def test_zero_noise_and_a_perfect_network_cost_nothing(self):
        """With no noise the target is x * (1 - c_skip) / c_out; feed that back."""
        sigma_data = torch.full((4,), 0.5)
        x = torch.randn(2, 6, 4)
        sigma = torch.rand(2) + 0.1
        c_skip, c_out, _ = karras_scalings(sigma, sigma_data)
        target = (x - c_skip * x) / c_out

        class Perfect(torch.nn.Module):
            def forward(self, inputs, sigma_, **kwargs):
                return target

        denoiser = Denoiser(Perfect(), sigma_data=0.5, device="cpu")

        loss = denoiser.loss(x.clone(), torch.zeros_like(x), sigma)

        torch.testing.assert_close(loss, torch.zeros(2), atol=1e-6, rtol=0)

    def test_l1_and_l2_differ(self):
        x = torch.randn(3, 6, 4)
        noise = torch.randn(3, 6, 4)
        sigma = torch.rand(3) + 0.1
        l2 = Denoiser(ConstantInner(), sigma_data=0.5, device="cpu", diffusion_loss="l2")
        l1 = Denoiser(ConstantInner(), sigma_data=0.5, device="cpu", diffusion_loss="l1")

        assert not torch.allclose(
            l2.loss(x.clone(), noise, sigma), l1.loss(x.clone(), noise, sigma)
        )

    def test_l1_is_the_mean_absolute_error(self):
        denoiser = Denoiser(
            ConstantInner(0.0), sigma_data=0.5, device="cpu", diffusion_loss="l1"
        )
        x = torch.randn(3, 6, 4)
        noise = torch.randn(3, 6, 4)
        sigma = torch.rand(3) + 0.1

        loss = denoiser.loss(x.clone(), noise, sigma)

        c_skip, c_out, _ = karras_scalings(sigma, denoiser.sigma_data)
        noised = x + noise * sigma.reshape(-1, 1, 1)
        target = (x - c_skip * noised) / c_out
        torch.testing.assert_close(loss, target.abs().flatten(1).mean(1))

    def test_an_unknown_loss_name_is_rejected(self):
        denoiser = Denoiser(
            ConstantInner(), sigma_data=0.5, device="cpu", diffusion_loss="huber"
        )

        with pytest.raises(ValueError, match="l1 or l2"):
            denoiser.loss(torch.randn(2, 6, 4), torch.randn(2, 6, 4), torch.rand(2) + 0.1)

    def test_is_non_negative(self, denoiser):
        loss = denoiser.loss(
            torch.randn(3, 6, 4), torch.randn(3, 6, 4), torch.rand(3) + 0.1
        )

        assert (loss >= 0).all()

    def test_keyword_arguments_are_forwarded(self, denoiser):
        context = torch.randn(3, 4)

        denoiser.loss(
            torch.randn(3, 6, 4), torch.randn(3, 6, 4), torch.rand(3) + 0.1,
            context=context,
        )

        assert denoiser.inner_model.calls[0]["kwargs"]["context"] is context

    def test_input_mask_is_not_forwarded_to_the_network(self, denoiser):
        denoiser.loss(
            torch.randn(3, 6, 4),
            torch.randn(3, 6, 4),
            torch.rand(3) + 0.1,
            input_mask=torch.ones(3, 6, dtype=torch.bool),
        )

        assert "input_mask" not in denoiser.inner_model.calls[0]["kwargs"]

    def test_an_all_true_mask_changes_nothing(self, denoiser):
        x = torch.randn(3, 6, 4)
        noise = torch.randn(3, 6, 4)
        sigma = torch.rand(3) + 0.1

        with_mask = denoiser.loss(
            x.clone(), noise, sigma, input_mask=torch.ones(3, 6, dtype=torch.bool)
        )
        without = denoiser.loss(x.clone(), noise, sigma, input_mask=None)

        torch.testing.assert_close(with_mask, without)

    def test_masked_rows_are_overwritten_in_place(self, denoiser):
        """``input[i][~event] = ...`` mutates the caller's tensor."""
        x = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
        before = x.clone()
        mask = torch.tensor([[True, True, True, True, False, False]])

        denoiser.loss(x, torch.zeros(1, 6, 4), torch.rand(1) + 0.1, input_mask=mask)

        assert not torch.equal(x, before)
        torch.testing.assert_close(x[:, :4], before[:, :4])  # kept rows untouched

    def test_masked_rows_are_replaced_by_copies_of_kept_rows(self, denoiser):
        x = torch.arange(24, dtype=torch.float32).reshape(1, 6, 4)
        kept = x[0, :4].clone()
        mask = torch.tensor([[True, True, True, True, False, False]])

        denoiser.loss(x, torch.zeros(1, 6, 4), torch.rand(1) + 0.1, input_mask=mask)

        for row in x[0, 4:]:
            assert any(torch.equal(row, candidate) for candidate in kept)

    def test_only_events_that_need_it_are_touched(self, denoiser):
        x = torch.randn(2, 6, 4)
        before = x.clone()
        mask = torch.ones(2, 6, dtype=torch.bool)
        mask[1, -1] = False

        denoiser.loss(x, torch.zeros(2, 6, 4), torch.rand(2) + 0.1, input_mask=mask)

        torch.testing.assert_close(x[0], before[0])

    def test_a_fully_masked_event_is_an_error(self, denoiser):
        """No rows left to sample a replacement from -- randint(0, 0) explodes."""
        mask = torch.ones(2, 6, dtype=torch.bool)
        mask[0] = False

        with pytest.raises(RuntimeError):
            denoiser.loss(
                torch.randn(2, 6, 4), torch.randn(2, 6, 4), torch.rand(2) + 0.1,
                input_mask=mask,
            )

    def test_the_mask_must_be_one_value_per_point(self, denoiser):
        """The docstring says "same shape as input", but only (B, N) works."""
        mask = torch.ones(2, 6, 4, dtype=torch.bool)
        mask[0, 0, 0] = False

        with pytest.raises((RuntimeError, TypeError, ValueError)):
            denoiser.loss(
                torch.randn(2, 6, 4), torch.randn(2, 6, 4), torch.rand(2) + 0.1,
                input_mask=mask,
            )

    def test_an_input_that_requires_grad_cannot_be_masked(self, denoiser):
        """The in-place fill is illegal on a leaf that requires grad."""
        x = torch.randn(1, 6, 4, requires_grad=True)
        mask = torch.tensor([[True, True, True, True, True, False]])

        with pytest.raises(RuntimeError):
            denoiser.loss(x, torch.zeros(1, 6, 4), torch.rand(1) + 0.1, input_mask=mask)

    def test_gradients_flow_back_into_the_network(self):
        config = model_config()
        net = PointwiseNet_kDiffusion(config)
        denoiser = Denoiser(net, sigma_data=0.5, device="cpu")

        loss = denoiser.loss(
            torch.randn(2, 6, 4),
            torch.randn(2, 6, 4),
            torch.rand(2) + 0.1,
            context=torch.randn(2, 4),
        )
        loss.mean().backward()

        for name, parameter in net.named_parameters():
            assert parameter.grad is not None, name

    def test_each_event_is_scored_independently(self, denoiser):
        x = torch.randn(3, 6, 4)
        noise = torch.randn(3, 6, 4)
        sigma = torch.rand(3) + 0.1

        together = denoiser.loss(x.clone(), noise, sigma)
        alone = denoiser.loss(x[1:2].clone(), noise[1:2], sigma[1:2])

        torch.testing.assert_close(together[1:2], alone)


# --------------------------------------------------------------------------- #
# Denoiser -- consistency loss
# --------------------------------------------------------------------------- #
class TestDenoiserConsistencyLoss:
    @pytest.fixture
    def config(self):
        # num_steps=2 makes ``indices`` degenerate to 0, so t == sigma_max and
        # t2 == sigma_min for every event and the test is deterministic.
        return model_config(num_steps=2)

    @pytest.fixture
    def trio(self, config):
        def build():
            return Denoiser(
                PointwiseNet_kDiffusion(config),
                sigma_data=config["training"]["sigma_data"],
                device="cpu",
                distillation=True,
                sigma_min=config["model"]["sigma_min"],
            )

        return build(), build(), build()

    def test_returns_one_number_per_event(self, trio, config):
        student, teacher, target = trio

        loss = student.consistency_loss(
            torch.randn(3, 6, 4), teacher, target, config, context=torch.randn(3, 4)
        )

        assert loss.shape == (3,)
        assert torch.isfinite(loss).all()

    def test_is_non_negative(self, trio, config):
        student, teacher, target = trio

        loss = student.consistency_loss(
            torch.randn(3, 6, 4), teacher, target, config, context=torch.randn(3, 4)
        )

        assert (loss >= 0).all()

    def test_a_missing_teacher_falls_back_to_the_euler_solver(self, trio, config):
        student, _, target = trio

        loss = student.consistency_loss(
            torch.randn(3, 6, 4), None, target, config, context=torch.randn(3, 4)
        )

        assert loss.shape == (3,)
        assert torch.isfinite(loss).all()

    def test_is_reproducible_under_a_fixed_seed(self, trio, config):
        student, teacher, target = trio
        x = torch.randn(3, 6, 4)
        context = torch.randn(3, 4)

        torch.manual_seed(7)
        first = student.consistency_loss(
            x.clone(), teacher, target, config, context=context
        )
        torch.manual_seed(7)
        second = student.consistency_loss(
            x.clone(), teacher, target, config, context=context
        )

        torch.testing.assert_close(first, second)

    def test_gradients_reach_the_student_but_not_the_target(self, trio, config):
        student, teacher, target = trio

        student.consistency_loss(
            torch.randn(2, 6, 4), teacher, target, config, context=torch.randn(2, 4)
        ).mean().backward()

        assert any(p.grad is not None for p in student.inner_model.parameters())
        assert all(p.grad is None for p in target.inner_model.parameters())

    def test_an_identical_teacher_and_target_still_costs_something(self, trio, config):
        """The two branches are evaluated at different sigmas, so it is not 0."""
        student, teacher, _ = trio

        loss = student.consistency_loss(
            torch.randn(3, 6, 4), teacher, student, config, context=torch.randn(3, 4)
        )

        assert (loss > 0).any()

    def test_does_not_modify_its_input(self, trio, config):
        student, teacher, target = trio
        x = torch.randn(3, 6, 4)
        before = x.clone()

        student.consistency_loss(x, teacher, target, config, context=torch.randn(3, 4))

        torch.testing.assert_close(x, before)


# --------------------------------------------------------------------------- #
# Diffusion -- construction
# --------------------------------------------------------------------------- #
class TestDiffusionConstruction:
    def test_wires_a_pointwise_net_into_a_denoiser(self):
        model = Diffusion(model_config())

        assert isinstance(model.diffusion, Denoiser)
        assert isinstance(model.diffusion.inner_model, PointwiseNet_kDiffusion)

    def test_keeps_the_config_and_the_flag(self):
        config = model_config()

        model = Diffusion(config, distillation=False)

        assert model.config is config
        assert model.distillation is False

    def test_the_default_is_not_distilled(self):
        assert Diffusion(model_config()).distillation is False
        assert Diffusion(model_config()).diffusion.distillation is False

    def test_the_distillation_flag_reaches_the_denoiser(self):
        model = Diffusion(model_config(), distillation=True)

        assert model.distillation is True
        assert model.diffusion.distillation is True

    def test_sigma_data_comes_from_the_training_section(self):
        model = Diffusion(model_config(training={"sigma_data": [1.0, 2.0, 3.0, 4.0]}))

        torch.testing.assert_close(
            model.diffusion.sigma_data, torch.tensor([1.0, 2.0, 3.0, 4.0])
        )

    def test_the_diffusion_loss_comes_from_the_training_section(self):
        model = Diffusion(model_config(training={"diffusion_loss": "l1"}))

        assert model.diffusion.diffusion_loss == "l1"

    def test_distillation_ignores_the_configured_diffusion_loss(self):
        """The distilled branch never passes ``diffusion_loss`` on, so an ``l1``
        config silently trains the student against the ``l2`` default."""
        model = Diffusion(model_config(training={"diffusion_loss": "l1"}),
                          distillation=True)

        assert model.diffusion.diffusion_loss == "l2"

    def test_sigma_min_only_reaches_the_denoiser_when_distilling(self):
        config = model_config(model={"sigma_min": 0.25})

        plain = Diffusion(config, distillation=False)
        distilled = Diffusion(config, distillation=True)

        assert plain.diffusion.sigma_min == 0.002  # the Denoiser default
        assert distilled.diffusion.sigma_min == 0.25

    def test_carries_an_unused_kld_loss(self):
        model = Diffusion(model_config())

        assert isinstance(model.kld, KLDloss)

    def test_the_state_dict_is_reloadable(self):
        config = model_config()
        first, second = Diffusion(config), Diffusion(config)

        second.load_state_dict(first.state_dict())

        x = torch.randn(2, 6, 4)
        sigma = torch.rand(2) + 0.1
        context = torch.randn(2, 4)
        torch.testing.assert_close(
            first.diffusion(x, sigma, context=context),
            second.diffusion(x, sigma, context=context),
        )

    def test_parameters_are_registered(self):
        model = Diffusion(model_config())

        names = [name for name, _ in model.named_parameters()]
        assert any(name.startswith("diffusion.inner_model.layers") for name in names)


# --------------------------------------------------------------------------- #
# Diffusion -- get_loss
# --------------------------------------------------------------------------- #
def positive_points(batch=3, points=6):
    """Points whose 4th feature (energy) is safely above the padding cut."""
    x = torch.randn(batch, points, 4)
    x[..., 3] = torch.rand(batch, points) + 0.5
    return x


class TestDiffusionGetLoss:
    @pytest.fixture
    def model(self):
        return Diffusion(model_config())

    def test_returns_a_scalar(self, model):
        loss = model.get_loss(
            positive_points(), torch.randn(3, 6, 4), torch.rand(3) + 0.1,
            torch.randn(3, 4),
        )

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_is_the_mean_of_the_per_event_denoiser_loss(self, model, mocker):
        per_event = torch.tensor([1.0, 2.0, 6.0])
        mocker.patch.object(model.diffusion, "loss", return_value=per_event)

        loss = model.get_loss(
            positive_points(), torch.randn(3, 6, 4), torch.rand(3) + 0.1,
            torch.randn(3, 4),
        )

        torch.testing.assert_close(loss, torch.tensor(3.0))

    def test_conditioning_is_passed_through_as_context(self, model, mocker):
        spy = mocker.patch.object(
            model.diffusion, "loss", return_value=torch.zeros(3)
        )
        cond = torch.randn(3, 4)

        model.get_loss(positive_points(), torch.randn(3, 6, 4), torch.rand(3) + 0.1, cond)

        assert spy.call_args.kwargs["context"] is cond

    def test_log_energies_are_masked_on_finiteness(self, mocker):
        """``logarithmic_point_energy`` true -> padding shows up as -inf."""
        model = Diffusion(model_config(model={"logarithmic_point_energy": True}))
        spy = mocker.patch.object(model.diffusion, "loss", return_value=torch.zeros(1))
        x = positive_points(batch=1)
        x[0, -1, 3] = float("-inf")

        model.get_loss(x, torch.randn(1, 6, 4), torch.rand(1) + 0.1, torch.randn(1, 4))

        mask = spy.call_args.kwargs["input_mask"]
        assert mask.shape == (1, 6)
        assert mask[0, -1].item() is False
        assert mask[0, :-1].all()

    def test_linear_energies_are_masked_on_a_threshold(self, mocker):
        """``logarithmic_point_energy`` false -> padding is a near-zero energy."""
        model = Diffusion(model_config(model={"logarithmic_point_energy": False}))
        spy = mocker.patch.object(model.diffusion, "loss", return_value=torch.zeros(1))
        x = positive_points(batch=1)
        x[0, -1, 3] = 0.0
        x[0, -2, 3] = 1e-9

        model.get_loss(x, torch.randn(1, 6, 4), torch.rand(1) + 0.1, torch.randn(1, 4))

        mask = spy.call_args.kwargs["input_mask"]
        assert not mask[0, -1] and not mask[0, -2]
        assert mask[0, :-2].all()

    def test_only_the_energy_column_decides_the_mask(self, mocker):
        model = Diffusion(model_config(model={"logarithmic_point_energy": False}))
        spy = mocker.patch.object(model.diffusion, "loss", return_value=torch.zeros(1))
        x = positive_points(batch=1)
        x[0, 2, :3] = 0.0  # a point at the origin is still a real point

        model.get_loss(x, torch.randn(1, 6, 4), torch.rand(1) + 0.1, torch.randn(1, 4))

        assert spy.call_args.kwargs["input_mask"].all()

    def test_padding_is_overwritten_in_the_caller_s_tensor(self):
        """``get_loss`` mutates ``x`` -- callers that hold a view lose their padding.

        ``training.teacher.ValidationChecker.loss`` passes ``self.target[start:end]``,
        which is a view, so the stored validation set is rewritten the first time
        the loss is evaluated.
        """
        model = Diffusion(model_config(model={"logarithmic_point_energy": False}))
        stored = positive_points(batch=4)
        stored[1, -1, 3] = 0.0
        before = stored.clone()

        model.get_loss(
            stored[:2], torch.randn(2, 6, 4), torch.rand(2) + 0.1, torch.randn(2, 4)
        )

        assert not torch.equal(stored, before)
        assert stored[1, -1, 3] > 1e-5  # the padding row is now a real point
        torch.testing.assert_close(stored[2:], before[2:])  # untouched events

    def test_a_batch_with_no_padding_is_left_alone(self, model):
        x = positive_points()
        before = x.clone()

        model.get_loss(x, torch.randn(3, 6, 4), torch.rand(3) + 0.1, torch.randn(3, 4))

        torch.testing.assert_close(x, before)

    def test_gradients_reach_the_network(self, model):
        loss = model.get_loss(
            positive_points(), torch.randn(3, 6, 4), torch.rand(3) + 0.1,
            torch.randn(3, 4),
        )

        loss.backward()

        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert grads
        assert all(torch.isfinite(g).all() for g in grads)

    def test_the_unused_fourier_buffer_gets_no_gradient(self, model):
        model.get_loss(
            positive_points(), torch.randn(3, 6, 4), torch.rand(3) + 0.1,
            torch.randn(3, 4),
        ).backward()

        buffer = model.diffusion.inner_model.timestep_embed[0].weight
        assert buffer.grad is None


# --------------------------------------------------------------------------- #
# Diffusion -- sample
# --------------------------------------------------------------------------- #
SAMPLERS = [
    "euler",
    "heun",
    "dpmpp_2m",
    "dpmpp_2s_ancestral",
    "sample_euler_ancestral",
    "sample_lms",
    "sample_dpmpp_2m_sde",
]


class TestDiffusionSample:
    @pytest.mark.parametrize("sampler", SAMPLERS)
    def test_every_named_sampler_produces_the_right_shape(self, sampler):
        model = Diffusion(model_config(sampler=sampler))

        out = model.sample(torch.randn(2, 4), num_points=5)

        assert out.shape == (2, 5, 4)
        assert torch.isfinite(out).all()

    def test_an_unknown_sampler_is_rejected(self):
        model = Diffusion(model_config(sampler="ddim"))

        with pytest.raises(NotImplementedError, match="ddim"):
            model.sample(torch.randn(2, 4), num_points=5)

    def test_the_sampler_is_only_looked_up_when_sampling(self):
        """Construction does not validate ``config["sampler"]``."""
        Diffusion(model_config(sampler="ddim"))

    def test_an_empty_batch_short_circuits(self):
        model = Diffusion(model_config())

        out = model.sample(torch.randn(0, 4), num_points=5)

        assert out.shape == (0, 5, 4)

    def test_an_empty_batch_never_reaches_the_sampler(self, mocker):
        model = Diffusion(model_config())
        spy = mocker.patch("k_diffusion.sampling.sample_euler")

        model.sample(torch.randn(0, 4), num_points=5)

        spy.assert_not_called()

    def test_a_feature_dim_other_than_four_builds_but_cannot_run(self):
        """``feature_dim`` is configurable, but ``sigma_data`` is fixed at four.

        ``Denoiser.__init__`` insists on exactly four ``sigma_data`` entries, so
        the preconditioning scalings are (B, 1, 4) whatever ``feature_dim`` says.
        The mismatch only surfaces once something actually runs through the
        model, which makes a mis-set ``feature_dim`` look fine at start-up.
        """
        model = Diffusion(model_config(model={"feature_dim": 6}))

        with pytest.raises(RuntimeError, match="must match"):
            model.sample(torch.randn(2, 4), num_points=3)

    def test_the_conditioning_is_handed_to_the_sampler(self, mocker):
        model = Diffusion(model_config(sampler="euler"))
        spy = mocker.patch(
            "k_diffusion.sampling.sample_euler", return_value=torch.zeros(2, 5, 4)
        )
        cond = torch.randn(2, 4)

        model.sample(cond, num_points=5)

        assert spy.call_args.kwargs["extra_args"] == {"context": cond}
        assert spy.call_args.kwargs["disable"] is True
        assert spy.call_args.args[0] is model.diffusion

    def test_heun_also_forwards_the_churn_settings(self, mocker):
        model = Diffusion(
            model_config(sampler="heun", model={"s_churn": 0.3, "s_noise": 1.2})
        )
        spy = mocker.patch(
            "k_diffusion.sampling.sample_heun", return_value=torch.zeros(2, 5, 4)
        )

        model.sample(torch.randn(2, 4), num_points=5)

        assert spy.call_args.kwargs["s_churn"] == 0.3
        assert spy.call_args.kwargs["s_noise"] == 1.2

    def test_other_samplers_do_not_get_churn_settings(self, mocker):
        model = Diffusion(model_config(sampler="euler"))
        spy = mocker.patch(
            "k_diffusion.sampling.sample_euler", return_value=torch.zeros(2, 5, 4)
        )

        model.sample(torch.randn(2, 4), num_points=5)

        assert "s_churn" not in spy.call_args.kwargs

    def test_the_karras_schedule_is_built_from_the_config(self, mocker):
        config = model_config(
            sampler="euler", num_steps=7, model={"sigma_min": 0.01, "sigma_max": 50.0,
                                                 "rho": 5.0}
        )
        model = Diffusion(config)
        spy = mocker.spy(k_diffusion.sampling, "get_sigmas_karras")
        mocker.patch(
            "k_diffusion.sampling.sample_euler", return_value=torch.zeros(2, 5, 4)
        )

        model.sample(torch.randn(2, 4), num_points=5)

        assert spy.call_args.args[:3] == (7, 0.01, 50.0)
        assert spy.call_args.kwargs["rho"] == 5.0

    def test_the_starting_noise_is_scaled_by_sigma_max(self, mocker):
        model = Diffusion(model_config(sampler="euler", model={"sigma_max": 80.0}))
        captured = {}

        def capture(inner, x, sigmas, **kwargs):
            captured["x"] = x
            return torch.zeros_like(x)

        mocker.patch("k_diffusion.sampling.sample_euler", side_effect=capture)

        model.sample(torch.randn(64, 4), num_points=32)

        # x_T ~ N(0, sigma_max^2)
        assert 40.0 < captured["x"].std().item() < 130.0

    def test_the_distilled_path_takes_a_single_step(self, mocker):
        model = Diffusion(model_config(), distillation=True)
        spy = mocker.patch("k_diffusion.sampling.sample_euler")

        out = model.sample(torch.randn(2, 4), num_points=5)

        spy.assert_not_called()
        assert out.shape == (2, 5, 4)

    def test_the_distilled_path_denoises_from_sigma_max(self, mocker):
        model = Diffusion(model_config(model={"sigma_max": 80.0}), distillation=True)
        spy = mocker.spy(model.diffusion, "forward")

        model.sample(torch.randn(2, 4), num_points=5)

        assert spy.call_args.args[1] == 80.0

    def test_the_output_follows_the_conditioning_device(self):
        model = Diffusion(model_config())

        out = model.sample(torch.randn(2, 4), num_points=5)

        assert out.device == torch.device("cpu")

    def test_a_one_dimensional_conditioning_tensor_is_rejected(self):
        """``batch_size, _ = cond_feats.size()`` needs exactly two axes."""
        model = Diffusion(model_config())

        with pytest.raises(ValueError):
            model.sample(torch.randn(4), num_points=5)

    def test_a_single_point_can_be_sampled(self):
        model = Diffusion(model_config())

        assert model.sample(torch.randn(2, 4), num_points=1).shape == (2, 1, 4)

    def test_sampling_is_reproducible_under_a_fixed_seed(self):
        model = Diffusion(model_config())
        cond = torch.randn(2, 4)

        torch.manual_seed(99)
        first = model.sample(cond, num_points=5)
        torch.manual_seed(99)
        second = model.sample(cond, num_points=5)

        torch.testing.assert_close(first, second)

    def test_sampling_does_not_need_gradients(self):
        model = Diffusion(model_config())

        with torch.no_grad():
            out = model.sample(torch.randn(2, 4), num_points=5)

        assert not out.requires_grad


# --------------------------------------------------------------------------- #
# Diffusion -- get_cd_loss
# --------------------------------------------------------------------------- #
class TestDiffusionGetCdLoss:
    @pytest.fixture
    def config(self):
        return model_config(num_steps=2)

    def test_returns_a_scalar(self, config):
        student = Diffusion(config, distillation=True)
        teacher = Diffusion(config, distillation=True)
        target = Diffusion(config, distillation=True)

        loss = student.get_cd_loss(
            torch.randn(3, 6, 4), torch.randn(3, 4), teacher, target, config
        )

        assert loss.shape == ()
        assert torch.isfinite(loss)

    def test_unwraps_the_inner_denoisers(self, config, mocker):
        student = Diffusion(config, distillation=True)
        teacher = Diffusion(config, distillation=True)
        target = Diffusion(config, distillation=True)
        spy = mocker.patch.object(
            student.diffusion, "consistency_loss", return_value=torch.zeros(3)
        )
        cond = torch.randn(3, 4)

        student.get_cd_loss(torch.randn(3, 6, 4), cond, teacher, target, config)

        assert spy.call_args.args[1] is teacher.diffusion
        assert spy.call_args.args[2] is target.diffusion
        assert spy.call_args.kwargs["context"] is cond

    def test_is_the_mean_over_the_batch(self, config, mocker):
        student = Diffusion(config, distillation=True)
        mocker.patch.object(
            student.diffusion,
            "consistency_loss",
            return_value=torch.tensor([1.0, 2.0, 6.0]),
        )

        loss = student.get_cd_loss(
            torch.randn(3, 6, 4), torch.randn(3, 4), student, student, config
        )

        torch.testing.assert_close(loss, torch.tensor(3.0))

    def test_gradients_reach_the_student(self, config):
        student = Diffusion(config, distillation=True)
        teacher = Diffusion(config, distillation=True)
        target = Diffusion(config, distillation=True)

        student.get_cd_loss(
            torch.randn(2, 6, 4), torch.randn(2, 4), teacher, target, config
        ).backward()

        assert any(p.grad is not None for p in student.parameters())
