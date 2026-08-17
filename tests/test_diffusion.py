"""Tests for ``src/diffusion.py``.

The whole file skips when ``k_diffusion`` is missing: the module imports it at
the top level and builds its Fourier time embedding and samplers out of it, so
it cannot be stubbed.  Everything runs on the CPU with networks shrunk through
the ``diffusion_pointwise_hidden_l*`` config keys, so no test needs more than
a few milliseconds.

The stochastic parts (the padding resample in ``Denoiser.loss``, ``x_T`` in
``sample``) are pinned by seeding ``torch.manual_seed`` and replaying, rather
than by statistical assertions.

Tests carrying a docstring record current behaviour that looks unintended.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
k_diffusion = pytest.importorskip("k_diffusion")

from src import diffusion  # noqa: E402

from conftest import base_config  # noqa: E402


# config string -> attribute of k_diffusion.sampling it dispatches to
SAMPLER_NAMES = {
    "euler": "sample_euler",
    "heun": "sample_heun",
    "dpmpp_2m": "sample_dpmpp_2m",
    "dpmpp_2s_ancestral": "sample_dpmpp_2s_ancestral",
    "sample_euler_ancestral": "sample_euler_ancestral",
    "sample_lms": "sample_lms",
    "sample_dpmpp_2m_sde": "sample_dpmpp_2m_sde",
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def diffusion_config():
    """``base_config`` plus the diffusion entries of ``config/default.yaml``.

    ``diffusion_pointwise_hidden_l1`` is set to 8, which chains the whole net
    down to a few thousand parameters.
    """
    config = base_config("unused")
    config["sampler"] = "heun"
    config["num_steps"] = 4
    config["model"].update(
        {
            "logarithmic_point_energy": True,
            "sigma_min": 0.002,
            "sigma_max": 80.0,
            "rho": 7.0,
            "s_churn": 0.0,
            "s_noise": 1.0,
            "diffusion_pointwise_hidden_l1": 8,
        }
    )
    config["training"].update({"sigma_data": 0.5, "diffusion_loss": "l2"})
    return config


class RecordingInner(torch.nn.Module):
    """Stands in for the pointwise net: returns a constant, records its calls."""

    def __init__(self, value=0.0):
        super().__init__()
        self.value = value
        self.calls = []

    def forward(self, x, sigma, **kwargs):
        self.calls.append({"x": x, "sigma": sigma, **kwargs})
        return torch.full_like(x, self.value)


def zero_denoiser(sigma_data=(0.5, 0.5, 0.5, 0.5), **kwargs):
    """A Denoiser around an inner model that always returns zeros."""
    return diffusion.Denoiser(
        RecordingInner(0.0), sigma_data=list(sigma_data), device="cpu", **kwargs
    )


def event_batch(batch_size=2, n_points=5, n_features=4, seed=0):
    """Points with strictly positive energies, so nothing counts as padding."""
    rng = torch.Generator().manual_seed(seed)
    x = torch.randn(batch_size, n_points, n_features, generator=rng)
    x[..., 3] = x[..., 3].abs() + 0.1
    return x


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture
def config():
    return diffusion_config()


@pytest.fixture
def model(config):
    torch.manual_seed(0)
    return diffusion.Diffusion(config)


@pytest.fixture
def distilled(config):
    torch.manual_seed(0)
    return diffusion.Diffusion(config, distillation=True)


# --------------------------------------------------------------------------- #
# mean_flat
# --------------------------------------------------------------------------- #
class TestMeanFlat:
    def test_reduces_everything_but_the_batch_axis(self):
        tensor = torch.arange(24.0).reshape(2, 3, 4)

        result = diffusion.mean_flat(tensor)

        torch.testing.assert_close(result, tensor.reshape(2, -1).mean(1))

    def test_a_2d_tensor_gives_row_means(self):
        tensor = torch.tensor([[1.0, 3.0], [5.0, 7.0]])

        torch.testing.assert_close(
            diffusion.mean_flat(tensor), torch.tensor([2.0, 6.0])
        )

    def test_the_batch_axis_length_survives(self):
        assert diffusion.mean_flat(torch.zeros(7, 2, 2, 2)).shape == (7,)

    def test_a_1d_tensor_collapses_to_a_scalar(self):
        """An empty ``dim`` list means every dim to torch, so the batch axis of
        a 1-D tensor is reduced too, instead of the tensor passing through."""
        result = diffusion.mean_flat(torch.tensor([1.0, 2.0, 3.0]))

        assert result.ndim == 0
        assert result.item() == 2.0


# --------------------------------------------------------------------------- #
# KLDloss
# --------------------------------------------------------------------------- #
class TestKLDloss:
    def test_a_standard_normal_costs_nothing(self):
        loss = diffusion.KLDloss()(torch.zeros(3, 2), torch.zeros(3, 2))

        torch.testing.assert_close(loss, torch.tensor(0.0))

    def test_a_shifted_mean_matches_the_closed_form(self):
        # KL(N(1, 1) || N(0, 1)) = mu^2 / 2 = 0.5
        loss = diffusion.KLDloss()(torch.tensor([[1.0]]), torch.tensor([[0.0]]))

        torch.testing.assert_close(loss, torch.tensor(0.5))

    def test_a_widened_variance_matches_the_closed_form(self):
        # KL(N(0, 2) || N(0, 1)) = 0.5 * (2 - 1 - ln 2)
        logvar = torch.tensor([[np.log(2.0)]], dtype=torch.float32)

        loss = diffusion.KLDloss()(torch.zeros(1, 1), logvar)

        torch.testing.assert_close(loss, torch.tensor(0.5 * (1 - np.log(2.0))).float())

    def test_normalised_by_batch_size_not_by_elements(self):
        mu, logvar = torch.tensor([[1.0, 1.0]]), torch.zeros(1, 2)
        single = diffusion.KLDloss()(mu, logvar)

        doubled = diffusion.KLDloss()(mu.repeat(2, 1), logvar.repeat(2, 1))

        torch.testing.assert_close(doubled, single)
        torch.testing.assert_close(single, torch.tensor(1.0))  # 2 * mu^2/2

    def test_never_negative(self):
        rng = torch.Generator().manual_seed(0)
        for _ in range(20):
            mu = torch.randn(4, 3, generator=rng)
            logvar = torch.randn(4, 3, generator=rng)

            assert diffusion.KLDloss()(mu, logvar) >= 0

    def test_the_cuda_flag_mirrors_availability(self):
        assert diffusion.KLDloss().use_cuda is torch.cuda.is_available()


# --------------------------------------------------------------------------- #
# ConcatSquashLinear
# --------------------------------------------------------------------------- #
class TestConcatSquashLinear:
    @pytest.fixture
    def layer(self):
        """dim_in == dim_out == 2, ctx dim 3, all weights hand set to zero."""
        layer = diffusion.ConcatSquashLinear(2, 2, 3)
        with torch.no_grad():
            layer._layer.weight.copy_(torch.eye(2))
            layer._layer.bias.zero_()
            layer._hyper_gate.weight.zero_()
            layer._hyper_gate.bias.zero_()
            layer._hyper_bias.weight.zero_()
        return layer

    def test_a_saturated_gate_passes_the_linear_layer_through(self, layer):
        with torch.no_grad():
            layer._hyper_gate.bias.fill_(100.0)  # sigmoid -> 1
        x = torch.randn(1, 4, 2)

        out = layer(ctx=torch.zeros(1, 1, 3), x=x)

        torch.testing.assert_close(out, x)

    def test_a_closed_gate_leaves_only_the_hyper_bias(self, layer):
        with torch.no_grad():
            layer._hyper_gate.bias.fill_(-100.0)  # sigmoid -> 0
            layer._hyper_bias.weight.copy_(torch.ones(2, 3))
        x = torch.randn(1, 4, 2)

        out = layer(ctx=torch.ones(1, 1, 3), x=x)

        torch.testing.assert_close(out, torch.full_like(x, 3.0))

    def test_a_zero_gate_bias_halves_the_linear_output(self, layer):
        x = torch.randn(1, 4, 2)

        out = layer(ctx=torch.zeros(1, 1, 3), x=x)  # sigmoid(0) = 0.5

        torch.testing.assert_close(out, 0.5 * x)

    def test_the_context_broadcasts_across_points(self, layer):
        out = layer(ctx=torch.randn(3, 1, 3), x=torch.randn(3, 7, 2))

        assert out.shape == (3, 7, 2)

    def test_each_batch_row_sees_its_own_context(self, layer):
        with torch.no_grad():
            layer._hyper_bias.weight.copy_(torch.ones(2, 3))
        ctx = torch.stack(
            [torch.zeros(1, 3), torch.ones(1, 3)]
        )  # (2, 1, 3), rows differ
        x = torch.zeros(2, 4, 2)

        out = layer(ctx=ctx, x=x)

        torch.testing.assert_close(out[0], torch.zeros(4, 2))
        torch.testing.assert_close(out[1], torch.full((4, 2), 3.0))


# --------------------------------------------------------------------------- #
# PointwiseNet_kDiffusion
# --------------------------------------------------------------------------- #
class TestPointwiseNet:
    def test_six_concat_squash_layers(self, config):
        net = diffusion.PointwiseNet_kDiffusion(config)

        assert len(net.layers) == 6
        assert all(
            isinstance(layer, diffusion.ConcatSquashLinear) for layer in net.layers
        )

    def test_points_enter_and_leave_at_feature_dim(self, config):
        net = diffusion.PointwiseNet_kDiffusion(config)

        assert net.layers[0]._layer.in_features == config["model"]["feature_dim"]
        assert net.layers[-1]._layer.out_features == config["model"]["feature_dim"]

    def test_the_default_hidden_chain(self, config):
        for key in list(config["model"]):
            if key.startswith("diffusion_pointwise_hidden"):
                del config["model"][key]

        net = diffusion.PointwiseNet_kDiffusion(config)

        widths = [layer._layer.out_features for layer in net.layers[:-1]]
        assert widths == [128, 256, 512, 256, 128]

    def test_setting_the_first_width_rescales_the_chain(self, config):
        config["model"]["diffusion_pointwise_hidden_l1"] = 8

        net = diffusion.PointwiseNet_kDiffusion(config)

        widths = [layer._layer.out_features for layer in net.layers[:-1]]
        assert widths == [8, 16, 32, 16, 8]

    def test_each_width_can_be_set_individually(self, config):
        config["model"]["diffusion_pointwise_hidden_l1"] = 8
        config["model"]["diffusion_pointwise_hidden_l3"] = 7

        net = diffusion.PointwiseNet_kDiffusion(config)

        widths = [layer._layer.out_features for layer in net.layers[:-1]]
        assert widths == [8, 16, 7, 16, 8]

    def test_the_context_width_is_cond_dim_plus_the_time_embedding(self, config):
        net = diffusion.PointwiseNet_kDiffusion(config)

        for layer in net.layers:
            assert layer._hyper_gate.in_features == config["model"]["cond_dim"] + 64

    def test_the_time_embedding_is_fourier_then_linear(self, config):
        net = diffusion.PointwiseNet_kDiffusion(config)

        assert isinstance(net.timestep_embed[0], k_diffusion.layers.FourierFeatures)
        assert isinstance(net.timestep_embed[1], torch.nn.Linear)
        assert net.timestep_embed[1].out_features == 64

    def test_forward_preserves_the_point_cloud_shape(self, config):
        net = diffusion.PointwiseNet_kDiffusion(config)

        out = net(torch.randn(2, 5, 4), torch.rand(2) + 0.5, torch.randn(2, 4))

        assert out.shape == (2, 5, 4)

    def test_the_output_depends_on_the_context(self, config):
        torch.manual_seed(0)
        net = diffusion.PointwiseNet_kDiffusion(config)
        x, sigma = torch.randn(1, 5, 4), torch.ones(1)

        first = net(x, sigma, torch.zeros(1, 4))
        second = net(x, sigma, torch.ones(1, 4))

        assert not torch.allclose(first, second)

    def test_the_output_depends_on_sigma(self, config):
        torch.manual_seed(0)
        net = diffusion.PointwiseNet_kDiffusion(config)
        x, context = torch.randn(1, 5, 4), torch.randn(1, 4)

        first = net(x, torch.tensor([0.1]), context)
        second = net(x, torch.tensor([10.0]), context)

        assert not torch.allclose(first, second)

    def test_sigma_zero_gives_non_finite_output(self, config):
        """The time conditioning is ``sigma.log() / 4``, so a sigma of exactly
        zero puts ``-inf`` into the Fourier features and the whole output goes
        non-finite instead of raising a clear error."""
        net = diffusion.PointwiseNet_kDiffusion(config)

        out = net(torch.randn(2, 3, 4), torch.zeros(2), torch.randn(2, 4))

        assert not torch.isfinite(out).all()


# --------------------------------------------------------------------------- #
# Diffusion construction
# --------------------------------------------------------------------------- #
class TestDiffusionConstruction:
    def test_it_is_a_torch_module(self, model):
        assert isinstance(model, torch.nn.Module)

    def test_a_scalar_sigma_data_is_broadcast_to_every_feature(self, config):
        config["training"]["sigma_data"] = 0.5

        model = diffusion.Diffusion(config)

        torch.testing.assert_close(
            model.diffusion.sigma_data, torch.tensor([0.5, 0.5, 0.5, 0.5])
        )

    def test_a_per_feature_sigma_data_is_kept(self, config):
        config["training"]["sigma_data"] = [0.3, 0.5, 0.7, 1.1]

        model = diffusion.Diffusion(config)

        torch.testing.assert_close(
            model.diffusion.sigma_data, torch.tensor([0.3, 0.5, 0.7, 1.1])
        )

    @pytest.mark.parametrize("bad", [[0.5], [0.5] * 3, [0.5] * 5])
    def test_a_wrong_length_sigma_data_raises(self, config, bad):
        config["training"]["sigma_data"] = bad

        with pytest.raises(ValueError, match="sigma_data must be"):
            diffusion.Diffusion(config)

    def test_the_inner_model_is_a_pointwise_net(self, model):
        assert isinstance(
            model.diffusion.inner_model, diffusion.PointwiseNet_kDiffusion
        )

    def test_the_denoiser_reads_the_configured_loss(self, config):
        config["training"]["diffusion_loss"] = "l1"

        assert diffusion.Diffusion(config).diffusion.diffusion_loss == "l1"

    def test_the_distillation_flag_reaches_the_denoiser(self, config):
        model = diffusion.Diffusion(config, distillation=True)

        assert model.distillation is True
        assert model.diffusion.distillation is True
        assert model.diffusion.sigma_min == config["model"]["sigma_min"]

    def test_a_plain_model_is_not_distilled(self, model):
        assert model.distillation is False
        assert model.diffusion.distillation is False

    def test_a_distilled_denoiser_always_uses_l2(self, config):
        """The distillation branch of ``__init__`` never reads
        ``training.diffusion_loss``, so a distilled Denoiser is always l2 even
        when the config asks for l1."""
        config["training"]["diffusion_loss"] = "l1"

        model = diffusion.Diffusion(config, distillation=True)

        assert model.diffusion.diffusion_loss == "l2"

    def test_the_kld_head_is_attached(self, model):
        assert isinstance(model.kld, diffusion.KLDloss)


# --------------------------------------------------------------------------- #
# Denoiser construction and scalings
# --------------------------------------------------------------------------- #
class TestDenoiserScalings:
    def test_the_device_argument_is_ignored(self):
        """``device`` is accepted and never used: the ``sigma_data`` buffer is
        created on the CPU whatever the argument says."""
        denoiser = zero_denoiser()  # device="cpu" via helper

        loud = diffusion.Denoiser(RecordingInner(), [0.5] * 4, device="cuda")

        assert denoiser.sigma_data.device.type == "cpu"
        assert loud.sigma_data.device.type == "cpu"

    def test_get_scalings_matches_the_karras_formulas(self):
        sigma_data = np.array([0.3, 0.5, 0.7, 1.1])
        denoiser = zero_denoiser(sigma_data=sigma_data)
        sigma = torch.tensor([0.7, 1.3])

        c_skip, c_out, c_in = denoiser.get_scalings(sigma)

        s = sigma.numpy()[:, None]
        np.testing.assert_allclose(
            c_skip.numpy(), sigma_data**2 / (s**2 + sigma_data**2), rtol=1e-6
        )
        np.testing.assert_allclose(
            c_out.numpy(),
            s * sigma_data / (s**2 + sigma_data**2) ** 0.5,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            c_in.numpy(), 1 / (s**2 + sigma_data**2) ** 0.5, rtol=1e-6
        )

    def test_scalings_are_batch_by_feature(self):
        denoiser = zero_denoiser()

        for c in denoiser.get_scalings(torch.tensor([0.7, 1.3, 2.0])):
            assert c.shape == (3, 4)

    def test_each_feature_gets_its_own_scaling(self):
        denoiser = zero_denoiser(sigma_data=(0.3, 0.5, 0.7, 1.1))

        c_skip, _, _ = denoiser.get_scalings(torch.ones(1))

        assert len(set(c_skip[0].tolist())) == 4

    def test_the_boundary_scalings_close_the_gap_at_sigma_min(self):
        denoiser = zero_denoiser(distillation=True, sigma_min=0.002)
        sigma = torch.tensor([0.002])

        c_skip, c_out, _ = denoiser.get_scalings_for_boundary_condition(sigma)

        torch.testing.assert_close(c_skip, torch.ones(1, 4))
        torch.testing.assert_close(c_out, torch.zeros(1, 4))

    def test_boundary_scalings_with_sigma_min_zero_are_the_plain_ones(self):
        denoiser = zero_denoiser(distillation=True, sigma_min=0.0)
        sigma = torch.tensor([0.7, 1.3])

        plain = denoiser.get_scalings(sigma)
        boundary = denoiser.get_scalings_for_boundary_condition(sigma)

        for p, b in zip(plain, boundary):
            torch.testing.assert_close(p, b)


# --------------------------------------------------------------------------- #
# Denoiser.forward
# --------------------------------------------------------------------------- #
class TestDenoiserForward:
    def test_the_karras_recombination(self):
        """forward == inner(x * c_in) * c_out + x * c_skip, checked with an
        inner model that returns ones."""
        denoiser = diffusion.Denoiser(RecordingInner(1.0), [0.5] * 4, device="cpu")
        x, sigma = torch.randn(2, 5, 4), torch.tensor([0.7, 1.3])

        out = denoiser(x, sigma)

        c_skip, c_out, _ = [c.unsqueeze(1) for c in denoiser.get_scalings(sigma)]
        torch.testing.assert_close(out, torch.ones_like(x) * c_out + x * c_skip)

    def test_the_inner_model_sees_the_scaled_input(self):
        inner = RecordingInner(0.0)
        denoiser = diffusion.Denoiser(inner, [0.5] * 4, device="cpu")
        x, sigma = torch.randn(2, 5, 4), torch.tensor([0.7, 1.3])

        denoiser(x, sigma)

        _, _, c_in = [c.unsqueeze(1) for c in denoiser.get_scalings(sigma)]
        torch.testing.assert_close(inner.calls[0]["x"], x * c_in)

    def test_a_scalar_sigma_is_expanded_to_a_column(self):
        inner = RecordingInner(0.0)
        denoiser = diffusion.Denoiser(inner, [0.5] * 4, device="cpu")

        denoiser(torch.randn(3, 5, 4), 0.7)

        sigma_seen = inner.calls[0]["sigma"]
        assert sigma_seen.shape == (3, 1)
        torch.testing.assert_close(sigma_seen, torch.full((3, 1), 0.7))

    def test_kwargs_reach_the_inner_model(self):
        inner = RecordingInner(0.0)
        denoiser = diffusion.Denoiser(inner, [0.5] * 4, device="cpu")
        context = torch.randn(2, 4)

        denoiser(torch.randn(2, 5, 4), torch.ones(2), context=context)

        assert inner.calls[0]["context"] is context

    def test_a_distilled_model_at_sigma_min_is_the_identity(self):
        """The consistency boundary condition: c_skip is 1 and c_out is 0 at
        sigma_min, so the input comes back exactly, whatever the net says."""
        denoiser = diffusion.Denoiser(
            lambda x, s, **kw: torch.randn_like(x),
            [0.5] * 4,
            device="cpu",
            distillation=True,
            sigma_min=0.002,
        )
        x = torch.randn(3, 5, 4)

        assert torch.equal(denoiser(x, 0.002), x)

    def test_an_undistilled_model_at_sigma_min_is_not_the_identity(self):
        denoiser = diffusion.Denoiser(
            lambda x, s, **kw: torch.randn_like(x), [0.5] * 4, device="cpu"
        )
        x = torch.randn(3, 5, 4)

        assert not torch.allclose(denoiser(x, 0.002), x)


# --------------------------------------------------------------------------- #
# Denoiser.loss
# --------------------------------------------------------------------------- #
class TestDenoiserLoss:
    @pytest.fixture
    def inputs(self):
        rng = torch.Generator().manual_seed(0)
        inp = torch.randn(2, 4, 4, generator=rng)
        noise = torch.randn(2, 4, 4, generator=rng)
        sigma = torch.tensor([0.7, 1.3])
        return inp, noise, sigma

    def test_one_loss_per_batch_row(self, inputs):
        loss = zero_denoiser().loss(*inputs)

        assert loss.shape == (2,)
        assert torch.isfinite(loss).all()

    def test_the_l2_loss_matches_the_closed_form(self, inputs):
        inp, noise, sigma = inputs
        denoiser = zero_denoiser()

        loss = denoiser.loss(inp, noise, sigma)

        c_skip, c_out, _ = [c.unsqueeze(1) for c in denoiser.get_scalings(sigma)]
        noised = inp + noise * sigma.view(-1, 1, 1)
        target = (inp - c_skip * noised) / c_out
        torch.testing.assert_close(loss, target.pow(2).flatten(1).mean(1))

    def test_the_l1_loss_matches_the_closed_form(self, inputs):
        inp, noise, sigma = inputs
        denoiser = zero_denoiser(diffusion_loss="l1")

        loss = denoiser.loss(inp, noise, sigma)

        c_skip, c_out, _ = [c.unsqueeze(1) for c in denoiser.get_scalings(sigma)]
        noised = inp + noise * sigma.view(-1, 1, 1)
        target = (inp - c_skip * noised) / c_out
        torch.testing.assert_close(loss, target.abs().flatten(1).mean(1))

    def test_an_unknown_loss_name_raises(self, inputs):
        denoiser = zero_denoiser(diffusion_loss="huber")

        with pytest.raises(ValueError, match="must be either l1 or l2"):
            denoiser.loss(*inputs)

    def test_the_inner_model_sees_the_kwargs(self, inputs):
        denoiser = zero_denoiser()
        context = torch.randn(2, 4)

        denoiser.loss(*inputs, context=context)

        assert denoiser.inner_model.calls[0]["context"] is context

    def test_masked_points_cannot_change_the_loss(self, inputs):
        """The mask is per point, ``(B, N)`` -- the docstring's claim that it
        matches the input shape is out of date.  Masked rows are overwritten
        with resampled real rows before any noise is added, so garbage in the
        padding is invisible given the same RNG state."""
        inp, noise, sigma = inputs
        mask = torch.ones(2, 4, dtype=torch.bool)
        mask[1, 2:] = False
        garbled = inp.clone()
        garbled[1, 2:] = 999.0

        torch.manual_seed(0)
        clean_loss = zero_denoiser().loss(inp, noise, sigma, input_mask=mask)
        torch.manual_seed(0)
        garbled_loss = zero_denoiser().loss(garbled, noise, sigma, input_mask=mask)

        torch.testing.assert_close(clean_loss, garbled_loss)

    def test_without_the_mask_the_same_garbage_matters(self, inputs):
        inp, noise, sigma = inputs
        garbled = inp.clone()
        garbled[1, 2:] = 999.0

        clean_loss = zero_denoiser().loss(inp, noise, sigma)
        garbled_loss = zero_denoiser().loss(garbled, noise, sigma)

        assert not torch.allclose(clean_loss, garbled_loss)

    def test_the_callers_tensor_is_not_modified(self, inputs):
        inp, noise, sigma = inputs
        mask = torch.ones(2, 4, dtype=torch.bool)
        mask[1, 2:] = False
        before = inp.clone()

        zero_denoiser().loss(inp, noise, sigma, input_mask=mask)

        torch.testing.assert_close(inp, before)

    def test_an_all_true_mask_is_a_no_op(self, inputs):
        inp, noise, sigma = inputs

        bare = zero_denoiser().loss(inp, noise, sigma)
        masked = zero_denoiser().loss(
            inp, noise, sigma, input_mask=torch.ones(2, 4, dtype=torch.bool)
        )

        torch.testing.assert_close(bare, masked)

    def test_an_event_with_no_real_points_crashes(self, inputs):
        """A fully padded event leaves ``torch.randint`` nothing to draw from,
        so the resampling loop raises rather than skipping the event or
        zeroing its loss."""
        inp, noise, sigma = inputs
        mask = torch.ones(2, 4, dtype=torch.bool)
        mask[1] = False

        with pytest.raises(RuntimeError, match="from.*less than.*to"):
            zero_denoiser().loss(inp, noise, sigma, input_mask=mask)


# --------------------------------------------------------------------------- #
# Diffusion.get_loss
# --------------------------------------------------------------------------- #
class TestGetLoss:
    def test_returns_one_finite_scalar(self, model):
        """The docstring promises ``(loss, loss_prior)`` but a single scalar
        tensor is all that ever comes back."""
        x = event_batch()

        loss = model.get_loss(x, torch.randn_like(x), torch.rand(2) + 0.5, torch.randn(2, 4))

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_gradients_reach_every_parameter(self, model):
        x = event_batch()

        loss = model.get_loss(x, torch.randn_like(x), torch.rand(2) + 0.5, torch.randn(2, 4))
        loss.backward()

        for parameter in model.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()

    def test_logarithmic_energies_mask_non_finite_rows(self, config):
        """With ``logarithmic_point_energy`` a point is padding when its energy
        is not finite, so garbage coordinates in ``-inf`` rows do not move the
        loss (given the same RNG state)."""
        torch.manual_seed(0)
        model = diffusion.Diffusion(config)
        x = event_batch()
        x[1, 2:, 3] = float("-inf")
        garbled = x.clone()
        garbled[1, 2:, :3] = 999.0
        noise, sigma, cond = torch.randn_like(x), torch.rand(2) + 0.5, torch.randn(2, 4)

        torch.manual_seed(1)
        clean = model.get_loss(x, noise, sigma, cond)
        torch.manual_seed(1)
        dirty = model.get_loss(garbled, noise, sigma, cond)

        torch.testing.assert_close(clean, dirty)

    def test_linear_energies_mask_rows_at_or_below_the_threshold(self, config):
        config["model"]["logarithmic_point_energy"] = False
        torch.manual_seed(0)
        model = diffusion.Diffusion(config)
        x = event_batch()
        x[1, 2:, 3] = 0.0  # padding by energy
        garbled = x.clone()
        garbled[1, 2:, :3] = 999.0
        noise, sigma, cond = torch.randn_like(x), torch.rand(2) + 0.5, torch.randn(2, 4)

        torch.manual_seed(1)
        clean = model.get_loss(x, noise, sigma, cond)
        torch.manual_seed(1)
        dirty = model.get_loss(garbled, noise, sigma, cond)

        torch.testing.assert_close(clean, dirty)

    def test_real_rows_do_move_the_loss(self, model):
        x = event_batch()
        moved = x.clone()
        moved[0, 0, 0] += 5.0
        noise, sigma, cond = torch.randn_like(x), torch.rand(2) + 0.5, torch.randn(2, 4)

        torch.manual_seed(1)
        first = model.get_loss(x, noise, sigma, cond)
        torch.manual_seed(1)
        second = model.get_loss(moved, noise, sigma, cond)

        assert not torch.allclose(first, second)

    def test_the_conditioning_reaches_the_inner_model(self, model):
        recorder = RecordingInner(0.0)
        model.diffusion.inner_model = recorder
        x = event_batch()
        cond = torch.randn(2, 4)

        model.get_loss(x, torch.randn_like(x), torch.rand(2) + 0.5, cond)

        assert recorder.calls[0]["context"] is cond


# --------------------------------------------------------------------------- #
# Diffusion.sample
# --------------------------------------------------------------------------- #
class TestSample:
    @pytest.fixture
    def patched_samplers(self, mocker):
        """Every k_diffusion sampler replaced by a mock returning zeros."""
        mocks = {}
        for attribute in set(SAMPLER_NAMES.values()):
            mocks[attribute] = mocker.patch(
                f"k_diffusion.sampling.{attribute}",
                return_value=torch.zeros(2, 6, 4),
            )
        return mocks

    def test_an_empty_batch_short_circuits(self, model, patched_samplers):
        out = model.sample(torch.zeros(0, 4), num_points=6)

        assert out.shape == (0, 6, 4)
        assert all(mock.call_count == 0 for mock in patched_samplers.values())

    @pytest.mark.parametrize("name, attribute", sorted(SAMPLER_NAMES.items()))
    def test_each_name_dispatches_to_its_own_sampler(
        self, config, patched_samplers, name, attribute
    ):
        """The accepted names are inconsistently prefixed: four are bare
        ("euler", ..., "dpmpp_2s_ancestral") and three carry ``sample_``."""
        config["sampler"] = name
        model = diffusion.Diffusion(config)

        model.sample(torch.randn(2, 4), num_points=6)

        assert patched_samplers[attribute].call_count == 1
        for other, mock in patched_samplers.items():
            if other != attribute:
                assert mock.call_count == 0

    def test_the_sampler_gets_the_denoiser_noise_and_schedule(
        self, model, patched_samplers
    ):
        cond = torch.randn(2, 4)

        result = model.sample(cond, num_points=6)

        args, kwargs = patched_samplers["sample_heun"].call_args
        assert args[0] is model.diffusion
        assert args[1].shape == (2, 6, 4)  # x_T
        assert kwargs["extra_args"]["context"] is cond
        assert kwargs["disable"] is True
        assert result is patched_samplers["sample_heun"].return_value

    def test_the_schedule_is_karras_from_sigma_max_to_zero(
        self, model, config, patched_samplers
    ):
        model.sample(torch.randn(2, 4), num_points=6)

        sigmas = patched_samplers["sample_heun"].call_args[0][2]
        assert len(sigmas) == config["num_steps"] + 1
        assert sigmas[0] == pytest.approx(config["model"]["sigma_max"], rel=1e-4)
        assert sigmas[-1] == 0.0
        assert (sigmas.diff() < 0).all()

    def test_x_T_is_unit_noise_scaled_by_sigma_max(
        self, model, config, patched_samplers
    ):
        torch.manual_seed(123)
        model.sample(torch.randn(2, 4), num_points=6)

        x_T = patched_samplers["sample_heun"].call_args[0][1]
        torch.manual_seed(123)
        torch.randn(2, 4)  # replay the cond draw, which came first off the stream
        expected = torch.randn([2, 6, 4]) * config["model"]["sigma_max"]
        torch.testing.assert_close(x_T, expected)

    def test_only_heun_receives_the_churn_settings(self, config, patched_samplers):
        config["model"]["s_churn"] = 0.3
        config["model"]["s_noise"] = 1.1

        heun_model = diffusion.Diffusion({**config, "sampler": "heun"})
        heun_model.sample(torch.randn(2, 4), num_points=6)
        euler_model = diffusion.Diffusion({**config, "sampler": "euler"})
        euler_model.sample(torch.randn(2, 4), num_points=6)

        heun_kwargs = patched_samplers["sample_heun"].call_args[1]
        assert heun_kwargs["s_churn"] == 0.3
        assert heun_kwargs["s_noise"] == 1.1
        assert "s_churn" not in patched_samplers["sample_euler"].call_args[1]

    def test_an_unknown_sampler_raises(self, config):
        config["sampler"] = "midpoint"
        model = diffusion.Diffusion(config)

        with pytest.raises(NotImplementedError, match="midpoint"):
            model.sample(torch.randn(2, 4), num_points=6)

    @pytest.mark.parametrize("name", sorted(SAMPLER_NAMES))
    def test_every_sampler_runs_end_to_end(self, config, name):
        config["sampler"] = name
        torch.manual_seed(0)
        model = diffusion.Diffusion(config)

        out = model.sample(torch.randn(2, 4), num_points=6)

        assert out.shape == (2, 6, 4)
        assert torch.isfinite(out).all()

    def test_a_distilled_model_samples_in_one_step(
        self, distilled, config, patched_samplers
    ):
        recorder = RecordingInner(0.0)
        distilled.diffusion.inner_model = recorder

        out = distilled.sample(torch.randn(2, 4), num_points=6)

        assert all(mock.call_count == 0 for mock in patched_samplers.values())
        assert out.shape == (2, 6, 4)
        assert len(recorder.calls) == 1
        torch.testing.assert_close(
            recorder.calls[0]["sigma"],
            torch.full((2, 1), config["model"]["sigma_max"]),
        )


# --------------------------------------------------------------------------- #
# consistency loss / get_cd_loss
# --------------------------------------------------------------------------- #
class TestConsistencyLoss:
    @pytest.fixture
    def trio(self, config):
        """(student, teacher, target), all distilled, all identically seeded."""
        models = []
        for _ in range(3):
            torch.manual_seed(0)
            models.append(diffusion.Diffusion(config, distillation=True))
        return models

    def test_one_step_is_refused(self, trio, config):
        student, teacher, target = trio
        config["num_steps"] = 1

        with pytest.raises(AssertionError, match="one step"):
            student.diffusion.consistency_loss(
                torch.randn(3, 5, 4),
                teacher.diffusion,
                target.diffusion,
                config,
                context=torch.randn(3, 4),
            )

    def test_one_loss_per_batch_row(self, trio, config):
        student, teacher, target = trio

        loss = student.diffusion.consistency_loss(
            torch.randn(3, 5, 4),
            teacher.diffusion,
            target.diffusion,
            config,
            context=torch.randn(3, 4),
        )

        assert loss.shape == (3,)
        assert torch.isfinite(loss).all()
        assert (loss >= 0).all()

    def test_consistency_training_runs_without_a_teacher(self, trio, config):
        student, _, target = trio

        loss = student.diffusion.consistency_loss(
            torch.randn(3, 5, 4), None, target.diffusion, config,
            context=torch.randn(3, 4),
        )

        assert loss.shape == (3,)
        assert torch.isfinite(loss).all()

    def test_get_cd_loss_returns_a_scalar(self, trio, config):
        student, teacher, target = trio

        loss = student.get_cd_loss(
            event_batch(3), torch.randn(3, 4), teacher, target, config
        )

        assert loss.ndim == 0
        assert torch.isfinite(loss)

    def test_only_the_student_collects_gradients(self, trio, config):
        student, teacher, target = trio

        loss = student.get_cd_loss(
            event_batch(3), torch.randn(3, 4), teacher, target, config
        )
        loss.backward()

        assert any(p.grad is not None for p in student.parameters())
        assert all(p.grad is None for p in teacher.parameters())
        assert all(p.grad is None for p in target.parameters())
