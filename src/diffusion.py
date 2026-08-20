import torch
import numbers
from torch.nn import Module, ModuleList, functional, Linear
import k_diffusion


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class KLDloss(Module):
    def __init__(self):
        super(KLDloss, self).__init__()
        self.use_cuda = torch.cuda.is_available()

    def forward(self, mu, logvar):
        B = logvar.size(0)
        KLD = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())) / (B)
        return KLD


def is_v2_config(config):
    v2_keys = ["use_generalist", "use_experts", "use_specialists"]
    return any(key in config["model"] for key in v2_keys)


class Diffusion(Module):
    """
    Default Caloclouds diffusion model for generating points.
    This diffusion model produces indeviual points,
    drawn independently from the distribution.
    It is conditioned on number of points and incident energy.
    """

    def __init__(self, config, distillation=False):
        """
        Constructor for the model, for both training and sampling.

        Parameters
        ----------
        config : dictionary
        distillation : bool, optional
            If true, this model is disstilld from recent iterations of the
            primary model being trained.

        """
        super().__init__()
        self.config = config
        self.distillation = distillation
        device = config["device"]

        if is_v2_config(config):
            print("Using PointwiseNet_kDiffusion_v2")
            net = PointwiseNet_kDiffusion_v2(config=config)
        else:
            print("Using PointwiseNet_kDiffusion")
            net = PointwiseNet_kDiffusion(config=config)

        # set up the denoiser
        sigma_data = config["training"]["sigma_data"]
        n_features = config["model"]["feature_dim"]
        if isinstance(sigma_data, numbers.Real):
            sigma_data = [sigma_data] * n_features
        if len(sigma_data) != n_features:
            raise ValueError(
                "sigma_data must be either a float or a list of n_features floats."
            )
        if not distillation:
            self.diffusion = Denoiser(
                net,
                sigma_data=sigma_data,
                device=device,
                diffusion_loss=config["training"]["diffusion_loss"],
            )
        else:
            self.diffusion = Denoiser(
                net,
                sigma_data=sigma_data,
                device=device,
                distillation=True,
                sigma_min=config["model"]["sigma_min"],
            )

        self.kld = KLDloss()

    def get_loss(
        self,
        x,
        noise,
        sigma,
        cond_feats,
    ):
        """
        Calculate the loss of one or mode batches of data.
        Data may be zero padded, points with 0 energy are padding.

        Parameters
        ----------
        x : torch.Tensor (batch_size, max_num_points, features)
            Data to compare to.
        noise : torch.Tensor (batch_size, max_num_points, features)
            Random tensor for adding noise to data.
        sigma : torch.Tensor (batch_size)
            Time for diffusion.
        cond_feats : torch.Tensor (batch_size, cond_features)
            Features to condition on, normally energy and number of points.

        Returns
        -------
        loss : torch.Tensor (1,)
            Loss of the model.
        loss_prior : torch.Tensor (1,), or None
            If there is a flow model forming the latent dimension,
            this creates a prior loss for the flow model.

        """
        data_mask = None
        if not self.config["model"]["logarithmic_point_energy"]:
            # loss_diffusion = self.diffusion.get_loss(x, cond_feats)    # diffusion loss
            data_mask = (
                x[..., 3] > 1e-5
            )  # anything with very small energy is considered padding
        else:
            data_mask = torch.isfinite(x[..., 3])

        loss_diffusion = self.diffusion.loss(
            x, noise, sigma, context=cond_feats, input_mask=data_mask
        ).mean()  # diffusion loss

        # Total loss
        loss = loss_diffusion

        return loss

    def sample(self, cond_feats, num_points):
        batch_size, _ = cond_feats.size()
        if batch_size == 0:
            return torch.zeros(
                0,
                num_points,
                self.config["model"]["feature_dim"],
                device=cond_feats.device,
            )

        x_T = (
            torch.randn(
                [cond_feats.size(0), num_points, self.config["model"]["feature_dim"]],
                device=cond_feats.device,
            )
            * self.config["model"]["sigma_max"]
        )

        if not self.distillation:
            sigmas = k_diffusion.sampling.get_sigmas_karras(
                self.config["num_steps"],
                self.config["model"]["sigma_min"],
                self.config["model"]["sigma_max"],
                rho=self.config["model"]["rho"],
                device=cond_feats.device,
            )

            sampler_kw_args = {"extra_args": {"context": cond_feats}, "disable": True}
            if self.config["sampler"] == "euler":
                sampler_class = k_diffusion.sampling.sample_euler
            elif self.config["sampler"] == "heun":
                sampler_class = k_diffusion.sampling.sample_heun
                sampler_kw_args["s_churn"] = self.config["model"]["s_churn"]
                sampler_kw_args["s_noise"] = self.config["model"]["s_noise"]
            elif self.config["sampler"] == "dpmpp_2m":
                sampler_class = k_diffusion.sampling.sample_dpmpp_2m
            elif self.config["sampler"] == "dpmpp_2s_ancestral":
                sampler_class = k_diffusion.sampling.sample_dpmpp_2s_ancestral
            elif self.config["sampler"] == "sample_euler_ancestral":
                sampler_class = k_diffusion.sampling.sample_euler_ancestral
            elif self.config["sampler"] == "sample_lms":
                sampler_class = k_diffusion.sampling.sample_lms
            elif self.config["sampler"] == "sample_dpmpp_2m_sde":
                sampler_class = k_diffusion.sampling.sample_dpmpp_2m_sde
            else:
                raise NotImplementedError(
                    f"Sampler {self.config['sampler']} not implemented"
                )
            x_0 = sampler_class(self.diffusion, x_T, sigmas, **sampler_kw_args)

        else:  # one step for consistency model
            x_0 = self.diffusion.forward(
                x_T, self.config["model"]["sigma_max"], context=cond_feats
            )

        return x_0

    def get_cd_loss(self, x, cond_feats, model_teacher, model_target, config):
        """
        Args:
            x:  Input point clouds, (B, N, d).
            cond_feats: conditional features, (B, C)
            model_teacher: teacher model as score function for ODE solver
            model_ema_target: target model
            config: dict
        """
        loss = self.diffusion.consistency_loss(
            x,
            model_teacher.diffusion,
            model_target.diffusion,
            config,
            context=cond_feats,
        ).mean()  # consistency loss

        return loss


class Denoiser(torch.nn.Module):
    """
    A Karras et al. preconditioner for denoising diffusion models.
    from: https://github.com/crowsonkb/k-diffusion/blob/master/k_diffusion/layers.py#L12

    Slightly extended to allow filtering of input data when calculating loss.
    """

    def __init__(
        self,
        inner_model,
        sigma_data: list[float],
        device="cuda",
        distillation=False,
        sigma_min=0.002,
        diffusion_loss="l2",
    ):
        super().__init__()
        self.inner_model = inner_model
        self.register_buffer("sigma_data", torch.tensor(sigma_data))
        self.distillation = distillation
        self.sigma_min = sigma_min
        self.diffusion_loss = diffusion_loss

    def get_scalings(self, sigma):  # B,
        sigma_data = self.sigma_data.expand(sigma.shape[0], -1)  # B, 4
        sigma = k_diffusion.utils.append_dims(sigma, sigma_data.ndim)  # B, 4
        c_skip = sigma_data**2 / (sigma**2 + sigma_data**2)  # B, 4
        c_out = sigma * sigma_data / (sigma**2 + sigma_data**2) ** 0.5  # B, 4
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5  # B, 4
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(
        self, sigma
    ):  # B,   # for consistency model
        sigma_data = self.sigma_data.expand(sigma.shape[0], -1)  # B, 4
        sigma = k_diffusion.utils.append_dims(sigma, sigma_data.ndim)  # B, 4
        c_skip = sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + sigma_data**2
        )  # B, 4
        c_out = (
            (sigma - self.sigma_min)
            * sigma_data
            / (sigma**2 + sigma_data**2) ** 0.5
        )  # B, 4
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5  # B, 4
        return c_skip, c_out, c_in

    def loss(self, input, noise, sigma, input_mask=None, **kwargs):
        """
        Calculate the loss of the model for a batch of data.

        Parameters
        ----------
        input : torch.Tensor (batch_size, ...)
            Data to compare to.
        noise : torch.Tensor, same shape as input
            Random tensor for adding noise to data.
        sigma : torch.Tensor, (batch_size,)
            Time for diffusion.
        input_mask : torch.Tensor, same shape as input, optional
            Mask for filtering input data, only calculate loss for True values.
        **kwargs
            Passed straight to the inner model.

        Returns
        -------
        loss : torch.Tensor (batch_size,)
            Mean loss of each batch.

        """
        c_skip, c_out, c_in = [
            x.unsqueeze(1) for x in self.get_scalings(sigma)
        ]  # B,1,4
        if not (input_mask is None or input_mask.all()):
            # we need to fill some values of the input from the rest of the input
            changes_needed = (~input_mask).sum(1)
            input = input.clone()
            for i in torch.where(changes_needed > 0)[0]:
                event = input_mask[i]
                possible = torch.where(event)[0]
                idxs = possible[torch.randint(0, len(possible), (changes_needed[i],))]
                input[i][~event] = input[i][idxs]

        noised_input = input + noise * k_diffusion.utils.append_dims(sigma, input.ndim)
        model_output = self.inner_model(noised_input * c_in, sigma, **kwargs)
        target = (input - c_skip * noised_input) / c_out
        if self.diffusion_loss == "l2":
            distance = (model_output - target).pow(2)
        elif self.diffusion_loss == "l1":
            distance = (model_output - target).abs()
        else:
            raise ValueError("diffusion_loss must be either l1 or l2")
        # now the dimensionality is taken down to the batch size.
        mean_distance = distance.flatten(1).mean(1)
        return mean_distance

    def forward(
        self, input, sigma, **kwargs
    ):  # same as "denoise" in KarrasDenoiser of CM code
        if isinstance(sigma, float) or isinstance(sigma, int):
            sigma = (
                torch.tensor([sigma] * input.shape[0], dtype=torch.float32)
                .to(input.device)
                .unsqueeze(1)
            )
        if not self.distillation:
            c_skip, c_out, c_in = [
                x.unsqueeze(1) for x in self.get_scalings(sigma)
            ]  # B,1,4
        else:
            c_skip, c_out, c_in = [
                x.unsqueeze(1) for x in self.get_scalings_for_boundary_condition(sigma)
            ]
        # CM code did an additional resacling of the time sigma for the time conditing
        return self.inner_model(input * c_in, sigma, **kwargs) * c_out + input * c_skip

    # inspired by
    # https://github.com/openai/consistency_models/blob/main/cm/karras_diffusion.py#L106
    def consistency_loss(self, input, teacher_model, target_model, config, **kwargs):
        noise = torch.randn_like(input)
        dims = input.ndim
        num_scales = config["num_steps"]
        assert num_scales > 1, "if you want one step you need a distilled model"

        def denoise_fn(x, t):  # t = sigma
            return self(x, t, **kwargs)

        @torch.no_grad()
        def target_denoise_fn(x, t):
            return target_model(x, t, **kwargs)

        @torch.no_grad()
        def teacher_denoise_fn(x, t):
            return teacher_model(x, t, **kwargs)

        @torch.no_grad()
        def heun_solver(samples, t, next_t, x0):
            x = samples
            denoiser = teacher_denoise_fn(x, t)

            d = (x - denoiser) / k_diffusion.utils.append_dims(t, dims)
            samples = x + d * k_diffusion.utils.append_dims(next_t - t, dims)
            if teacher_model is None:
                # but this would not be the correct Heun method any more?
                # anyway, without teacher model it's using Euler
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(samples, next_t)

            next_d = (samples - denoiser) / k_diffusion.utils.append_dims(next_t, dims)
            samples = x + (d + next_d) * k_diffusion.utils.append_dims(
                (next_t - t) / 2, dims
            )

            return samples

        @torch.no_grad()
        def euler_solver(samples, t, next_t, x0):
            x = samples
            if teacher_model is None:
                denoiser = x0
            else:
                denoiser = teacher_denoise_fn(x, t)
            d = (x - denoiser) / k_diffusion.utils.append_dims(t, dims)
            samples = x + d * k_diffusion.utils.append_dims(next_t - t, dims)

            return samples

        # Get random sigmas / EDM boundaries
        # same as k_diffusion.utils.get_sigmas_karras()   with t and t+1
        indices = torch.randint(
            0, num_scales - 1, (input.shape[0],), device=input.device
        )

        # in paper: t + 1
        t = config["model"]["sigma_max"] ** (1 / config["model"]["rho"]) + indices / (
            num_scales - 1
        ) * (
            config["model"]["sigma_min"] ** (1 / config["model"]["rho"])
            - config["model"]["sigma_max"] ** (1 / config["model"]["rho"])
        )
        t = t ** config["model"]["rho"]

        # in paper: t    --> so t > t2
        t2 = config["model"]["sigma_max"] ** (1 / config["model"]["rho"]) + (
            indices + 1
        ) / (num_scales - 1) * (
            config["model"]["sigma_min"] ** (1 / config["model"]["rho"])
            - config["model"]["sigma_max"] ** (1 / config["model"]["rho"])
        )
        t2 = t2 ** config["model"]["rho"]

        x_t = input + noise * k_diffusion.utils.append_dims(
            t, dims
        )  # calculate x_t at time step t+1 from data

        dropout_state = (
            torch.get_rng_state()
        )  # get state of the random number generator
        distiller = denoise_fn(
            x_t, t
        )  # denoise x_t completely x_t (t + 1) --> x_0 = input

        if teacher_model is None:
            x_t2 = euler_solver(
                x_t, t, t2, input
            ).detach()  # for consistency training, not used
        else:
            x_t2 = heun_solver(x_t, t, t2, input).detach()
            # for consistency distllation, one solver step to get from t+1 to t

        torch.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2)
        # target model (ema, not trained)
        # denoises data completely from time t to t=0 / x_0 / input
        distiller_target = distiller_target.detach()

        weights = 1.0  # paper: uniform weights work well in their experiments

        # l2 loss / MSE loss
        diffs = (distiller - distiller_target) ** 2
        loss = mean_flat(diffs) * weights

        return loss


class ConcatSquashLinear(Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        ret = self._layer(x) * gate + bias
        return ret


class PointwiseNet_kDiffusion(Module):
    def __init__(self, config):
        super().__init__()
        context_dim = config["model"]["cond_dim"]
        point_dim = config["model"]["feature_dim"]
        time_dim = 64
        fourier_scale = (
            16  # 1 in k-diffusion, 16 in EDM, 30 in Score-based generative modeling
        )

        self.act = functional.leaky_relu
        hidden_1 = config["model"].get("diffusion_pointwise_hidden_l1", 128)
        hidden_2 = config["model"].get("diffusion_pointwise_hidden_l2", hidden_1 * 2)
        hidden_3 = config["model"].get("diffusion_pointwise_hidden_l3", hidden_2 * 2)
        hidden_4 = config["model"].get("diffusion_pointwise_hidden_l4", hidden_2)
        hidden_5 = config["model"].get("diffusion_pointwise_hidden_l5", hidden_1)
        self.layers = ModuleList(
            [
                ConcatSquashLinear(point_dim, hidden_1, context_dim + time_dim),
                ConcatSquashLinear(hidden_1, hidden_2, context_dim + time_dim),
                ConcatSquashLinear(hidden_2, hidden_3, context_dim + time_dim),
                ConcatSquashLinear(hidden_3, hidden_4, context_dim + time_dim),
                ConcatSquashLinear(hidden_4, hidden_5, context_dim + time_dim),
                ConcatSquashLinear(hidden_5, point_dim, context_dim + time_dim),
            ]
        )

        self.timestep_embed = torch.nn.Sequential(
            k_diffusion.layers.FourierFeatures(1, time_dim, std=fourier_scale),
            # 1D Fourier features --> with register_buffer, so weights are not trained
            torch.nn.Linear(time_dim, time_dim),  # this is a trainable layer
        )

    def forward(self, x, sigma, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            sigma:     Time. (B, ).  --> becomes "sigma" in k-diffusion
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        sigma = sigma.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        # formulation from EDM paper / k-diffusion
        c_noise = sigma.log() / 4  # (B, 1, 1)
        time_emb = self.act(self.timestep_embed(c_noise))  # (B, 1, T)

        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+T)
        # TODO: might want to add additional linear embedding net
        # for context or only cond_feats

        out = x
        for i, layer in enumerate(self.layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        return out


# --------------------------------------------------------------------------- #
# locating a conditioning feature inside the context vector
# --------------------------------------------------------------------------- #
# Widths of the conditioning features as they arrive in ``cond_feats``.
# ``incident_pdg`` is one-hot encoded over ``config["simulate_pdgs"]``
# (see ``dataset.pdgs_to_onehot`` / ``inference.pdg_to_onehot_in_full_cond``)
# so its width is not fixed and is worked out per config.
COND_FEATURE_WIDTHS = {
    "incident_energy": 1,
    "incident_direction": 3,
}


def cond_feature_range(config, name):
    """Columns of ``cond_feats`` holding one conditioning feature.

    Parameters
    ----------
    config : dict
    name : str
        A member of ``config["model"]["cond_features"]``.

    Returns
    -------
    (int, int) or None
        ``None`` if the feature is not being conditioned on.

    Raises
    ------
    ValueError
        If the implied width disagrees with ``config["model"]["cond_dim"]``,
        which means this table has gone stale.
    """
    cond_features = list(config["model"]["cond_features"])
    widths = []
    for feature in cond_features:
        if feature == "incident_pdg":
            widths.append(len(config["simulate_pdgs"]))
        elif feature in COND_FEATURE_WIDTHS:
            widths.append(COND_FEATURE_WIDTHS[feature])
        else:
            raise NotImplementedError(f"Unknown conditioning feature {feature}")
    total = sum(widths)
    if total != config["model"]["cond_dim"]:
        raise ValueError(
            f"cond_features {cond_features} imply a context of width {total}, "
            f"but config['model']['cond_dim'] is {config['model']['cond_dim']}"
        )
    if name not in cond_features:
        return None
    position = cond_features.index(name)
    start = sum(widths[:position])
    return start, start + widths[position]


# --------------------------------------------------------------------------- #
# the network
# --------------------------------------------------------------------------- #
class PointwiseNet_kDiffusion_v2(Module):
    """
    We can have 1 generalist (the old model)
    Any number of experts who combine their output all at once.
    Any number of specialists, only one of whom is active at once.
    If more than once network was active, we also have a pooling network at the end.

    Every branch is a ConcatSquashLinear stack of the same shape as the v1 net:
    it takes the ``(B, N, point_dim)`` points and returns ``(B, N, point_dim)``.
    The branch outputs are concatenated on the feature axis and reduced back to
    ``point_dim`` by the pooling stack, which is itself conditioned on the
    context, so the pooling layers act as the router that weighs the branches
    against each other.  With a single branch the pooling stack is skipped
    entirely and the module is numerically identical to
    ``PointwiseNet_kDiffusion`` (given the same hidden widths).

    Config keys, all under ``config["model"]``::

        use_generalist:   bool                         (default True)
        use_experts:      false | int | "incident_pdg" (default False)
        use_specialists:   false | "incident_pdg"       (default False)
        diffusion_generalist_hidden_dims: list[int]
        diffusion_expert_hidden_dims:     list[int]
        diffusion_specialist_hidden_dims: list[int]
        diffusion_pooling_hidden_dims:    list[int]
    """

    DEFAULT_HIDDEN_DIMS = [128, 256, 512, 256, 128]
    DEFAULT_POOLING_DIMS = [128]

    def __init__(self, config):
        super().__init__()
        context_dim = config["model"]["cond_dim"]
        point_dim = config["model"]["feature_dim"]
        time_dim = 64
        fourier_scale = (
            16  # 1 in k-diffusion, 16 in EDM, 30 in Score-based generative modeling
        )
        ctx_dim = context_dim + time_dim

        self.act = functional.leaky_relu

        n_outputs = 0

        self.generalist = None
        if config["model"].get("use_generalist", True):
            self.generalist = self._setup_generalist(
                config, point_dim, context_dim, time_dim
            )
            n_outputs += 1

        self.experts = None
        if config["model"].get("use_experts", False):
            self.experts = self._setup_experts(config, point_dim, context_dim, time_dim)
            n_outputs += len(self.experts)

        self.specialists = None
        self.specialist_routing_range = None
        if config["model"].get("use_specialists", False):
            # routing is on the one-hot pdg block of the context
            self.specialist_routing_range = cond_feature_range(config, "incident_pdg")
            self.specialists = self._setup_specialists(
                config, point_dim, context_dim, time_dim
            )
            n_outputs += 1

        if n_outputs == 0:
            raise ValueError(
                "At least one of use_generalist, use_experts, "
                "or use_specialists must be True"
            )
        self.use_pooling = n_outputs > 1
        self.pooling = None
        if self.use_pooling:
            self.pooling = self._setup_pooling(
                config, n_outputs * point_dim, point_dim, ctx_dim
            )

        self.timestep_embed = torch.nn.Sequential(
            k_diffusion.layers.FourierFeatures(1, time_dim, std=fourier_scale),
            # 1D Fourier features --> with register_buffer, so weights are not trained
            torch.nn.Linear(time_dim, time_dim),  # this is a trainable layer
        )

    # ----------------------------------------------------------------- #
    # construction
    # ----------------------------------------------------------------- #
    @staticmethod
    def _stack(hidden_dims, dim_in, dim_out, dim_ctx):
        """One ConcatSquashLinear branch, dim_in -> hidden_dims -> dim_out."""
        all_dims = [dim_in] + list(hidden_dims) + [dim_out]
        return ModuleList(
            [
                ConcatSquashLinear(one_in, one_out, dim_ctx)  # noqa: F821
                for one_in, one_out in zip(all_dims[:-1], all_dims[1:])
            ]
        )

    @staticmethod
    def _count_branches(config, key, allow_int):
        """How many sub-networks ``config["model"][key]`` asks for."""
        value = config["model"][key]
        if isinstance(value, str) and value == "incident_pdg":
            if "incident_pdg" not in config["model"]["cond_features"]:
                raise ValueError(
                    f"config['model']['{key}'] routes on incident_pdg, so "
                    "incident_pdg must be in config['model']['cond_features']"
                )
            return len(config["simulate_pdgs"])
        # bool is a subclass of int, so `use_experts: true` would silently
        # become a single expert without this guard
        if allow_int and isinstance(value, int) and not isinstance(value, bool):
            if value < 1:
                raise ValueError(f"config['model']['{key}'] must be at least 1")
            return int(value)
        raise NotImplementedError(
            f"unknown value for config['model']['{key}']: {value}"
        )

    def _setup_generalist(self, config, point_dim, context_dim, time_dim):
        hidden_dims = config["model"].get(
            "diffusion_generalist_hidden_dims", self.DEFAULT_HIDDEN_DIMS
        )
        return self._stack(hidden_dims, point_dim, point_dim, context_dim + time_dim)

    def _setup_experts(self, config, point_dim, context_dim, time_dim):
        n_experts = self._count_branches(config, "use_experts", allow_int=True)
        hidden_dims = config["model"].get(
            "diffusion_expert_hidden_dims", self.DEFAULT_HIDDEN_DIMS
        )
        return ModuleList(
            [
                self._stack(hidden_dims, point_dim, point_dim, context_dim + time_dim)
                for _ in range(n_experts)
            ]
        )

    def _setup_specialists(self, config, point_dim, context_dim, time_dim):
        n_specialists = self._count_branches(config, "use_specialists", allow_int=False)
        if self.specialist_routing_range is None and n_specialists > 1:
            raise ValueError(
                "n_specialists > 1, but specialist_routing_range is not set"
            )
        routing_length = (
            self.specialist_routing_range[1] - self.specialist_routing_range[0]
        )
        specialist_context_dim = context_dim + time_dim - routing_length
        hidden_dims = config["model"].get(
            "diffusion_specialist_hidden_dims", self.DEFAULT_HIDDEN_DIMS
        )
        return ModuleList(
            [
                self._stack(hidden_dims, point_dim, point_dim, specialist_context_dim)
                for _ in range(n_specialists)
            ]
        )

    def _setup_pooling(self, config, dim_in, dim_out, dim_ctx):
        """Reduce the concatenated branch outputs back to one point.

        Returns ``None`` when only one branch is active, in which case
        ``forward`` hands that branch's output straight back.
        """
        if dim_in == dim_out:
            return None
        hidden_dims = config["model"].get(
            "diffusion_pooling_hidden_dims", self.DEFAULT_POOLING_DIMS
        )
        return self._stack(hidden_dims, dim_in, dim_out, dim_ctx)

    # ----------------------------------------------------------------- #
    # evaluation
    # ----------------------------------------------------------------- #
    def _run_stack(self, layers, x, ctx_emb):
        """Apply one branch; activations between layers but not on the output."""
        out = x
        for i, layer in enumerate(layers):
            out = layer(ctx=ctx_emb, x=out)
            if i < len(layers) - 1:
                out = self.act(out)
        return out

    def _run_specialists(self, x, ctx_emb, context):
        """Hard routing: each event goes through exactly one specialist.

        ``context`` is ``(B, 1, C)``.  The pdg block is one-hot, and the
        conditioning preprocessing leaves it untouched, so an ``argmax`` over
        that block recovers the index into ``config["simulate_pdgs"]``.
        """
        # there is a 0 because context has been broadcast up from (B, C) to (B, 1, C)
        route = context[
            :, 0, self.specialist_routing_range[0] : self.specialist_routing_range[1]
        ].argmax(
            dim=-1
        )  # (B,)

        out = torch.zeros_like(x)
        for index, specialist in enumerate(self.specialists):
            chosen = route == index
            if not bool(chosen.any()):
                continue
            out[chosen] = self._run_stack(specialist, x[chosen], ctx_emb[chosen])
        return out

    def forward(self, x, sigma, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            sigma:     Time. (B, ).  --> becomes "sigma" in k-diffusion
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        sigma = sigma.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        # formulation from EDM paper / k-diffusion
        c_noise = sigma.log() / 4  # (B, 1, 1)
        time_emb = self.act(self.timestep_embed(c_noise))  # (B, 1, T)

        ctx_emb = torch.cat([time_emb, context], dim=-1)  # (B, 1, F+T)
        # TODO: might want to add additional linear embedding net
        # for context or only cond_feats

        branch_outputs = []
        if self.generalist is not None:
            branch_outputs.append(self._run_stack(self.generalist, x, ctx_emb))
        if self.experts is not None:
            for expert in self.experts:
                branch_outputs.append(self._run_stack(expert, x, ctx_emb))
        if self.specialists is not None:
            specialist_ctx_emb = torch.cat(
                [
                    time_emb,
                    context[:, :, : self.specialist_routing_range[0]],
                    context[:, :, self.specialist_routing_range[1] :],
                ],
                dim=-1,
            )
            branch_outputs.append(self._run_specialists(x, specialist_ctx_emb, context))

        if self.pooling is None:
            return branch_outputs[0]

        out = torch.cat(branch_outputs, dim=-1)  # (B, N, n_outputs*d)
        return self._run_stack(self.pooling, out, ctx_emb)
