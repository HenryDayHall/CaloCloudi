"""Standalone inference for the distilled (consistency) CaloClouds model.

This module reproduces exactly the inference path that
``evaluation.inference.Sampler.sample`` takes for a *distilled* checkpoint,
with no imports from the project and no ``k_diffusion``.

Imports: ``torch``, ``yaml``, ``numpy``, and the standard library.  Nothing
else.  (``verify_against_project`` imports the project, but it is only called
from ``--verify``; nothing is imported from the project at module import time.)

Why this is short
-----------------
The distilled path touches only two things in ``k_diffusion``:

* ``k_diffusion.utils.append_dims``   -- three lines, reproduced below.
* ``k_diffusion.layers.FourierFeatures`` -- one module holding one *buffer*,
  reproduced below.  The buffer is in the checkpoint, so only its shape and
  its ``forward`` matter here; the random initialisation never runs.

Everything else in ``k_diffusion`` (``get_sigmas_karras``, the samplers,
``EMAWarmup``) is used by the teacher and by training only.  A distilled model
takes exactly one step, at ``sigma_max``, so none of it is reachable.

State dict compatibility
------------------------
The module names here match ``src/diffusion.py`` one for one, so an existing
checkpoint loads with a strict ``load_state_dict``:

    diffusion.sigma_data
    diffusion.inner_model.layers.<i>._layer.{weight,bias}
    diffusion.inner_model.layers.<i>._hyper_bias.weight
    diffusion.inner_model.layers.<i>._hyper_gate.{weight,bias}
    diffusion.inner_model.timestep_embed.0.weight      (Fourier buffer)
    diffusion.inner_model.timestep_embed.1.{weight,bias}

``Diffusion.kld`` is dropped: ``KLDloss`` has no parameters and no buffers, so
it contributes nothing to the state dict.

Usage
-----
    from standalone_distilled import StandaloneSampler

    sampler = StandaloneSampler.from_checkpoint("…/xxx_model.pt", "…/config.yaml")
    points = sampler.sample(cond, num_points)      # numpy in, numpy out

Or with no Python file at all, after ``--export``:

    core = torch.jit.load("distilled.ts.pt")
    points = core(cond, torch.randn(B, N, F), num_points)

Command line
------------
    python standalone_distilled.py export   --checkpoint CKPT --config CFG --out OUT
    python standalone_distilled.py verify   --checkpoint CKPT --config CFG
    python standalone_distilled.py selftest --config CFG
"""

from __future__ import annotations

import argparse
import math
import numbers
import os
import warnings

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import Linear, ModuleList
from torch.nn import functional as F

__all__ = [
    "StandaloneSampler",
    "DistilledModel",
    "PipelineCore",
    "load_torchscript",
    "sample_with_torchscript",
]

TIME_DIM = 64
FOURIER_SCALE = 16


# --------------------------------------------------------------------------- #
# the two pieces of k_diffusion the distilled path needs
# --------------------------------------------------------------------------- #
def append_dims(x, target_dims):
    """``k_diffusion.utils.append_dims``: trailing singleton axes onto ``x``."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


class FourierFeatures(nn.Module):
    """``k_diffusion.layers.FourierFeatures``.

    ``weight`` is a buffer, not a parameter: it is fixed at construction and
    saved in the checkpoint.  Loading a checkpoint overwrites it, so the ``std``
    used here is irrelevant to inference and is kept only so that a
    from-scratch build has the right scale.
    """

    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        if out_features % 2 != 0:
            raise ValueError("out_features must be even")
        self.register_buffer(
            "weight", torch.randn([out_features // 2, in_features]) * std
        )

    def forward(self, x):
        f = 2 * math.pi * x @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


# --------------------------------------------------------------------------- #
# config helpers (copied from src/diffusion.py, no behaviour change)
# --------------------------------------------------------------------------- #
COND_FEATURE_WIDTHS = {"incident_energy": 1, "incident_direction": 3}


def is_v2_config(config):
    v2_keys = ["use_generalist", "use_experts", "use_specialists"]
    return any(key in config["model"] for key in v2_keys)


def cond_feature_range(config, name):
    """Columns of ``cond_feats`` holding one conditioning feature, or None."""
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
class ConcatSquashLinear(nn.Module):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super().__init__()
        self._layer = Linear(dim_in, dim_out)
        self._hyper_bias = Linear(dim_ctx, dim_out, bias=False)
        self._hyper_gate = Linear(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = torch.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        return self._layer(x) * gate + bias


class Stack(ModuleList):
    """A ConcatSquashLinear branch that can be called.

    Subclasses ``ModuleList`` rather than wrapping one so that the state dict
    keys stay ``<branch>.<i>.<...>``, exactly as the project writes them.

    The activation is applied *before* every layer but the first, which is the
    same function as the project's "after every layer but the last".
    """

    def forward(self, ctx, x):
        out = x
        for i, layer in enumerate(self):
            if i > 0:
                out = F.leaky_relu(out)
            out = layer(ctx, out)
        return out


def _stack(hidden_dims, dim_in, dim_out, dim_ctx):
    all_dims = [dim_in] + list(hidden_dims) + [dim_out]
    return Stack(
        [
            ConcatSquashLinear(one_in, one_out, dim_ctx)
            for one_in, one_out in zip(all_dims[:-1], all_dims[1:])
        ]
    )


def _timestep_embed():
    return nn.Sequential(
        FourierFeatures(1, TIME_DIM, std=FOURIER_SCALE),
        Linear(TIME_DIM, TIME_DIM),
    )


class PointwiseNet(nn.Module):
    """``PointwiseNet_kDiffusion``."""

    def __init__(self, config):
        super().__init__()
        context_dim = config["model"]["cond_dim"]
        point_dim = config["model"]["feature_dim"]
        ctx_dim = context_dim + TIME_DIM

        hidden_1 = config["model"].get("diffusion_pointwise_hidden_l1", 128)
        hidden_2 = config["model"].get("diffusion_pointwise_hidden_l2", hidden_1 * 2)
        hidden_3 = config["model"].get("diffusion_pointwise_hidden_l3", hidden_2 * 2)
        hidden_4 = config["model"].get("diffusion_pointwise_hidden_l4", hidden_2)
        hidden_5 = config["model"].get("diffusion_pointwise_hidden_l5", hidden_1)
        hidden = [hidden_1, hidden_2, hidden_3, hidden_4, hidden_5]

        self.layers = _stack(hidden, point_dim, point_dim, ctx_dim)
        self.timestep_embed = _timestep_embed()

    def forward(self, x, sigma, context):
        batch_size = x.size(0)
        sigma = sigma.view(batch_size, 1, 1)
        # the project writes .view(batch_size, 1, -1); naming the last axis is
        # the same reshape for a (B, C) context and stays legal when B == 0
        context = context.reshape(batch_size, 1, context.shape[-1])

        c_noise = sigma.log() / 4
        time_emb = F.leaky_relu(self.timestep_embed(c_noise))
        ctx_emb = torch.cat([time_emb, context], dim=-1)

        return self.layers(ctx_emb, x)


class PointwiseNetV2(nn.Module):
    """``PointwiseNet_kDiffusion_v2``: generalist / experts / specialists."""

    DEFAULT_HIDDEN_DIMS = [128, 256, 512, 256, 128]
    DEFAULT_POOLING_DIMS = [128]

    def __init__(self, config):
        super().__init__()
        context_dim = config["model"]["cond_dim"]
        point_dim = config["model"]["feature_dim"]
        ctx_dim = context_dim + TIME_DIM
        model_cfg = config["model"]

        n_outputs = 0

        self.generalist = None
        if model_cfg.get("use_generalist", True):
            hidden = model_cfg.get(
                "diffusion_generalist_hidden_dims", self.DEFAULT_HIDDEN_DIMS
            )
            self.generalist = _stack(hidden, point_dim, point_dim, ctx_dim)
            n_outputs += 1

        self.experts = None
        if model_cfg.get("use_experts", False):
            n_experts = self._count_branches(config, "use_experts", allow_int=True)
            hidden = model_cfg.get(
                "diffusion_expert_hidden_dims", self.DEFAULT_HIDDEN_DIMS
            )
            self.experts = ModuleList(
                [_stack(hidden, point_dim, point_dim, ctx_dim) for _ in range(n_experts)]
            )
            n_outputs += n_experts

        self.specialists = None
        self.specialist_routing_range = None
        if model_cfg.get("use_specialists", False):
            self.specialist_routing_range = cond_feature_range(config, "incident_pdg")
            n_specialists = self._count_branches(
                config, "use_specialists", allow_int=False
            )
            if self.specialist_routing_range is None and n_specialists > 1:
                raise ValueError(
                    "n_specialists > 1, but specialist_routing_range is not set"
                )
            routing_length = (
                self.specialist_routing_range[1] - self.specialist_routing_range[0]
            )
            hidden = model_cfg.get(
                "diffusion_specialist_hidden_dims", self.DEFAULT_HIDDEN_DIMS
            )
            self.specialists = ModuleList(
                [
                    _stack(hidden, point_dim, point_dim, ctx_dim - routing_length)
                    for _ in range(n_specialists)
                ]
            )
            n_outputs += 1

        if n_outputs == 0:
            raise ValueError(
                "At least one of use_generalist, use_experts, "
                "or use_specialists must be True"
            )

        self.pooling = None
        if n_outputs > 1:
            hidden = model_cfg.get(
                "diffusion_pooling_hidden_dims", self.DEFAULT_POOLING_DIMS
            )
            self.pooling = _stack(hidden, n_outputs * point_dim, point_dim, ctx_dim)

        self.timestep_embed = _timestep_embed()

    @staticmethod
    def _count_branches(config, key, allow_int):
        value = config["model"][key]
        if isinstance(value, str) and value == "incident_pdg":
            if "incident_pdg" not in config["model"]["cond_features"]:
                raise ValueError(
                    f"config['model']['{key}'] routes on incident_pdg, so "
                    "incident_pdg must be in config['model']['cond_features']"
                )
            return len(config["simulate_pdgs"])
        if allow_int and isinstance(value, int) and not isinstance(value, bool):
            if value < 1:
                raise ValueError(f"config['model']['{key}'] must be at least 1")
            return int(value)
        raise NotImplementedError(f"unknown value for config['model']['{key}']: {value}")

    def _run_specialists(self, x, ctx_emb, context):
        start, stop = self.specialist_routing_range
        route = context[:, 0, start:stop].argmax(dim=-1)
        out = torch.zeros_like(x)
        for index, specialist in enumerate(self.specialists):
            chosen = route == index
            if not bool(chosen.any()):
                continue
            out[chosen] = specialist(ctx_emb[chosen], x[chosen])
        return out

    def forward(self, x, sigma, context):
        batch_size = x.size(0)
        sigma = sigma.view(batch_size, 1, 1)
        context = context.reshape(batch_size, 1, context.shape[-1])

        c_noise = sigma.log() / 4
        time_emb = F.leaky_relu(self.timestep_embed(c_noise))
        ctx_emb = torch.cat([time_emb, context], dim=-1)

        branch_outputs = []
        if self.generalist is not None:
            branch_outputs.append(self.generalist(ctx_emb, x))
        if self.experts is not None:
            for expert in self.experts:
                branch_outputs.append(expert(ctx_emb, x))
        if self.specialists is not None:
            start, stop = self.specialist_routing_range
            specialist_ctx_emb = torch.cat(
                [time_emb, context[:, :, :start], context[:, :, stop:]], dim=-1
            )
            branch_outputs.append(
                self._run_specialists(x, specialist_ctx_emb, context)
            )

        if self.pooling is None:
            return branch_outputs[0]
        return self.pooling(ctx_emb, torch.cat(branch_outputs, dim=-1))


# --------------------------------------------------------------------------- #
# the preconditioner, boundary-condition branch only
# --------------------------------------------------------------------------- #
class Denoiser(nn.Module):
    """``Denoiser`` with ``distillation=True``; the teacher branch is dropped."""

    def __init__(self, inner_model, sigma_data, sigma_min=0.002):
        super().__init__()
        self.inner_model = inner_model
        self.register_buffer("sigma_data", torch.tensor(sigma_data))
        self.sigma_min = float(sigma_min)

    def get_scalings_for_boundary_condition(self, sigma):
        sigma_data = self.sigma_data.expand(sigma.shape[0], -1)
        sigma = append_dims(sigma, sigma_data.ndim)
        c_skip = sigma_data**2 / ((sigma - self.sigma_min) ** 2 + sigma_data**2)
        c_out = (
            (sigma - self.sigma_min)
            * sigma_data
            / (sigma**2 + sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def forward(self, x, sigma, context):
        c_skip, c_out, c_in = [
            scale.unsqueeze(1)
            for scale in self.get_scalings_for_boundary_condition(sigma)
        ]
        return self.inner_model(x * c_in, sigma, context) * c_out + x * c_skip


class DistilledModel(nn.Module):
    """``Diffusion(config, distillation=True)``, sampling only.

    The attribute is called ``diffusion`` so the checkpoint's keys line up.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        net = PointwiseNetV2(config) if is_v2_config(config) else PointwiseNet(config)

        sigma_data = config["training"]["sigma_data"]
        n_features = config["model"]["feature_dim"]
        if isinstance(sigma_data, numbers.Real):
            sigma_data = [sigma_data] * n_features
        if len(sigma_data) != n_features:
            raise ValueError(
                "sigma_data must be either a float or a list of n_features floats."
            )

        self.diffusion = Denoiser(
            net, sigma_data=sigma_data, sigma_min=config["model"]["sigma_min"]
        )
        self.sigma_max = float(config["model"]["sigma_max"])
        self.feature_dim = n_features

    def denoise(self, x_T, cond_feats):
        """One consistency step at ``sigma_max``.  Trace-safe."""
        # The project builds this with torch.tensor([sigma] * B), which a
        # tracer would freeze at the example batch size; full_like does not.
        sigma = torch.full_like(
            x_T[:, :1, 0], self.sigma_max, dtype=torch.float32
        )  # (B, 1)
        return self.diffusion(x_T, sigma, cond_feats)

    def sample(self, cond_feats, num_points, generator=None):
        """``Diffusion.sample``, distilled branch."""
        batch_size = cond_feats.size(0)
        if batch_size == 0:
            return torch.zeros(
                0, num_points, self.feature_dim, device=cond_feats.device
            )
        x_T = (
            torch.randn(
                [batch_size, num_points, self.feature_dim],
                device=cond_feats.device,
                generator=generator,
            )
            * self.sigma_max
        )
        return self.denoise(x_T, cond_feats)


# --------------------------------------------------------------------------- #
# preprocessing (copied from src/data/transforms.py)
# --------------------------------------------------------------------------- #
TRANSFORM_NAMES = [
    "Partial",
    "Identity",
    "Log",
    "LogIt",
    "Affine",
    "Clamp",
    "StandardScaler",
    "Dequantize",
]


class Transformation(nn.Module):
    def forward(self, x):
        raise NotImplementedError()

    def inverse(self, x):
        raise NotImplementedError()


class Sequence(Transformation):
    def __init__(self, modules):
        super().__init__()
        self.sub_modules = ModuleList(modules)

    def forward(self, x):
        for module in self.sub_modules:
            x = module.forward(x)
        return x

    def inverse(self, x):
        for module in self.sub_modules[::-1]:
            x = module.inverse(x)
        return x


class Identity(Transformation):
    def forward(self, x):
        return x

    def inverse(self, x):
        return x


class Log(Transformation):
    def __init__(self, alpha=1e-6, base=math.e):
        super().__init__()
        self.alpha = alpha
        self.log_base = math.log(base)

    def forward(self, x):
        return torch.log(x + self.alpha) / self.log_base

    def inverse(self, x):
        return torch.exp(self.log_base * x) - self.alpha


class LogIt(Transformation):
    def __init__(self, alpha=1e-6):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        x = (1 - 2 * self.alpha) * x + self.alpha
        return torch.log(x / (1 - x))

    def inverse(self, x):
        x = torch.sigmoid(x)
        return (x - self.alpha) / (1 - 2 * self.alpha)


class Affine(Transformation):
    def __init__(self, scale=1.0, shift=0.0, inverse_scale=None, neg_shift=None):
        super().__init__()
        if inverse_scale is not None:
            scale = 1 / inverse_scale
        if neg_shift is not None:
            shift = -neg_shift
        self.a = scale
        self.b = shift

    def forward(self, x):
        return self.a * (x + self.b)

    def inverse(self, x):
        return x / self.a - self.b


class Clamp(Transformation):
    def __init__(self, min=0.0, max=1.0):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x):
        return torch.clamp(x, self.min, self.max)

    def inverse(self, x):
        return x


class StandardScaler(Transformation):
    """Kept for completeness.

    ``Sampler`` builds preprocessing from the config and never calls ``fit``,
    so at inference this is always the identity with ``mean=0``, ``std=1``.
    Included so that a config naming it still loads.
    """

    def __init__(self, shape):
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("std", torch.ones(shape))
        self.shape = shape

    def forward(self, x):
        return (x - self.mean) / self.std

    def inverse(self, x):
        return x * self.std + self.mean


class Dequantize(Transformation):
    def forward(self, x):
        return x + torch.rand_like(x)

    def inverse(self, x):
        return torch.floor(x)


class Partial(Transformation):
    def __init__(self, split_indices, components, axis=-1):
        super().__init__()
        self.split_indices = self._check_split_indices(split_indices)
        self.axis = axis
        components = list(components)
        expected = len(self.split_indices) + 1
        if len(components) != expected:
            raise ValueError(
                f"split_indices={self.split_indices} describes {expected} slices "
                f"along axis {axis}, but {len(components)} components were given"
            )
        self.components = ModuleList(
            self._build_component(component) for component in components
        )

    @staticmethod
    def _check_split_indices(split_indices):
        indices = list(split_indices)
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError(f"split_indices must be integers, got {index!r}")
        if any(index < 1 for index in indices):
            raise ValueError(f"split_indices must be positive, got {indices}")
        if any(later <= earlier for earlier, later in zip(indices, indices[1:])):
            raise ValueError(f"split_indices must be strictly increasing, got {indices}")
        return indices

    @staticmethod
    def _build_component(component):
        if isinstance(component, Transformation):
            return component
        if isinstance(component, nn.Module):
            raise TypeError(
                f"components must be Transformations, got {type(component).__name__}"
            )
        return compose(component)

    def _chunk(self, x):
        # The guard is Python-level only; under a trace ``x.shape[axis]`` is a
        # tensor and comparing it would raise a TracerWarning for a check that
        # never enters the graph.
        if not torch.jit.is_tracing():
            if not -x.ndim <= self.axis < x.ndim:
                raise ValueError(
                    f"axis {self.axis} is out of range for a {x.ndim}d tensor"
                )
            length = x.shape[self.axis]
            if self.split_indices and self.split_indices[-1] >= length:
                raise ValueError(
                    f"split_indices={self.split_indices} do not fit along axis "
                    f"{self.axis}, which has length {length}"
                )
        return torch.tensor_split(x, self.split_indices, dim=self.axis)

    def forward(self, x):
        return torch.cat(
            [
                component.forward(chunk)
                for component, chunk in zip(self.components, self._chunk(x))
            ],
            dim=self.axis,
        )

    def inverse(self, x):
        return torch.cat(
            [
                component.inverse(chunk)
                for component, chunk in zip(self.components, self._chunk(x))
            ],
            dim=self.axis,
        )


_TRANSFORMS = {
    "Partial": Partial,
    "Identity": Identity,
    "Log": Log,
    "LogIt": LogIt,
    "Affine": Affine,
    "Clamp": Clamp,
    "StandardScaler": StandardScaler,
    "Dequantize": Dequantize,
}


def compose(transformation):
    if transformation is None:
        return Sequence([Identity()])
    trafo_list = []
    for element in transformation:
        if element[0] not in _TRANSFORMS:
            raise ValueError(f"Invalid transformation: {element[0]}")
        Trafo = _TRANSFORMS[element[0]]
        if len(element) == 1 or element[1] is None:
            trafo_list.append(Trafo())
        elif isinstance(element[1], list):
            trafo_list.append(Trafo(*element[1]))
        elif isinstance(element[1], dict):
            trafo_list.append(Trafo(**element[1]))
        else:
            raise ValueError(
                f"argument for {element[0]} must be a list or a dict "
                f"not {type(element[1])}"
            )
    return Sequence(trafo_list)


def _resolve_value(configs, value):
    if (not hasattr(value, "__iter__")) or isinstance(value, str):
        return value
    try:
        part = configs
        for key in value:
            if not isinstance(key, str):
                return value
            part = part[key]
        return part
    except KeyError:
        return value


def _leaf_like(value):
    if not hasattr(value, "__iter__"):
        return True
    if isinstance(value, list):
        return all(_leaf_like(v) for v in value)
    return isinstance(value, str)


def fetch_values(configs, nested_iterable):
    if _leaf_like(nested_iterable):
        return _resolve_value(configs, nested_iterable)
    if isinstance(nested_iterable, dict):
        return {k: fetch_values(configs, v) for k, v in nested_iterable.items()}
    if isinstance(nested_iterable, list):
        return [fetch_values(configs, element) for element in nested_iterable]
    return nested_iterable


def preprocessing(configs, part):
    return compose(fetch_values(configs, configs["preprocessing"][part]))


# --------------------------------------------------------------------------- #
# the whole pipeline as one traceable module
# --------------------------------------------------------------------------- #
class PipelineCore(nn.Module):
    """``raw cond`` + ``noise`` + ``num_points`` -> physical points.

    Noise is an *input* rather than drawn inside, so the module is
    deterministic and its trace does not freeze the batch or point count.

    forward
    -------
    cond : (B, cond_dim) float
        Raw conditioning, one-hot already applied to any pdg block.
    noise : (B, N, feature_dim) float
        Standard normal.  ``N`` is the largest ``num_points`` in the batch.
    num_points : (B,) long
        Points to keep per event; the rest are zeroed lowest-energy-first.
    """

    def __init__(self, model, preprocess_conditioning, preprocess_features):
        super().__init__()
        self.model = model
        self.preprocess_conditioning = preprocess_conditioning
        self.preprocess_features = preprocess_features

    def forward(self, cond, noise, num_points):
        cond = self.preprocess_conditioning.forward(cond)
        x_T = noise * self.model.sigma_max
        output = self.model.denoise(x_T, cond)
        output = self.preprocess_features.inverse(output)

        max_points = noise.shape[1]
        energies = output[:, :, 3]
        # rank of each point by energy, 0 = lowest
        energy_order = energies.argsort(dim=1).argsort(dim=1)
        remove_from_event = max_points - num_points
        remove = energy_order < remove_from_event.unsqueeze(1)
        return output.masked_fill(remove.unsqueeze(-1), 0.0)


# --------------------------------------------------------------------------- #
# the user-facing object
# --------------------------------------------------------------------------- #
class StandaloneSampler:
    """``evaluation.inference.Sampler`` for a distilled checkpoint, standalone."""

    def __init__(self, config, model=None, device=None, dtype=None):
        self.config = config
        self.device = torch.device(device or config.get("device", "cpu"))
        self.dtype = dtype or getattr(torch, config["training"]["dtype"])
        self.model = DistilledModel(config) if model is None else model
        self.model.to(self.device, dtype=self.dtype).eval().requires_grad_(False)
        self.preprocess_conditioning = preprocessing(config, "conditioning").to(
            self.device, dtype=self.dtype
        )
        self.preprocess_features = preprocessing(config, "features").to(
            self.device, dtype=self.dtype
        )
        self.core = PipelineCore(
            self.model, self.preprocess_conditioning, self.preprocess_features
        )

    # ---------------------------------------------------------------- #
    @classmethod
    def from_checkpoint(cls, checkpoint, config=None, device=None, dtype=None):
        """Build from a ``.pt`` state dict and a config.

        ``config`` may be a path, a dict, or ``None`` to look for
        ``config.yaml`` in the run's log directory (two levels up from the
        checkpoint, matching ``Sampler.get_config_from_model_path``).
        """
        if config is None:
            config = find_config(checkpoint)
        if isinstance(config, (str, os.PathLike)):
            with open(config, "r") as handle:
                config = yaml.safe_load(handle)
        device = device or config.get("device", "cpu")
        if device != "cpu" and not torch.cuda.is_available():
            warnings.warn(f"config asks for {device!r} but no CUDA; using cpu")
            device = "cpu"

        sampler = cls(config, device=device, dtype=dtype)
        state = _load_state_dict(checkpoint, device)
        sampler.model.load_state_dict(state, strict=True)
        sampler.model.to(sampler.device, dtype=sampler.dtype)
        return sampler

    # ---------------------------------------------------------------- #
    def sample(self, cond, num_points, generator=None):
        """Identical in behaviour to ``Sampler.sample``.

        Parameters
        ----------
        cond : (B, cond_dim) array
            Raw conditioning, one-hot already applied to any pdg column.
        num_points : (B,) int array
        """
        cond = torch.as_tensor(np.asarray(cond)).to(self.device, dtype=self.dtype)
        num_points = np.asarray(num_points).astype(np.int64)
        max_points = int(np.max(num_points)) if num_points.size else 0
        if cond.shape[0] == 0:
            # Diffusion.sample short-circuits an empty batch; so does this
            return np.zeros((0, max_points, self.model.feature_dim), dtype=np.float32)
        num_points_t = torch.as_tensor(num_points, device=self.device)

        with torch.no_grad():
            noise = torch.randn(
                [cond.shape[0], max_points, self.model.feature_dim],
                device=self.device,
                generator=generator,
            )
            output = self.core(cond, noise, num_points_t)
        return output.cpu().numpy()

    # ---------------------------------------------------------------- #
    def export_torchscript(self, path, batch=4, points=64, check=True):
        """Trace the pipeline and save it.

        The result loads with ``torch.jit.load`` and needs no Python file at
        all -- not even this one.
        """
        if getattr(self.model.diffusion.inner_model, "specialists", None) is not None:
            raise RuntimeError(
                "This config uses specialists, whose routing is data dependent. "
                "Tracing would freeze the routing of the example batch. Use the "
                "eager StandaloneSampler for these, or script the routing by hand."
            )
        cond_dim = self.config["model"]["cond_dim"]
        example = (
            torch.randn(batch, cond_dim, device=self.device, dtype=self.dtype).abs() + 1,
            torch.randn(
                batch, points, self.model.feature_dim, device=self.device,
                dtype=self.dtype,
            ),
            torch.full((batch,), points // 2, device=self.device, dtype=torch.long),
        )
        with torch.no_grad():
            traced = torch.jit.trace(self.core, example, check_trace=False)
            traced = torch.jit.freeze(traced.eval())
            if check:
                _check_shape_independence(self.core, traced, self.config, self)
        torch.jit.save(traced, path)
        return path


# --------------------------------------------------------------------------- #
# loading a saved TorchScript module -- no project, no this file
# --------------------------------------------------------------------------- #
def load_torchscript(path, device="cpu"):
    return torch.jit.load(path, map_location=device).eval()


def sample_with_torchscript(module, cond, num_points, feature_dim=4, device="cpu"):
    """Convenience wrapper matching ``StandaloneSampler.sample``."""
    cond = torch.as_tensor(np.asarray(cond), dtype=torch.float32, device=device)
    num_points = np.asarray(num_points).astype(np.int64)
    max_points = int(np.max(num_points))
    noise = torch.randn(cond.shape[0], max_points, feature_dim, device=device)
    with torch.no_grad():
        out = module(cond, noise, torch.as_tensor(num_points, device=device))
    return out.cpu().numpy()


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def find_config(checkpoint):
    """``<log_dir>/config.yaml`` for ``<log_dir>/checkpoints/<file>.pt``."""
    log_dir = os.path.dirname(os.path.dirname(os.path.abspath(checkpoint)))
    for name in ("config.yaml", "config.yml"):
        candidate = os.path.join(log_dir, name)
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        f"No config.yaml in {log_dir}; pass one with --config"
    )


def _load_state_dict(checkpoint, device):
    try:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
    except TypeError:  # torch < 1.13
        state = torch.load(checkpoint, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # KLDloss holds nothing, but tolerate a checkpoint that somehow named it
    return {k: v for k, v in state.items() if not k.startswith("kld.")}


def _check_shape_independence(eager, traced, config, sampler):
    """Run both at a different batch and point count than the trace."""
    cond_dim = config["model"]["cond_dim"]
    batch, points = 7, 37
    cond = (
        torch.randn(batch, cond_dim, device=sampler.device, dtype=sampler.dtype).abs()
        + 1
    )
    noise = torch.randn(
        batch, points, sampler.model.feature_dim,
        device=sampler.device, dtype=sampler.dtype,
    )
    keep = torch.randint(1, points + 1, (batch,), device=sampler.device)
    with torch.no_grad():
        a = eager(cond, noise, keep)
        b = traced(cond, noise, keep)
    if a.shape != b.shape:
        raise RuntimeError(f"trace froze a shape: {a.shape} vs {b.shape}")
    gap = (a - b).abs().max().item()
    if gap > 1e-5:
        raise RuntimeError(f"traced module disagrees with eager by {gap:g}")
    return gap


# --------------------------------------------------------------------------- #
# optional: compare against the project.  Only --verify calls this, and the
# project import happens here rather than at module level.
# --------------------------------------------------------------------------- #
def verify_against_project(checkpoint, config_path, batch=8, points=53, seed=0):
    """Run the project's Sampler and this one on the same noise and compare.

    Run from the repo root, in an environment that has ``k_diffusion``.
    """
    from src.diffusion import Diffusion  # noqa: PLC0415
    from src.data import transforms as project_transforms  # noqa: PLC0415

    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = dict(config)
    config["device"] = device
    dtype = getattr(torch, config["training"]["dtype"])

    state = _load_state_dict(checkpoint, device)

    reference = Diffusion(config, distillation=True)
    reference.load_state_dict(state)
    reference.to(device, dtype=dtype).eval().requires_grad_(False)

    mine = StandaloneSampler.from_checkpoint(checkpoint, config, device=device)

    ref_cond_pre = project_transforms.preprocessing(config, "conditioning").to(
        device, dtype=dtype
    )
    ref_feat_pre = project_transforms.preprocessing(config, "features").to(
        device, dtype=dtype
    )

    torch.manual_seed(seed)
    cond_dim = config["model"]["cond_dim"]
    cond = torch.rand(batch, cond_dim, device=device, dtype=dtype) * 90 + 10
    feature_dim = config["model"]["feature_dim"]
    noise = torch.randn(batch, points, feature_dim, device=device, dtype=dtype)
    keep = torch.randint(1, points + 1, (batch,), device=device)

    with torch.no_grad():
        # project: preprocess, one consistency step at sigma_max, invert
        ref_cond = ref_cond_pre.forward(cond)
        x_T = noise * config["model"]["sigma_max"]
        ref_out = reference.diffusion.forward(
            x_T, config["model"]["sigma_max"], context=ref_cond
        )
        ref_out = ref_feat_pre.inverse(ref_out)

        mine_out = mine.core(cond, noise, keep)

    # the trimming is applied only by mine.core, so compare before it
    ref_np = ref_out.cpu().numpy()
    energy_order = np.argsort(np.argsort(ref_np[:, :, 3], axis=1))
    remove = energy_order < (points - keep.cpu().numpy())[:, None]
    ref_np[remove] = 0

    gap = float(np.abs(ref_np - mine_out.cpu().numpy()).max())
    print(f"max |project - standalone| = {gap:.3e}")
    if gap > 1e-5:
        raise SystemExit("MISMATCH")
    print("identical within tolerance")
    return gap


# --------------------------------------------------------------------------- #
# selftest: build random weights from a config, check eager == traced
# --------------------------------------------------------------------------- #
def selftest(config_path):
    with open(config_path, "r") as handle:
        config = yaml.safe_load(handle)
    config["device"] = "cpu"
    sampler = StandaloneSampler(config, device="cpu")
    path = "/tmp/_selftest.ts.pt"
    sampler.export_torchscript(path, check=True)
    loaded = load_torchscript(path)
    cond = np.abs(np.random.randn(5, config["model"]["cond_dim"])) + 1
    keep = np.random.randint(1, 40, size=5)
    a = sampler.sample(cond, keep, generator=torch.Generator().manual_seed(0))
    out = sample_with_torchscript(
        loaded, cond, keep, feature_dim=config["model"]["feature_dim"]
    )
    print(f"eager sample  {a.shape}, scripted sample {out.shape}")
    print("selftest passed")


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="trace and save a TorchScript module")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--config", default=None)
    export.add_argument("--out", required=True)
    export.add_argument("--device", default="cpu")

    verify = sub.add_parser("verify", help="compare against the project (needs repo)")
    verify.add_argument("--checkpoint", required=True)
    verify.add_argument("--config", required=True)

    test = sub.add_parser("selftest", help="random weights, eager vs traced")
    test.add_argument("--config", required=True)

    args = parser.parse_args()
    if args.command == "export":
        sampler = StandaloneSampler.from_checkpoint(
            args.checkpoint, args.config, device=args.device
        )
        sampler.export_torchscript(args.out)
        print(f"wrote {args.out}")
    elif args.command == "verify":
        verify_against_project(args.checkpoint, args.config)
    elif args.command == "selftest":
        selftest(args.config)


if __name__ == "__main__":
    main()
