"""How much does the pooling stack listen to each branch, and does the context change that?

``PointwiseNet_kDiffusion_v2`` concatenates its branch outputs on the feature
axis and hands the result to ``self.pooling``::

    out = torch.cat(branch_outputs, dim=-1)      # (B, N, n_branches * point_dim)
    return self._run_stack(self.pooling, out, ctx_emb)

so the pooling input is partitioned into contiguous ``point_dim``-wide *blocks*,
one per branch, in the order generalist, experts, specialist.  This module
measures how much each block drives the pooling output, and how that split moves
as the conditioning changes.

Why the answer is not obvious from the weights
----------------------------------------------
A ``ConcatSquashLinear`` layer computes ``(W x) * sigmoid(G c) + B c``.  The gate
scales *rows* (output units), not *columns*, so at the first pooling layer every
block's contribution to a given hidden unit is scaled by the same gate value:
the relative weighting of blocks in the pre-activation is context independent.
Context can only re-weight blocks by

1. moving the leaky-ReLU activation pattern (via the hyper-bias ``B c``), and
2. re-weighting hidden units, which blocks project onto unequally, at the
   later layers.

For a two-layer pooling stack (``diffusion_pooling_hidden_dims: [128]``) the
Jacobian is exactly

    J = diag(s2) W2 diag(alpha) diag(s1) W1

with ``s1``, ``s2`` the gates and ``alpha`` the leaky-ReLU slopes.  The block
split lives entirely in the columns of ``W1``; the context enters only through
``s1``, ``s2`` and ``alpha``.  So a *flat* result (no context dependence) is a
real possible outcome, not a bug, and the study is built to tell the two apart.

Because the activations are piecewise linear, the local Jacobian is exact inside
its linear region and the block contributions ``J_k x_k`` sum to the output up to
a context-only offset.  Nothing here is a first-order approximation.

Conventions
-----------
``x_cat``   (G, P, n_branches * point_dim)  pooling input, G contexts, P points
``ctx_emb`` (G, 1, 64 + cond_dim)           ``[time_emb, context]`` -- time FIRST
``jac``     (G, P, point_dim, n_branches * point_dim)

Two private members of the net are used deliberately -- ``net._run_stack`` and
``net.act`` -- so that this module runs the same code path as ``forward`` rather
than a copy of it that can silently drift.
"""

import itertools
import math

import numpy as np
import torch

from .inference import evaluating
from ..diffusion import PointwiseNet_kDiffusion_v2, cond_feature_range


#: Width of the Fourier time embedding, hard coded in ``PointwiseNet_kDiffusion_v2``.
#: ``ctx_emb`` is ``[time_emb, context]``, so the conditioning starts at this column.
TIME_DIM = 64

#: Above this many branches, exact Shapley enumeration stops being cheap.
MAX_EXACT_SHAPLEY_BLOCKS = 10


# --------------------------------------------------------------------------- #
# getting at the network
# --------------------------------------------------------------------------- #
def get_v2_net(model):
    """The ``PointwiseNet_kDiffusion_v2`` inside a ``Diffusion``, with a pooling stack.

    Parameters
    ----------
    model : Diffusion or PointwiseNet_kDiffusion_v2

    Raises
    ------
    TypeError
        If the checkpoint holds a v1 net.
    ValueError
        If only one branch is active, in which case ``forward`` returns that
        branch directly and there is no pooling stack to study.
    """
    net = getattr(getattr(model, "diffusion", None), "inner_model", model)
    if not isinstance(net, PointwiseNet_kDiffusion_v2):
        raise TypeError(
            f"expected PointwiseNet_kDiffusion_v2, got {type(net).__name__}; "
            "this checkpoint has no branch structure to attribute"
        )
    if net.pooling is None:
        raise ValueError(
            "this model has a single branch, so forward() returns it unpooled "
            "and there is nothing to attribute"
        )
    return net


def block_names(net):
    """Branch labels in the order ``forward`` concatenates them.

    The specialist slot is a single block whose *identity* changes with the
    routed pdg, so never average it across pdg values.
    """
    names = []
    if net.generalist is not None:
        names.append("generalist")
    if net.experts is not None:
        names += [f"expert_{i}" for i in range(len(net.experts))]
    if net.specialists is not None:
        names.append("specialist")
    return names


def check_layout(net, config):
    """Fail loudly if the block partition or the ctx layout is not what we assume."""
    point_dim = config["model"]["feature_dim"]
    n_blocks = len(block_names(net))
    expected_in = n_blocks * point_dim
    actual_in = net.pooling[0]._layer.in_features
    if actual_in != expected_in:
        raise ValueError(
            f"pooling takes {actual_in} inputs but {n_blocks} branches of "
            f"width {point_dim} imply {expected_in}; block_names() is stale"
        )
    actual_out = net.pooling[-1]._layer.out_features
    if actual_out != point_dim:
        raise ValueError(
            f"pooling returns {actual_out} features, expected {point_dim}"
        )
    expected_ctx = TIME_DIM + config["model"]["cond_dim"]
    actual_ctx = net.pooling[0]._hyper_gate.in_features
    if actual_ctx != expected_ctx:
        raise ValueError(
            f"pooling context is {actual_ctx} wide, expected {expected_ctx} "
            f"(TIME_DIM={TIME_DIM} + cond_dim={config['model']['cond_dim']})"
        )
    return n_blocks, point_dim


def block_slice(index, point_dim):
    """Columns of the pooling input belonging to branch ``index``."""
    return slice(index * point_dim, (index + 1) * point_dim)


# --------------------------------------------------------------------------- #
# building and capturing the pooling input
# --------------------------------------------------------------------------- #
class capture_pooling_input:
    """Context manager that records the arguments handed to ``pooling[0]``.

    Using a hook rather than reimplementing ``forward`` means the captured
    tensors are by construction the ones the model actually pools.

    Examples
    --------
    >>> with capture_pooling_input(net) as captured:  # doctest: +SKIP
    ...     model.diffusion(noised, sigma, context=context)
    >>> captured.x.shape  # doctest: +SKIP
    """

    def __init__(self, net):
        self.net = net
        self.x = None
        self.ctx = None
        self._handle = None

    def _hook(self, module, args, kwargs):
        # ConcatSquashLinear.forward(self, ctx, x); _run_stack calls it by keyword
        ctx = kwargs.get("ctx", args[0] if len(args) > 0 else None)
        x = kwargs.get("x", args[1] if len(args) > 1 else None)
        if ctx is None or x is None:
            raise RuntimeError("could not read ctx/x from the pooling call")
        self.ctx = ctx.detach()
        self.x = x.detach()

    def __enter__(self):
        self._handle = self.net.pooling[0].register_forward_pre_hook(
            self._hook, with_kwargs=True
        )
        return self

    def __exit__(self, *exc_info):
        self._handle.remove()
        return False


def make_ctx_emb(net, context, sigma):
    """Rebuild ``ctx_emb`` for arbitrary context and sigma.

    Mirrors ``PointwiseNet_kDiffusion_v2.forward``: the time embedding comes
    first, the conditioning second.

    Parameters
    ----------
    context : torch.Tensor (G, cond_dim)
        Already preprocessed, i.e. what ``forward`` receives.
    sigma : torch.Tensor (G,)
        Strictly positive; ``forward`` takes ``sigma.log()``.

    Returns
    -------
    torch.Tensor (G, 1, TIME_DIM + cond_dim)
    """
    if torch.any(sigma <= 0):
        raise ValueError(
            "sigma must be strictly positive; get_sigmas_karras appends a "
            "trailing zero, drop it before calling this"
        )
    batch_size = context.shape[0]
    sigma = sigma.reshape(batch_size, 1, 1)
    c_noise = sigma.log() / 4
    time_emb = net.act(net.timestep_embed(c_noise))
    return torch.cat([time_emb, context.view(batch_size, 1, -1)], dim=-1)


def branch_outputs(model, net, points, sigma, context):
    """Run the branches and return the concatenated pooling input.

    The full ``Denoiser`` is called so the ``c_in`` preconditioning is applied
    exactly as during sampling.

    Parameters
    ----------
    points : torch.Tensor (G, P, point_dim)
        Noised points, *before* preconditioning.
    sigma : torch.Tensor (G,)
    context : torch.Tensor (G, cond_dim)
        Preprocessed conditioning.

    Returns
    -------
    x_cat : torch.Tensor (G, P, n_blocks * point_dim)
    ctx_emb : torch.Tensor (G, 1, TIME_DIM + cond_dim)
    """
    with evaluating(model), torch.no_grad():
        with capture_pooling_input(net) as captured:
            model.diffusion(points, sigma, context=context)
    rebuilt = make_ctx_emb(net, context, sigma)
    if not torch.allclose(rebuilt, captured.ctx, atol=1e-5, rtol=1e-4):
        raise RuntimeError(
            "make_ctx_emb disagrees with the context the model actually used; "
            "the ctx_emb layout in forward() has changed"
        )
    return captured.x, captured.ctx


def apply_pooling(net, x_cat, ctx_emb):
    """The pooling stack alone, on the same code path as ``forward``."""
    return net._run_stack(net.pooling, x_cat, ctx_emb)


# --------------------------------------------------------------------------- #
# measure 1: static, weights only
# --------------------------------------------------------------------------- #
def static_block_weight(net, n_blocks, point_dim):
    """Column norms of the first pooling layer, per block.

    Context free, data free.  This is the flat line every context-dependent
    measure should be compared against: if the study finds no context effect,
    this is all there is.

    Returns
    -------
    numpy.ndarray (n_blocks,)
    """
    weight = net.pooling[0]._layer.weight.detach()  # (hidden, n_blocks * point_dim)
    return np.array(
        [
            float(torch.linalg.norm(weight[:, block_slice(k, point_dim)]))
            for k in range(n_blocks)
        ]
    )


# --------------------------------------------------------------------------- #
# measure 2: exact local Jacobian
# --------------------------------------------------------------------------- #
def block_jacobian(net, x_cat, ctx_emb):
    """Jacobian of the pooling output with respect to its input, per point.

    Cheap because the stack is pointwise: one backward pass per output feature
    (``point_dim``, normally 4) yields the whole batch's Jacobian.

    Returns
    -------
    jac : torch.Tensor (G, P, point_dim, n_blocks * point_dim)
    out : torch.Tensor (G, P, point_dim)
    """
    x_cat = x_cat.detach().clone().requires_grad_(True)
    out = apply_pooling(net, x_cat, ctx_emb)
    d_out = out.shape[-1]
    rows = []
    for j in range(d_out):
        (grad,) = torch.autograd.grad(
            out[..., j].sum(), x_cat, retain_graph=(j < d_out - 1)
        )
        rows.append(grad)
    return torch.stack(rows, dim=-2).detach(), out.detach()


def jacobian_block_norms(jac, n_blocks, point_dim, per_output=False):
    """Frobenius norm of each block's Jacobian columns.

    Returns
    -------
    torch.Tensor
        ``(G, P, n_blocks)``, or ``(G, P, point_dim, n_blocks)`` when
        ``per_output``, which separates e.g. "who decides the energy" from
        "who decides the position".
    """
    parts = []
    for k in range(n_blocks):
        columns = jac[..., block_slice(k, point_dim)]
        if per_output:
            parts.append(torch.linalg.norm(columns, dim=-1))
        else:
            parts.append(torch.linalg.norm(columns.flatten(start_dim=-2), dim=-1))
    return torch.stack(parts, dim=-1)


def linear_contributions(jac, x_cat, n_blocks, point_dim):
    """The exactly additive split ``out = sum_k J_k x_k + offset``.

    The offset is the context-only part of the affine map (the hyper-bias path);
    it is returned so the decomposition can be checked to close.

    Returns
    -------
    contributions : torch.Tensor (G, P, n_blocks, point_dim)
    """
    parts = []
    for k in range(n_blocks):
        sl = block_slice(k, point_dim)
        parts.append((jac[..., sl] * x_cat.unsqueeze(-2)[..., sl]).sum(-1))
    return torch.stack(parts, dim=-2)


# --------------------------------------------------------------------------- #
# measure 3 and 4: ablation and exact Shapley
# --------------------------------------------------------------------------- #
def mean_baseline(x_cat, point_mask=None):
    """Per-block mean over points: "this branch says whatever it usually says"."""
    if point_mask is None:
        return x_cat.mean(dim=(0, 1), keepdim=True)
    weights = point_mask.unsqueeze(-1).to(x_cat.dtype)
    return (x_cat * weights).sum(dim=(0, 1), keepdim=True) / weights.sum().clamp(min=1)


def _keep_blocks(x_cat, baseline, subset, n_blocks, point_dim):
    """``x_cat`` with every block outside ``subset`` replaced by the baseline."""
    out = baseline.expand_as(x_cat).clone()
    for k in subset:
        sl = block_slice(k, point_dim)
        out[..., sl] = x_cat[..., sl]
    return out


def subset_outputs(net, x_cat, ctx_emb, baseline, n_blocks, point_dim):
    """Pooling output for every subset of blocks. ``2 ** n_blocks`` forward passes."""
    if n_blocks > MAX_EXACT_SHAPLEY_BLOCKS:
        raise ValueError(
            f"{n_blocks} branches means {2 ** n_blocks} forward passes; "
            "use ablation_effects instead, or sample the subsets"
        )
    values = {}
    with torch.no_grad():
        for size in range(n_blocks + 1):
            for subset in itertools.combinations(range(n_blocks), size):
                masked = _keep_blocks(x_cat, baseline, subset, n_blocks, point_dim)
                values[subset] = apply_pooling(net, masked, ctx_emb)
    return values


def shapley_blocks(values, n_blocks):
    """Exact Shapley value of each block, per output feature.

    Model faithful (no linearisation) and exactly additive:
    ``sum_k phi_k == f(all blocks) - f(no blocks)``.

    Returns
    -------
    torch.Tensor (G, P, n_blocks, point_dim)
    """
    full = values[tuple(range(n_blocks))]
    phi = torch.zeros(
        full.shape[:-1] + (n_blocks, full.shape[-1]),
        dtype=full.dtype,
        device=full.device,
    )
    total = math.factorial(n_blocks)
    for k in range(n_blocks):
        others = [i for i in range(n_blocks) if i != k]
        for size in range(n_blocks):
            weight = math.factorial(size) * math.factorial(n_blocks - size - 1) / total
            for subset in itertools.combinations(others, size):
                with_k = tuple(sorted(subset + (k,)))
                phi[..., k, :] += weight * (values[with_k] - values[subset])
    return phi


def ablation_effects(values, n_blocks):
    """Leave-one-out and only-one effects, as norms.

    ``leave_out[k]`` is how much the output moves when block ``k`` alone is
    removed; ``only[k]`` is how far block ``k`` alone moves it from the
    all-baseline output.  The two disagree exactly when the branches interact.

    Returns
    -------
    (torch.Tensor, torch.Tensor), each (G, P, n_blocks)
    """
    everything = tuple(range(n_blocks))
    full = values[everything]
    empty = values[()]
    leave_out, only = [], []
    for k in range(n_blocks):
        without = tuple(i for i in everything if i != k)
        leave_out.append(torch.linalg.norm(full - values[without], dim=-1))
        only.append(torch.linalg.norm(values[(k,)] - empty, dim=-1))
    return torch.stack(leave_out, dim=-1), torch.stack(only, dim=-1)


# --------------------------------------------------------------------------- #
# turning per-point numbers into per-context shares
# --------------------------------------------------------------------------- #
def to_shares(per_block, point_mask=None, pool="mean"):
    """Normalise a positive per-block quantity to sum to one over blocks.

    Parameters
    ----------
    per_block : torch.Tensor (G, P, n_blocks)
    point_mask : torch.Tensor (G, P) or None
        False marks padding, which is excluded.
    pool : {"mean", "total"}
        ``"mean"`` averages the per-point shares, giving every point equal say
        and shrugging off a few extreme points.  ``"total"`` takes the share of
        the summed magnitudes, so loud points dominate.  They can disagree
        sharply when one branch is small everywhere but huge on a few points --
        that disagreement is itself a result worth reporting.

    Returns
    -------
    numpy.ndarray (G, n_blocks)
    """
    per_block = per_block.abs()
    weights = (
        torch.ones_like(per_block[..., 0])
        if point_mask is None
        else point_mask.to(per_block.dtype)
    ).unsqueeze(-1)
    if pool == "total":
        totals = (per_block * weights).sum(dim=1)
        return (totals / totals.sum(-1, keepdim=True).clamp(min=1e-30)).cpu().numpy()
    if pool != "mean":
        raise ValueError(f"pool must be 'mean' or 'total', not {pool!r}")
    shares = per_block / per_block.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    summed = (shares * weights).sum(dim=1)
    return (summed / weights.sum(dim=1).clamp(min=1)).cpu().numpy()


def to_output_shares(per_block_per_output, point_mask=None):
    """Per-block share, computed separately for each output feature.

    This is where specialisation usually shows up first: one branch can own the
    energy channel while another owns position, and that is invisible once the
    output features are collapsed into a single norm.

    Parameters
    ----------
    per_block_per_output : torch.Tensor (G, P, point_dim, n_blocks)

    Returns
    -------
    numpy.ndarray (G, point_dim, n_blocks)
    """
    values = per_block_per_output.abs()
    shares = values / values.sum(dim=-1, keepdim=True).clamp(min=1e-30)
    if point_mask is None:
        return shares.mean(dim=1).cpu().numpy()
    weights = point_mask.to(shares.dtype)[:, :, None, None]
    summed = (shares * weights).sum(dim=1)
    return (summed / weights.sum(dim=1).clamp(min=1)).cpu().numpy()


def offset_fraction(measures, point_mask=None):
    """Fraction of the output carried by the context-only offset, not by any branch.

    ``out = sum_k J_k x_k + offset``.  When this approaches one the pooling stack
    is largely ignoring its inputs and emitting a context-driven constant, and
    every block share becomes a ratio between negligible quantities.  Check this
    before reading anything else -- it is invisible to measures normalised over
    blocks.

    Returns
    -------
    numpy.ndarray (G,)
    """
    total = measures["contribution"].sum(-1)
    fraction = measures["offset_norm"] / (measures["offset_norm"] + total).clamp(
        min=1e-30
    )
    if point_mask is None:
        return fraction.mean(dim=1).cpu().numpy()
    weights = point_mask.to(fraction.dtype)
    summed = (fraction * weights).sum(dim=1)
    return (summed / weights.sum(dim=1).clamp(min=1)).cpu().numpy()


def bootstrap_share_ci(
    per_block, point_mask=None, n_resamples=200, seed=0, alpha=0.32, pool="mean"
):
    """Percentile bootstrap over points: the error bar on each plotted share.

    This is sampling uncertainty on the mean only.  It is the right error bar
    for the plot, but it is *not* the right null for a frozen-probe sweep,
    where the same points are reused at every grid point and this uncertainty
    is common to all of them.  See :func:`matched_random_contexts`.

    Returns
    -------
    (numpy.ndarray, numpy.ndarray), each (G, n_blocks)
    """
    rng = np.random.default_rng(seed)
    n_points = per_block.shape[1]
    draws = []
    for _ in range(n_resamples):
        idx = torch.as_tensor(
            rng.integers(0, n_points, n_points), device=per_block.device
        )
        sub_mask = None if point_mask is None else point_mask[:, idx]
        draws.append(to_shares(per_block[:, idx], sub_mask, pool=pool))
    draws = np.stack(draws)
    return (
        np.quantile(draws, alpha / 2, axis=0),
        np.quantile(draws, 1 - alpha / 2, axis=0),
    )


def matched_random_contexts(context_grid, n_draws=20, seed=0):
    """Placebo contexts: same displacement sizes as the real sweep, random directions.

    The null for a frozen-probe sweep is not "no change" -- moving the context
    at all moves the gates a little.  The question is whether moving *this*
    feature moves the routing more than an arbitrary context perturbation of the
    same magnitude does.  Each draw walks the same distances from the grid's
    first row as the real sweep, in a random direction.

    Feed the result through the same measurement path as the real sweep and
    compare ``numpy.ptp`` of the shares.  If the real range does not clear the
    upper tail of the placebo ranges, the routing responds to context size, not
    to the feature's meaning.

    Caveat: random directions ignore the structure of the context, so a draw
    can break the pdg one-hot or push the direction off the unit sphere.  If
    off-manifold contexts destabilise the gates this null is conservative, and
    a feature has to beat a somewhat inflated bar.

    Parameters
    ----------
    context_grid : torch.Tensor (G, cond_dim)
        Preprocessed contexts, as handed to :func:`make_ctx_emb`.

    Returns
    -------
    torch.Tensor (n_draws, G, cond_dim)
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    base = context_grid[:1]
    steps = torch.linalg.norm(context_grid - base, dim=-1, keepdim=True)  # (G, 1)
    draws = []
    for _ in range(n_draws):
        direction = torch.randn(
            context_grid.shape, generator=generator, dtype=torch.float32
        ).to(context_grid.device, dtype=context_grid.dtype)
        direction = direction / torch.linalg.norm(direction, dim=-1, keepdim=True)
        draws.append(base + steps * direction)
    return torch.stack(draws)


def summarise_sweep(shares, low=None, high=None, placebo_shares=None):
    """Reduce a sweep to a few numbers per block.

    ``range`` is the practical answer -- how many share points the block moves
    across the sweep.  ``significance`` and ``placebo_p`` say whether that
    movement is real, by two different arguments:

    * ``significance`` divides the range by the bootstrap width.  It grows like
      the square root of the probe size, so with tens of thousands of points
      everything is "significant".  Read ``range`` to decide whether it matters.
    * ``placebo_p`` is the fraction of matched random-direction sweeps whose
      range met or beat the real one.  Small means the routing responds to this
      feature specifically, not merely to context motion.

    Parameters
    ----------
    shares : numpy.ndarray (G, n_blocks)
    low, high : numpy.ndarray (G, n_blocks), optional
        From :func:`bootstrap_share_ci`.
    placebo_shares : numpy.ndarray (n_draws, G, n_blocks), optional
        Shares measured on :func:`matched_random_contexts`.

    Returns
    -------
    dict of numpy.ndarray, each (n_blocks,)
    """
    observed = np.ptp(shares, axis=0)
    summary = {
        "range": observed,
        "min": shares.min(axis=0),
        "max": shares.max(axis=0),
        "argmax": shares.argmax(axis=0),
        "mean": shares.mean(axis=0),
    }
    if low is not None and high is not None:
        width = np.mean(high - low, axis=0)
        summary["significance"] = observed / np.clip(width, 1e-12, None)
    if placebo_shares is not None:
        null_ranges = np.ptp(placebo_shares, axis=1)  # (n_draws, n_blocks)
        summary["placebo_range_95"] = np.quantile(null_ranges, 0.95, axis=0)
        summary["placebo_p"] = (null_ranges >= observed).mean(axis=0)
    return summary


# --------------------------------------------------------------------------- #
# context grids
# --------------------------------------------------------------------------- #
def sigma_grid(config, n_sigma=None):
    """Karras sigmas actually used at sampling time, with the trailing zero dropped."""
    import k_diffusion

    steps = n_sigma or config["num_steps"]
    sigmas = k_diffusion.sampling.get_sigmas_karras(
        steps,
        config["model"]["sigma_min"],
        config["model"]["sigma_max"],
        rho=config["model"]["rho"],
    )
    return sigmas[:-1]  # the last entry is 0 and sigma.log() would be -inf


def base_context(config, cond_raw):
    """A representative conditioning row that stays on the data manifold.

    A plain column-wise median is not safe here.  Each pdg one-hot column is
    zero for most events, so its median is zero and the whole one-hot block
    collapses -- ``_run_specialists`` takes an ``argmax`` of that block, so
    every event would silently route to specialist 0.  A component-wise median
    of a direction vector is likewise not a unit vector.

    So: median for scalars, the modal category for the pdg one-hot, and a
    direction rescaled to the median observed length.

    Parameters
    ----------
    cond_raw : numpy.ndarray (n_events, cond_dim)

    Returns
    -------
    numpy.ndarray (cond_dim,)
    """
    base = np.median(cond_raw, axis=0)
    for feature in config["model"]["cond_features"]:
        span = cond_feature_range(config, feature)
        if span is None:
            continue
        start, stop = span
        if feature == "incident_pdg":
            base[start:stop] = 0.0
            base[start + int(np.argmax(cond_raw[:, start:stop].sum(axis=0)))] = 1.0
        elif stop - start > 1:
            block = base[start:stop]
            length = np.linalg.norm(block)
            if length > 0:
                typical = np.median(np.linalg.norm(cond_raw[:, start:stop], axis=1))
                base[start:stop] = block / length * typical
    return base


def sweep_conditioning(config, cond_raw, feature, n_grid=9, quantile_range=(0.02, 0.98)):
    """Vary one conditioning feature over its empirical range, holding the rest fixed.

    Grids are built from the data, not from an arbitrary interval, so every
    context handed to the network is one the training distribution supports.
    Direction is swept along observed vectors rather than component by
    component, since sweeping components independently leaves the unit sphere
    and produces contexts the model has never seen.

    Parameters
    ----------
    cond_raw : numpy.ndarray (n_events, cond_dim)
        Conditioning as ``Sampler.get_cond`` returns it: physical units, pdg
        already one-hot expanded, *not* yet preprocessed.
    feature : str
        A member of ``config["model"]["cond_features"]``.

    Returns
    -------
    grid_values : numpy.ndarray (G,)
        Human readable x axis: the energy, the angle in degrees, or the pdg code.
    cond_grid : numpy.ndarray (G, cond_dim)
        Feed through ``preprocess_conditioning.forward`` before use.
    """
    span = cond_feature_range(config, feature)
    if span is None:
        raise ValueError(f"{feature} is not in config['model']['cond_features']")
    start, stop = span
    base = base_context(config, cond_raw)

    if feature == "incident_pdg":
        pdgs = np.array(config["simulate_pdgs"])
        cond_grid = np.tile(base, (len(pdgs), 1))
        cond_grid[:, start:stop] = np.eye(len(pdgs))
        return pdgs.astype(float), cond_grid

    if stop - start == 1:
        quantiles = np.linspace(*quantile_range, n_grid)
        values = np.quantile(cond_raw[:, start], quantiles)
        cond_grid = np.tile(base, (n_grid, 1))
        cond_grid[:, start] = values
        return values, cond_grid

    # a vector feature: walk along observed vectors, ordered by angle to the mean
    vectors = cond_raw[:, start:stop]
    mean_direction = vectors.mean(axis=0)
    mean_direction = mean_direction / np.linalg.norm(mean_direction)
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True).clip(1e-12)
    angles = np.degrees(np.arccos(np.clip(unit @ mean_direction, -1, 1)))
    order = np.argsort(angles)
    picks = order[
        np.clip(
            (np.linspace(*quantile_range, n_grid) * len(order)).astype(int),
            0,
            len(order) - 1,
        )
    ]
    cond_grid = np.tile(base, (n_grid, 1))
    cond_grid[:, start:stop] = vectors[picks]
    return angles[picks], cond_grid


# --------------------------------------------------------------------------- #
# probes
# --------------------------------------------------------------------------- #
def probe_points(sampler, target, n_points=2000, seed=0):
    """A pool of real, preprocessed, unpadded points to push through the branches.

    Padding is dropped before sampling, so every probe point is a point the
    model was actually trained on.

    Parameters
    ----------
    target : numpy.ndarray (n_events, max_points, point_dim)
        Raw points from ``Sampler.get_cond(..., return_target=True)``.

    Returns
    -------
    torch.Tensor (1, n_points, point_dim)
    """
    real = target[target[..., 3] > 0]
    if len(real) == 0:
        raise ValueError("no unpadded points in the target")
    rng = np.random.default_rng(seed)
    picked = real[rng.choice(len(real), min(n_points, len(real)), replace=False)]
    device = sampler.config["device"]
    points = torch.from_numpy(picked).to(device, dtype=sampler.datatype)
    return sampler.preprocess_features.forward(points).unsqueeze(0)


def noise_points(points, sigma, seed=0):
    """Karras forward noising, with one shared noise draw across every context.

    Sharing the draw makes the sweep paired: two grid points differ by their
    context and nothing else.

    Parameters
    ----------
    points : torch.Tensor (1, P, point_dim)
    sigma : torch.Tensor (G,)

    Returns
    -------
    torch.Tensor (G, P, point_dim)
    """
    generator = torch.Generator(device="cpu").manual_seed(seed)
    noise = torch.randn(points.shape[1:], generator=generator, dtype=points.dtype)
    noise = noise.to(points.device).unsqueeze(0)
    return points + noise * sigma.reshape(-1, 1, 1)


def measure_blocks(net, x_cat, ctx_emb, n_blocks, point_dim, with_shapley=True):
    """Every per-block measure at once, for one set of contexts.

    Returns
    -------
    dict
        ``jac_norm`` (G, P, n_blocks), ``jac_norm_per_output``
        (G, P, point_dim, n_blocks), ``contribution`` (G, P, n_blocks) as the
        norm of ``J_k x_k``, and when ``with_shapley`` also ``shapley``
        (G, P, n_blocks), ``leave_out`` and ``only`` (G, P, n_blocks).
    """
    jac, out = block_jacobian(net, x_cat, ctx_emb)
    contributions = linear_contributions(jac, x_cat, n_blocks, point_dim)
    measures = {
        "jac_norm": jacobian_block_norms(jac, n_blocks, point_dim),
        "jac_norm_per_output": jacobian_block_norms(
            jac, n_blocks, point_dim, per_output=True
        ),
        "contribution": torch.linalg.norm(contributions, dim=-1),
        "output_norm": torch.linalg.norm(out, dim=-1),
        # what the output would be with no branch signal at all: the pure
        # context offset.  If this dwarfs the contributions, the pooling stack
        # is mostly ignoring its inputs and writing a context-driven constant.
        "offset_norm": torch.linalg.norm(out - contributions.sum(-2), dim=-1),
    }
    if with_shapley:
        baseline = mean_baseline(x_cat)
        values = subset_outputs(net, x_cat, ctx_emb, baseline, n_blocks, point_dim)
        measures["shapley"] = torch.linalg.norm(
            shapley_blocks(values, n_blocks), dim=-1
        )
        leave_out, only = ablation_effects(values, n_blocks)
        measures["leave_out"] = leave_out
        measures["only"] = only
    return measures


def per_event_points(sampler, target, n_points=256, seed=0):
    """Up to ``n_points`` real points per event, gathered forward, plus a mask.

    Gathering means the padding convention (front or back) stops mattering and
    the tensors stay small enough to differentiate.

    Returns
    -------
    (torch.Tensor (E, n_points, point_dim), torch.Tensor (E, n_points))
    """
    rng = np.random.default_rng(seed)
    n_events, _, point_dim = target.shape
    gathered = np.zeros((n_events, n_points, point_dim), dtype=target.dtype)
    mask = np.zeros((n_events, n_points), dtype=bool)
    for event in range(n_events):
        real = np.flatnonzero(target[event, :, 3] > 0)
        if len(real) == 0:
            continue
        take = min(n_points, len(real))
        gathered[event, :take] = target[event, rng.choice(real, take, replace=False)]
        mask[event, :take] = True
    device = sampler.config["device"]
    points = torch.from_numpy(gathered).to(device, dtype=sampler.datatype)
    points = sampler.preprocess_features.forward(points)
    return points, torch.from_numpy(mask).to(device)


# --------------------------------------------------------------------------- #
# the study
# --------------------------------------------------------------------------- #
def run_study(
    model_path,
    data_part="test",
    n_events=256,
    n_probe_points=2000,
    n_event_points=256,
    features=None,
    n_grid=9,
    n_sigma=6,
    arms=("frozen", "coupled", "observational"),
    with_shapley=True,
    n_placebo=20,
    seed=0,
    printer=print,
):
    """Measure branch influence against conditioning, in up to three arms.

    frozen
        Branch outputs captured once at a reference context and then held
        fixed while the pooling context is swept.  Isolates the routing map:
        any movement is the pooling stack changing its mind about inputs that
        did not change.  The ``(x, context)`` pairs are off-distribution, so
        read this as a property of the learned function, not of deployment.
    coupled
        Branch outputs recomputed at every swept context, from the same points
        and the same noise draw.  On-distribution, and answers the question that
        matters in practice: as the conditioning moves, which branch ends up
        driving the output?  It cannot separate "pooling re-weighted" from
        "the branch outputs themselves changed".
    observational
        Real events with their own conditioning.  No intervention, so features
        stay correlated exactly as they are in the data; this is what actually
        happens at sampling time, confounders and all.

    Running frozen and coupled together is the point: agreement means the
    pooling stack is doing the routing, disagreement means the branches are.

    Parameters
    ----------
    model_path : str
        A ``.pt`` checkpoint in a run's ``checkpoints`` directory.
    features : list of str or None
        Conditioning features to sweep; defaults to all of them.

    Returns
    -------
    dict
        Nested results; pass to :func:`save_study` or :func:`plot_study`.
    """
    from .inference import Sampler

    sampler = Sampler.from_model_path(model_path)
    config = sampler.config
    model = sampler.model
    net = get_v2_net(model)
    n_blocks, point_dim = check_layout(net, config)
    names = block_names(net)
    printer(f"{n_blocks} branches: {', '.join(names)}")

    if net.specialists is not None:
        printer(
            "note: the specialist block changes identity with the routed pdg, "
            "so do not read its curve across an incident_pdg sweep as one network"
        )

    printer(f"Reading {n_events} events from the {data_part} set")
    cond_raw, _, target = sampler.get_cond(
        data_part, total_size=n_events, return_target=True
    )
    device = config["device"]
    dtype = sampler.datatype

    def preprocess(cond_array):
        tensor = torch.from_numpy(np.asarray(cond_array)).to(device, dtype=dtype)
        return sampler.preprocess_conditioning.forward(tensor)

    sigmas = sigma_grid(config, n_sigma).to(device, dtype=dtype)
    printer(f"sigma grid: {np.round(sigmas.cpu().numpy(), 4)}")

    results = {
        "block_names": names,
        "static_weight": static_block_weight(net, n_blocks, point_dim),
        "sigma_grid": sigmas.cpu().numpy(),
        "config_model": dict(config["model"]),
        "sweeps": {},
    }

    probe = probe_points(sampler, target, n_probe_points, seed=seed)
    reference_context = preprocess(base_context(config, cond_raw)[None])
    reference_sigma = sigmas[len(sigmas) // 2].reshape(1)
    frozen_x = branch_outputs(
        model, net, noise_points(probe, reference_sigma, seed), reference_sigma,
        reference_context,
    )[0]
    printer(
        f"probe: {probe.shape[1]} points, frozen branch outputs taken at "
        f"sigma={float(reference_sigma):.4g} and the median context"
    )

    features = features or list(config["model"]["cond_features"])
    n_steps = len(sigmas)

    for feature in features:
        printer(f"--- sweeping {feature} ---")
        grid, cond_grid = sweep_conditioning(config, cond_raw, feature, n_grid)
        n_points_grid = len(grid)
        # every (sigma, grid) pair as one flat batch of contexts
        context = preprocess(np.repeat(cond_grid[None], n_steps, axis=0).reshape(
            n_steps * n_points_grid, -1
        ))
        row_sigma = sigmas.repeat_interleave(n_points_grid)
        entry = {"grid": grid}

        for arm in arms:
            if arm == "observational":
                continue
            if arm == "frozen":
                x_cat = frozen_x.expand(len(row_sigma), -1, -1)
            else:
                x_cat = branch_outputs(
                    model, net,
                    noise_points(probe, row_sigma, seed),
                    row_sigma, context,
                )[0]
            ctx_emb = make_ctx_emb(net, context, row_sigma)
            measures = measure_blocks(
                net, x_cat, ctx_emb, n_blocks, point_dim, with_shapley
            )
            arm_result = {}
            for key, value in measures.items():
                if value.ndim != 3:
                    continue
                shares = to_shares(value).reshape(n_steps, n_points_grid, n_blocks)
                arm_result[f"{key}_share"] = shares
            # these two are not (G, P, n_blocks) and would otherwise be dropped,
            # yet they are the first things worth looking at
            arm_result["jac_norm_share_per_output"] = to_output_shares(
                measures["jac_norm_per_output"]
            ).reshape(n_steps, n_points_grid, point_dim, n_blocks)
            arm_result["offset_fraction"] = offset_fraction(measures).reshape(
                n_steps, n_points_grid
            )
            low, high = bootstrap_share_ci(measures["jac_norm"], seed=seed)
            arm_result["jac_norm_share_low"] = low.reshape(
                n_steps, n_points_grid, n_blocks
            )
            arm_result["jac_norm_share_high"] = high.reshape(
                n_steps, n_points_grid, n_blocks
            )

            if arm == "frozen" and n_placebo and n_points_grid > 1:
                # one null per sigma, so each step is judged against a placebo
                # measured under its own noise level
                placebo = []
                for draw in matched_random_contexts(
                    preprocess(cond_grid), n_draws=n_placebo, seed=seed
                ):
                    draw_context = draw.repeat(n_steps, 1)
                    draw_jac, _ = block_jacobian(
                        net, x_cat,
                        make_ctx_emb(net, draw_context, row_sigma),
                    )
                    placebo.append(
                        to_shares(
                            jacobian_block_norms(draw_jac, n_blocks, point_dim)
                        ).reshape(n_steps, n_points_grid, n_blocks)
                    )
                arm_result["placebo_shares"] = np.stack(placebo)

            placebo_shares = arm_result.get("placebo_shares")
            low_grid = low.reshape(n_steps, n_points_grid, n_blocks)
            high_grid = high.reshape(n_steps, n_points_grid, n_blocks)
            arm_result["summary"] = [
                summarise_sweep(
                    arm_result["jac_norm_share"][step],
                    low_grid[step],
                    high_grid[step],
                    None if placebo_shares is None else placebo_shares[:, step],
                )
                for step in range(n_steps)
            ]
            entry[arm] = arm_result
            printer(
                f"  {arm}: share range per block = "
                f"{np.round(np.ptp(arm_result['jac_norm_share'], axis=1).max(0), 4)}"
            )
        results["sweeps"][feature] = entry

    if "observational" in arms:
        printer("--- observational arm ---")
        event_points, event_mask = per_event_points(
            sampler, target, n_event_points, seed=seed
        )
        event_context = preprocess(cond_raw)
        shares_by_sigma = []
        for sigma in sigmas:
            row_sigma = sigma.repeat(event_context.shape[0])
            x_cat = branch_outputs(
                model, net, noise_points(event_points, row_sigma, seed),
                row_sigma, event_context,
            )[0]
            ctx_emb = make_ctx_emb(net, event_context, row_sigma)
            jac, _ = block_jacobian(net, x_cat, ctx_emb)
            norms = jacobian_block_norms(jac, n_blocks, point_dim)
            shares_by_sigma.append(to_shares(norms, event_mask))
        results["observational"] = {
            "cond_raw": cond_raw,
            "jac_norm_share": np.stack(shares_by_sigma),  # (n_sigma, n_events, n_blocks)
        }
    return results


# --------------------------------------------------------------------------- #
# saving and plotting
# --------------------------------------------------------------------------- #
def default_output_path(model_path, suffix="_pooling_influence"):
    """``<checkpoint>.pt`` -> ``<checkpoint>_pooling_influence.npz``."""
    return model_path[:-3] + suffix + ".npz"


def save_study(results, output_path):
    """Flatten the nested results into a single npz."""
    flat = {
        "block_names": np.array(results["block_names"]),
        "static_weight": results["static_weight"],
        "sigma_grid": results["sigma_grid"],
    }
    for feature, entry in results["sweeps"].items():
        flat[f"{feature}/grid"] = entry["grid"]
        for arm, arm_result in entry.items():
            if arm == "grid":
                continue
            for key, value in arm_result.items():
                if key == "summary":
                    for name in value[0]:
                        flat[f"{feature}/{arm}/summary/{name}"] = np.stack(
                            [step[name] for step in value]
                        )
                else:
                    flat[f"{feature}/{arm}/{key}"] = np.asarray(value)
    for key, value in results.get("observational", {}).items():
        flat[f"observational/{key}"] = np.asarray(value)
    np.savez_compressed(output_path, **flat)
    return output_path


def plot_sweep(results, feature, arm="frozen", measure="jac_norm_share", ax_width=4):
    """Share of influence against one conditioning feature, one panel per sigma.

    The dashed line is the context-free baseline from the first pooling layer's
    column norms: where the curves sit on it, the context is not routing.
    The grey band, when present, is the middle 90 percent of the matched
    random-direction placebo, drawn around each block's mean share -- a curve
    that stays inside it has not beaten an arbitrary context nudge.
    """
    import matplotlib.pyplot as plt

    from .plotting import nice_hex

    entry = results["sweeps"][feature]
    if arm not in entry:
        raise KeyError(f"arm {arm!r} was not run for {feature!r}")
    shares = entry[arm][measure]  # (n_sigma, n_grid, n_blocks)
    grid = entry["grid"]
    names = results["block_names"]
    sigmas = results["sigma_grid"]
    colours = (nice_hex[0] + nice_hex[1] + nice_hex[2])[: len(names)]
    static = results["static_weight"] / results["static_weight"].sum()

    n_steps = shares.shape[0]
    fig, axes = plt.subplots(
        1, n_steps, figsize=(ax_width * n_steps, 3.6), sharey=True, squeeze=False
    )
    axes = axes[0]
    low = entry[arm].get(f"{measure}_low")
    high = entry[arm].get(f"{measure}_high")
    placebo = entry[arm].get("placebo_shares")

    for step, ax in enumerate(axes):
        for block, (name, colour) in enumerate(zip(names, colours)):
            ax.plot(grid, shares[step, :, block], color=colour, marker="o",
                    markersize=3, label=name)
            if low is not None:
                ax.fill_between(grid, low[step, :, block], high[step, :, block],
                                color=colour, alpha=0.25, linewidth=0)
            if placebo is not None:
                centre = shares[step, :, block].mean()
                spread = np.ptp(placebo[:, step, :, block], axis=1)
                half = np.quantile(spread, 0.95) / 2
                ax.axhspan(centre - half, centre + half, color=colour, alpha=0.07)
            ax.axhline(static[block], color=colour, linestyle="--", linewidth=0.8,
                       alpha=0.6)
        ax.set_title(f"$\\sigma$ = {sigmas[step]:.3g}")
        ax.set_xlabel(feature)
        if feature == "incident_energy":
            ax.set_xscale("log")
    axes[0].set_ylabel(f"share of influence ({measure.replace('_share', '')})")
    axes[-1].legend(fontsize="small")
    fig.suptitle(f"{arm} arm: branch influence vs {feature}")
    fig.tight_layout()
    return fig


def plot_sigma_heatmap(results, feature, arm="frozen", measure="jac_norm_share"):
    """Blocks against sigma at the middle of the sweep: who owns which noise scale."""
    import matplotlib.pyplot as plt

    entry = results["sweeps"][feature]
    shares = entry[arm][measure]
    middle = shares.shape[1] // 2
    fig, ax = plt.subplots(figsize=(6, 3.5))
    mesh = ax.pcolormesh(shares[:, middle, :].T, cmap="magma")
    ax.set_yticks(np.arange(len(results["block_names"])) + 0.5)
    ax.set_yticklabels(results["block_names"])
    ax.set_xticks(np.arange(len(results["sigma_grid"])) + 0.5)
    ax.set_xticklabels([f"{s:.3g}" for s in results["sigma_grid"]], rotation=45)
    ax.set_xlabel("$\\sigma$")
    fig.colorbar(mesh, ax=ax, label="share of influence")
    fig.tight_layout()
    return fig


def plot_observational(results, config, sigma_index=None, n_bins=8):
    """Per-event influence share against each real conditioning feature.

    No intervention here, so the features stay correlated as they are in the
    data.  A trend that shows up here but not in the frozen or coupled sweeps
    is a confounder, not routing.
    """
    import matplotlib.pyplot as plt

    from .plotting import nice_hex

    observational = results["observational"]
    shares = observational["jac_norm_share"]
    if sigma_index is None:
        sigma_index = shares.shape[0] // 2
    shares = shares[sigma_index]
    cond_raw = observational["cond_raw"]
    names = results["block_names"]
    colours = (nice_hex[0] + nice_hex[1] + nice_hex[2])[: len(names)]

    features = list(config["model"]["cond_features"])
    fig, axes = plt.subplots(
        1, len(features), figsize=(4 * len(features), 3.6), sharey=True, squeeze=False
    )
    for ax, feature in zip(axes[0], features):
        start, stop = cond_feature_range(config, feature)
        if stop - start == 1:
            values = cond_raw[:, start]
        elif feature == "incident_pdg":
            values = np.array(config["simulate_pdgs"])[
                cond_raw[:, start:stop].argmax(axis=1)
            ].astype(float)
        else:
            unit = cond_raw[:, start:stop]
            unit = unit / np.linalg.norm(unit, axis=1, keepdims=True).clip(1e-12)
            mean_direction = unit.mean(axis=0)
            mean_direction /= np.linalg.norm(mean_direction)
            values = np.degrees(np.arccos(np.clip(unit @ mean_direction, -1, 1)))
        edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        which = np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)
        centres = 0.5 * (edges[:-1] + edges[1:])
        for block, (name, colour) in enumerate(zip(names, colours)):
            binned = [shares[which == b, block] for b in range(len(centres))]
            means = np.array([b.mean() if len(b) else np.nan for b in binned])
            errs = np.array(
                [b.std() / max(np.sqrt(len(b)), 1) if len(b) else np.nan
                 for b in binned]
            )
            ax.errorbar(centres, means, yerr=errs, color=colour, marker="o",
                        markersize=3, label=name)
        ax.set_xlabel(feature)
        if feature == "incident_energy":
            ax.set_xscale("log")
    axes[0][0].set_ylabel("share of influence")
    axes[0][-1].legend(fontsize="small")
    fig.suptitle(
        f"observational arm, $\\sigma$ = {results['sigma_grid'][sigma_index]:.3g}"
    )
    fig.tight_layout()
    return fig


def plot_per_output(results, feature, arm="frozen", grid_index=None):
    """Which branch drives which output feature, across the sweep.

    A branch owning one channel and not others is the clearest signature of
    real specialisation, and it is invisible in the collapsed share.
    """
    import matplotlib.pyplot as plt

    entry = results["sweeps"][feature]
    shares = entry[arm]["jac_norm_share_per_output"]  # (n_sigma, n_grid, d, n_blocks)
    if grid_index is None:
        grid_index = shares.shape[1] // 2
    names = results["block_names"]
    labels = ["x", "y", "z", "E"][: shares.shape[2]] or None
    n_steps = shares.shape[0]
    fig, axes = plt.subplots(
        1, n_steps, figsize=(3.2 * n_steps, 3.2), sharey=True, squeeze=False
    )
    for step, ax in enumerate(axes[0]):
        mesh = ax.pcolormesh(shares[step, grid_index].T, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(np.arange(shares.shape[2]) + 0.5)
        ax.set_xticklabels(labels or np.arange(shares.shape[2]))
        ax.set_title(f"$\\sigma$ = {results['sigma_grid'][step]:.3g}")
    axes[0][0].set_yticks(np.arange(len(names)) + 0.5)
    axes[0][0].set_yticklabels(names)
    fig.colorbar(mesh, ax=axes[0].tolist(), label="share of influence")
    fig.suptitle(f"{arm} arm: branch influence per output feature ({feature})")
    return fig
