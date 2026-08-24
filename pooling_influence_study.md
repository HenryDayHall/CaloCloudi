# Does the context change which branch the pooling stack listens to?

A study design for `PointwiseNet_kDiffusion_v2` in `src/diffusion.py`.

## 1. The question

`PointwiseNet_kDiffusion_v2` runs several branches on the same points, concatenates
their outputs on the feature axis, and reduces them back to `point_dim` with a
pooling stack that is itself conditioned on the context:

```python
out = torch.cat(branch_outputs, dim=-1)      # (B, N, n_branches * point_dim)
return self._run_stack(self.pooling, out, ctx_emb)
```

The class docstring claims the pooling layers "act as the router that weighs the
branches against each other". This study tests that claim. Concretely:

> For each contiguous `point_dim`-wide **block** of the pooling input — one per
> branch, in the order generalist, experts, specialist — how much does that block
> drive the pooling output, and how does that split move as the conditioning
> changes?

The answer decides whether the architecture is doing what it was built to do. If
the split is flat across the conditioning, the extra branches are an ensemble with
a fixed mixing weight, and the conditioning on the pooling stack is dead weight.

## 2. What the architecture already tells us

Worth working out before measuring anything, because it says what a null result
would look like.

A `ConcatSquashLinear` layer computes `(W x) * sigmoid(G c) + B c`. The gate scales
**rows** — output units — not columns. So at the first pooling layer, block `k`'s
contribution to hidden unit `h` is `s_h * (W[h, block_k] . x_k)`, and the gate
`s_h` multiplies every block's contribution to that unit identically. **The relative
weighting of blocks in the layer-1 pre-activation is context independent.**

Context can only re-weight blocks through:

1. the hyper-bias `B c` moving the leaky-ReLU activation pattern, and
2. hidden-unit re-weighting at later layers, which blocks feel unequally because
   they project onto hidden units differently.

For the default two-layer stack (`diffusion_pooling_hidden_dims: [128]`) the
Jacobian is exactly

```
J = diag(s2) . W2 . diag(alpha) . diag(s1) . W1
```

with `s1`, `s2` the gates and `alpha` the leaky-ReLU slopes. The block partition
lives entirely in the columns of `W1`; the context enters only via `s1`, `s2`,
`alpha`. Routing is real but indirect — mediated by which hidden units are live.

Three consequences for the design:

- **A flat result is a genuine possible outcome, not a bug.** The study needs a
  null it can accept, not just an effect it can find.
- **The mechanism is hidden-unit re-weighting**, so the hidden gate and activation
  pattern are worth recording as a secondary observable when a result needs
  explaining.
- **The activations are piecewise linear**, so the local Jacobian is *exact* inside
  its linear region, not a first-order approximation. Attribution here can be
  exact in a way it usually is not.

## 3. What "influence" means — four measures

Increasing cost and fidelity. All four are computed; they are cross-checks on each
other, and where they disagree that disagreement is the finding.

| # | Measure | Cost | What it answers |
|---|---------|------|-----------------|
| 1 | **Static column norms** of `pooling[0]._layer.weight` per block | free | The context-free baseline. The flat line every other measure is read against. |
| 2 | **Jacobian block norms**, `‖J[:, block_k]‖` | `point_dim` backward passes | Local sensitivity. Also available per output feature, separating "who decides the energy" from "who decides the position". |
| 3 | **Ablation**, leave-one-out and only-one | `2 n_branches` forwards | Model-faithful, catches saturation the Jacobian misses. Their disagreement measures branch interaction. |
| 4 | **Exact Shapley over blocks** | `2**n_branches` forwards | The headline. Model-faithful, handles interactions, exactly additive. With ≤ 6 branches this is ≤ 64 forward passes — trivial. |

Measure 2 also yields an exactly additive split, `out = Σ_k J_k x_k + offset`, where
`offset` is the context-only affine part. That offset is reported: **if it dwarfs
the block contributions, the pooling stack is largely ignoring its inputs and
emitting a context-driven constant**, which would be the most important possible
result and is invisible to any measure normalised over blocks.

Shares can be pooled two ways — mean of per-point shares (every point equal) or
share of summed magnitudes (loud points dominate). Both are computed. They diverge
when a branch is quiet everywhere but enormous on a few points, which is exactly
what a specialist should look like.

## 4. Three arms, because one is not enough

The pooling input is not data — it is the branch outputs. Change the context and
*both* the pooling weights and the branch outputs move. Attributing a shift to
"routing" without separating those is the central trap.

**Frozen.** Capture branch outputs once at a reference context, hold them fixed,
sweep the pooling context. Any movement is the pooling stack changing its mind
about inputs that did not change: this isolates the routing map.
*Caveat:* the `(x, context)` pairs are off-distribution — the pooling stack never
saw them in training. Read it as a property of the learned function, not of
deployment.

**Coupled.** Recompute branch outputs at every swept context, from the same points
and the same noise draw. On-distribution and paired. Answers the deployment
question — as conditioning moves, which branch ends up driving the output — but
cannot separate "pooling re-weighted" from "branch outputs changed".

**Observational.** Real events with their own conditioning, no intervention.
Features stay correlated exactly as in the data. This is what actually happens at
sampling time, confounders included.

Running frozen and coupled together is the point of the design:

| frozen | coupled | reading |
|--------|---------|---------|
| moves | moves | the pooling stack is routing |
| flat | moves | the branches specialise; pooling mixes them at a fixed ratio |
| moves | flat | pooling re-weights, branches compensate — net effect cancels |
| flat | flat | no conditional specialisation anywhere |

A trend in the observational arm that appears in neither sweep is a confounder,
not routing.

## 5. Context axes

**Conditioning features**, one at a time, others held at an on-manifold base row.
Grids come from data quantiles (2%–98%), never from arbitrary intervals, and the
sweep is built in physical units *before* preprocessing so the axis is readable.

- *Scalar* (`incident_energy`): empirical quantiles.
- *Vector* (`incident_direction`): walked along **observed** vectors ordered by
  angle to the mean direction. Sweeping components independently would leave the
  unit sphere and manufacture contexts the model has never seen.
- *Categorical* (`incident_pdg`): the one-hot categories.

**Sigma** is the second axis, not an afterthought. It enters `ctx_emb` through the
Fourier time embedding and is plausibly the strongest driver of branch weighting —
at high noise every branch should agree, at low noise they can specialise. The grid
is the Karras schedule from the config, **with the trailing zero dropped**, since
`forward` takes `sigma.log()`.

The base row deserves a warning. A plain column-wise median is unsafe: each pdg
one-hot column is zero for most events, so its median is zero and the whole block
collapses — and `_run_specialists` takes an `argmax` of that block, so every event
would silently route to specialist 0. `base_context` uses the median for scalars,
the **modal** category for the one-hot, and a direction rescaled to unit length.

## 6. Controls and nulls

**The static baseline** (measure 1) is drawn on every plot. Curves sitting on it
mean the context is not routing.

**Bootstrap over probe points** gives the error bar on each plotted share. It is
the right error bar and the *wrong* null for the frozen arm, where the same points
are reused at every grid point so this uncertainty is common to all of them. It
also scales as `1/sqrt(P)`, so with tens of thousands of probe points everything
becomes "significant". Read the raw share **range** to decide whether an effect
matters.

**Matched random-direction placebo** is the real null for the frozen arm. Moving
the context at all moves the gates a little; the question is whether moving *this
feature* moves routing more than an arbitrary context perturbation *of the same
magnitude*. Each placebo draw walks the same distances from the base row in a random
direction. `placebo_p` is the fraction of draws whose share range met or beat the
real one.

This null is not decoration. On an untrained network with random weights, the
bootstrap reports the energy sweep at 20–170 sigma, while the placebo returns
`p` between 0.17 and 1.0 — correctly concluding there is no feature-specific
routing. The naive statistic would have reported a strong effect on a network that
has learned nothing.

*Caveat:* random directions ignore context structure, so a draw can break the pdg
one-hot or push the direction off the sphere. If off-manifold contexts destabilise
the gates the null is conservative and a real feature must clear a somewhat
inflated bar.

**Single-branch control.** Run the same pipeline on a checkpoint with one branch;
it should refuse, since `forward` returns that branch unpooled.

## 7. Reading the output

`<checkpoint>_pooling_influence.npz` plus figures. Order of interrogation:

1. **Is the offset term large?** If yes, stop — the pooling stack is mostly writing
   a context-driven constant and block shares are a distraction.
2. **Do Shapley and Jacobian shares agree?** Disagreement means saturation or
   strong interaction; trust Shapley.
3. **Does the frozen sweep clear its placebo?** If not, no feature-specific routing,
   whatever the significance says.
4. **Does frozen agree with coupled?** Use the table in §4.
5. **Per output feature**: a branch can own the energy channel while another owns
   position. This is where genuine specialisation usually shows up first.
6. **Sigma heatmap**: branch ownership versus noise scale.

## 8. Pitfalls, all guarded in code

- `ctx_emb` is `[time_emb, context]` — **time first**. The conditioning starts at
  column 64. An off-by-64 here is silent and ruinous.
- The **specialist block changes identity** with the routed pdg. Never average it
  across an `incident_pdg` sweep; it is not one network.
- `get_sigmas_karras` appends a trailing zero and `sigma.log()` is `-inf`.
- Branch outputs must be captured through the `Denoiser`, so the `c_in`
  preconditioning matches sampling.
- Padding points must be dropped before they reach the branches — after the log
  preprocessing on energy, a zero-energy pad becomes a finite but meaningless value.
- The study uses `net._run_stack` and a forward pre-hook on `pooling[0]` rather than
  reimplementing `forward`, so it runs the model's own code path and cannot drift
  from it. `check_layout` fails loudly if the block partition or context width stops
  matching the config.

## 9. Running it

```bash
python -m scripts.run_pooling_influence_study <checkpoint.pt>

# quick look, sweeps only
python -m scripts.run_pooling_influence_study <checkpoint.pt> \
    --arms frozen coupled --features incident_energy --n-probe-points 500

# many branches: skip the 2**n Shapley enumeration
python -m scripts.run_pooling_influence_study <checkpoint.pt> --no-shapley
```

Cost is dominated by `n_grid * n_sigma` contexts times `n_probe_points`, with
`2**n_branches` forward passes each when Shapley is on. Defaults are a few minutes
on one GPU.

## 10. What would count as an answer

- **Positive:** a block's share moves by more than a few points across a feature
  sweep, clears the placebo, and moves the same way in frozen and coupled. The
  architecture routes, and the per-output breakdown says on what.
- **Negative:** all shares sit within the placebo band of the static baseline
  across every feature and every sigma. Then the branches are a fixed-weight
  ensemble, the pooling stack's conditioning is unused, and
  `diffusion_pooling_hidden_dims` could be cut to a plain linear mixer at no loss.

Both outcomes are publishable inside the project; the design is built so the second
is a conclusion rather than a failure to find anything.
