# Method notes

## What the score estimates

The target is the change in a scalar output metric when one internal
component receives a specified activation replacement. The reference API
expands at the clean activation and uses the fixed direction from clean to
corrupt. The resulting score is patched metric minus clean metric.

For an activation-space function `metric_fn(a)`, only that component should
vary as `a` changes. For a coordinate-space function `F(z)`, use the path
`0 -> e_i` to score component `i`. The path `0 -> ones(N)` instead scores a
joint intervention and can include cross-component interactions.

## HVP and the exposure diagnostic

The HVP estimate adds the quadratic directional term to the AtP estimate.
`hvp(...)` returns these two terms separately, their sum, and an optional
exposure ratio `abs(correction) / abs(atp)`. The ratio is undefined when the
AtP score is exactly zero, so the implementation returns `None` in that case.
The absolute terms remain available.

For nonzero but very small AtP scores, applications should interpret absolute
errors in native metric units. No universal small-denominator threshold is
part of the correction algorithm. Exposure measures the size of the leading
quadratic contribution; it is not a probability that AtP is wrong, and small
exposure alone does not bound higher-order error.

## Multi-step HVP

For `steps=K`, the implementation uses points

$$a_k = a_{\mathrm{clean}} + k\delta/K,$$

and sums

$$
\sum_{k=0}^{K-1}\left[
  \nabla M(a_k)^\top\frac{\delta}{K}
  +\frac12\left(\frac{\delta}{K}\right)^\top
    H(a_k)\left(\frac{\delta}{K}\right)
\right].
$$

Both displacements in the quadratic term are divided by `K`. This matters:
the quadratic contribution for a substep scales with `1/K^2`.
`steps=1` recovers the single-step HVP estimate.

If the Hessian is Lipschitz between every pair of points on this path with a
valid constant $L_3$, the accumulated remainder is at most

$$\frac{L_3\lVert\delta\rVert^3}{6K^2}.$$

This is a conditional theorem. Running HVP or MS-HVP does not require an
estimate of $L_3$. The CPU tests include a cubic for which the accumulated
error is exactly proportional to `1/K^2`.

## Attention-head interventions

The Hugging Face adapter changes the input to the attention output projection,
where individual head outputs are concatenated. At a fixed token position,
the direction is the corrupt head activation minus the clean head activation.
The hook adds the scaled direction to the current live activation. This
preserves effects propagated from upstream coordinates.

Each isolated coordinate endpoint is compared with a separate replacement
hook. The adapter requires evaluation mode, eager attention, one prompt at a
time, position-matched clean/corrupt token sequences, and float32 or float64
weights. Tiny GPT-2 and grouped-query Llama tests exercise these hooks without
downloading model weights.

## Compute and interpretation

One isolated HVP estimate requires a gradient and a Hessian-vector product.
MS-HVP requires these quantities at each of `K` substeps. Query counts are not
fixed wall-clock multipliers: graph reuse, batching, architecture, and sequence
length affect runtime. This reference API prioritizes a readable isolated
implementation and recomputes each component path explicitly.

More accurate component-effect estimates do not guarantee that selecting the
largest scores yields a more faithful circuit at every circuit size. Evaluate
component estimation and circuit behavior with their respective metrics.
