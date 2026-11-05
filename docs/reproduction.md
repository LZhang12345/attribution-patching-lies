# Examples, benchmarks, and reproduction scope

## What this release contains

The package provides the paper's HVP and MS-HVP formulas, a first-order AtP
implementation and direct endpoint evaluation for accuracy checks. Other
baseline methods are not bundled. It also includes attention-head hooks, a runner for
user-supplied prompt pairs, and the isolated-edge MIB scoring core.

The cubic demo and miniature decoder tests are reproducible on CPU without
model downloads. The pretrained runner obtains weights from Hugging Face and
saves each score and the resolved model revision. It uses next-token logit
difference as its scalar metric.

The two sample IOI pairs are illustrative examples, not the paper's full
benchmark cohorts. This release does not bundle the original cluster launchers,
model weights, evaluation datasets, or complete experiment outputs. Running
the sample command alone does not reproduce every paper table.

## Input format for the head runner

Each JSONL record has four strings:

```json
{
  "clean": "When John and Mary went to the store, John gave a book to",
  "corrupt": "When Mary and John went to the store, Mary gave a book to",
  "correct": " Mary",
  "counterfactual": " John"
}
```

Clean and corrupt strings must tokenize to equal lengths. Each answer must be
a single token; leading spaces are significant for many tokenizers. Token
alignment must also make sense for the chosen intervention. The runner scores
each supplied head independently, records absolute errors, and checks its
coordinate endpoint against direct replacement.

## MIB circuit evaluation

[`hvp_patching.mib`](../src/hvp_patching/mib.py) contains the original pure
scoring utilities used by the exact isolated-edge HVP/MS-HVP implementation.
It includes destination grouping, independent edge updates on replicated
batches, Gemma attention endpoint handling, directional Taylor terms, and
per-record validation. It is a scoring core, not a standalone MIB runner.

In this implementation each candidate edge is assigned to a distinct
replicated batch block. Its loss must depend only on that edge's scalar path
coordinate. Under this isolation invariant, two vector-Jacobian products
recover the first and second directional derivatives for every block.
Applying the same summed-derivative shortcut to jointly coupled coordinates
would introduce cross terms and compute a different quantity.

The production plan's `split: train` field describes discovery of edge scores.
Final circuit evaluation uses MIB's public test split. The evaluation protocol
also distinguishes selection rules:

| Metric | Edge selection | Direction |
| :--- | :--- | :--- |
| CPR | Signed / high-value scores | Higher is better |
| CMD | Absolute / high-magnitude scores | Lower is better |

Use the [official MIB circuit repository](https://github.com/hannamw/MIB-circuit-track)
for graph construction and evaluation, and retain the exact upstream commit,
dataset revision, split, edge selection rule, and evaluator settings with the
results. The selection distinction is specified in the circuit-construction
appendix of [Mueller et al., MIB](https://proceedings.mlr.press/v267/mueller25a.html).

## Regenerating the README plot

```bash
python -m pip install -e '.[plots]'
python scripts/build_readme_assets.py
python scripts/build_method_animation.py
```

Both scripts call the package's estimators and save numerical values alongside
their images. The animation draws the first-order estimate, the HVP quadratic
correction, and the sum of five local MS-HVP corrections. Its intermediate
segments accumulate estimated changes; they do not use exact intervention
effects to reset the curve at each step. A static PNG is included as an
alternative to the looping GIF.
