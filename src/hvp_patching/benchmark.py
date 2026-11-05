"""A component benchmark runner with per-intervention endpoint checks."""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import torch

from . import __version__
from .estimators import activation_patching, hvp, integrated_gradients, multi_step_hvp
from .heads import Head, HeadInterventions


def score_heads(
    adapter, clean, corrupt, heads, correct, counter, position=-1, steps=5, ig_steps=10
):
    metric = adapter.metric(clean, corrupt, heads, correct, counter, position)
    base = torch.zeros(len(heads), dtype=adapter.dtype, device=adapter.device)
    records = []
    for index, head in enumerate(heads):
        endpoint = base.clone()
        endpoint[index] = 1
        result = hvp(metric, base, endpoint)
        direct = activation_patching(metric, base, endpoint)
        replacement = adapter.replacement_effect(clean, corrupt, head, correct, counter, position)
        if not torch.allclose(direct, replacement, rtol=1e-4, atol=2e-5):
            raise RuntimeError(
                f"coordinate endpoint disagrees with direct replacement for {head.name}"
            )
        records.append(
            {
                "head": head.name,
                "activation_patching": direct.item(),
                "atp": result.atp.item(),
                "quadratic_correction": result.correction.item(),
                "hvp": result.estimate.item(),
                "ms_hvp": multi_step_hvp(metric, base, endpoint, steps).item(),
                "ig_midpoint": integrated_gradients(metric, base, endpoint, ig_steps).item(),
            }
        )
    return records


def run_head_benchmark(args):
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if args.output.exists():
        raise FileExistsError(f"choose a new output path: {args.output}")
    heads = [Head(*map(int, item.split(":"))) for item in args.heads.split(",")]
    pairs = [json.loads(line) for line in args.pairs.read_text().splitlines() if line.strip()]
    if not pairs:
        raise ValueError("no prompt pairs found")
    cache = str(args.cache_dir) if args.cache_dir else None
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision, cache_dir=cache)
    model = (
        AutoModelForCausalLM.from_pretrained(
            args.model,
            revision=args.revision,
            cache_dir=cache,
            torch_dtype=torch.float32,
            attn_implementation="eager",
        )
        .to(args.device)
        .eval()
    )
    model.requires_grad_(False)
    adapter = HeadInterventions(model)
    output = []
    started = time.perf_counter()
    for index, pair in enumerate(pairs):
        clean = tokenizer(
            pair["clean"], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(args.device)
        corrupt = tokenizer(
            pair["corrupt"], return_tensors="pt", add_special_tokens=False
        ).input_ids.to(args.device)
        answer = tokenizer.encode(pair["correct"], add_special_tokens=False)
        counter = tokenizer.encode(pair["counterfactual"], add_special_tokens=False)
        if len(answer) != 1 or len(counter) != 1:
            raise ValueError(
                "answer strings must each encode to exactly one token, including leading space"
            )
        scores = score_heads(
            adapter,
            clean,
            corrupt,
            heads,
            answer[0],
            counter[0],
            args.position,
            args.steps,
            args.ig_steps,
        )
        output.extend({"prompt_index": index, **score} for score in scores)
    metadata = {
        "package_version": __version__,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model": args.model,
        "requested_revision": args.revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "device": args.device,
        "dtype": "float32",
        "attention": "eager",
        "metric": "correct-minus-counterfactual next-token logit difference",
        "direction": "corrupt-minus-clean; expansion at clean",
        "position": args.position,
        "steps": args.steps,
        "ig_steps": args.ig_steps,
        "ig_quadrature": "midpoint",
        "pairs": pairs,
        "elapsed_seconds": time.perf_counter() - started,
    }
    mae = {
        method: sum(abs(row[method] - row["activation_patching"]) for row in output) / len(output)
        for method in ("atp", "hvp", "ms_hvp", "ig_midpoint")
    }
    payload = {"metadata": metadata, "mean_absolute_error": mae, "records": output}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"output": str(Path(args.output)), "mean_absolute_error": mae}, indent=2))
