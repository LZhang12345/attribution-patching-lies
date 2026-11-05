"""Small reproducible entry points for the reference implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .estimators import activation_patching, hvp, integrated_gradients, multi_step_hvp


def demo() -> dict[str, float]:
    """A cubic whose exact endpoint and Taylor remainders are known."""
    clean = torch.zeros(1, dtype=torch.float64)
    corrupt = torch.ones_like(clean)

    def metric(a):
        t = a[0]
        return t + 2 * t.square() + 0.2 * t.pow(3)

    result = hvp(metric, clean, corrupt)
    return {
        "activation_patching": activation_patching(metric, clean, corrupt).item(),
        "attribution_patching": result.atp.item(),
        "hvp": result.estimate.item(),
        "ms_hvp_k5": multi_step_hvp(metric, clean, corrupt, steps=5).item(),
        "ig_midpoint_s10": integrated_gradients(metric, clean, corrupt, steps=10).item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="hvp-patching")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("demo", help="run the CPU cubic example; no model download")
    benchmark = commands.add_parser("heads", help="score isolated attention-head interventions")
    benchmark.add_argument("--model", default="gpt2")
    benchmark.add_argument("--revision", default="main")
    benchmark.add_argument("--pairs", type=Path, required=True)
    benchmark.add_argument("--heads", required=True, help="comma-separated layer:head pairs")
    benchmark.add_argument("--position", type=int, default=-1)
    benchmark.add_argument("--steps", type=int, default=5)
    benchmark.add_argument("--ig-steps", type=int, default=10)
    benchmark.add_argument("--device", default="cpu")
    benchmark.add_argument("--cache-dir", type=Path)
    benchmark.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "demo":
        print(json.dumps(demo(), indent=2))
    else:
        from .benchmark import run_head_benchmark

        run_head_benchmark(args)
