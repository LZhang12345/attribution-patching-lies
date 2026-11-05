"""Animate AtP, HVP, and accumulated local MS-HVP estimates.

The scalar example and every displayed coefficient come from the released
estimators. MS-HVP curves accumulate local estimated changes; they are not
anchored to exact metric values at intermediate points.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image
import torch

from hvp_patching import activation_patching, hvp, multi_step_hvp


COLORS = {"truth": "#20364e", "atp": "#7b8fa9", "hvp": "#149d81", "ms": "#7253c8"}
PHASES = [
    ("AtP uses the starting slope", "One gradient gives a straight-line estimate."),
    ("HVP adds the leading curvature", "A directional HVP supplies the quadratic correction."),
    ("MS-HVP takes smaller steps", "Recompute the local correction on each of five path segments."),
]


def example_data():
    def metric(a):
        return (a + 2 * a.square() + 0.2 * a.pow(3)).sum()

    zero = torch.zeros(1, dtype=torch.float64)
    one = torch.ones_like(zero)
    result = hvp(metric, zero, one)
    truth = activation_patching(metric, zero, one).item()
    k = 5
    segments = []
    total = 0.0
    for index in range(k):
        start = torch.tensor([index / k], dtype=torch.float64)
        end = torch.tensor([(index + 1) / k], dtype=torch.float64)
        local = hvp(metric, start, end)
        segments.append(
            {
                "start": index / k,
                "end": (index + 1) / k,
                "estimated_change_before_segment": total,
                "first_order": local.atp.item(),
                "quadratic": local.correction.item(),
            }
        )
        total += local.estimate.item()
    direct_ms = multi_step_hvp(metric, zero, one, steps=k).item()
    if not np.isclose(total, direct_ms, atol=1e-12, rtol=0):
        raise AssertionError("animated segment sum does not match the released MS-HVP estimator")
    return {
        "example": "analytic illustration, not a language-model benchmark",
        "function": "t + 2*t^2 + 0.2*t^3",
        "activation_patching": truth,
        "atp": result.atp.item(),
        "hvp": result.estimate.item(),
        "quadratic": result.correction.item(),
        "ms_hvp": direct_ms,
        "steps": k,
        "segments": segments,
    }


def ms_curve(x, data):
    """Cumulative Taylor estimate at each location on the displayed path."""
    k = data["steps"]
    indices = np.minimum(np.floor(x * k).astype(int), k - 1)
    fraction = x * k - indices
    prefixes = np.array([row["estimated_change_before_segment"] for row in data["segments"]])
    first = np.array([row["first_order"] for row in data["segments"]])
    quadratic = np.array([row["quadratic"] for row in data["segments"]])
    return prefixes[indices] + fraction * first[indices] + fraction**2 * quadratic[indices]


def render_frame(fig, data, phase, progress):
    fig.clear()
    active_color = (COLORS["atp"], COLORS["hvp"], COLORS["ms"])[phase]
    fig.text(
        0.065,
        0.942,
        "A better estimate of the same intervention",
        fontsize=22,
        weight="bold",
        color=COLORS["truth"],
    )
    fig.text(
        0.065,
        0.875,
        f"{phase + 1} / 3   {PHASES[phase][0]}",
        fontsize=16,
        weight="bold",
        color=active_color,
    )
    fig.text(0.065, 0.832, PHASES[phase][1], fontsize=12, color="#62748a")

    axis = fig.add_axes([0.078, 0.23, 0.54, 0.53])
    x = np.linspace(0, 1, 401)
    exact = x + 2 * x**2 + 0.2 * x**3
    axis.plot(x, exact, color=COLORS["truth"], lw=2.4, label="Direct patching")
    axis.scatter([1], [data["activation_patching"]], color=COLORS["truth"], s=28, zorder=6)

    methods = (
        ("AtP", COLORS["atp"], data["atp"] * x, "--"),
        ("HVP", COLORS["hvp"], data["atp"] * x + data["quadratic"] * x**2, "-"),
        ("MS-HVP", COLORS["ms"], ms_curve(x, data), "-"),
    )
    for index, (name, color, values, style) in enumerate(methods):
        if index > phase:
            continue
        if index == phase:
            current = np.linspace(0, progress, max(2, int(progress * 400) + 1))
            if index == 0:
                current_y = data["atp"] * current
            elif index == 1:
                current_y = data["atp"] * current + data["quadratic"] * current**2
            else:
                current_y = ms_curve(current, data)
            axis.plot(current, current_y, color=color, ls=style, lw=3, label=name, zorder=5)
            axis.scatter([current[-1]], [current_y[-1]], color=color, s=55, zorder=7)
        else:
            axis.plot(x, values, color=color, ls=style, lw=1.6, alpha=0.55, label=name)

    if phase == 2:
        for index in range(1, data["steps"]):
            boundary = index / data["steps"]
            if boundary <= progress:
                axis.axvline(boundary, color=COLORS["ms"], lw=0.8, alpha=0.18)
                y = ms_curve(np.array([boundary]), data)[0]
                axis.scatter([boundary], [y], color=COLORS["ms"], s=24, zorder=6)
    axis.set_xlim(-0.015, 1.04)
    axis.set_ylim(-0.05, 3.55)
    axis.set_xticks([0, 0.5, 1], labels=["0 · Clean", "0.5", "1 · Corrupt"])
    axis.set_yticks([0, 1, 2, 3])
    axis.set_xlabel("Intervention strength", labelpad=10, fontsize=12)
    axis.set_ylabel("Change in output metric", labelpad=8, fontsize=12)
    axis.grid(axis="y", color="#e8edf3", lw=0.8)
    axis.legend(loc="upper left", frameon=False, fontsize=10)

    summary = fig.add_axes([0.69, 0.21, 0.27, 0.56])
    summary.set_axis_off()
    summary.text(0, 1.02, "Full intervention", fontsize=14, weight="bold", color=COLORS["truth"])
    summary.text(
        0, 0.96, f"Direct patching: {data['activation_patching']:.3f}", fontsize=12, color="#62748a"
    )
    rows = [
        ("AtP", "atp", COLORS["atp"]),
        ("HVP", "hvp", COLORS["hvp"]),
        ("MS-HVP · K=5", "ms_hvp", COLORS["ms"]),
    ]
    for index, (name, key, color) in enumerate(rows):
        bottom = 0.68 - index * 0.29
        active = index == phase
        visible = index <= phase
        summary.add_patch(
            FancyBboxPatch(
                (0.015, bottom),
                0.96,
                0.225,
                boxstyle="round,pad=0.012,rounding_size=0.025",
                facecolor="#f8fafc" if not active else "#f3f7fb",
                edgecolor=color if active else "#e1e7ef",
                lw=1.7 if active else 0.9,
            )
        )
        summary.text(
            0.045,
            bottom + 0.16,
            name,
            fontsize=12,
            weight="bold",
            color=color if visible else "#9eaaba",
        )
        value = f"{data[key]:.3f}" if visible else "—"
        summary.text(
            0.93,
            bottom + 0.15,
            value,
            ha="right",
            fontsize=18,
            weight="bold",
            color=COLORS["truth"] if visible else "#b5bfcd",
        )
        if visible:
            error = abs(data[key] - data["activation_patching"])
            summary.text(
                0.045, bottom + 0.05, f"Absolute error: {error:.3f}", fontsize=11, color="#62748a"
            )
        else:
            summary.text(
                0.045, bottom + 0.05, "Shown in the next stage", fontsize=10, color="#9eaaba"
            )

    fig.text(
        0.065,
        0.10,
        "One isolated component. One fixed clean → corrupt activation path.",
        fontsize=11,
        color="#465a71",
    )
    fig.text(
        0.065,
        0.055,
        "Analytic example: M(t) = t + 2t² + 0.2t³. No model benchmark data.",
        fontsize=10,
        color="#7b8da2",
    )
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "assets"
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c9d4e0",
            "axes.labelcolor": "#40556d",
            "xtick.color": "#60748b",
            "ytick.color": "#60748b",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    data = example_data()
    grid = np.linspace(0, 1, 401)
    k = data["steps"]
    segment_index = np.minimum(np.floor(grid * k).astype(int), k - 1)
    fraction = grid * k - segment_index
    actual_curve = grid + 2 * grid**2 + 0.2 * grid**3
    # Each completed cubic substep leaves 0.2/K^3 remainder; a partial
    # substep leaves that quantity times its fractional progress cubed.
    expected_remainder = 0.2 * (segment_index + fraction**3) / k**3
    if not np.allclose(actual_curve - ms_curve(grid, data), expected_remainder, atol=1e-12, rtol=0):
        raise AssertionError("animated curve disagrees with the independent cubic remainder")
    fig = plt.figure(figsize=(12, 6.5), dpi=100)
    poster = render_frame(fig, data, 2, 1)
    poster.save(args.output_dir / "method_animation_poster.png", optimize=True)
    palette = poster.quantize(colors=160, method=Image.Quantize.MEDIANCUT)
    frames, durations = [], []
    for phase, count in enumerate([21, 21, 41]):
        for index, progress in enumerate(np.linspace(0, 1, count)):
            image = render_frame(fig, data, phase, float(progress))
            frames.append(image.quantize(palette=palette, dither=Image.Dither.NONE))
            durations.append(100 if index < count - 1 else (2500 if phase == 2 else 1200))
    target = args.output_dir / "method_animation.gif"
    frames[0].save(
        target,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=1,
    )
    plt.close(fig)
    data["animation"] = {"frames": len(frames), "duration_ms": sum(durations), "size": [1200, 650]}
    (args.output_dir / "method_animation.json").write_text(json.dumps(data, indent=2) + "\n")
    print(
        json.dumps(
            {
                "gif": str(target),
                "bytes": target.stat().st_size,
                "frames": len(frames),
                "duration_seconds": sum(durations) / 1000,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
