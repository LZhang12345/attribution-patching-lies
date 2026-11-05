"""Generate the README's analytic plot from actual estimator calls.

This cubic illustration is separate from the language-model experiments.
Run from any directory after installing the package with the plots extra.
"""

import json
from io import StringIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from hvp_patching import activation_patching, hvp, multi_step_hvp
from hvp_patching.cli import demo


def main():
    assets = Path(__file__).resolve().parents[1] / "assets"
    assets.mkdir(exist_ok=True)
    dtype = torch.float64
    zero, one = torch.zeros(1, dtype=dtype), torch.ones(1, dtype=dtype)

    def metric(a):
        return (a + 2 * a.square() + 0.2 * a.pow(3)).sum()

    coefficients = hvp(metric, zero, one)
    truth = activation_patching(metric, zero, one).item()
    steps = [1, 2, 3, 5, 8, 12, 20]
    errors = [abs(truth - multi_step_hvp(metric, zero, one, k).item()) for k in steps]
    payload = {
        "function": "t + 2*t^2 + 0.2*t^3",
        "demo": demo(),
        "steps": steps,
        "ms_hvp_absolute_errors": errors,
    }
    (assets / "cubic_demo.json").write_text(json.dumps(payload, indent=2) + "\n")

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#c4cfdc",
            "axes.labelcolor": "#3a4b60",
            "text.color": "#20364e",
            "xtick.color": "#60748b",
            "ytick.color": "#60748b",
            "axes.titleweight": "bold",
            "axes.titlepad": 15,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "svg.hashsalt": "hvp-patching",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.6), layout="constrained")
    t = np.linspace(0, 1, 250)
    target = t + 2 * t**2 + 0.2 * t**3
    first = coefficients.atp.item() * t
    second = first + coefficients.correction.item() * t**2
    axes[0].plot(t, target, color="#23364c", lw=2.6, label="Direct intervention")
    axes[0].plot(t, first, color="#7b8fa9", lw=2.1, ls="--", label="AtP")
    axes[0].plot(t, second, color="#1baf91", lw=2.2, label="HVP")
    axes[0].fill_between(t, first, second, color="#1baf91", alpha=0.09)
    axes[0].set(
        title="A. The gradient misses the bend",
        xlabel="Intervention strength",
        ylabel="Metric change",
    )
    axes[0].set_xlim(0, 1.03)
    axes[0].set_ylim(0, 3.5)
    axes[0].legend(frameon=False, loc="upper left", fontsize=10)
    axes[0].grid(axis="y", color="#e8edf3", lw=0.8)
    axes[1].loglog(steps, errors, color="#6553c7", lw=2.4, marker="o", ms=6, label="MS-HVP")
    axes[1].axhline(
        abs(truth - coefficients.atp.item()), color="#7b8fa9", ls="--", lw=1.5, label="AtP"
    )
    axes[1].scatter([5], [errors[3]], color="#1baf91", s=85, zorder=4)
    axes[1].annotate(
        "K=5: error 0.008",
        xy=(5, errors[3]),
        xytext=(7.1, 0.025),
        arrowprops={"arrowstyle": "-", "color": "#87a899"},
        color="#19836e",
        fontsize=10,
    )
    axes[1].set(
        title="B. Smaller steps reduce the remaining error",
        xlabel="Path segments (K)",
        ylabel="Absolute error",
    )
    axes[1].set_xticks([1, 2, 5, 10, 20], labels=["1", "2", "5", "10", "20"])
    axes[1].grid(axis="y", which="major", color="#e8edf3", lw=0.8)
    axes[1].legend(frameon=False, loc="lower left", fontsize=10)
    fig.suptitle("An analytic cubic example", fontsize=14, weight="bold")
    fig.savefig(assets / "cubic_demo.png", dpi=200)
    vector = StringIO()
    fig.savefig(vector, format="svg", metadata={"Date": None})
    (assets / "cubic_demo.svg").write_text(
        "\n".join(line.rstrip() for line in vector.getvalue().splitlines()) + "\n"
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
