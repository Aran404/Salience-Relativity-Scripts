"""Epistemic-decay and repetition simulation.

This toy simulation checks the internal consistency of the claim behind P2:
repetition/re-engagement probability separates by epistemic-decay rate within
each reward-magnitude level, a pattern that a reward-magnitude-only account
cannot generate under the stated equations.

The simulation is not intended to model real subjects. It generates synthetic
agents in a 2x2 design: reward magnitude x epistemic-decay rate. It computes
beta(x) from Equation 3 at a fixed late time point, converts beta into a
re-engagement probability through an illustrative monotone link, and checks
for a within-reward-level separation.

It does not test the acute-contingency-change signature distinguishing route
one from habitization; that comparison remains a design prediction.

No parameter is fitted to data; all constants are illustrative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, TypedDict

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

BETA_FLOOR: Final = 1.0
K: Final = 4.0
DEFAULT_SEED: Final = 1
DEFAULT_OUTPUT_PATH: Final = Path("fig3_epistemic_decay_repetition.png")


class Condition(TypedDict):
    """Parameters defining one cell of the 2x2 simulation."""

    tau: float
    reward: float


class SimulationResult(TypedDict):
    """Summary statistics for one simulated condition."""

    mean: float
    standard_error: float


def beta(v_epistemic: float | FloatArray) -> FloatArray:
    """Compute policy-selection temperature from epistemic value."""
    values = np.asarray(v_epistemic, dtype=np.float64)
    return BETA_FLOOR + K * (1.0 - np.exp(-values))


def epistemic_value_at(
    t: float,
    tau: float,
    *,
    e0: float = 2.0,
) -> float:
    """Return model-relative epistemic value after exposure time ``t``."""
    return e0 * np.exp(-t / tau)


def reengagement_probability(
    beta_value: float | FloatArray,
    reward: float,
    *,
    beta_ref: float = 3.0,
    reward_weight: float = 0.35,
) -> FloatArray:
    """Map beta and reward to an illustrative re-engagement probability.

    Lower beta increases re-engagement probability in this toy model.
    Reward contributes an independent additive term.
    """
    beta_array = np.asarray(beta_value, dtype=np.float64)

    beta_term = 1.0 / (1.0 + np.exp((beta_array - beta_ref) * 1.5))
    probability = 0.15 + reward_weight * reward + 0.5 * beta_term

    return np.clip(probability, 0.0, 1.0)


def simulate_condition(
    rng: np.random.Generator,
    tau: float,
    reward: float,
    *,
    n_subjects: int = 200,
    t_final: float = 40.0,
) -> SimulationResult:
    """Simulate one reward x decay-rate condition."""
    v_epistemic = epistemic_value_at(t_final, tau=tau)
    v_epistemic_samples = np.clip(
        v_epistemic + rng.normal(0.0, 0.05, size=n_subjects),
        0.0,
        None,
    )

    beta_values = beta(v_epistemic_samples)
    probabilities = reengagement_probability(beta_values, reward)

    draws = rng.binomial(1, probabilities)

    return {
        "mean": float(draws.mean()),
        "standard_error": float(draws.std(ddof=1) / np.sqrt(n_subjects)),
    }


def run_2x2(
    rng: np.random.Generator,
    *,
    n_subjects: int = 200,
) -> dict[str, SimulationResult]:
    """Run the complete 2x2 reward x epistemic-decay simulation."""
    conditions: dict[str, Condition] = {
        "low_reward_fast_decay": {"tau": 4.0, "reward": 0.2},
        "low_reward_slow_decay": {"tau": 25.0, "reward": 0.2},
        "high_reward_fast_decay": {"tau": 4.0, "reward": 0.9},
        "high_reward_slow_decay": {"tau": 25.0, "reward": 0.9},
    }

    return {
        name: simulate_condition(
            rng,
            n_subjects=n_subjects,
            **parameters,
        )
        for name, parameters in conditions.items()
    }


def summarize_and_check(
    results: dict[str, SimulationResult],
) -> tuple[float, float]:
    """Print simulation results and verify the predicted within-level gaps."""
    print("=== 2x2 design: reward magnitude x epistemic-decay rate ===")

    for name, result in results.items():
        print(
            f"{name:>28s}: repetition probability = "
            f"{result['mean']:.3f} (SE {result['standard_error']:.3f})"
        )

    low_reward_gap = (
        results["low_reward_fast_decay"]["mean"]
        - results["low_reward_slow_decay"]["mean"]
    )
    high_reward_gap = (
        results["high_reward_fast_decay"]["mean"]
        - results["high_reward_slow_decay"]["mean"]
    )

    assert low_reward_gap > 0, (
        "FAILED: fast decay should exceed slow decay within low reward"
    )
    assert high_reward_gap > 0, (
        "FAILED: fast decay should exceed slow decay within high reward"
    )

    print(
        "\nCheck passed: decay-rate gap is nonzero within both reward levels "
        f"(low-reward gap = {low_reward_gap:+.3f}, "
        f"high-reward gap = {high_reward_gap:+.3f}). "
        "Fast decay drives V_epistemic toward zero sooner; under Equation 3 "
        "this lowers beta(x) sooner, which under the illustrative "
        "re-engagement link raises repetition probability within both reward "
        "levels. The effect is generated by construction because decay rate "
        "enters through beta(x) independently of the additive reward term."
    )

    return low_reward_gap, high_reward_gap


def make_figure(
    results: dict[str, SimulationResult],
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Generate and save the epistemic-decay repetition figure."""
    order = [
        "low_reward_slow_decay",
        "low_reward_fast_decay",
        "high_reward_slow_decay",
        "high_reward_fast_decay",
    ]
    labels = [
        "Low reward\nslow decay",
        "Low reward\nfast decay",
        "High reward\nslow decay",
        "High reward\nfast decay",
    ]

    means = [results[name]["mean"] for name in order]
    errors = [results[name]["standard_error"] for name in order]
    colors = ["#2980b9", "#c0392b", "#2980b9", "#c0392b"]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    ax.bar(
        labels,
        means,
        yerr=errors,
        capsize=5,
        color=colors,
    )
    ax.set_ylabel("final repetition probability")
    ax.set_ylim(0.0, 1.0)
    ax.set_title(
        "Repetition separates by decay rate\nwithin both reward levels"
    )

    ax.annotate(
        "",
        xy=(1, means[1] + 0.04),
        xytext=(0, means[0] + 0.04),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1},
    )
    ax.annotate(
        "",
        xy=(3, means[3] + 0.04),
        xytext=(2, means[2] + 0.04),
        arrowprops={"arrowstyle": "->", "color": "black", "lw": 1},
    )

    legend_elements = [
        Patch(facecolor="#2980b9", label="slow decay"),
        Patch(facecolor="#c0392b", label="fast decay (route one)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigure written to {output_path}")


def main() -> None:
    """Run the simulation, checks, and figure generation."""
    rng = np.random.default_rng(DEFAULT_SEED)
    results = run_2x2(rng)

    summarize_and_check(results)
    make_figure(results)


if __name__ == "__main__":
    main()
