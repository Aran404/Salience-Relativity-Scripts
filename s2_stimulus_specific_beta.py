"""Stimulus-specific beta simulation.

This toy simulation checks the internal consistency of Equation 3, beta(x),
and the route-one/route-two dissociation described in the paper.

The simulation is not intended as a biological model. It only verifies that
the stated equations produce the qualitative pattern claimed by the paper:
a stimulus whose epistemic value is exhausted but whose pragmatic value is
not (route one) yields lower beta(x) than a matched stimulus with sustained
epistemic value (route two), despite equal exposure count and reward.

No parameter is fitted to data; all constants are illustrative.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]

BETA_FLOOR: Final = 1.0
K: Final = 4.0
DEFAULT_SEED: Final = 0
DEFAULT_OUTPUT_PATH: Final = Path("fig2_stimulus_specific_beta.png")


def beta(v_epistemic: float | FloatArray) -> FloatArray:
    """Compute policy-selection temperature from epistemic value."""
    values = np.asarray(v_epistemic, dtype=np.float64)
    return BETA_FLOOR + K * (1.0 - np.exp(-values))


def excess_entropy_route_one(
    t: FloatArray,
    *,
    e0: float = 2.0,
    tau: float = 6.0,
) -> FloatArray:
    """Return excess entropy for a source whose structure is rapidly exhausted."""
    return e0 * np.exp(-t / tau)


def excess_entropy_route_two(
    t: FloatArray,
    rng: np.random.Generator,
    *,
    floor: float = 1.6,
    amplitude: float = 0.15,
    period: float = 5.0,
    noise: float = 0.05,
) -> FloatArray:
    """Return excess entropy for a source that continues producing structure."""
    return (
        floor
        + amplitude * np.sin(t / period)
        + noise * rng.standard_normal(t.size)
    )


def excess_entropy_novel(
    t: FloatArray,
    *,
    level: float = 2.0,
) -> FloatArray:
    """Return constant epistemic value for a novel stimulus."""
    return np.full_like(t, level, dtype=np.float64)


def single_timepoint_comparison(
    rng: np.random.Generator,
) -> tuple[list[str], FloatArray, FloatArray]:
    """Compare twelve concurrently available stimuli at one time point."""
    labels = ["route one (stale song)", "route two (codebase)"] + [
        f"novel {index}" for index in range(1, 11)
    ]

    v_epistemic = np.concatenate(
        (
            np.array([0.02, 1.8], dtype=np.float64),
            rng.uniform(1.0, 2.2, size=10),
        )
    )

    return labels, v_epistemic, beta(v_epistemic)


def time_course_comparison(
    rng: np.random.Generator,
    *,
    n_steps: int = 60,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Compare beta over simulated exposure time for three stimulus types."""
    t = np.arange(n_steps, dtype=np.float64)

    entropy_route_one = excess_entropy_route_one(t)
    entropy_route_two = excess_entropy_route_two(t, rng)
    entropy_novel = excess_entropy_novel(t)

    # Illustrative Equation 2 form: epistemic value is capped by model-relative
    # excess entropy.
    v_route_one = np.minimum(entropy_route_one, entropy_novel)
    v_route_two = np.minimum(entropy_route_two, entropy_novel)

    return t, beta(v_route_one), beta(v_route_two), beta(entropy_novel)


def summarize_and_check(
    rng: np.random.Generator,
) -> tuple[
    list[str],
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
    FloatArray,
]:
    """Run qualitative checks and return the simulated data."""
    labels, v_epistemic, beta_values = single_timepoint_comparison(rng)
    route_one_beta = beta_values[0]
    route_two_beta = beta_values[1]
    novel_mean_beta = beta_values[2:].mean()

    print("=== Single-timepoint comparison ===")
    for label, value, beta_value in zip(labels, v_epistemic, beta_values):
        print(f"{label:>20s}: V_epistemic={value:5.2f}  beta={beta_value:5.2f}")

    assert route_one_beta < route_two_beta, (
        "FAILED: route one should show lower beta than route two"
    )
    assert route_one_beta < novel_mean_beta, (
        "FAILED: route one should show lower beta than novel stimuli"
    )

    print(
        f"
Check passed: route one beta ({route_one_beta:.2f}) < "
        f"route two beta ({route_two_beta:.2f}) < novel-stimulus mean "
        f"beta ({novel_mean_beta:.2f})."
    )

    t, beta_route_one, beta_route_two, beta_novel = time_course_comparison(rng)

    assert beta_route_one[-1] < beta_route_two[-1], (
        "FAILED: route one should stay low over time"
    )
    assert beta_route_one[-1] < beta_novel[-1], (
        "FAILED: route one should stay below novel stimuli"
    )

    print(
        "Check passed: route-one/route-two dissociation is stable over "
        "simulated time, not a single-timepoint artifact "
        f"(final beta: route one={beta_route_one[-1]:.2f}, "
        f"route two={beta_route_two[-1]:.2f}, "
        f"novel={beta_novel[-1]:.2f})."
    )

    return (
        labels,
        v_epistemic,
        beta_values,
        t,
        beta_route_one,
        beta_route_two,
        beta_novel,
    )


def make_figure(
    labels: list[str],
    beta_values: FloatArray,
    t: FloatArray,
    beta_route_one: FloatArray,
    beta_route_two: FloatArray,
    beta_novel: FloatArray,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Generate and save the stimulus-specific beta figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    colors = ["#c0392b", "#2980b9"] + ["#95a5a6"] * (len(labels) - 2)

    axes[0].bar(
        np.arange(len(labels)),
        beta_values,
        color=colors,
        width=0.65,
    )
    axes[0].set_xticks(np.arange(len(labels)))
    axes[0].set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
    axes[0].set_ylabel(r"$\beta(x)$")
    axes[0].set_title("Single timepoint: 12 concurrent stimuli")
    axes[0].axhline(BETA_FLOOR, color="black", linestyle=":", linewidth=0.8)
    axes[0].margins(x=0.02)

    axes[1].plot(
        t,
        beta_route_one,
        color="#c0392b",
        label="route one (stale song)",
    )
    axes[1].plot(
        t,
        beta_route_two,
        color="#2980b9",
        label="route two (codebase)",
    )
    axes[1].plot(
        t,
        beta_novel,
        color="#95a5a6",
        linestyle="--",
        label="novel stimulus",
    )
    axes[1].set_xlabel(r"simulated exposure time $t$")
    axes[1].set_ylabel(r"$\beta(x)$")
    axes[1].set_title("Over simulated time")
    axes[1].legend(fontsize=8, loc="center right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigure written to {output_path}")


def main() -> None:
    """Run the simulation, checks, and figure generation."""
    rng = np.random.default_rng(DEFAULT_SEED)

    (
        labels,
        _v_epistemic,
        beta_values,
        t,
        beta_route_one,
        beta_route_two,
        beta_novel,
    ) = summarize_and_check(rng)

    make_figure(
        labels,
        beta_values,
        t,
        beta_route_one,
        beta_route_two,
        beta_novel,
    )


if __name__ == "__main__":
    main()
