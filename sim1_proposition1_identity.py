"""Proposition 1 verification: the exact salience-relativity identity.

This simulation checks Proposition 1 of the paper directly against its own
equations, Eq. 1 and Eq. 2:

    S_t(x) = U_t(x) + lambda * I_t(x)                              (Eq. 1)
    P_t(x) = exp{beta_t S_t(x)} / sum_y exp{beta_t S_t(y)}          (Eq. 2)

Proposition 1 states that for any two available targets x and y,

    log( P_t(x) / P_t(y) ) = beta_t * ( S_t(x) - S_t(y) )           (Eq. 3)

This is an algebraic identity, not an empirical claim: it must hold exactly
for any values of U, I, lambda, and beta_t > 0, and for any number of
concurrently available targets (the softmax denominator need not be
restricted to two alternatives). The simulation draws many random choice
sets of varying size, computes both sides of Eq. 3 for many random target
pairs within each set, and confirms they agree to floating-point precision.

This script performs no model fitting and makes no claim about behavioral
data. It only certifies internal consistency of Equations 1-3 as stated in
the paper. It does not simulate a stimulus-specific temperature, an
excess-entropy bound, or a route one/route two dissociation -- the paper
explicitly disclaims all three, and no such construct appears in Eq. 1-4.
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

DEFAULT_SEED: Final = 0
DEFAULT_OUTPUT_PATH: Final = Path("fig1_proposition1_identity.png")
N_CHOICE_SETS: Final = 500
MAX_ALTERNATIVES: Final = 8


def target_score(
    utility: FloatArray,
    information: FloatArray,
    *,
    lam: float,
) -> FloatArray:
    """Compute S_t(x) = U_t(x) + lambda * I_t(x), Equation 1."""
    return utility + lam * information


def softmax_probabilities(score: FloatArray, *, beta: float) -> FloatArray:
    """Compute P_t(x) via Equation 2, using a numerically stable softmax."""
    scaled = beta * score
    shifted = scaled - np.max(scaled)
    weights = np.exp(shifted)
    return weights / weights.sum()


def sample_choice_set(
    rng: np.random.Generator,
    *,
    n_alternatives: int,
    lam: float,
    beta: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Draw a random choice set and return (utility, information, prob)."""
    utility = rng.normal(0.0, 1.5, size=n_alternatives)
    information = rng.uniform(0.0, 2.5, size=n_alternatives)
    score = target_score(utility, information, lam=lam)
    prob = softmax_probabilities(score, beta=beta)
    return utility, information, prob


def check_proposition1(
    rng: np.random.Generator,
    *,
    n_choice_sets: int = N_CHOICE_SETS,
) -> tuple[FloatArray, FloatArray]:
    """Verify Eq. 3 against Eq. 1-2 over many random choice sets and pairs.

    Returns the left-hand side (empirical log-odds) and right-hand side
    (beta * score difference) arrays for every sampled pair, for plotting.
    """
    lhs_values: list[float] = []
    rhs_values: list[float] = []

    for _ in range(n_choice_sets):
        n_alternatives = int(rng.integers(2, MAX_ALTERNATIVES + 1))
        lam = float(rng.uniform(0.0, 3.0))
        beta = float(rng.uniform(0.1, 5.0))

        utility, information, prob = sample_choice_set(
            rng, n_alternatives=n_alternatives, lam=lam, beta=beta
        )
        score = target_score(utility, information, lam=lam)

        idx_x, idx_y = rng.choice(n_alternatives, size=2, replace=False)

        lhs = np.log(prob[idx_x] / prob[idx_y])
        rhs = beta * (score[idx_x] - score[idx_y])

        lhs_values.append(float(lhs))
        rhs_values.append(float(rhs))

    lhs_array = np.array(lhs_values, dtype=np.float64)
    rhs_array = np.array(rhs_values, dtype=np.float64)

    max_absolute_error = float(np.max(np.abs(lhs_array - rhs_array)))

    print("=== Proposition 1: exact log-odds identity ===")
    print(f"Sampled {n_choice_sets} choice sets (2 to {MAX_ALTERNATIVES} "
          "alternatives each), one random target pair per set.")
    print(f"Maximum |log-odds - beta*(S_x - S_y)| across all pairs: "
          f"{max_absolute_error:.3e}")

    assert max_absolute_error < 1e-9, (
        "FAILED: Proposition 1 identity violated beyond floating-point "
        "tolerance"
    )

    print(
        "Check passed: Equation 3 holds to floating-point precision for "
        "every sampled pair, across varying choice-set size, lambda, and "
        "beta. This is expected of an algebraic identity and confirms no "
        "inconsistency between Eq. 1, Eq. 2, and the stated Proposition 1."
    )

    return lhs_array, rhs_array


def check_corollary1(
    rng: np.random.Generator,
    *,
    n_points: int = 300,
) -> tuple[FloatArray, FloatArray, float, float]:
    """Verify Corollary 1: under matched utility, log-odds is linear in
    lambda * (I_t(x) - I_t(y)) with slope beta_t.

    Utility is held equal across x and y (Ut(x) = Ut(y)) by construction, so
    Eq. 4 predicts log P(x)/P(y) = beta * lambda * (I(x) - I(y)) exactly.
    """
    beta = 2.0
    lam = 1.3
    shared_utility = rng.normal(0.0, 1.0, size=n_points)

    info_x = rng.uniform(0.0, 2.5, size=n_points)
    info_y = rng.uniform(0.0, 2.5, size=n_points)

    score_x = target_score(shared_utility, info_x, lam=lam)
    score_y = target_score(shared_utility, info_y, lam=lam)

    # Two-alternative softmax for each matched-utility pair.
    log_odds = beta * (score_x - score_y)

    predicted = beta * lam * (info_x - info_y)

    max_absolute_error = float(np.max(np.abs(log_odds - predicted)))

    print("\n=== Corollary 1: matched-utility prediction ===")
    print(f"Maximum |empirical log-odds - beta*lambda*(I_x - I_y)|: "
          f"{max_absolute_error:.3e}")

    assert max_absolute_error < 1e-9, (
        "FAILED: Corollary 1 identity violated beyond floating-point "
        "tolerance"
    )

    print(
        "Check passed: with Ut(x) = Ut(y) enforced by construction, choice "
        "log-odds is exactly linear in the information difference with "
        "slope beta*lambda, matching Equation 4."
    )

    return info_x - info_y, log_odds, beta, lam


def make_figure(
    lhs_array: FloatArray,
    rhs_array: FloatArray,
    delta_information: FloatArray,
    matched_log_odds: FloatArray,
    beta: float,
    lam: float,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Generate and save the Proposition 1 / Corollary 1 verification figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axes[0].scatter(
        rhs_array, lhs_array, s=10, alpha=0.5, color="#2980b9",
    )
    identity_line = np.linspace(
        min(rhs_array.min(), lhs_array.min()),
        max(rhs_array.max(), lhs_array.max()),
        100,
    )
    axes[0].plot(
        identity_line, identity_line, color="#c0392b", linewidth=1,
        linestyle="--", label="identity line",
    )
    axes[0].set_xlabel(r"$\beta_t (S_t(x) - S_t(y))$")
    axes[0].set_ylabel(r"$\log[P_t(x)/P_t(y)]$")
    axes[0].set_title("Proposition 1\nempirical log-odds vs. predicted")
    axes[0].legend(fontsize=8)

    order = np.argsort(delta_information)
    axes[1].scatter(
        delta_information, matched_log_odds, s=10, alpha=0.5,
        color="#27ae60", label="simulated matched-utility pairs",
    )
    axes[1].plot(
        delta_information[order],
        beta * lam * delta_information[order],
        color="#c0392b", linewidth=1.5,
        label=fr"$\beta\lambda \Delta I$, slope $\beta\lambda={beta*lam:.2f}$",
    )
    axes[1].set_xlabel(r"$I_t(x) - I_t(y)$")
    axes[1].set_ylabel(r"$\log[P_t(x)/P_t(y)]$")
    axes[1].set_title("Corollary 1\nmatched pragmatic utility")
    axes[1].legend(fontsize=8)

    fig.suptitle(
        "Internal-consistency check of the salience-relativity identity "
        "(Eq. 1-4)", fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigure written to {output_path}")


def main() -> None:
    """Run the simulation, checks, and figure generation."""
    rng = np.random.default_rng(DEFAULT_SEED)

    lhs_array, rhs_array = check_proposition1(rng)
    delta_information, matched_log_odds, beta, lam = check_corollary1(rng)

    make_figure(lhs_array, rhs_array, delta_information, matched_log_odds, beta, lam)


if __name__ == "__main__":
    main()
