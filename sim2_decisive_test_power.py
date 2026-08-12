"""Decisive-test design simulation: M0 versus M1 under matched controls.

This simulation checks the internal logic of Section 6's decisive empirical
test. It does not use or generate real behavioral data. It builds synthetic
choice data from a generative process consistent with the paper's own model
(Eq. 1-2), for a within-subject pair of targets x and y whose pragmatic
utility, exposure, and recency are matched (Eq. 7), while their independently
identified residual information It(x) - It(y) varies across trials.

It then fits the two nested models named in Section 6:

    M0: S_t(x) = U_t(x)                          (no information term)
    M1: S_t(x) = U_t(x) + lambda * I_t(x)         (Eq. 1)

by maximum likelihood on the binary choice outcome, and checks that:
  (1) when data are generated with lambda > 0, M1 recovers a positive
      lambda and out-performs M0 on held-out trials, and
  (2) when data are generated with lambda = 0 (information genuinely does
      not matter), M1 does not spuriously outperform M0 out-of-sample.

This demonstrates that the proposed model comparison is falsifiable in the
sense claimed by the paper: it can distinguish a world in which residual
information drives re-engagement from a world in which it does not, using
only the quantities the paper's identification procedure (Section 5) makes
available. It is a design/power check, not a test of any real hypothesis
about human or animal behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize_scalar
from scipy.special import expit

FloatArray = NDArray[np.float64]

DEFAULT_SEED: Final = 0
DEFAULT_OUTPUT_PATH: Final = Path("fig2_decisive_test_power.png")
TRUE_BETA: Final = 1.0
N_TRIALS_PER_SUBJECT: Final = 40
N_SUBJECTS: Final = 60
N_SIMULATIONS: Final = 200
HOLDOUT_FRACTION: Final = 0.3


@dataclass(frozen=True)
class TrialData:
    """One simulated within-subject decisive-test dataset."""

    delta_information: FloatArray  # I_t(x) - I_t(y), matched utility/exposure/recency
    choice_x: FloatArray  # 1 if x chosen over y, else 0


def generate_dataset(
    rng: np.random.Generator,
    *,
    true_lambda: float,
    n_trials: int = N_TRIALS_PER_SUBJECT * N_SUBJECTS,
    beta: float = TRUE_BETA,
) -> TrialData:
    """Simulate binary choices under Eq. 1-2 with matched utility (Eq. 7).

    Because Ut(x) = Ut(y) is enforced, the log-odds of choosing x reduces to
    beta * true_lambda * delta_information (Eq. 4), which is the generative
    process used here.
    """
    delta_information = rng.uniform(-2.0, 2.0, size=n_trials)
    log_odds = beta * true_lambda * delta_information
    prob_choose_x = expit(log_odds)
    choice_x = rng.binomial(1, prob_choose_x).astype(np.float64)

    return TrialData(delta_information=delta_information, choice_x=choice_x)


def negative_log_likelihood_m1(
    lam: float,
    data: TrialData,
    *,
    beta: float = TRUE_BETA,
) -> float:
    """Negative log-likelihood of M1 (information term included) at a given lambda."""
    log_odds = beta * lam * data.delta_information
    prob_choose_x = expit(log_odds)
    prob_choose_x = np.clip(prob_choose_x, 1e-9, 1.0 - 1e-9)
    log_likelihood = (
        data.choice_x * np.log(prob_choose_x)
        + (1.0 - data.choice_x) * np.log(1.0 - prob_choose_x)
    )
    return -float(log_likelihood.sum())


def negative_log_likelihood_m0(data: TrialData) -> float:
    """Negative log-likelihood of M0 (lambda fixed at 0, i.e. chance choice
    once utility is matched, since Ut(x) = Ut(y) leaves no basis for
    systematic preference under M0)."""
    return negative_log_likelihood_m1(0.0, data)


def fit_lambda_mle(data: TrialData) -> float:
    """Maximum-likelihood estimate of lambda under M1."""
    result = minimize_scalar(
        negative_log_likelihood_m1,
        args=(data,),
        bounds=(-5.0, 5.0),
        method="bounded",
    )
    return float(result.x)


def train_test_split(
    data: TrialData, rng: np.random.Generator, *, holdout_fraction: float
) -> tuple[TrialData, TrialData]:
    """Split trials into a training set (for fitting) and a held-out set."""
    n_trials = data.delta_information.size
    indices = rng.permutation(n_trials)
    n_holdout = int(n_trials * holdout_fraction)
    holdout_idx, train_idx = indices[:n_holdout], indices[n_holdout:]

    train = TrialData(
        delta_information=data.delta_information[train_idx],
        choice_x=data.choice_x[train_idx],
    )
    holdout = TrialData(
        delta_information=data.delta_information[holdout_idx],
        choice_x=data.choice_x[holdout_idx],
    )
    return train, holdout


def run_single_simulation(
    rng: np.random.Generator, *, true_lambda: float
) -> tuple[float, float]:
    """Simulate one dataset, fit M0 and M1, return held-out log-lik advantage
    of M1 over M0, and the fitted lambda."""
    data = generate_dataset(rng, true_lambda=true_lambda)
    train, holdout = train_test_split(data, rng, holdout_fraction=HOLDOUT_FRACTION)

    fitted_lambda = fit_lambda_mle(train)

    holdout_nll_m1 = negative_log_likelihood_m1(fitted_lambda, holdout)
    holdout_nll_m0 = negative_log_likelihood_m0(holdout)

    # Positive means M1 fits held-out data better (lower NLL) than M0.
    log_lik_advantage = holdout_nll_m0 - holdout_nll_m1

    return log_lik_advantage, fitted_lambda


def run_power_simulation(
    rng: np.random.Generator,
    *,
    true_lambda: float,
    n_simulations: int = N_SIMULATIONS,
) -> tuple[FloatArray, FloatArray]:
    """Repeat the decisive test many times for one ground-truth lambda."""
    advantages = np.empty(n_simulations, dtype=np.float64)
    fitted_lambdas = np.empty(n_simulations, dtype=np.float64)

    for i in range(n_simulations):
        advantages[i], fitted_lambdas[i] = run_single_simulation(
            rng, true_lambda=true_lambda
        )

    return advantages, fitted_lambdas


def summarize_and_check(
    rng: np.random.Generator,
) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
    """Run the null (lambda=0) and alternative (lambda>0) simulations and
    check that the design distinguishes them, as Section 6 requires."""
    print("=== Decisive-test design check: M0 vs. M1 ===")
    print(
        f"{N_SIMULATIONS} simulated experiments per condition, "
        f"{N_TRIALS_PER_SUBJECT * N_SUBJECTS} matched-utility trials each."
    )

    null_advantages, null_lambdas = run_power_simulation(rng, true_lambda=0.0)
    alt_advantages, alt_lambdas = run_power_simulation(rng, true_lambda=0.9)

    null_false_positive_rate = float(np.mean(null_advantages > 0))
    alt_true_positive_rate = float(np.mean(alt_advantages > 0))

    print(
        f"\nWhen lambda_true = 0 (information does not matter): "
        f"M1 spuriously beats M0 out-of-sample on "
        f"{null_false_positive_rate:.1%} of simulated experiments "
        f"(mean fitted lambda = {null_lambdas.mean():+.3f})."
    )
    print(
        f"When lambda_true = 0.9 (information matters): "
        f"M1 correctly beats M0 out-of-sample on "
        f"{alt_true_positive_rate:.1%} of simulated experiments "
        f"(mean fitted lambda = {alt_lambdas.mean():+.3f})."
    )

    assert null_false_positive_rate < 0.5, (
        "FAILED: M1 spuriously outperforms M0 more often than chance under "
        "the null; held-out comparison is not conservative"
    )
    assert alt_true_positive_rate > null_false_positive_rate, (
        "FAILED: the design does not discriminate lambda>0 from lambda=0"
    )

    print(
        "\nCheck passed: held-out model comparison distinguishes a world in "
        "which residual information drives matched-utility choice from a "
        "world in which it does not, using only quantities available under "
        "the Section 5 identification procedure. This does not validate the "
        "theory against real data; it only confirms the proposed test is "
        "capable, in principle, of falsifying it."
    )

    return null_advantages, alt_advantages, null_lambdas, alt_lambdas


def make_figure(
    null_advantages: FloatArray,
    alt_advantages: FloatArray,
    null_lambdas: FloatArray,
    alt_lambdas: FloatArray,
    *,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> None:
    """Generate and save the decisive-test design figure."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    bins = np.linspace(
        min(null_advantages.min(), alt_advantages.min()),
        max(null_advantages.max(), alt_advantages.max()),
        30,
    )
    axes[0].hist(
        null_advantages, bins=bins, alpha=0.6, color="#7f8c8d",
        label=r"$\lambda_{\mathrm{true}}=0$ (M1 should not win)",
    )
    axes[0].hist(
        alt_advantages, bins=bins, alpha=0.6, color="#2980b9",
        label=r"$\lambda_{\mathrm{true}}=0.9$ (M1 should win)",
    )
    axes[0].axvline(0.0, color="#c0392b", linestyle="--", linewidth=1)
    axes[0].set_xlabel("held-out log-likelihood advantage, M1 over M0")
    axes[0].set_ylabel("simulated experiments")
    axes[0].set_title("Held-out model comparison\nacross repeated simulated experiments")
    axes[0].legend(fontsize=8)

    axes[1].scatter(
        np.zeros_like(null_lambdas) + rng_jitter(null_lambdas.size),
        null_lambdas, s=10, alpha=0.5, color="#7f8c8d",
        label=r"$\lambda_{\mathrm{true}}=0$",
    )
    axes[1].scatter(
        np.ones_like(alt_lambdas) + rng_jitter(alt_lambdas.size),
        alt_lambdas, s=10, alpha=0.5, color="#2980b9",
        label=r"$\lambda_{\mathrm{true}}=0.9$",
    )
    axes[1].axhline(0.0, color="#7f8c8d", linestyle=":", linewidth=1)
    axes[1].axhline(0.9, color="#2980b9", linestyle=":", linewidth=1)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels([r"$\lambda_{\mathrm{true}}=0$", r"$\lambda_{\mathrm{true}}=0.9$"])
    axes[1].set_ylabel(r"fitted $\hat\lambda$ (training trials)")
    axes[1].set_title("Recovered information weight\nacross repeated simulated experiments")

    fig.suptitle(
        "Design check for the Section 6 decisive test (M0 vs. M1)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"\nFigure written to {output_path}")


def rng_jitter(n: int, *, scale: float = 0.06) -> FloatArray:
    """Small fixed jitter for scatter-plot readability (not a random draw
    from the simulation's own generator, purely cosmetic)."""
    local_rng = np.random.default_rng(12345)
    return local_rng.normal(0.0, scale, size=n)


def main() -> None:
    """Run the simulation, checks, and figure generation."""
    rng = np.random.default_rng(DEFAULT_SEED)

    null_advantages, alt_advantages, null_lambdas, alt_lambdas = summarize_and_check(rng)

    make_figure(null_advantages, alt_advantages, null_lambdas, alt_lambdas)


if __name__ == "__main__":
    main()
