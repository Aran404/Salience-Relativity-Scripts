"""
Toy model 1: Salience as fractional-knapsack allocation.

Verifies the claim of Eq. 4 / Sec. "Salience" in Salience Relativity:
that ratio-ranking S_i = V_i / C_i is the EXACT solution to

    max_{r in [0,1]^n}  sum_i r_i * V_i
    s.t.                sum_i r_i * C_i <= W

for a single, shared, continuously divisible budget W across n
simultaneously competing discrepancies.

Method
------
For each of N_TRIALS random problem instances:
  1. Draw n candidate discrepancies with random (V_i, C_i) > 0.
  2. Draw a random budget W in (0, sum(C_i)).
  3. Solve the LP exactly with scipy.optimize.linprog (ground truth).
  4. Solve the same instance with the closed-form greedy ratio rule
     derived in the paper: sort by V_i/C_i descending, fill r_i = 1
     until the budget is exhausted, set the marginal item's r_i
     fractionally, set the rest to 0.
  5. Compare objective values and allocations.

If the paper's derivation is correct, the two should agree to within
floating-point tolerance on every instance, with equality holding
exactly at the optimum (this is a textbook result -- Dantzig 1957 --
but the point of the script is to verify no error was introduced in
transcribing it into the paper's notation, and to make the threshold
policy rule tau* / lambda* visually and numerically concrete rather
than asserted).

Usage
-----
    python knapsack_salience.py
    python knapsack_salience.py --n-trials 20000 --max-items 50
"""

from __future__ import annotations

import argparse
import dataclasses
from collections.abc import Sequence

import numpy as np
from numpy.random import Generator, default_rng
from scipy.optimize import linprog


@dataclasses.dataclass(frozen=True, slots=True)
class Instance:
    """A single salience-allocation problem instance."""

    values: np.ndarray  # V_i, shape (n,)
    costs: np.ndarray  # C_i, shape (n,), all > 0
    budget: float  # W, the shared binding budget


@dataclasses.dataclass(frozen=True, slots=True)
class Solution:
    """An allocation r_i in [0, 1]^n together with its objective value."""

    allocation: np.ndarray
    objective: float
    threshold_lambda: float  # lambda*: the shared cutoff on V_i/C_i


def sample_instance(rng: Generator, n_items: int) -> Instance:
    """Draw a random instance with strictly positive values and costs."""
    values = rng.uniform(0.1, 10.0, size=n_items)
    costs = rng.uniform(0.1, 10.0, size=n_items)
    total_cost = float(costs.sum())
    # Keep the budget strictly interior so the constraint actually binds
    # (binding W is the paper's stated precondition for the ratio-ranking
    # result to apply -- see Sec. "Salience", first caveat).
    budget = rng.uniform(0.05 * total_cost, 0.95 * total_cost)
    return Instance(values=values, costs=costs, budget=budget)


def solve_lp_exact(instance: Instance) -> Solution:
    """Ground truth via linear programming (linprog minimizes, so negate)."""
    n = len(instance.values)
    result = linprog(
        c=-instance.values,
        A_ub=instance.costs.reshape(1, -1),
        b_ub=np.array([instance.budget]),
        bounds=[(0.0, 1.0)] * n,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"LP solver failed: {result.message}")
    allocation = result.x
    objective = float(instance.values @ allocation)
    # Recover lambda* as the LP's dual value on the budget constraint.
    threshold_lambda = float(result.ineqlin.marginals[0] * -1.0)
    return Solution(allocation, objective, threshold_lambda)


def solve_greedy_ratio(instance: Instance) -> Solution:
    """
    The paper's closed-form policy: greedily fill by decreasing V_i/C_i
    until the budget runs out; the marginal item is taken fractionally.
    This is Dantzig's (1957) classical solution to the fractional
    knapsack problem, applied here to the salience ratio S_i = V_i/C_i.
    """
    n = len(instance.values)
    ratios = instance.values / instance.costs
    order = np.argsort(-ratios)  # descending salience

    allocation = np.zeros(n)
    remaining = instance.budget
    threshold_lambda = float(ratios[order[-1]])  # fallback: budget never binds

    for idx in order:
        cost_i = instance.costs[idx]
        if remaining <= 0.0:
            break
        if cost_i <= remaining:
            allocation[idx] = 1.0
            remaining -= cost_i
        else:
            allocation[idx] = remaining / cost_i
            remaining = 0.0
        # lambda* is the ratio of the marginal (last touched) item.
        threshold_lambda = float(ratios[idx])

    objective = float(instance.values @ allocation)
    return Solution(allocation, objective, threshold_lambda)


def run_trials(
    n_trials: int, max_items: int, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run n_trials random instances, return arrays of (objective_gap,
    allocation_gap) where objective_gap = |LP_opt - greedy_opt| and
    allocation_gap = max_i |r_i^LP - r_i^greedy| (allowing for the
    LP's non-uniqueness at ties, we compare objectives primarily).
    """
    rng = default_rng(seed)
    objective_gaps = np.empty(n_trials)
    allocation_gaps = np.empty(n_trials)

    for t in range(n_trials):
        n_items = int(rng.integers(2, max_items + 1))
        instance = sample_instance(rng, n_items)
        lp_sol = solve_lp_exact(instance)
        greedy_sol = solve_greedy_ratio(instance)

        objective_gaps[t] = abs(lp_sol.objective - greedy_sol.objective)
        allocation_gaps[t] = float(
            np.max(np.abs(lp_sol.allocation - greedy_sol.allocation))
        )

    return objective_gaps, allocation_gaps


def demo_single_instance() -> None:
    """Print one worked instance so the threshold rule is visible by eye."""
    rng = default_rng(42)
    instance = sample_instance(rng, n_items=6)
    lp_sol = solve_lp_exact(instance)
    greedy_sol = solve_greedy_ratio(instance)

    ratios = instance.values / instance.costs
    order = np.argsort(-ratios)

    print("=" * 70)
    print("Worked instance: 6 discrepancies competing for one shared budget")
    print("=" * 70)
    print(f"{'item':>4} {'V_i':>8} {'C_i':>8} {'S_i=V/C':>10} "
          f"{'r_i (LP)':>10} {'r_i (greedy)':>12}")
    for idx in order:
        print(
            f"{idx:>4} {instance.values[idx]:>8.3f} {instance.costs[idx]:>8.3f} "
            f"{ratios[idx]:>10.3f} {lp_sol.allocation[idx]:>10.3f} "
            f"{greedy_sol.allocation[idx]:>12.3f}"
        )
    print(f"\nBudget W = {instance.budget:.3f}")
    print(f"LP optimal objective:     {lp_sol.objective:.6f}")
    print(f"Greedy-ratio objective:   {greedy_sol.objective:.6f}")
    print(f"lambda* (LP dual):        {lp_sol.threshold_lambda:.6f}")
    print(f"lambda* (greedy cutoff):  {greedy_sol.threshold_lambda:.6f}")
    print(
        "\nNote the greedy rule fills every r_i = 1 whenever S_i > lambda*, "
        "r_i = 0 whenever S_i < lambda*, exactly as claimed in Sec. Salience."
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=5000)
    parser.add_argument("--max-items", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args(argv)

    demo_single_instance()

    print("\n" + "=" * 70)
    print(
        f"Running {args.n_trials} random trials "
        f"(2 to {args.max_items} competing discrepancies each)..."
    )
    print("=" * 70)
    objective_gaps, allocation_gaps = run_trials(
        args.n_trials, args.max_items, seed=args.seed
    )

    n_exact = int(np.sum(objective_gaps < args.tol))
    print(f"Objective-value match (|LP - greedy| < {args.tol}): "
          f"{n_exact}/{args.n_trials} trials")
    print(f"Max objective gap observed:   {objective_gaps.max():.3e}")
    print(f"Mean objective gap observed:  {objective_gaps.mean():.3e}")
    print(f"Max allocation gap observed:  {allocation_gaps.max():.3e}")

    if n_exact == args.n_trials:
        print(
            "\nCONFIRMED: greedy ratio-ranking on S_i = V_i/C_i matches the "
            "exact LP optimum on every trial, to floating-point precision. "
            "This is the numerical content of the claim in Sec. 'Salience' "
            "that S = V/C is not a free choice of functional form but the "
            "unique solution (up to ties) of the stated allocation problem."
        )
    else:
        failed = args.n_trials - n_exact
        print(
            f"\nWARNING: {failed} trial(s) disagree beyond tolerance. "
            "Inspect the corresponding instances -- this would falsify "
            "the paper's claim as stated, not just this script."
        )


if __name__ == "__main__":
    main()