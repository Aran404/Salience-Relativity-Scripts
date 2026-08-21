"""
Toy model 2: Mutual information vs. viability-relevant epistemic value.

Verifies the claim of Sec. "Why Information Has Value At All" in
Salience Relativity: that maximizing I(S;O), expected information
gain / mutual information between a hidden state and an observation,
is a DIFFERENT objective from maximizing

    V_epi^V(x, tau) = E[V_prag*(M_{t+1}) - V_prag*(M_t)]

the expected gain in future viable control -- and that a system
optimizing the former can prefer strictly less decision-relevant
information over strictly more decision-relevant information,
whenever the extra bits target a part of the world that does not
bear on the viable set V.

Construction
------------
Two independent hidden variables:

    Z_irrelevant ~ Uniform{0, ..., 2^K - 1}   K bits of entropy.
                    Does not affect which action keeps the agent
                    inside the viable set V, and does not affect
                    survival probability at all.
    Z_pivotal    ~ Bernoulli(1/2)             1 bit of entropy.
                    Determines which of two actions keeps the agent
                    inside V.

Two available observations, each a channel P(o | Z_irrelevant, Z_pivotal):

    A: noiselessly reveals Z_irrelevant, reveals nothing about
       Z_pivotal (channel independent of Z_pivotal).
       I(Z_irrelevant, Z_pivotal ; O_A) = K bits, all of it about
       the irrelevant variable. K is set to 100 below, matching the
       paper's worked example ("Learning A yields 100 bits").

    B: reveals nothing about Z_irrelevant, reveals Z_pivotal through
       a binary symmetric channel with small flip probability eps.
       I(Z_irrelevant, Z_pivotal ; O_B) = 1 - H_b(eps) bits (at most
       1), where H_b is binary entropy -- strictly less than K for
       any eps in (0, 1) once K > 1.

We compute exactly (closed-form entropy expressions, no sampling):
    I(S; O_A), I(S; O_B)                       -- mutual information
    V_prag  before / V_prag* after each obs.    -- via Eq. 1
    V_epi^V(A), V_epi^V(B)                      -- via Eq. 2

and confirm I(S;O_A) = K bits >> I(S;O_B) <= 1 bit, while
V_epi^V(A) = 0 exactly (resolving an irrelevant variable cannot move
the agent's best achievable survival probability, by construction)
and V_epi^V(B) > 0 strictly whenever eps < 1/2. This reproduces the
paper's worked example ("no system optimizing I(S;O) alone can
prefer B to A ... and it is wrong to") as a computed result rather
than an assertion.

Usage
-----
    python epistemic_value_pomdp.py
    python epistemic_value_pomdp.py --k-bits 100 --flip-prob 0.02
"""

from __future__ import annotations

import argparse
import dataclasses
import math
from collections.abc import Sequence


def binary_entropy(p: float) -> float:
    """H_b(p) in bits, with the 0 log 0 = 0 convention."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -p * math.log2(p) - (1.0 - p) * math.log2(1.0 - p)


@dataclasses.dataclass(frozen=True, slots=True)
class ObservationResult:
    """Exact information- and value-theoretic scores for one observation."""

    name: str
    mutual_information_bits: float
    v_prag_after: float
    v_epi: float


def v_prag(p_survive: float) -> float:
    """Eq. 1: V_prag ~ E[log P(survive)]. A pure function of survival prob."""
    return math.log(p_survive)


def score_observation_a(k_bits: int, p_survive_uninformed: float) -> ObservationResult:
    """
    Observation A: noiselessly resolves K bits of a variable that does
    not affect survival at all. The agent's best achievable policy is
    unchanged by learning it -- there is nothing to act differently
    on -- so its best achievable survival probability after observing
    A is, by construction, identical to the uninformed baseline.
    """
    v_after = v_prag(p_survive_uninformed)  # nothing actionable changed
    v_before = v_prag(p_survive_uninformed)
    return ObservationResult(
        name=f"A ({k_bits} bits, viability-irrelevant)",
        mutual_information_bits=float(k_bits),
        v_prag_after=v_after,
        v_epi=v_after - v_before,
    )


def score_observation_b(
    flip_prob: float,
    p_survive_uninformed: float,
    p_survive_correct_action: float,
    p_survive_wrong_action: float,
) -> ObservationResult:
    """
    Observation B: a binary-symmetric-channel reading of the single
    pivotal bit, correct with probability (1 - flip_prob). The agent
    acts on the posterior: with probability (1 - flip_prob) it reads
    correctly and takes the survival-maximizing action; with
    probability flip_prob it is fooled and takes the wrong one. This
    is the exact expected log-survival-probability under the
    posterior-optimal policy (a closed-form expectation, not a
    Monte Carlo estimate).
    """
    mi_bits = 1.0 - binary_entropy(flip_prob)
    v_after = (1.0 - flip_prob) * v_prag(p_survive_correct_action) + flip_prob * v_prag(
        p_survive_wrong_action
    )
    v_before = v_prag(p_survive_uninformed)
    return ObservationResult(
        name=f"B (~{mi_bits:.3f} bits, viability-pivotal)",
        mutual_information_bits=mi_bits,
        v_prag_after=v_after,
        v_epi=v_after - v_before,
    )


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k-bits", type=int, default=100,
        help="Bits of irrelevant information carried by observation A.",
    )
    parser.add_argument(
        "--flip-prob", type=float, default=0.02,
        help="Error rate of observation B's binary symmetric channel on Z_pivotal.",
    )
    args = parser.parse_args(argv)

    # World parameters. Uninformed policy is a coin flip over the two
    # actions under a uniform prior on Z_pivotal, so its survival rate
    # is exactly the average of the two conditional rates -- enforced
    # below by construction, not assumed.
    p_survive_correct_action = 0.95
    p_survive_wrong_action = 0.05
    p_survive_uninformed = 0.5 * p_survive_correct_action + 0.5 * p_survive_wrong_action
    assert math.isclose(p_survive_uninformed, 0.5)

    result_a = score_observation_a(args.k_bits, p_survive_uninformed)
    result_b = score_observation_b(
        args.flip_prob, p_survive_uninformed, p_survive_correct_action, p_survive_wrong_action
    )

    print("=" * 78)
    print("Mutual information vs. viability-relevant epistemic value")
    print("=" * 78)
    print(f"{'Observation':<42}{'I(S;O) [bits]':>16}{'V_epi^V [nats]':>20}")
    for r in (result_a, result_b):
        print(f"{r.name:<42}{r.mutual_information_bits:>16.4f}{r.v_epi:>20.6f}")

    print(f"\nV_prag before any observation:  {v_prag(p_survive_uninformed):.6f} nats "
          f"(log {p_survive_uninformed} survival)")
    print(f"V_prag* after observing A:      {result_a.v_prag_after:.6f} nats "
          "(identical to baseline: A carries 0 bits about Z_pivotal)")
    print(f"V_prag* after observing B:      {result_b.v_prag_after:.6f} nats")

    mi_favors_a = result_a.mutual_information_bits > result_b.mutual_information_bits
    vepi_favors_b = result_b.v_epi > result_a.v_epi
    v_epi_a_is_exactly_zero = math.isclose(result_a.v_epi, 0.0, abs_tol=1e-12)

    print("\n" + "-" * 78)
    print(f"I(S;O) ranks A above B:      {mi_favors_a}  "
          f"({result_a.mutual_information_bits:.2f} vs "
          f"{result_b.mutual_information_bits:.2f} bits)")
    print(f"V_epi^V ranks B above A:     {vepi_favors_b}  "
          f"({result_b.v_epi:.4f} vs {result_a.v_epi:.4f} nats)")
    print(f"V_epi^V(A) is exactly zero:  {v_epi_a_is_exactly_zero}")
    print("-" * 78)

    if not vepi_favors_b and result_b.v_epi < 0.0:
        print(
            "\nNOTE: at this flip_prob, observation B is noisy enough that "
            "acting on it is worse than ignoring it and keeping the "
            "uninformed policy (V_epi^V(B) < 0). This is expected, not an "
            "error: the paper's claim requires B's channel to be "
            "informative enough that acting on the posterior beats the "
            "prior policy. B remains uninformative about the irrelevant "
            "variable and informative (however weakly) about the pivotal "
            "one -- the qualitative asymmetry the example targets -- but "
            "the specific numerical claim V_epi^V(B) > V_epi^V(A) = 0 only "
            "holds once B clears that threshold. Try a smaller --flip-prob."
        )

    if mi_favors_a and vepi_favors_b and v_epi_a_is_exactly_zero:
        print(
            "\nCONFIRMED: the two objectives disagree, in the direction the "
            "paper's worked example claims. A mutual-information maximizer "
            f"strictly prefers observation A ({args.k_bits} bits vs "
            f"{result_b.mutual_information_bits:.2f}); a viability-relevant-"
            "value maximizer strictly prefers B, and is exactly indifferent "
            "to A (V_epi^V(A) = 0 to machine precision), because A does not "
            "touch the variable the agent's future control depends on. No "
            "system optimizing I(S;O) alone can prefer B to A here; by "
            "construction it prefers A, and by construction (since resolving "
            "Z_irrelevant changes nothing the agent can do) it is wrong to."
        )
    else:
        print(
            "\nWARNING: the constructed instance does not reproduce the "
            "claimed divergence -- inspect parameters."
        )


if __name__ == "__main__":
    main()