"""The race as an equation, before it is a simulation.

A great deal of pit strategy is decided by arithmetic that does not need a
Monte Carlo, and knowing exactly where that arithmetic stops being enough
is most of what this project is for. So the closed form comes first.

**The model.** A lap costs

    base + degradation * (tyre age) + fuel * (race lap)

`degradation` is positive: an older set is slower. `fuel` is negative: the
car gets lighter and the track rubbers in, both of which make later laps
quicker. Both numbers are measurements, not guesses - they come out of the
same fixed-effects fit as the degradation project, on the same 2023 laps.

**Why the plan only enters through two terms.** Over a stint of `m` laps
the age term contributes `degradation * m(m+1)/2`. Summed over a plan whose
stints are m_1 ... m_k with sum N:

    T = N*base + fuel*N(N+1)/2 + (degradation/2)*(sum of m_k^2 + N)
        + stops * loss

The first two terms do not depend on the plan at all: every plan runs the
same N laps and passes the same race laps. Only `sum of m_k^2` and the
number of stops do. That is the whole of deterministic strategy, and it has
two consequences that are worth stating separately because they are proved
rather than simulated:

1. For a fixed number of stints, `sum of m_k^2` is minimised by making the
   stints as equal as the integers allow. Convexity; nothing else.
2. Substituting N^2/k for the sum at the equal split and differentiating in
   k gives the best number of stints outright:

       k* = N * sqrt(degradation / (2 * loss))

   Round to the integers either side and compare. No search required.

**Where it stops working.** The closed form assumes the age penalty is
linear and that nothing random happens. Real strategy is decided by the
things that break those assumptions - a tyre that falls off a cliff rather
than fading, a safety car that makes a stop cost a third of what it should,
a lap time distribution with a tail. Each of those is a reason to simulate,
and `montecarlo` quantifies what each one is worth. The point of this
module is that everything the closed form already handles is handled here,
exactly, so that the simulation is answering only the part that is actually
uncertain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Params:
    """Everything a race needs. Defaults are placeholders; `run.py`
    replaces them with the measured 2023 values."""

    laps: int = 57
    base: float = 95.0
    degradation: float = 0.032   # s per lap of tyre age
    fuel: float = -0.055         # s per lap of race distance
    loss: float = 21.75          # green-flag pit loss, seconds
    cliff_age: int | None = None   # age at which the extra penalty starts
    cliff_rate: float = 0.0        # extra s per lap of age beyond the cliff

    def __post_init__(self) -> None:
        if self.laps < 1:
            raise ValueError("a race needs at least one lap")
        if self.degradation < 0:
            raise ValueError(
                "negative degradation means tyres get faster with age, which "
                "no number of stops can exploit; the optimiser would return "
                "the longest possible stint for a reason that is a data "
                "problem, not a strategy")
        if self.loss <= 0:
            raise ValueError("a free pit stop makes the problem unbounded")


def tyre_penalty(age: int, p: Params) -> float:
    """The age cost of one lap, cliff included."""
    penalty = p.degradation * age
    if p.cliff_age is not None and age > p.cliff_age:
        penalty += p.cliff_rate * (age - p.cliff_age)
    return penalty


def lap_time(age: int, lap: int, p: Params) -> float:
    return p.base + tyre_penalty(age, p) + p.fuel * lap


def check_plan(plan: tuple[int, ...], p: Params) -> None:
    if not plan:
        raise ValueError("a plan needs at least one stint")
    if any(m < 1 for m in plan):
        raise ValueError(f"stint lengths must be positive: {plan}")
    if sum(plan) != p.laps:
        raise ValueError(
            f"the plan runs {sum(plan)} laps but the race is {p.laps}")


def race_time(plan: tuple[int, ...], p: Params) -> float:
    """Total race time, summed lap by lap. The reference implementation.

    Nothing clever: it exists so the algebra below has something honest to
    be checked against.
    """
    check_plan(plan, p)
    total = 0.0
    lap = 0
    for stint in plan:
        for age in range(1, stint + 1):
            lap += 1
            total += lap_time(age, lap, p)
    return total + (len(plan) - 1) * p.loss


def race_time_closed(plan: tuple[int, ...], p: Params) -> float:
    """The same number from the decomposition, valid only without a cliff."""
    check_plan(plan, p)
    if p.cliff_age is not None and p.cliff_rate:
        raise ValueError(
            "the closed form assumes a linear age penalty; with a cliff the "
            "sum of squares no longer carries the whole plan dependence")
    n = p.laps
    squares = sum(m * m for m in plan)
    return (n * p.base
            + p.fuel * n * (n + 1) / 2
            + p.degradation * (squares + n) / 2
            + (len(plan) - 1) * p.loss)


def balanced(laps: int, stints: int) -> tuple[int, ...]:
    """The plan that splits `laps` into `stints` as evenly as integers allow.

    This is the minimiser of the sum of squares at that stint count, so for
    a linear age penalty it is the best plan with that many stops. The
    remainder laps go to the earlier stints, which is arbitrary without a
    cliff: every arrangement of the same multiset is the same race time.
    """
    if stints < 1 or laps < stints:
        raise ValueError(f"cannot split {laps} laps into {stints} stints")
    base, extra = divmod(laps, stints)
    return tuple([base + 1] * extra + [base] * (stints - extra))


def continuous_optimum(p: Params) -> float:
    """k* = N sqrt(d / 2L), the real-valued best number of stints.

    Rarely an integer, and never the answer on its own - but it says
    immediately whether the race is near a boundary between stop counts,
    which is where strategy is actually interesting.
    """
    return p.laps * math.sqrt(p.degradation / (2.0 * p.loss))


def closed_form_plan(p: Params, *, max_stints: int = 8
                     ) -> tuple[tuple[int, ...], float, int]:
    """The best plan by arithmetic alone: balanced splits, k* rounded.

    Returns the plan, its race time, and the stint count. Only the two
    integers either side of k* are evaluated, because the objective is
    convex in k.
    """
    if p.cliff_age is not None and p.cliff_rate:
        raise ValueError(
            "with a cliff the best plan is not a balanced split and this "
            "shortcut does not apply; use strategy.best_plan")
    target = continuous_optimum(p)
    candidates = {max(1, min(max_stints, int(math.floor(target)))),
                  max(1, min(max_stints, int(math.ceil(target))))}
    best = min(((balanced(p.laps, k), k) for k in sorted(candidates)
                if k <= p.laps),
               key=lambda item: race_time(item[0], p))
    return best[0], race_time(best[0], p), best[1]


def plan_cost(plan: tuple[int, ...], p: Params) -> float:
    """Only the part of race time that the plan can change.

    Every plan runs the same N laps and passes the same race laps, so the
    base and fuel terms are identical across plans and carry no
    information. What is left is the age penalty, stint by stint, plus the
    stops - and that has a closed form even with a cliff, because the
    extra slope beyond the knot is also an arithmetic series:

        stint of m laps, knot at a:
            age term   = degradation * m(m+1)/2
            cliff term = rate * (m-a)(m-a+1)/2   when m > a, else 0

    So a plan costs O(number of stints) to evaluate rather than O(laps).
    That is the difference between an optimiser that can be put inside a
    bisection and one that cannot. `race_time` remains the reference and
    the two are required to differ by a constant that depends only on the
    race, which is what the tests check.
    """
    check_plan(plan, p)
    total = (len(plan) - 1) * p.loss
    for stint in plan:
        total += p.degradation * stint * (stint + 1) / 2.0
        if p.cliff_age is not None and p.cliff_rate and stint > p.cliff_age:
            over = stint - p.cliff_age
            total += p.cliff_rate * over * (over + 1) / 2.0
    return total
