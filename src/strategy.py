"""Searching over plans, and knowing when the data cannot rank them.

**Order never matters.** A plan's race time depends on the multiset of
stint lengths and nothing else. The age penalty is counted from the start
of each stint, so it cannot tell which stint it is in; the fuel term is a
sum over race laps, which every plan covers identically. So 20-20-17 and
17-20-20 are the same race time to the last decimal, and the search space
is the partitions of the race distance, not its compositions.

That is a property of the model, not of racing, and it is the sharpest
statement of what this model leaves out. Order matters enormously in a real
race, because stopping earlier than the car behind is an undercut and
stopping later is an overcut, and both are about where you come out in
traffic rather than what your own stopwatch says. None of that is here. See
the scope section of the README: this is a single-car time trial.

**Refusal.** The optimiser will rank plans for any positive degradation
handed to it, including one whose confidence interval covers zero. At those
circuits the ranking is an artefact of a point estimate. `across_interval`
re-solves at both ends of the interval and reports the set of stop counts
that are optimal somewhere inside it; when that set has more than one
member the honest answer is that the data does not choose, and the caller
is told so rather than handed the midpoint's answer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache

from race import Params, balanced, closed_form_plan, plan_cost, race_time

MIN_STINT = 2    # a one-lap stint is a drive-through, not a strategy
MAX_STINTS = 6


@lru_cache(maxsize=None)
def partitions(laps: int, stints: int, *, minimum: int = MIN_STINT,
               maximum: int | None = None) -> tuple[tuple[int, ...], ...]:
    """Every multiset of `stints` stint lengths summing to `laps`.

    Non-increasing order, so each distinct plan appears exactly once.
    """
    cap = maximum if maximum is not None else laps

    def walk(remaining: int, parts: int, largest: int):
        if parts == 0:
            if remaining == 0:
                yield ()
            return
        top = min(largest, cap, remaining - minimum * (parts - 1))
        for size in range(top, minimum - 1, -1):
            for rest in walk(remaining - size, parts - 1, size):
                yield (size, *rest)

    return tuple(walk(laps, stints, cap))


@dataclass(frozen=True)
class Best:
    plan: tuple[int, ...]
    time: float
    stints: int
    runner_up: tuple[int, ...] | None
    margin: float          # how much slower the best plan with a different
                           # stop count is, in seconds

    @property
    def stops(self) -> int:
        return self.stints - 1


def best_plan(p: Params, *, max_stints: int = MAX_STINTS,
              minimum: int = MIN_STINT, maximum: int | None = None) -> Best:
    """Exhaustive search over partitions, at every stint count.

    Slower than the closed form and correct in cases the closed form is
    not: a cliff, or a cap on how long a set can be run. With neither, the
    two agree exactly, which is a test rather than a hope.
    """
    ranked: list[tuple[float, tuple[int, ...], int]] = []
    for stints in range(1, max_stints + 1):
        options = partitions(p.laps, stints, minimum=minimum, maximum=maximum)
        if not options:
            continue
        plan = min(options, key=lambda candidate: plan_cost(candidate, p))
        ranked.append((race_time(plan, p), plan, stints))
    if not ranked:
        raise ValueError(
            f"no plan can cover {p.laps} laps with stints of at least "
            f"{minimum}" + (f" and at most {maximum}" if maximum else ""))
    ranked.sort()
    time, plan, stints = ranked[0]
    other = next((row for row in ranked[1:] if row[2] != stints), None)
    return Best(plan=plan, time=time, stints=stints,
                runner_up=other[1] if other else None,
                margin=(other[0] - time) if other else float("inf"))


def across_interval(p: Params, low: float, high: float, **kwargs) -> dict:
    """Re-solve at both ends of a degradation confidence interval.

    The optimum is monotone in degradation - more degradation never means
    fewer stops - so the two ends bracket every stop count that is optimal
    anywhere inside. If they disagree, the data does not pick a strategy,
    and that is the result.
    """
    if low > high:
        raise ValueError("interval ends are the wrong way round")
    ends = []
    for value in (low, high):
        if value <= 0:
            # A non-positive degradation is not a slower optimum, it is a
            # different problem: with no age penalty every stop is pure
            # loss and the answer is one stint whatever else is true.
            ends.append((1, (p.laps,)))
            continue
        solved = best_plan(replace(p, degradation=value), **kwargs)
        ends.append((solved.stints, solved.plan))
    decided = ends[0][0] == ends[1][0]
    return {
        "low_stints": ends[0][0], "low_plan": ends[0][1],
        "high_stints": ends[1][0], "high_plan": ends[1][1],
        "decided": decided,
        "range": (min(ends[0][0], ends[1][0]), max(ends[0][0], ends[1][0])),
    }


def table(p: Params, *, max_stints: int = MAX_STINTS,
          minimum: int = MIN_STINT) -> list[dict]:
    """Best plan at each stint count, and the cost of being wrong.

    The gap to the best overall is the number that decides whether any of
    this matters. Where two stop counts are within a second of each other,
    the choice between them is not an optimisation problem; it is a bet on
    a safety car.
    """
    rows = []
    for stints in range(1, max_stints + 1):
        options = partitions(p.laps, stints, minimum=minimum)
        if not options:
            continue
        plan = min(options, key=lambda candidate: plan_cost(candidate, p))
        rows.append({"stints": stints, "stops": stints - 1, "plan": plan,
                     "time": race_time(plan, p)})
    floor = min(row["time"] for row in rows)
    for row in rows:
        row["gap"] = row["time"] - floor
    return rows


def agrees_with_closed_form(p: Params, **kwargs) -> bool:
    """Does the search find what the arithmetic already knew?"""
    searched = best_plan(p, **kwargs)
    plan, _, stints = closed_form_plan(p)
    return (searched.stints == stints
            and sorted(searched.plan, reverse=True)
            == sorted(balanced(p.laps, stints), reverse=True))
