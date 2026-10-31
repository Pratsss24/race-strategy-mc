"""The part the arithmetic cannot do, and only that part.

Before writing a simulator it is worth being precise about what it can
possibly add, because a Monte Carlo that answers a question the closed form
already answered is decoration.

**Independent lap-time noise adds nothing to the ranking.** Race time is a
sum of lap times, and the expectation of a sum is the sum of expectations,
so adding a zero-mean disturbance to every lap leaves every plan's mean
exactly where it was. It widens the distribution of outcomes without
moving the order of the means. This is testable and it is tested: with the
safety car switched off, the simulated mean converges on the deterministic
race time and the ranking of plans is the deterministic ranking, however
large the noise.

So the simulation exists for one reason: **a safety car is not zero-mean in
the plan's coordinates.** It costs every plan the same amount of lap time -
that part is a constant - but it discounts a pit stop by about two thirds
(measured, not assumed), and only a plan that has a stop left to make can
collect the discount. That makes the value of an extra stop depend on the
probability of a neutralisation, which is a genuinely stochastic question
with no closed form once the plan can react to what happens.

**Common random numbers.** Plans are compared on the *same* simulated
races: one draw of safety car timing and lap noise, every plan run through
it. Without that, the comparison between two plans whose means differ by a
few seconds is buried under the variance of the races themselves, and the
win probability - which is the output that matters - would be measuring
the wrong thing entirely. It also makes the comparison paired, so the
standard error of the difference is far smaller than the standard error of
either mean.

**What a reactive plan does.** A fixed plan stops on its planned laps. A
reactive plan brings its next stop forward if a neutralisation starts
within a window of laps before it. That is the cheapest possible version of
real strategy and it is the only reactive rule here: no responding to
rivals, because there are no rivals in this model.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from race import Params, check_plan, lap_time


@dataclass(frozen=True)
class Safety:
    """The neutralisation process, measured from the season in `run.py`."""

    hazard: float = 0.0196      # probability a period starts on a given lap
    length: float = 2.91        # mean laps per period
    factor: float = 1.35        # how much slower a neutralised lap runs
    discount: float = 0.342     # pit loss multiplier while neutralised

    def __post_init__(self) -> None:
        if not 0.0 <= self.hazard <= 1.0:
            raise ValueError("a hazard is a probability")
        if self.length < 1.0:
            raise ValueError("a neutralisation lasts at least one lap")
        if self.factor < 1.0:
            raise ValueError(
                "a neutralised lap cannot be quicker than a green one")


def draw_neutralisations(laps: int, sc: Safety, rng: np.random.Generator
                         ) -> np.ndarray:
    """A boolean lap mask: which laps of this race are neutralised.

    Periods start with probability `hazard` on each lap that is not already
    neutralised, and run for a geometric number of laps with the measured
    mean. Geometric because the hazard of a period ending is roughly flat
    once it has started - a safety car is called in when the track is
    clear, not after a fixed count.
    """
    mask = np.zeros(laps + 1, dtype=bool)
    lap = 1
    while lap <= laps:
        if rng.random() < sc.hazard:
            span = int(rng.geometric(1.0 / sc.length))
            end = min(lap + span - 1, laps)
            mask[lap:end + 1] = True
            lap = end + 1
        else:
            lap += 1
    return mask


def stop_laps(plan: tuple[int, ...]) -> list[int]:
    """The laps a plan pits on: the cumulative stint lengths but the last."""
    out, total = [], 0
    for stint in plan[:-1]:
        total += stint
        out.append(total)
    return out


def schedule(plan: tuple[int, ...], p: Params, neutral: np.ndarray, *,
             reactive: bool, window: int) -> list[int]:
    """The laps this plan actually pits on, given what the race did.

    A fixed plan returns its planned laps. A reactive plan brings its next
    stop forward to the current lap when a neutralisation is running and
    the stop was due within `window` laps. Later stops stay where they
    were planned, so the stint after an early stop is longer than the plan
    intended - a real cost of reacting, and one the caller pays rather
    than being quietly rebalanced away.
    """
    planned = stop_laps(plan)
    if not reactive:
        return planned
    taken, pending = [], list(planned)
    for lap in range(1, p.laps):
        if not pending:
            break
        if pending[0] == lap or (neutral[lap] and pending[0] - lap <= window):
            pending.pop(0)
            taken.append(lap)
    return taken + pending


def _age_profile(stops: list[int], laps: int) -> np.ndarray:
    """Tyre age on each lap 1..laps, given the laps pitted on."""
    age = np.empty(laps, dtype=float)
    current = 0
    boundaries = set(stops)
    for index in range(laps):
        current += 1
        age[index] = current
        if (index + 1) in boundaries:
            current = 0
    return age


def run_one(plan: tuple[int, ...], p: Params, sc: Safety,
            neutral: np.ndarray, noise: np.ndarray, *,
            reactive: bool = False, window: int = 5) -> float:
    """One race, lap by lap. The reference implementation.

    Deliberately plain. `run_vector` is what the Monte Carlo actually
    calls, and the two are required to agree to floating point in the
    tests - which is the only reason to trust the fast one.
    """
    check_plan(plan, p)
    stops = schedule(plan, p, neutral, reactive=reactive, window=window)
    total, age = 0.0, 0
    for lap in range(1, p.laps + 1):
        age += 1
        cost = lap_time(age, lap, p) + noise[lap - 1]
        if neutral[lap]:
            cost *= sc.factor
        total += cost
        if lap in stops:
            total += p.loss * (sc.discount if neutral[lap] else 1.0)
            age = 0
    return total


def run_vector(plan: tuple[int, ...], p: Params, sc: Safety,
               neutral: np.ndarray, noise: np.ndarray, *,
               reactive: bool = False, window: int = 5) -> float:
    """The same race in array form."""
    check_plan(plan, p)
    stops = schedule(plan, p, neutral, reactive=reactive, window=window)
    age = _age_profile(stops, p.laps)
    lap = np.arange(1, p.laps + 1, dtype=float)

    penalty = p.degradation * age
    if p.cliff_age is not None and p.cliff_rate:
        penalty += p.cliff_rate * np.maximum(age - p.cliff_age, 0.0)
    cost = p.base + penalty + p.fuel * lap + noise

    flags = neutral[1:p.laps + 1]
    cost = np.where(flags, cost * sc.factor, cost)
    pit = sum(p.loss * (sc.discount if neutral[stop] else 1.0)
              for stop in stops)
    return float(cost.sum() + pit)


def compare(plans: dict[str, tuple[int, ...]], p: Params, sc: Safety, *,
            draws: int = 20_000, sigma: float = 0.8, reactive: bool = False,
            window: int = 5, seed: int = 0) -> dict:
    """Run every plan through the same `draws` races.

    Returns per-plan summaries and the paired win probability of each plan
    against the plan with the lowest mean.
    """
    rng = np.random.default_rng(seed)
    names = list(plans)
    times = {name: np.empty(draws) for name in names}
    neutralised = np.empty(draws)

    for draw in range(draws):
        neutral = draw_neutralisations(p.laps, sc, rng)
        noise = rng.normal(0.0, sigma, p.laps)
        neutralised[draw] = neutral[1:].sum()
        for name in names:
            times[name][draw] = run_vector(plans[name], p, sc, neutral,
                                           noise, reactive=reactive,
                                           window=window)

    means = {name: float(times[name].mean()) for name in names}
    champion = min(means, key=means.get)
    # Which plan was quickest in each individual race, not on average.
    # A plan can hold the best mean and still lose most races, and that
    # distinction is the entire reason to run the simulation.
    stacked = np.vstack([times[name] for name in names])
    winners = np.argmin(stacked, axis=0)
    share = {name: float((winners == index).mean())
             for index, name in enumerate(names)}
    rows = []
    for name in names:
        gap = times[name] - times[champion]
        rows.append({
            "name": name, "plan": plans[name],
            "mean": means[name],
            "sd": float(times[name].std(ddof=1)),
            "p05": float(np.percentile(times[name], 5)),
            "p50": float(np.percentile(times[name], 50)),
            "p95": float(np.percentile(times[name], 95)),
            "win_share": share[name],
            "gap_mean": float(gap.mean()),
            # Paired, so this is the error on the difference and not on
            # either mean. It is the number that says whether a gap of a
            # second is real.
            "gap_error": (float(gap.std(ddof=1) / np.sqrt(draws))
                          if name != champion else 0.0),
        })
    return {
        "rows": sorted(rows, key=lambda row: row["mean"]),
        "best": champion,
        "draws": draws,
        "neutralised_mean": float(neutralised.mean()),
        "neutralised_any": float((neutralised > 0).mean()),
        "times": times,
    }


def hazard_crossover(plans: dict[str, tuple[int, ...]], p: Params, sc: Safety,
                     *, hazards: tuple[float, ...], draws: int = 8_000,
                     sigma: float = 0.8, reactive: bool = True,
                     window: int = 5, seed: int = 1) -> list[dict]:
    """Which plan wins as neutralisations get more likely.

    The deterministic optimum is the answer at hazard zero by
    construction. The interesting number is the hazard at which it stops
    being the answer, because that is a probability a strategist can
    actually have an opinion about before the race.
    """
    from dataclasses import replace

    rows = []
    for hazard in hazards:
        result = compare(plans, p, replace(sc, hazard=hazard), draws=draws,
                         sigma=sigma, reactive=reactive, window=window,
                         seed=seed)
        row = {"hazard": hazard, "best": result["best"],
               "neutralised_any": result["neutralised_any"]}
        for entry in result["rows"]:
            row[entry["name"]] = entry["mean"]
        rows.append(row)
    return rows
