"""The penalty the lap times cannot see, and what it would have to be.

The deterministic optimum, fed the degradation measured from 2023 lap times
and the pit loss measured from 2023 pit stops, recommends fewer stops than
the field actually made at almost every circuit, and never more. A
one-directional error across eighteen independent races is not noise. Some
cost of running a long stint is missing from the model.

The obvious candidate is the one every engineer would name first: tyre
performance is not linear in age. A set fades gently and then falls away.
A linear fit through a stint that was *ended before it fell away* measures
the gentle part and extrapolates it, which makes long stints look cheaper
than they are, which recommends fewer stops. Exactly the error observed.

**This cannot be settled by fitting harder, and that is the point.** Teams
choose when to stop, and they stop before the cliff. So the laps beyond the
age where a cliff would start are the laps that mostly do not exist in the
data - and the ones that do exist belong to cars that were running long
*because* their tyres were holding on, which is selection straight onto the
outcome. Two things are measured here instead of one:

`hinge`      fits an extra slope beyond a candidate age and tests it.
             This is the falsification attempt, and it reports how many
             laps are actually available beyond the knot, because a
             coefficient estimated from forty laps of a nineteen-thousand
             lap season is not evidence of absence.

`required`   inverts the question. Rather than asking what the cliff is,
             it asks what the cliff would have to be for the field's own
             choice to be the optimal one. That number is well defined
             whether or not the data can see it, and whether it comes out
             plausible or absurd is informative either way.

The inversion is not a measurement of the cliff and is not presented as
one. It is a statement of the form: *if the only thing missing from this
model is a cliff, here is how big it has to be.*
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

import model
from race import Params
from strategy import MAX_STINTS, MIN_STINT, best_plan


def hinge(arrays: dict, knot: int) -> dict:
    """Fit lap time on age, race lap, and max(age - knot, 0).

    The third column is the extra seconds per lap charged beyond the knot.
    Positive means a cliff. The fit is the same within estimator and the
    same cluster-robust errors as the degradation project; what is added
    here is the count of laps that inform the new column, which is the
    number that decides whether the test means anything.
    """
    age = arrays["age"]
    extra = np.maximum(age - knot, 0.0)
    beyond = int((age > knot).sum())
    if beyond < 30:
        return {"knot": knot, "beyond": beyond, "testable": False}

    matrix = np.column_stack([age, arrays["lap"], extra])
    fit = model.fit_matrix(
        arrays["seconds"], matrix, ("age", "lap", "beyond knot"),
        arrays["group"], arrays["n_groups"], name=f"hinge at {knot}")
    low, high = fit.interval("beyond knot")
    return {
        "knot": knot, "beyond": beyond,
        "share": beyond / age.size,
        "testable": True,
        "extra": fit["beyond knot"],
        "low": low, "high": high,
        "p": fit.p_value("beyond knot"),
        "age": fit["age"],
    }


def age_coverage(arrays: dict, ages: tuple[int, ...]) -> list[dict]:
    """How much of the record survives to each tyre age.

    The denominator of every claim about what old tyres do.
    """
    age = arrays["age"]
    return [{"age": value,
             "laps": int((age >= value).sum()),
             "share": float((age >= value).mean())}
            for value in ages]


def required(p: Params, target_stops: int, knot: int, *,
             ceiling: float = 2.0, tolerance: float = 1e-4,
             max_stints: int = MAX_STINTS, minimum: int = MIN_STINT
             ) -> float | None:
    """The smallest cliff rate at which `target_stops` becomes optimal.

    Bisection on the cliff rate. More cliff never means fewer stops, so
    the optimal stop count is monotone in the rate and the bracket is
    valid. Returns None when even `ceiling` seconds per lap beyond the
    knot is not enough, which happens when the knot sits beyond the
    longest stint any plan would run anyway - a cliff nobody reaches
    cannot change anybody's plan.
    """
    if target_stops < 0:
        raise ValueError("a negative number of stops is not a strategy")

    def stops_at(rate: float) -> int:
        trial = replace(p, cliff_age=knot, cliff_rate=rate)
        return best_plan(trial, max_stints=max_stints,
                         minimum=minimum).stops

    if stops_at(0.0) >= target_stops:
        return 0.0
    if stops_at(ceiling) < target_stops:
        return None

    low, high = 0.0, ceiling
    while high - low > tolerance:
        middle = (low + high) / 2.0
        if stops_at(middle) >= target_stops:
            high = middle
        else:
            low = middle
    return high


def reconcile(circuits: list[dict], knots: tuple[int, ...]) -> list[dict]:
    """The required cliff for every circuit, at every candidate knot.

    `circuits` carries the measured degradation, pit loss, race distance
    and the stop count the field most often chose.
    """
    rows = []
    for entry in circuits:
        if entry["degradation"] <= 0:
            # The optimiser refuses a non-positive degradation, and so
            # should this: there is no cliff that repairs a tyre model
            # whose linear term says tyres improve with age.
            rows.append({**entry, "rates": {knot: None for knot in knots},
                         "refused": True})
            continue
        p = Params(laps=entry["laps"], degradation=entry["degradation"],
                   loss=entry["loss"])
        rates = {knot: required(p, entry["ran"], knot) for knot in knots}
        rows.append({**entry, "rates": rates, "refused": False})
    return rows


def selection_demo(*, cars: int = 500, laps: int = 60, knot: int = 20,
                   rate: float = 0.080, degradation: float = 0.040,
                   spread: float = 0.030, fuel: float = -0.055,
                   noise: float = 0.8, threshold: float = 1.6,
                   max_stint: int = 45, fixed_stints: int = 2,
                   seed: int = 0) -> dict:
    """Generate laps that really do have a cliff, then try to find it.

    Every car in both arms is given the **same** true cliff: an extra
    `rate` seconds per lap of age beyond `knot`. The arms differ only in
    how stints are ended.

    `fixed`      every car runs the same `fixed_stints` equal stints,
                 chosen before the race and never revised. The split has
                 to be long enough for the stints to cross the knot, or
                 the control arm has no laps beyond it to measure and the
                 comparison is empty.
    `selected`   a car ends a stint once its own tyre penalty passes
                 `threshold` seconds a lap. Cars are not identical - each
                 draws its own degradation - so this is a rule that stops
                 the cars with the worst tyres first.

    Nothing else changes. If the hinge estimate survives the first arm and
    collapses in the second, the collapse is caused by the stopping rule
    and by nothing else, which is the claim being tested. This is the same
    argument as the real data's negative hinge coefficient, run where the
    truth is known.
    """
    rng = np.random.default_rng(seed)
    out = {"truth": rate, "knot": knot, "arms": {}}

    for arm in ("fixed", "selected"):
        seconds, ages, race_laps, groups = [], [], [], []
        for car in range(cars):
            own = max(rng.normal(degradation, spread), 0.002)
            pace = rng.normal(0.0, 3.0)
            age, lap = 0, 0
            while lap < laps:
                lap += 1
                age += 1
                penalty = own * age + rate * max(age - knot, 0)
                seconds.append(95.0 + pace + penalty + fuel * lap
                               + rng.normal(0.0, noise))
                ages.append(age)
                race_laps.append(lap)
                groups.append(car)
                if arm == "fixed":
                    ends = age >= laps // fixed_stints and lap < laps
                else:
                    ends = (penalty > threshold or age >= max_stint) \
                        and lap < laps
                if ends:
                    age = 0

        age_array = np.array(ages, dtype=float)
        arrays = {"seconds": np.array(seconds), "age": age_array,
                  "lap": np.array(race_laps, dtype=float),
                  "group": np.array(groups), "n_groups": cars}
        found = hinge(arrays, knot)
        out["arms"][arm] = {
            **found,
            "mean_stint": float(np.mean(np.diff(
                np.flatnonzero(age_array == 1), append=age_array.size))),
            "max_age": int(age_array.max()),
        }
    return out
