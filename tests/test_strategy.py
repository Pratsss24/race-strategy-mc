"""What has to be true for any of the numbers to mean anything.

The anchors here are of three kinds, in descending order of how much they
are worth.

**Algebra.** The race time decomposition, the optimality of the balanced
split, and `k* = N sqrt(d/2L)` are provable, so they are checked against
exact values rather than against saved output. If the implementation
drifts, these fail with no judgement call involved.

**Agreement between two implementations.** The lap-by-lap race, the closed
form, and the vectorised simulator all compute the same quantity by
different routes. They are required to agree to floating point. That is the
only reason to trust the fast one, which is the one the Monte Carlo
actually runs.

**Recovery of a known truth.** The optimiser is given a race whose answer
is constructed, and the selection demonstration is given data with a cliff
written into it. Both are checked for recovering what was put in - and the
selection arm is checked for *failing* to, because that failure is the
project's central claim and a version of the code that quietly recovered
the cliff anyway would be wrong.

The real-data tests at the end pin the headline numbers. They skip rather
than fetch: run `python run.py` once to cache the season.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import cliff          # noqa: E402
import data           # noqa: E402
import model          # noqa: E402
import montecarlo as mc   # noqa: E402
import pitloss        # noqa: E402
import race           # noqa: E402
import stints         # noqa: E402
import strategy       # noqa: E402

PARAMS = race.Params(laps=57, degradation=0.032, fuel=-0.055, loss=21.75)


@pytest.fixture(scope="module")
def feed():
    """The cached season. The suite never goes to the network itself."""
    missing = [season for season in (2023,)
               if not (ROOT / "data" / f"season_{season}.json").exists()]
    if missing:
        pytest.skip("run `python run.py` once to cache the season")
    return data.load()


@pytest.fixture(scope="module")
def arrays(feed):
    laps, stops, results = feed
    rows, _ = stints.build(laps, stops, results)
    return stints.arrays(rows)


# ---------------------------------------------------------------- algebra

def test_the_closed_form_is_the_lap_by_lap_sum():
    """Not approximately. The decomposition is exact or it is wrong."""
    for plan in ((57,), (29, 28), (19, 19, 19), (30, 20, 7), (15, 14, 14, 14)):
        assert race.race_time(plan, PARAMS) == pytest.approx(
            race.race_time_closed(plan, PARAMS), abs=1e-9)


def test_plan_cost_differs_from_race_time_by_a_constant():
    """`plan_cost` drops the terms no plan can change, so it must shift
    every plan by the same amount - otherwise it would reorder them, and
    the optimiser runs on it."""
    gaps = {round(race.race_time(plan, PARAMS)
                  - race.plan_cost(plan, PARAMS), 9)
            for plan in strategy.partitions(57, 3)}
    assert len(gaps) == 1


def test_plan_cost_still_tracks_race_time_with_a_cliff():
    cliffed = race.Params(laps=57, degradation=0.032, loss=21.0,
                          cliff_age=18, cliff_rate=0.09)
    gaps = {round(race.race_time(plan, cliffed)
                  - race.plan_cost(plan, cliffed), 9)
            for plan in strategy.partitions(57, 3)}
    assert len(gaps) == 1


def test_stint_order_cannot_change_race_time():
    """A property of the model, and the sharpest statement of what it
    leaves out: there is no undercut in here."""
    assert race.race_time((30, 20, 7), PARAMS) == pytest.approx(
        race.race_time((7, 30, 20), PARAMS), abs=1e-9)
    assert race.race_time((30, 20, 7), PARAMS) == pytest.approx(
        race.race_time((20, 7, 30), PARAMS), abs=1e-9)


def test_the_balanced_split_is_optimal_at_every_stint_count():
    """Convexity, checked by exhaustion rather than asserted."""
    for laps in (44, 51, 57, 66, 70):
        for stints_ in range(1, 6):
            options = strategy.partitions(laps, stints_)
            if not options:
                continue
            p = race.Params(laps=laps, degradation=0.04, loss=21.0)
            found = min(options, key=lambda plan: race.plan_cost(plan, p))
            assert (sorted(found, reverse=True)
                    == sorted(race.balanced(laps, stints_), reverse=True))


def test_the_balanced_split_survives_a_cliff():
    """The per-stint cost is still convex with a hinge in it, so the
    balanced split is still the minimiser. If this failed, the optimiser
    would need a real search rather than a shortcut."""
    for rate in (0.05, 0.2, 0.8):
        for knot in (12, 20, 28):
            p = race.Params(laps=66, degradation=0.04, loss=21.0,
                            cliff_age=knot, cliff_rate=rate)
            for stints_ in range(1, 6):
                options = strategy.partitions(66, stints_)
                found = min(options, key=lambda plan: race.plan_cost(plan, p))
                assert (sorted(found, reverse=True)
                        == sorted(race.balanced(66, stints_), reverse=True))


def test_k_star_matches_a_hand_computation():
    """k* = N sqrt(d / 2L). Picked so the answer is exactly 3."""
    p = race.Params(laps=60, degradation=0.02, loss=4.0)
    # 60 * sqrt(0.02 / 8) = 60 * 0.05 = 3
    assert race.continuous_optimum(p) == pytest.approx(3.0)


def test_the_search_finds_what_the_arithmetic_already_knew():
    for degradation in (0.02, 0.05, 0.10, 0.19):
        for loss in (17.0, 21.75, 27.0):
            p = race.Params(laps=57, degradation=degradation, loss=loss)
            assert strategy.agrees_with_closed_form(p)


def test_more_degradation_never_means_fewer_stops():
    """Monotonicity. `across_interval` brackets the optimum with the two
    ends of a confidence interval, which is only valid if this holds."""
    previous = 0
    for degradation in np.linspace(0.005, 0.30, 40):
        p = race.Params(laps=57, degradation=float(degradation), loss=21.75)
        stops = strategy.best_plan(p).stops
        assert stops >= previous
        previous = stops


def test_a_dearer_stop_never_means_more_stops():
    previous = 99
    for loss in np.linspace(5.0, 60.0, 30):
        p = race.Params(laps=57, degradation=0.06, loss=float(loss))
        stops = strategy.best_plan(p).stops
        assert stops <= previous
        previous = stops


# --------------------------------------------------------------- refusals

def test_negative_degradation_is_refused_not_optimised():
    """Tyres that improve with age are a data problem. An optimiser that
    returned a plan here would be lending precision to a number with
    none."""
    with pytest.raises(ValueError, match="negative degradation"):
        race.Params(degradation=-0.01)


def test_a_free_pit_stop_is_refused():
    with pytest.raises(ValueError, match="unbounded"):
        race.Params(loss=0.0)


def test_a_plan_that_does_not_cover_the_race_is_refused():
    with pytest.raises(ValueError, match="runs 50 laps but the race is 57"):
        race.race_time((25, 25), PARAMS)


def test_the_closed_form_refuses_to_pretend_about_a_cliff():
    cliffed = race.Params(laps=57, degradation=0.032, loss=21.0,
                          cliff_age=18, cliff_rate=0.09)
    with pytest.raises(ValueError, match="linear age penalty"):
        race.race_time_closed((29, 28), cliffed)
    with pytest.raises(ValueError, match="does not apply"):
        race.closed_form_plan(cliffed)


def test_an_interval_spanning_zero_does_not_choose_a_strategy():
    """Silverstone's measured degradation is -0.0032 with an interval of
    -0.026 to +0.020. The honest output is a range, not a plan."""
    p = race.Params(laps=52, degradation=0.0197, loss=20.31)
    answer = strategy.across_interval(p, -0.0260, 0.0197)
    assert answer["low_stints"] == 1
    assert not answer["decided"] or answer["range"][0] == answer["range"][1]


def test_even_the_sharpest_circuit_leaves_the_stop_count_open():
    """Spa has the steepest and best-determined degradation of the season,
    +0.1912 with an interval of +0.151 to +0.231, and that interval still
    spans three and four stints. This was written expecting it to decide.
    It does not, and the measured season is full of the same thing: the
    input is nowhere near precise enough to pin a strategy."""
    p = race.Params(laps=44, degradation=0.1912, loss=17.02)
    answer = strategy.across_interval(p, 0.1511, 0.2312)
    assert not answer["decided"]
    assert answer["range"] == (3, 4)


def test_a_narrow_enough_interval_does_decide():
    """Monza's is narrow enough, which is what makes the Spa result a
    statement about precision rather than about the function."""
    p = race.Params(laps=51, degradation=0.0657, loss=24.92)
    answer = strategy.across_interval(p, 0.0557, 0.0757)
    assert answer["decided"]
    assert answer["range"] == (2, 2)


# -------------------------------------------- simulator against reference

def test_the_fast_simulator_matches_the_plain_one():
    """The vectorised path is what the Monte Carlo runs. It is trusted
    only because it agrees with the obvious implementation, fixed and
    reactive, with and without a cliff."""
    sc = mc.Safety()
    rng = np.random.default_rng(7)
    worst = 0.0
    for params in (PARAMS, race.Params(laps=57, degradation=0.032,
                                       loss=21.75, cliff_age=20,
                                       cliff_rate=0.08)):
        for _ in range(120):
            neutral = mc.draw_neutralisations(params.laps, sc, rng)
            noise = rng.normal(0.0, 0.7, params.laps)
            for plan in ((57,), (29, 28), (19, 19, 19), (15, 14, 14, 14)):
                for reactive in (False, True):
                    plain = mc.run_one(plan, params, sc, neutral, noise,
                                       reactive=reactive)
                    fast = mc.run_vector(plan, params, sc, neutral, noise,
                                         reactive=reactive)
                    worst = max(worst, abs(plain - fast))
    assert worst < 1e-9


def test_with_no_safety_car_the_simulation_is_the_deterministic_race():
    """Independent zero-mean noise cannot move a mean. If this drifts,
    the simulator has a bias in it somewhere."""
    quiet = mc.Safety(hazard=0.0)
    plans = {"1 stop": (29, 28), "2 stops": (19, 19, 19)}
    result = mc.compare(plans, PARAMS, quiet, draws=6_000, sigma=2.0, seed=4)
    for row in result["rows"]:
        exact = race.race_time(row["plan"], PARAMS)
        error = 2.0 * np.sqrt(PARAMS.laps) / np.sqrt(6_000)
        assert row["mean"] == pytest.approx(exact, abs=4 * error)


def test_noise_cannot_reorder_the_plans():
    """However large it is. The ranking of means is deterministic, which
    is why the simulation exists only for the safety car."""
    quiet = mc.Safety(hazard=0.0)
    plans = {"1 stop": (29, 28), "3 stops": (15, 14, 14, 14)}
    for sigma in (0.5, 4.0, 12.0):
        result = mc.compare(plans, PARAMS, quiet, draws=4_000, sigma=sigma,
                            seed=11)
        assert result["best"] == "1 stop"


def test_reacting_to_a_safety_car_is_never_worse_than_ignoring_it():
    """The reactive rule only ever brings a stop forward onto a discounted
    lap, so over enough shared races it cannot lose. Paired, so this is a
    comparison of the same races."""
    sc = mc.Safety()
    plan = {"2 stops": (19, 19, 19)}
    fixed = mc.compare(plan, PARAMS, sc, draws=8_000, sigma=0.7, seed=2,
                       reactive=False)
    reactive = mc.compare(plan, PARAMS, sc, draws=8_000, sigma=0.7, seed=2,
                          reactive=True)
    assert reactive["rows"][0]["mean"] < fixed["rows"][0]["mean"]


def test_a_bigger_discount_is_worth_more_to_a_plan_with_a_stop_left():
    """The mechanism the whole simulation turns on, isolated."""
    plan = {"2 stops": (19, 19, 19)}
    cheap = mc.compare(plan, PARAMS, mc.Safety(discount=0.1), draws=6_000,
                       sigma=0.7, seed=5, reactive=True)
    dear = mc.compare(plan, PARAMS, mc.Safety(discount=0.9), draws=6_000,
                      sigma=0.7, seed=5, reactive=True)
    assert cheap["rows"][0]["mean"] < dear["rows"][0]["mean"]


def test_the_neutralisation_mask_has_the_measured_frequency():
    sc = mc.Safety(hazard=0.02, length=3.0)
    rng = np.random.default_rng(0)
    draws = [mc.draw_neutralisations(60, sc, rng) for _ in range(4_000)]
    flagged = np.mean([mask[1:].sum() for mask in draws])
    # Roughly hazard * laps * length, allowing for periods truncated at
    # the end of the race and for laps already inside a period.
    assert 2.0 < flagged < 4.0


def test_a_neutralised_lap_cannot_be_quicker_than_a_green_one():
    with pytest.raises(ValueError, match="quicker"):
        mc.Safety(factor=0.9)


def test_the_schedule_only_ever_brings_a_stop_forward():
    plan = (19, 19, 19)
    neutral = np.zeros(58, dtype=bool)
    neutral[16:19] = True
    planned = mc.stop_laps(plan)
    taken = mc.schedule(plan, PARAMS, neutral, reactive=True, window=5)
    assert taken[0] == 16 and taken[0] < planned[0]
    assert all(a <= b for a, b in zip(taken, planned))
    assert len(taken) == len(planned)


def test_a_fixed_plan_ignores_the_safety_car_entirely():
    plan = (19, 19, 19)
    neutral = np.ones(58, dtype=bool)
    assert mc.schedule(plan, PARAMS, neutral, reactive=False,
                       window=5) == mc.stop_laps(plan)


# ------------------------------------------------- recovery and selection

def test_the_optimiser_recovers_a_constructed_answer():
    """A race built so that three stints is right by construction: k* is
    exactly 3, and the balanced split of 60 into 3 is 20-20-20."""
    p = race.Params(laps=60, degradation=0.02, loss=4.0)
    best = strategy.best_plan(p)
    assert best.stints == 3
    assert best.plan == (20, 20, 20)


def test_the_hinge_recovers_a_cliff_when_plans_are_fixed():
    """The control arm. The estimator is sound; everything after this
    about the cliff being invisible is about the data, not the method."""
    demo = cliff.selection_demo(seed=1)
    fixed = demo["arms"]["fixed"]
    assert fixed["testable"]
    assert fixed["low"] < demo["truth"] < fixed["high"]


def test_a_stopping_rule_destroys_the_cliff_the_data_contains():
    """The project's central claim, where the truth is known. The
    selection arm must recover materially less than the truth - a version
    of this code that found the cliff anyway would mean the argument
    about the real data was wrong."""
    demo = cliff.selection_demo(seed=1)
    selected = demo["arms"]["selected"]
    assert selected["extra"] < 0.5 * demo["truth"]
    assert selected["high"] < demo["truth"]


def test_the_attenuation_gets_worse_the_sooner_teams_stop():
    """Monotone in the stopping threshold, which is what makes it
    selection rather than an artefact of one arbitrary rule."""
    recovered = [cliff.selection_demo(threshold=value, seed=2)
                 ["arms"]["selected"]["extra"]
                 for value in (1.0, 1.6, 2.5)]
    assert recovered[0] < recovered[1] < recovered[2]


def test_the_hinge_refuses_a_knot_with_nothing_beyond_it():
    arrays_ = {"seconds": np.zeros(50), "age": np.arange(1.0, 51.0),
               "lap": np.arange(1.0, 51.0), "group": np.zeros(50, dtype=int),
               "n_groups": 1}
    assert not cliff.hinge(arrays_, 45)["testable"]


def test_the_required_cliff_is_zero_where_the_model_already_agrees():
    """Spa: the model and the field both say two stops, so nothing is
    missing and the inversion must say so rather than inventing a
    penalty."""
    p = race.Params(laps=44, degradation=0.1912, loss=17.02)
    assert strategy.best_plan(p).stops == 2
    assert cliff.required(p, 2, 20) == 0.0


def test_the_required_cliff_rises_with_the_number_of_stops_to_explain():
    """The knot has to sit below the stints being penalised. At a knot of
    18 over 57 laps a three-stop runs stints of 14 and never reaches it,
    so no cliff there can favour three stops - which the next test
    covers."""
    p = race.Params(laps=57, degradation=0.032, loss=21.75)
    for knot in (8, 10, 12):
        two = cliff.required(p, 2, knot)
        three = cliff.required(p, 3, knot)
        assert two is not None and three is not None
        assert three > two > 0


def test_a_knot_above_the_shorter_stints_cannot_favour_them():
    """A cliff only bites on stints long enough to reach it. Asking for
    three stops over 57 laps with a knot at 18 is asking a penalty that
    only touches the alternatives to pay for an extra 21.75 s stop, and
    within the ceiling it cannot."""
    p = race.Params(laps=57, degradation=0.032, loss=21.75)
    assert cliff.required(p, 3, 18) is None
    assert cliff.required(p, 2, 18) is not None


def test_a_cliff_beyond_every_stint_cannot_change_a_plan():
    """A knot no plan reaches is unidentified from strategy as well as
    from the data, and the inversion returns nothing rather than a
    number."""
    p = race.Params(laps=57, degradation=0.032, loss=21.75)
    assert cliff.required(p, 3, 56, ceiling=5.0) is None


# -------------------------------------------------------------- real data

def test_the_pit_loss_is_measured_not_read_from_the_feed(feed):
    """The feed's `duration` is a pit-lane transit, which is why it is not
    used as the pit loss. The two must differ, and the measured loss must
    be the smaller of the two on average."""
    laps, stops, results = feed
    losses, _ = pitloss.measure(laps, stops, results)
    summary = pitloss.summarise(losses)
    assert 18.0 < summary["green"] < 26.0
    assert summary["transit"] > summary["green"]


def test_a_stop_under_a_safety_car_is_cheaper(feed):
    """And by a lot. If this came out above one, the counterfactual for a
    neutralised lap would be wrong - which it was, the first time."""
    laps, stops, results = feed
    losses, _ = pitloss.measure(laps, stops, results)
    summary = pitloss.summarise(losses)
    assert 0.2 < summary["discount"] < 0.6
    assert summary["neutral"] < summary["green"]


def test_measuring_a_neutralised_stop_against_a_green_lap_inverts_it(feed):
    """The bug that was there first, kept as a test. Without scaling the
    counterfactual to the neutralised pace, a safety car stop looks more
    expensive than a green one instead of a third of the price."""
    laps, stops, results = feed
    losses, _ = pitloss.measure(laps, stops, results)
    import pitloss as module
    saved = module.neutralisation_factor
    try:
        module.neutralisation_factor = lambda rows: {}
        wrong, _ = module.measure(laps, stops, results)
    finally:
        module.neutralisation_factor = saved
    assert pitloss.summarise(wrong)["discount"] > 1.0
    assert pitloss.summarise(losses)["discount"] < 1.0


def test_the_degradation_headline_is_unchanged(feed):
    """The number this project takes from the last one."""
    laps, stops, results = feed
    rows, _ = stints.build(laps, stops, results)
    fit = model.fit_within(stints.arrays(rows))
    assert fit["age"] == pytest.approx(0.0316, abs=0.0005)
    assert fit["lap"] == pytest.approx(-0.0546, abs=0.0005)


def test_the_real_laps_show_no_cliff_and_the_sign_is_wrong(arrays):
    """Not a null result but a wrong-signed one, which is what the
    selection argument has to explain."""
    found = cliff.hinge(arrays, 25)
    assert found["testable"]
    assert found["extra"] < 0
    assert found["high"] < 0           # significantly negative


def test_the_model_underestimates_stops_and_never_the_other_way(feed):
    """The validation result, as a test. The direction is the finding, so
    if a change ever made the errors two-sided that would be a real
    change in what the project claims."""
    laps, stops, results = feed
    rows, _ = stints.build(laps, stops, results)
    losses, _ = pitloss.measure(laps, stops, results)
    loss_by = {row["circuit"]: row["loss"]
               for row in pitloss.by_circuit(losses)}

    import collections
    per_car: dict = collections.defaultdict(int)
    for stop in stops:
        per_car[(stop.season, stop.round, stop.driver)] += 1
    distance, chosen = {}, collections.defaultdict(list)
    for row in results:
        distance[row.circuit] = max(distance.get(row.circuit, 0), row.laps)
        if row.status in stints.FINISHED:
            chosen[row.circuit].append(
                per_car.get((row.season, row.round, row.driver), 0))
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[row.circuit].append(row)

    under = over = agree = 0
    for circuit, group in grouped.items():
        if len(group) < 400 or circuit not in loss_by:
            continue
        fit = model.fit_within(stints.arrays(group))
        if fit["age"] <= 0:
            continue
        p = race.Params(laps=distance[circuit], degradation=fit["age"],
                        loss=loss_by[circuit])
        modelled = strategy.best_plan(p).stops
        ran = collections.Counter(chosen[circuit]).most_common(1)[0][0]
        under += modelled < ran
        over += modelled > ran
        agree += modelled == ran
    # The direction is the whole finding; the margin is what it is. In
    # 2023 it is nine too few, five right, none too many.
    assert over == 0
    assert under > agree


def test_the_season_contains_the_neutralisations_the_simulation_assumes(feed):
    """The hazard is measured, so the simulated frequency of a quiet race
    has to look like the real one."""
    laps, stops, results = feed
    flagged = stints.neutralised_laps(laps)
    periods = 0
    for marked in flagged.values():
        runs = []
        for lap in sorted(marked):
            if runs and lap == runs[-1][-1] + 1:
                runs[-1].append(lap)
            else:
                runs.append([lap])
        periods += len(runs)
    assert 15 <= periods <= 60


def test_lap_time_parsing():
    assert data.parse_lap_time("1:33.996") == pytest.approx(93.996)
    assert data.parse_lap_time("93.996") == pytest.approx(93.996)
    assert data.parse_lap_time("") is None
    assert data.parse_lap_time("retired") is None
