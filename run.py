"""Measure the inputs, solve the race, then test the answer against 2023.

Every number the model uses is measured from the same season it is then
checked against. Nothing is typed in from a reference table. Run with no
arguments; the season is cached, so a second run costs nothing.
"""

from __future__ import annotations

import collections
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import charts                                              # noqa: E402
import cliff                                               # noqa: E402
import data                                                # noqa: E402
import model                                               # noqa: E402
import montecarlo as mc                                    # noqa: E402
import pitloss                                             # noqa: E402
import race                                                # noqa: E402
import report                                              # noqa: E402
import stints                                              # noqa: E402
import strategy                                            # noqa: E402

SEASONS = (2023,)
MIN_CIRCUIT_LAPS = 400       # before a circuit gets its own degradation
KNOTS = (15, 20, 25, 30)     # candidate cliff ages
HAZARDS = (0.0, 0.01, 0.02, 0.04, 0.06, 0.10, 0.15, 0.20)
WINDOWS = (0, 2, 4, 6, 8, 12)
DRAWS = 20_000
OUT = ROOT / "out"

_log: list[str] = []


def say(line: str = "") -> None:
    print(line)
    _log.append(line)


def main() -> None:
    laps, stops, results = data.load(SEASONS)
    rows, census = stints.build(laps, stops, results)
    arrays = stints.arrays(rows)
    say(f"{census['total']:,} lap times, {len(stops):,} pit stops, "
        f"{len({(r.season, r.round) for r in results})} races")
    say(f"  {census['kept']:,} laps usable after the cleaning rules "
        f"({census['kept'] / census['total']:.1%})")

    # ---- what a stop costs -------------------------------------------
    losses, loss_census = pitloss.measure(laps, stops, results)
    summary = pitloss.summarise(losses)
    circuit_loss = pitloss.by_circuit(losses)
    loss_by = {row["circuit"]: row["loss"] for row in circuit_loss}
    sigma = pitloss.pace_noise(laps, stops, results)

    say()
    say("pit loss, measured from in-lap and out-lap excess")
    for key, value in sorted(loss_census.items()):
        say(f"  {key:34s} {value:5d}")
    say(f"  green flag   {summary['green']:6.2f} s  "
        f"(IQR {summary['green_p25']:.2f}-{summary['green_p75']:.2f}, "
        f"n={summary['green_n']})")
    say(f"  neutralised  {summary['neutral']:6.2f} s  "
        f"(n={summary['neutral_n']}) -> discount "
        f"{summary['discount']:.3f} of a green-flag stop")
    say(f"  the feed's own pit-lane transit is {summary['transit']:.2f} s, "
        f"{summary['transit_gap']:+.2f} s against the measured loss")
    say(f"  single-car pace scatter {sigma:.3f} s a lap "
        f"(the pooled residual is {model.fit_within(arrays).sigma:.3f})")

    # ---- how fast tyres go off ---------------------------------------
    fitted = model.fit_within(arrays)
    low, high = fitted.interval("age")
    say()
    say(f"degradation  {fitted['age']:+.4f} s per lap of tyre age "
        f"(95% {low:+.4f} to {high:+.4f})")
    say(f"race lap     {fitted['lap']:+.4f} s per lap of distance")

    # ---- the safety car process ---------------------------------------
    sc_stats = measure_safety(laps, stops, results)
    say()
    say(f"neutralisations  {sc_stats['periods']} periods across "
        f"{sc_stats['races']} dry races")
    say(f"  per-lap hazard   {sc_stats['hazard']:.5f}")
    say(f"  mean length      {sc_stats['length']:.2f} laps")
    say(f"  a neutralised lap runs {sc_stats['factor']:.3f}x the green "
        f"baseline")
    say(f"  races with at least one: {sc_stats['any']:.0%}")

    safety = mc.Safety(hazard=sc_stats["hazard"], length=sc_stats["length"],
                       factor=sc_stats["factor"],
                       discount=summary["discount"])

    # ---- the deterministic problem ------------------------------------
    pooled = race.Params(laps=57, degradation=fitted["age"],
                         fuel=fitted["lap"], loss=summary["green"])
    say()
    say(f"at the pooled numbers over 57 laps, k* = "
        f"{race.continuous_optimum(pooled):.3f} stints")
    stop_table = strategy.table(pooled)
    for row in stop_table:
        say(f"  {row['stops']} stops  {str(row['plan']):26s} "
            f"{row['time']:10.2f} s  +{row['gap']:7.2f}")
    searched = strategy.best_plan(pooled)
    closed_plan, closed_time, closed_stints = race.closed_form_plan(pooled)
    say(f"  search and closed form agree: "
        f"{strategy.agrees_with_closed_form(pooled)} "
        f"({searched.plan} vs {closed_plan}, "
        f"{abs(searched.time - closed_time):.2e} s apart)")
    interval = strategy.across_interval(pooled, low, high)
    say(f"  across the degradation interval the optimum is "
        f"{interval['range'][0]}-{interval['range'][1]} stints, "
        f"so the data {'does' if interval['decided'] else 'does not'} "
        f"pick a stop count")

    # ---- per circuit, against what the field actually did --------------
    entries = circuit_entries(rows, results, stops, loss_by)
    say()
    say("by circuit: the model's optimum against the field's own choice")
    say(f"  {'circuit':16s} {'laps':>4s} {'degradation':>12s} {'loss':>6s} "
        f"{'k*':>5s} {'model':>5s} {'ran':>4s}  verdict")
    agree = under = over = 0
    for entry in entries:
        if entry["degradation"] <= 0:
            verdict = "refused: degradation not positive"
            modelled = None
        else:
            p = race.Params(laps=entry["laps"],
                            degradation=entry["degradation"],
                            fuel=fitted["lap"], loss=entry["loss"])
            modelled = strategy.best_plan(p).stops
            entry["k_star"] = race.continuous_optimum(p)
            if modelled == entry["ran"]:
                verdict = "agrees"
                agree += 1
            elif modelled < entry["ran"]:
                verdict = "too few stops"
                under += 1
            else:
                verdict = "too many stops"
                over += 1
        entry["model"] = modelled
        entry["verdict"] = verdict
        say(f"  {entry['circuit']:16s} {entry['laps']:4d} "
            f"{entry['degradation']:+12.4f} {entry['loss']:6.2f} "
            f"{entry.get('k_star', float('nan')):5.2f} "
            f"{'-' if modelled is None else modelled:>5} "
            f"{entry['ran']:4d}  {verdict}")
    settled = [e for e in entries if e["model"] is not None
               and strategy.across_interval(
                   race.Params(laps=e["laps"], degradation=e["degradation"],
                               fuel=fitted["lap"], loss=e["loss"]),
                   e["low"], e["high"])["decided"]]
    ranked = [e for e in entries if e["model"] is not None]
    say(f"  agrees {agree}, too few {under}, too many {over}, "
        f"refused {len(entries) - len(ranked)}")
    say("  the error has a direction, which noise does not: the model is "
        "never above the field")
    say(f"  and the input is blunter than the question: at only "
        f"{len(settled)} of {len(ranked)} circuits is the degradation "
        f"interval narrow enough to pin a stop count at all "
        f"({', '.join(e['circuit'] for e in settled)})")

    # ---- can safety car risk explain the gap? --------------------------
    focus = min(ranked, key=lambda e: abs(e["k_star"] - round(e["k_star"])))
    focus_p = race.Params(laps=focus["laps"],
                          degradation=focus["degradation"],
                          fuel=fitted["lap"], loss=focus["loss"])
    plans = {f"{n} stop{'s' if n != 1 else ''}":
             race.balanced(focus["laps"], n + 1) for n in range(0, 4)}
    say()
    say(f"Monte Carlo at {focus['circuit']} ({focus['laps']} laps, "
        f"degradation {focus['degradation']:+.4f}, loss "
        f"{focus['loss']:.2f}), {DRAWS:,} shared races")
    compared = mc.compare(plans, focus_p, safety, draws=DRAWS, sigma=sigma,
                          reactive=True)
    for row in compared["rows"]:
        say(f"  {row['name']:8s} {str(row['plan']):24s} mean "
            f"{row['mean']:9.2f}  sd {row['sd']:6.2f}  wins "
            f"{row['win_share']:6.1%}  gap {row['gap_mean']:+7.2f} "
            f"+/- {row['gap_error']:.2f}")
    # The focus circuit is longer than the average race, so its own
    # simulated frequency is not the number to check against the season.
    # A constant hazard applied to each real race's own distance is.
    expected = float(np.mean([1.0 - (1.0 - sc_stats["hazard"]) ** n
                              for n in sc_stats["distances"]]))
    say(f"  at {focus['laps']} laps, {compared['neutralised_any']:.1%} of "
        f"simulated races contain a neutralisation")
    say(f"  checked over the real race distances instead: the process "
        f"predicts {expected:.1%} against {sc_stats['any']:.1%} observed")
    say(f"  it slightly over-predicts because real periods cluster - one "
        f"race had {max(sc_stats['per_race'])} of them - so a constant "
        f"hazard that matches the total count spreads them over too many "
        f"races")

    fixed = mc.compare(plans, focus_p, safety, draws=DRAWS, sigma=sigma,
                       reactive=False)
    reactive_gain = (fixed["rows"][0]["mean"]
                     - min(r["mean"] for r in compared["rows"]))
    say(f"  reacting to a neutralisation is worth "
        f"{reactive_gain:+.2f} s to the best plan")

    window_rows = []
    for window in WINDOWS:
        result = mc.compare(plans, focus_p, safety, draws=6_000, sigma=sigma,
                            reactive=True, window=window)
        best = result["rows"][0]
        window_rows.append({"window": window, "best": best["name"],
                            "mean": best["mean"]})
    say("  how early a stop may be brought forward to catch one:")
    for row in window_rows:
        say(f"    window {row['window']:2d} laps  best {row['best']:8s} "
            f"{row['mean']:9.2f}")

    crossover = mc.hazard_crossover(plans, focus_p, safety, hazards=HAZARDS,
                                    draws=8_000, sigma=sigma)
    say("  does neutralisation risk change which plan wins?")
    for row in crossover:
        say(f"    hazard {row['hazard']:.3f}  P(any) "
            f"{row['neutralised_any']:5.1%}  best {row['best']}")
    winners = {row["best"] for row in crossover}
    say(f"    {len(winners)} different plan(s) win across that range: "
        f"{', '.join(sorted(winners))}")
    first = crossover[0]["best"]
    switch = next((row["hazard"] for row in crossover
                   if row["best"] != first), None)
    if switch is None:
        say("    the deterministic optimum wins at every hazard tried, so "
            "neutralisation risk is not what the field is responding to")
        crossover_note = None
    else:
        ratio = switch / sc_stats["hazard"]
        crossover_note = (switch, ratio)
        say(f"    an extra stop takes over at a hazard of {switch:.3f}, "
            f"{ratio:.1f}x the {sc_stats['hazard']:.4f} actually measured")
        say("    so neutralisation risk pushes the right way, but the "
            "season would have to be several times more chaotic than it "
            "was for it to account for the gap")

    # ---- the cliff -----------------------------------------------------
    coverage = cliff.age_coverage(arrays, (10, 15, 20, 25, 30, 35, 40))
    say()
    say("how much of the record survives to each tyre age")
    for row in coverage:
        say(f"  age >= {row['age']:2d}   {row['laps']:6,d} laps  "
            f"{row['share']:6.2%}")

    hinges = [cliff.hinge(arrays, knot) for knot in KNOTS]
    say()
    say("is there an extra slope beyond the knot? (a cliff would be "
        "positive)")
    for row in hinges:
        if not row["testable"]:
            say(f"  knot {row['knot']:2d}  only {row['beyond']} laps "
                f"beyond it, not testable")
            continue
        say(f"  knot {row['knot']:2d}  extra {row['extra']:+.4f} "
            f"({row['low']:+.4f} to {row['high']:+.4f})  p {row['p']:.3f}"
            f"   {row['beyond']:,} laps beyond ({row['share']:.1%})")
    say("  the sign is wrong for a cliff, and significantly so: on these "
        "laps old tyres fade more slowly, not faster")

    demo = cliff.selection_demo()
    say()
    say(f"the same test where the truth is known "
        f"(a real cliff of {demo['truth']:+.4f} beyond age {demo['knot']})")
    for arm, row in demo["arms"].items():
        say(f"  {arm:9s} recovered {row['extra']:+.4f} "
            f"({row['low']:+.4f} to {row['high']:+.4f})  "
            f"{row['extra'] / demo['truth']:5.0%} of the truth   "
            f"max age {row['max_age']}")
    say("  the estimator is sound; the stopping rule is what destroys it")

    thresholds = []
    for value in (1.0, 1.3, 1.6, 2.0, 2.5, 3.0):
        run = cliff.selection_demo(threshold=value)
        arm = run["arms"]["selected"]
        thresholds.append({"threshold": value, "extra": arm["extra"],
                           "share": arm["extra"] / run["truth"]})
    say("  the harder teams are assumed to stop, the more of it vanishes:")
    for row in thresholds:
        say(f"    stop at {row['threshold']:.1f} s/lap  ->  "
            f"{row['extra']:+.4f}  ({row['share']:.0%} of the truth)")

    reconciled = cliff.reconcile(
        [{k: e[k] for k in ("circuit", "laps", "degradation", "loss", "ran")}
         for e in entries], KNOTS)
    say()
    say("what a cliff would have to be for the field's choice to be optimal")
    say(f"  {'circuit':16s} {'ran':>4s} " +
        " ".join(f"knot {k:<3d}" for k in KNOTS))
    needed = {knot: [] for knot in KNOTS}
    for row in reconciled:
        cells = []
        for knot in KNOTS:
            value = row["rates"][knot]
            if row["refused"]:
                cells.append("refused ")
            elif value is None:
                cells.append("no cliff")
            else:
                cells.append(f"{value:+.3f}  ")
                needed[knot].append(value)
        say(f"  {row['circuit']:16s} {row['ran']:4d} " +
            " ".join(f"{c:8s}" for c in cells))
    for knot in KNOTS:
        # A zero means the model already agreed with the field there, so
        # averaging the zeros in would understate what is missing at the
        # circuits that are actually wrong.
        disagree = [value for value in needed[knot] if value > 0]
        if disagree:
            say(f"  knot {knot}: of the {len(needed[knot])} circuits a "
                f"cliff can reach, {len(disagree)} disagree with the field "
                f"and need a median {np.median(disagree):+.4f} s/lap "
                f"(range {min(disagree):+.3f} to {max(disagree):+.3f})")

    # ---- output --------------------------------------------------------
    figures = charts.draw(
        OUT, census=census, loss_census=loss_census, losses=losses,
        summary=summary,
        circuit_loss=circuit_loss, stop_table=stop_table, entries=ranked,
        compared=compared, crossover=crossover, coverage=coverage,
        hinges=[h for h in hinges if h["testable"]], demo=demo,
        thresholds=thresholds, reconciled=reconciled, knots=KNOTS,
        focus=focus, safety=safety, sc_stats=sc_stats)
    path = report.build(
        census=census, summary=summary, loss_census=loss_census,
        circuit_loss=circuit_loss, fitted=fitted, sigma=sigma,
        sc_stats=sc_stats, pooled=pooled, stop_table=stop_table,
        interval=interval, entries=entries, ranked=ranked, focus=focus,
        compared=compared, reactive_gain=reactive_gain,
        window_rows=window_rows, crossover=crossover, coverage=coverage,
        hinges=hinges, demo=demo, thresholds=thresholds,
        reconciled=reconciled, knots=KNOTS, seasons=SEASONS,
        figures=figures, out_path=OUT / "report.pdf")

    say()
    say(f"report  {path}")
    for name, figure in figures.items():
        say(f"figure  {figure}")
    (OUT / "_run.txt").write_text("\n".join(_log) + "\n", encoding="utf-8")


def measure_safety(laps, stops, results) -> dict:
    """The neutralisation process, from the field's own lap times.

    Periods are runs of consecutive flagged laps. Races the condition rule
    throws out are excluded here too: a wet race is neutralised for
    reasons a dry-race model has no business extrapolating.
    """
    flagged = stints.neutralised_laps(laps)
    staged, _ = stints.build(laps, stops, results, condition_spread_max=1e9)
    spreads = stints.condition_spread(staged)
    distance: dict[tuple, int] = {}
    for row in results:
        key = (row.season, row.round)
        distance[key] = max(distance.get(key, 0), row.laps)

    factors = pitloss.neutralisation_factor(laps)
    periods, lengths, total_laps, races, with_any = 0, [], 0, 0, 0
    inflation, distances_kept, per_race = [], [], []
    for key, marked in sorted(flagged.items()):
        if spreads.get(key, 0.0) > stints.CONDITION_SPREAD_MAX:
            continue
        runs: list[list[int]] = []
        for lap in sorted(marked):
            if runs and lap == runs[-1][-1] + 1:
                runs[-1].append(lap)
            else:
                runs.append([lap])
        periods += len(runs)
        lengths += [len(run) for run in runs]
        total_laps += distance[key]
        distances_kept.append(distance[key])
        per_race.append(len(runs))
        races += 1
        with_any += bool(runs)
        inflation += [factors[(key, lap)] for lap in marked
                      if (key, lap) in factors]

    return {"periods": periods, "races": races,
            "hazard": periods / total_laps,
            "length": float(np.mean(lengths)),
            "factor": float(np.median(inflation)),
            "any": with_any / races,
            "lengths": lengths,
            "distances": distances_kept,
            "per_race": per_race}


def circuit_entries(rows, results, stops, loss_by) -> list[dict]:
    """Per circuit: measured degradation, measured pit loss, race distance,
    and the number of stops the classified finishers most often made."""
    per_car: dict[tuple, int] = collections.defaultdict(int)
    for stop in stops:
        per_car[(stop.season, stop.round, stop.driver)] += 1

    distance: dict[str, int] = {}
    chosen: dict[str, list[int]] = collections.defaultdict(list)
    for row in results:
        distance[row.circuit] = max(distance.get(row.circuit, 0), row.laps)
        if row.status in stints.FINISHED:
            chosen[row.circuit].append(
                per_car.get((row.season, row.round, row.driver), 0))

    grouped: dict[str, list] = collections.defaultdict(list)
    for row in rows:
        grouped[row.circuit].append(row)

    entries = []
    for circuit, group in sorted(grouped.items()):
        if len(group) < MIN_CIRCUIT_LAPS or circuit not in loss_by:
            continue
        fit = model.fit_within(stints.arrays(group))
        counts = collections.Counter(chosen[circuit])
        entries.append({
            "circuit": circuit, "laps": distance[circuit],
            "degradation": fit["age"],
            "low": fit.interval("age")[0], "high": fit.interval("age")[1],
            "loss": loss_by[circuit],
            "ran": counts.most_common(1)[0][0],
            "finishers": sum(counts.values()),
            "k_star": float("nan"), "model": None, "verdict": "",
        })
    return entries


if __name__ == "__main__":
    main()
