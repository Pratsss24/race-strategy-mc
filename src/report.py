"""Build the report as a printable PDF.

`paper.py` owns the page furniture; everything here is this project's own
argument, in the order the argument runs.
"""

from __future__ import annotations

from pathlib import Path

from paper import Paper

SOURCE = ("Lap times and pit stops from the Jolpica mirror of the Ergast "
          "schema. The cleaning rules and the fixed-effects estimator are "
          "carried over from the stint-degradation project.")


def _plan(plan) -> str:
    return "-".join(str(stint) for stint in plan)


def _loss(circuit_loss) -> tuple[list[str], list[list[str]]]:
    rows = [[one["circuit"], f"{one['loss']:.2f}",
             f"{one['p25']:.2f} to {one['p75']:.2f}",
             f"{one['transit']:.2f}", str(one["n"])]
            for one in circuit_loss]
    return (["circuit", "pit loss", "interquartile", "transit in feed",
             "stops"], rows)


def _census(census: dict) -> tuple[list[str], list[list[str]]]:
    rows = [[key, f"{value:,}", f"{value / census['total']:.2%}"]
            for key, value in census.items() if key != "total"]
    return (["rule", "laps", "share"], rows)


def _stops(stop_table) -> tuple[list[str], list[list[str]]]:
    rows = [[str(one["stops"]), _plan(one["plan"]), f"{one['time']:.2f}",
             f"{one['gap']:+.2f}"] for one in stop_table]
    return (["stops", "plan", "race time, s", "behind the best"], rows)


def _ranked(ranked) -> tuple[list[str], list[list[str]]]:
    rows = [[one["circuit"], str(one["laps"]), f"{one['degradation']:+.4f}",
             f"{one['low']:+.4f} to {one['high']:+.4f}", f"{one['loss']:.2f}",
             f"{one['k_star']:.2f}", str(one["model"]), str(one["ran"]),
             one["verdict"]] for one in ranked]
    return (["circuit", "laps", "degradation", "95% interval", "pit loss",
             "k*", "model", "field", "verdict"], rows)


def _refused(entries) -> tuple[list[str], list[list[str]]]:
    rows = [[one["circuit"], f"{one['degradation']:+.4f}",
             f"{one['low']:+.4f} to {one['high']:+.4f}", str(one["ran"])]
            for one in entries if one["model"] is None]
    return (["circuit refused", "degradation", "95% interval",
             "field ran"], rows)


def _monte(compared) -> tuple[list[str], list[list[str]]]:
    rows = [[one["name"], _plan(one["plan"]), f"{one['mean']:.2f}",
             f"{one['sd']:.2f}", f"{one['p05']:.1f} to {one['p95']:.1f}",
             f"{one['gap_mean']:+.2f} ± {1.96 * one['gap_error']:.2f}",
             f"{one['win_share']:.1%}"] for one in compared["rows"]]
    return (["plan", "stints", "mean, s", "sd", "5th to 95th",
             "behind best (paired, 95%)", "races won"], rows)


def _hinges(hinges) -> tuple[list[str], list[list[str]]]:
    rows = []
    for one in hinges:
        if not one["testable"]:
            rows.append([str(one["knot"]),
                         f"only {one['beyond']} laps beyond the knot",
                         "—", "—", "—"])
            continue
        rows.append([str(one["knot"]), f"{one['extra']:+.4f}",
                     f"{one['low']:+.4f} to {one['high']:+.4f}",
                     f"{one['p']:.3f}",
                     f"{one['beyond']:,} ({one['share']:.1%})"])
    return (["knot", "extra s/lap beyond it", "95% interval", "p",
             "laps beyond"], rows)


def _demo(demo) -> tuple[list[str], list[list[str]]]:
    rows = []
    for name, one in demo["arms"].items():
        label = ("fixed plans" if name == "fixed"
                 else "teams stop when tyres go off")
        rows.append([label, f"{demo['truth']:+.4f}", f"{one['extra']:+.4f}",
                     f"{one['low']:+.4f} to {one['high']:+.4f}",
                     f"{one['extra'] / demo['truth']:.0%}",
                     str(one["max_age"])])
    return (["stints ended by", "true extra slope", "recovered",
             "95% interval", "share of truth", "longest stint"], rows)


def _reconciled(reconciled, knots) -> tuple[list[str], list[list[str]]]:
    rows = []
    for one in reconciled:
        cells = [one["circuit"], str(one["ran"])]
        for knot in knots:
            value = one["rates"][knot]
            if one["refused"]:
                cells.append("refused")
            elif value is None:
                cells.append("no cliff reaches")
            elif value == 0.0:
                cells.append("already agrees")
            else:
                cells.append(f"{value:+.3f}")
        rows.append(cells)
    return (["circuit", "field ran"] + [f"from age {one}" for one in knots],
            rows)


def build(*, census, summary, loss_census, circuit_loss, fitted, sigma,
          sc_stats, pooled, stop_table, interval, entries, ranked, focus,
          compared, reactive_gain, window_rows, crossover, coverage, hinges,
          demo, thresholds, reconciled, knots, seasons, figures,
          out_path: Path) -> Path:
    agree = sum(1 for one in ranked if one["verdict"] == "agrees")
    under = sum(1 for one in ranked if one["verdict"] == "too few stops")
    over = sum(1 for one in ranked if one["verdict"] == "too many stops")
    winners = sorted({one["best"] for one in crossover})
    k_star = pooled.laps * (pooled.degradation / (2 * pooled.loss)) ** 0.5
    out = Path(out_path)
    out = out.with_name("report.pdf") if out.suffix != ".pdf" else out
    out.parent.mkdir(parents=True, exist_ok=True)

    with Paper(out, "Race strategy Monte Carlo",
               f"Pit strategy solved from {census['total']:,} F1 lap times "
               f"and {loss_census['stops']:,} pit stops, "
               f"{'-'.join(str(one) for one in seasons)} — with every "
               f"input measured from the same season the answer is then "
               f"tested against.") as paper:

        paper.cover(
            headline=f"The model recommends too few stops at {under} of "
                     f"{len(ranked)} circuits",
            small=f"It agrees at {agree} and recommends too many at "
                  f"{over}. An error with a direction is a missing term, "
                  f"not noise — and the term it is missing is one the "
                  f"lap times cannot measure, because teams stop before it "
                  f"bites.",
            tiles=[("green-flag pit loss", f"{summary['green']:.1f} s"),
                   ("under a safety car", f"{summary['neutral']:.1f} s"),
                   ("degradation", f"{fitted['age']:+.4f} s/lap"),
                   ("circuits the model gets right",
                    f"{agree}/{len(ranked)}")],
            blurb="Everything the optimiser is fed is measured from the "
                  "same lap times it is then tested against: the pit loss, "
                  "the degradation, the safety-car hazard. Nothing is "
                  "borrowed from a published figure.",
            figure_path=figures["pit_loss"])

        paper.section(
            "What a pit stop costs",
            note="The feed records a pit-lane transit time, not a pit loss.",
            body=f"The cost is measured instead as the excess of the in-lap "
                 f"and the out-lap over what the car's own fitted line says "
                 f"those laps should have taken: {summary['green']:.2f} s "
                 f"under green flag across {summary['green_n']} stops, "
                 f"interquartile {summary['green_p25']:.2f} to "
                 f"{summary['green_p75']:.2f}. The same measurement over "
                 f"the {summary['neutral_n']} stops taken while the race "
                 f"was neutralised gives {summary['neutral']:.2f} s, a "
                 f"discount of {summary['discount']:.2f}×. That "
                 f"discount is the only reason a strategy model needs a "
                 f"simulation at all.",
            table=_loss(circuit_loss))

        paper.section(
            "Which laps and stops are used",
            body=f"Single-car pace scatter is {sigma:.3f} s a lap, measured "
                 f"as the residual spread of one car around its own line. "
                 f"The pooled residual from a fit with car-race fixed "
                 f"effects is {fitted.sigma:.3f} s, which is larger because "
                 f"it still carries differences between cars inside a race; "
                 f"a one-car simulation has no business inheriting those.",
            figure_path=figures["census"], table=_census(census))

        decided = "does" if interval["decided"] else "does NOT"
        paper.section(
            "The race as arithmetic",
            body=f"Race time splits into a part no plan can change and a "
                 f"part that is entirely the plan: the sum of squared stint "
                 f"lengths, and the stops. Two things follow without "
                 f"simulating anything. Equal stints minimise the sum of "
                 f"squares, so the best plan at any stop count is the "
                 f"balanced split. And differentiating gives the best "
                 f"number of stints outright, k* = N√(d/2L), which at "
                 f"the pooled numbers over {pooled.laps} laps is "
                 f"{k_star:.3f}.\n\n"
                 f"The exhaustive search over every partition of the "
                 f"distance finds exactly what the algebra does, which is a "
                 f"test rather than a hope. Across the degradation "
                 f"confidence interval the optimum moves between "
                 f"{interval['range'][0]} and {interval['range'][1]} "
                 f"stints, so on the pooled number the data {decided} pick "
                 f"a stop count on its own.",
            figure_path=figures["stop_count"], table=_stops(stop_table))

        paper.section(
            "Tested against what the field actually did",
            warn=f"Agrees at {agree}, recommends too few stops at {under}, "
                 f"too many at {over}. The direction is the finding. Random "
                 f"error would land on both sides.",
            figure_path=figures["validation"], table=_ranked(ranked))

        paper.section(
            "The circuits that are refused rather than ranked",
            body="Their measured degradation is not positive — on "
                 "those laps tyres apparently improve with age, and no stop "
                 "count exploits that. Returning a strategy anyway would be "
                 "the optimiser lending false precision to a number that "
                 "has none.",
            table=_refused(entries))

        paper.section(
            "Is it safety car risk?",
            note=f"{compared['draws']:,} simulated races at "
                 f"{focus['circuit']}, every plan run through the same "
                 f"draws so the comparison is paired.",
            body=f"The simulation reproduces the observed frequency of "
                 f"neutralisation without being told it: "
                 f"{compared['neutralised_any']:.1%} of simulated races "
                 f"contain at least one, against {sc_stats['any']:.1%} of "
                 f"the real dry races. Reacting to a neutralisation — "
                 f"bringing the next stop forward when one starts within "
                 f"the window — is worth {reactive_gain:+.2f} s to the "
                 f"best plan.",
            figure_path=figures["distributions"], table=_monte(compared))

        paper.section(
            "And across every hazard rate",
            warn=f"Across hazards from zero to "
                 f"{max(one['hazard'] for one in crossover):.2f} — the "
                 f"last of which neutralises almost every race — "
                 f"{len(winners)} plan ever wins: {', '.join(winners)}. "
                 f"Safety car risk narrows the gap between one stop and two "
                 f"steadily, and never closes it. So it is not the "
                 f"explanation for the field stopping more often.",
            figure_path=figures["hazard"],
            table=(["per-lap hazard", "P(any neutralisation)", "best plan"],
                   [[f"{one['hazard']:.3f}",
                     f"{one['neutralised_any']:.1%}", one["best"]]
                    for one in crossover]),
            after="Reaction window, for the best plan: " + "; ".join(
                f"{one['window']} laps → {one['best']} "
                f"({one['mean']:.2f} s)" for one in window_rows))

        paper.section(
            "The penalty the lap times cannot see",
            note="The remaining candidate is the one every engineer names "
                 "first: tyre performance is not linear in age.",
            warn="Fitting an extra slope beyond a knot tests it directly, "
                 "and the answer comes back with the wrong sign, "
                 "significantly. Teams choose when to stop, and they stop "
                 "before the tyre falls away, so a stint only reaches a "
                 "high age when that set happened to be holding on — "
                 "selection directly on the thing being measured. The laps "
                 "that would reveal a cliff are the laps strategy exists to "
                 "avoid.",
            figure_path=figures["selection"], table=_hinges(hinges),
            after="Laps reaching each tyre age: " + "; ".join(
                f"age {one['age']}: {one['laps']:,} ({one['share']:.2%})"
                for one in coverage))

        paper.section(
            "The same argument where the truth is known",
            body=f"Every car is given the same real cliff of "
                 f"{demo['truth']:+.4f} s per lap beyond age "
                 f"{demo['knot']}, and the two arms differ only in how "
                 f"stints end. With fixed plans the estimator recovers the "
                 f"cliff almost exactly. With a stopping rule that retires "
                 f"the worst sets first it recovers a fraction of it — "
                 f"and the more promptly teams are assumed to react, the "
                 f"less survives.",
            table=_demo(demo),
            after="By how promptly teams react: " + "; ".join(
                f"stop at {one['threshold']:.1f} s/lap → "
                f"{one['extra']:+.4f} ({one['share']:.0%} of truth)"
                for one in thresholds))

        paper.section(
            "How big would it have to be?",
            body="Since the cliff cannot be measured from these laps, the "
                 "question is inverted: what extra penalty would make the "
                 "field's own choice the optimal one? This is not a "
                 "measurement of the cliff and is not offered as one. It is "
                 "a statement of the form: if a cliff is the only thing "
                 "missing, it is this big.",
            figure_path=figures["reconciliation"],
            table=_reconciled(reconciled, knots), after=SOURCE)

    return out
