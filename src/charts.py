"""Figures. Committed to the repository so the write-up renders anywhere."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt        # noqa: E402
import numpy as np                     # noqa: E402

PALETTE = {"ink": "#16181d", "muted": "#5b6472", "grid": "#e3e6ea",
           "good": "#17803d", "bad": "#b4232c", "accent": "#2563a8",
           "warm": "#d9a441", "plum": "#6b4c9a"}
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 9,
    "axes.edgecolor": PALETTE["muted"], "axes.labelcolor": PALETTE["ink"],
    "text.color": PALETTE["ink"], "xtick.color": PALETTE["muted"],
    "ytick.color": PALETTE["muted"], "axes.titlesize": 10,
    "axes.titleweight": "bold", "figure.facecolor": "white",
})


def _save(fig, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.png"
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def _grid(ax):
    ax.grid(True, color=PALETTE["grid"], linewidth=0.7)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def pit_loss(losses, summary, circuit_loss, out_dir: Path) -> Path:
    """What a stop costs, green flag against neutralised, and by circuit."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2),
                             gridspec_kw={"width_ratios": [1, 1.25]})

    green = [row.total for row in losses if not row.neutralised]
    neutral = [row.total for row in losses if row.neutralised]
    bins = np.linspace(-5, 45, 51)
    axes[0].hist(green, bins=bins, color=PALETTE["accent"], alpha=0.75,
                 label=f"green flag (n={len(green)})")
    axes[0].hist(neutral, bins=bins, color=PALETTE["warm"], alpha=0.8,
                 label=f"neutralised (n={len(neutral)})")
    for value, colour in ((summary["green"], PALETTE["accent"]),
                          (summary["neutral"], PALETTE["warm"])):
        axes[0].axvline(value, color=colour, linewidth=1.6, linestyle="--")
    axes[0].set_xlabel("measured pit loss, seconds")
    axes[0].set_ylabel("stops")
    axes[0].set_title("A stop under a safety car costs a third as much")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].annotate(
        f"{summary['green']:.1f} s  ->  {summary['neutral']:.1f} s\n"
        f"discount {summary['discount']:.2f}x",
        xy=(0.62, 0.62), xycoords="axes fraction", fontsize=8.5,
        color=PALETTE["muted"])
    _grid(axes[0])

    names = [row["circuit"] for row in circuit_loss]
    values = [row["loss"] for row in circuit_loss]
    spread = [[row["loss"] - row["p25"] for row in circuit_loss],
              [row["p75"] - row["loss"] for row in circuit_loss]]
    positions = np.arange(len(names))
    axes[1].barh(positions, values, xerr=spread, color=PALETTE["accent"],
                 alpha=0.85, error_kw={"ecolor": PALETTE["muted"],
                                       "elinewidth": 0.9})
    axes[1].scatter([row["transit"] for row in circuit_loss], positions,
                    color=PALETTE["bad"], s=14, zorder=5,
                    label="pit-lane transit in the feed")
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(names, fontsize=7.5)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("green-flag pit loss, seconds (bars: interquartile)")
    axes[1].set_title("Nine seconds between the cheapest and dearest stop")
    axes[1].legend(frameon=False, fontsize=8, loc="lower right")
    _grid(axes[1])
    return _save(fig, out_dir, "pit_loss")


def stop_count(stop_table, pooled_laps: int, out_dir: Path) -> Path:
    """Race time against number of stops, and the cost of being wrong."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    stops = [row["stops"] for row in stop_table]
    gaps = [row["gap"] for row in stop_table]
    colours = [PALETTE["good"] if gap == 0 else PALETTE["accent"]
               for gap in gaps]
    ax.bar(stops, gaps, color=colours, alpha=0.85, width=0.62)
    for stop, gap, row in zip(stops, gaps, stop_table):
        ax.annotate("-".join(str(m) for m in row["plan"]),
                    xy=(stop, gap), xytext=(0, 4), textcoords="offset points",
                    ha="center", fontsize=7, color=PALETTE["muted"])
    ax.set_xlabel("pit stops")
    ax.set_ylabel("seconds slower than the best plan")
    ax.set_title(f"The optimum over {pooled_laps} laps, and what the "
                 f"alternatives cost")
    ax.set_xticks(stops)
    _grid(ax)
    return _save(fig, out_dir, "stop_count")


def validation(entries, out_dir: Path) -> Path:
    """The model's optimum against the stop count the field actually chose."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3),
                             gridspec_kw={"width_ratios": [1.3, 1]})

    order = sorted(entries, key=lambda row: -row["k_star"])
    positions = np.arange(len(order))
    axes[0].barh(positions, [row["k_star"] - 1 for row in order],
                 color=PALETTE["accent"], alpha=0.8,
                 label="model, before rounding (k* - 1)")
    axes[0].scatter([row["ran"] for row in order], positions,
                    color=PALETTE["bad"], s=26, zorder=5,
                    label="stops the field most often made")
    axes[0].set_yticks(positions)
    axes[0].set_yticklabels([row["circuit"] for row in order], fontsize=7.5)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("pit stops")
    axes[0].set_title("The field stops more often than the model says")
    axes[0].legend(frameon=False, fontsize=8, loc="lower right")
    _grid(axes[0])

    error = [row["model"] - row["ran"] for row in order]
    counts = {value: error.count(value) for value in sorted(set(error))}
    axes[1].bar(list(counts), list(counts.values()), width=0.6,
                color=[PALETTE["good"] if key == 0 else PALETTE["bad"]
                       for key in counts], alpha=0.85)
    axes[1].set_xlabel("model stops minus the field's stops")
    axes[1].set_ylabel("circuits")
    axes[1].set_xticks(list(counts))
    axes[1].set_title("The error only ever points one way")
    axes[1].annotate(
        f"a one-directional error across\n{len(order)} circuits is a "
        f"missing\nterm, not noise",
        xy=(0.03, 0.72), xycoords="axes fraction", fontsize=8.5,
        color=PALETTE["muted"])
    _grid(axes[1])
    return _save(fig, out_dir, "validation")


def distributions(compared, focus, out_dir: Path) -> Path:
    """Outcome distributions for each plan over the same simulated races."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))

    palette = [PALETTE["good"], PALETTE["accent"], PALETTE["warm"],
               PALETTE["plum"], PALETTE["bad"]]
    times = compared["times"]
    best = min(row["mean"] for row in compared["rows"])
    for index, row in enumerate(compared["rows"]):
        values = times[row["name"]] - best
        axes[0].hist(values, bins=70, histtype="step", linewidth=1.5,
                     color=palette[index % len(palette)],
                     label=f"{row['name']} ({row['win_share']:.0%} of races)")
    axes[0].set_xlabel("seconds relative to the best plan's mean")
    axes[0].set_ylabel("races")
    axes[0].set_title(f"{compared['draws']:,} shared races at "
                      f"{focus['circuit']}")
    axes[0].legend(frameon=False, fontsize=8)
    _grid(axes[0])

    names = [row["name"] for row in compared["rows"]]
    positions = np.arange(len(names))
    gaps = [row["gap_mean"] for row in compared["rows"]]
    errors = [1.96 * row["gap_error"] for row in compared["rows"]]
    axes[1].barh(positions, gaps, xerr=errors, color=PALETTE["accent"],
                 alpha=0.85, error_kw={"ecolor": PALETTE["ink"],
                                       "elinewidth": 1.1})
    for position, row in zip(positions, compared["rows"]):
        axes[1].annotate(f"{row['win_share']:.1%} of races won",
                         xy=(max(gaps) * 0.02, position), fontsize=7.5,
                         va="center", color=PALETTE["muted"])
    axes[1].set_yticks(positions)
    axes[1].set_yticklabels(names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("mean seconds behind the best plan (paired, 95%)")
    axes[1].set_title("A plan can lose on average and still win races")
    _grid(axes[1])
    return _save(fig, out_dir, "distributions")


def hazard(crossover, measured: float, out_dir: Path) -> Path:
    """Whether neutralisation risk ever changes which plan is best."""
    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    names = [key for key in crossover[0]
             if key not in ("hazard", "best", "neutralised_any")]
    hazards = [row["hazard"] for row in crossover]
    reference = [row[names[0]] for row in crossover]
    palette = [PALETTE["good"], PALETTE["accent"], PALETTE["warm"],
               PALETTE["plum"], PALETTE["bad"]]
    for index, name in enumerate(names):
        ax.plot(hazards, [row[name] - base
                          for row, base in zip(crossover, reference)],
                marker="o", markersize=3.5, linewidth=1.5,
                color=palette[index % len(palette)], label=name)
    ax.axvline(measured, color=PALETTE["bad"], linestyle="--", linewidth=1.2)
    ax.annotate(f"measured 2023\nhazard {measured:.4f}",
                xy=(measured, ax.get_ylim()[1] * 0.72), xytext=(6, 0),
                textcoords="offset points", fontsize=8, color=PALETTE["bad"])
    ax.axhline(0, color=PALETTE["muted"], linewidth=0.9)
    ax.set_xlabel("per-lap probability a neutralisation starts")
    ax.set_ylabel(f"seconds behind the {names[0]} plan")
    ax.set_title("Safety car risk narrows the gap without ever closing it")
    ax.legend(frameon=False, fontsize=8)
    _grid(ax)
    return _save(fig, out_dir, "hazard")


def selection(coverage, hinges, demo, thresholds, out_dir: Path) -> Path:
    """Why the lap times cannot see a cliff even if there is one."""
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1))

    ages = [row["age"] for row in coverage]
    axes[0].bar(ages, [row["share"] for row in coverage], width=3.2,
                color=PALETTE["accent"], alpha=0.85)
    for row in coverage:
        axes[0].annotate(f"{row['laps']:,}", xy=(row["age"], row["share"]),
                         xytext=(0, 3), textcoords="offset points",
                         ha="center", fontsize=7, color=PALETTE["muted"])
    axes[0].set_xlabel("tyre age, laps")
    axes[0].set_ylabel("share of the record reaching this age")
    axes[0].set_title("The old-tyre laps barely exist")
    _grid(axes[0])

    knots = [row["knot"] for row in hinges]
    values = [row["extra"] for row in hinges]
    lows = [row["extra"] - row["low"] for row in hinges]
    highs = [row["high"] - row["extra"] for row in hinges]
    axes[1].errorbar(knots, values, yerr=[lows, highs], fmt="o",
                     color=PALETTE["bad"], markersize=5, capsize=3,
                     linewidth=1.2)
    axes[1].axhline(0, color=PALETTE["muted"], linewidth=1.0)
    axes[1].fill_between([min(knots) - 1, max(knots) + 1], 0,
                         max(max(values) + max(highs), 0.01),
                         color=PALETTE["good"], alpha=0.07)
    axes[1].annotate("a cliff lives up here", xy=(0.38, 0.9),
                     xycoords="axes fraction", fontsize=8.5,
                     color=PALETTE["good"])
    axes[1].set_xlabel("knot: tyre age the extra slope starts at")
    axes[1].set_ylabel("extra seconds per lap beyond the knot")
    axes[1].set_title("Measured on real laps: the wrong sign")
    axes[1].set_xlim(min(knots) - 1, max(knots) + 1)
    _grid(axes[1])

    labels = list(demo["arms"])
    recovered = [demo["arms"][name]["extra"] for name in labels]
    axes[2].bar(np.arange(len(labels)), recovered, width=0.55,
                color=[PALETTE["good"], PALETTE["bad"]], alpha=0.85)
    axes[2].axhline(demo["truth"], color=PALETTE["ink"], linestyle="--",
                    linewidth=1.3)
    axes[2].annotate(f"truth {demo['truth']:+.3f}", xy=(0.56, 0.86),
                     xycoords="axes fraction", fontsize=8.5)
    for index, (name, value) in enumerate(zip(labels, recovered)):
        axes[2].annotate(f"{value / demo['truth']:.0%}\nof the truth",
                         xy=(index, value), xytext=(0, 5),
                         textcoords="offset points", ha="center", fontsize=8,
                         color=PALETTE["muted"])
    axes[2].set_xticks(np.arange(len(labels)))
    axes[2].set_xticklabels(["fixed plans", "teams stop when\ntyres go off"])
    axes[2].set_ylabel("recovered extra slope, s per lap")
    axes[2].set_title("Same cliff, same estimator, different stopping rule")
    axes[2].set_ylim(0, demo["truth"] * 1.3)
    _grid(axes[2])
    return _save(fig, out_dir, "selection")


def reconciliation(reconciled, knots, out_dir: Path) -> Path:
    """The cliff each circuit would need for the field to have been right."""
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    usable = [row for row in reconciled if not row["refused"]]
    positions = np.arange(len(usable))
    palette = [PALETTE["good"], PALETTE["accent"], PALETTE["warm"],
               PALETTE["plum"]]
    width = 0.8 / len(knots)
    for index, knot in enumerate(knots):
        values = [row["rates"][knot] if row["rates"][knot] is not None
                  else 0.0 for row in usable]
        ax.bar(positions + index * width - 0.4, values, width=width,
               color=palette[index % len(palette)], alpha=0.88,
               label=f"cliff from age {knot}")
    ax.set_xticks(positions)
    ax.set_xticklabels([row["circuit"] for row in usable], rotation=60,
                       ha="right", fontsize=7.5)
    ax.set_ylabel("extra seconds per lap needed beyond the knot")
    ax.set_title("What the missing penalty would have to be")
    ax.annotate("zero means the model already\nagreed with the field",
                xy=(0.02, 0.88), xycoords="axes fraction", fontsize=8,
                color=PALETTE["muted"])
    ax.legend(frameon=False, fontsize=8)
    _grid(ax)
    return _save(fig, out_dir, "reconciliation")


def census_chart(census, loss_census, out_dir: Path) -> Path:
    """Which laps and which stops the measurement is built from."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0))

    for ax, table, title, unit in (
            (axes[0], census, "Laps", "laps"),
            (axes[1], loss_census, "Pit stops", "stops")):
        items = [(key, value) for key, value in table.items()
                 if key not in ("total", "stops") and value]
        items.sort(key=lambda item: -item[1])
        names = [key for key, _ in items]
        values = [value for _, value in items]
        colours = [PALETTE["good"] if key.startswith("kept")
                   or key == "kept" else PALETTE["muted"] for key in names]
        positions = np.arange(len(names))
        ax.barh(positions, values, color=colours, alpha=0.85)
        for position, value in zip(positions, values):
            ax.annotate(f"{value:,}", xy=(value, position), xytext=(3, 0),
                        textcoords="offset points", va="center", fontsize=7.5,
                        color=PALETTE["muted"])
        ax.set_yticks(positions)
        ax.set_yticklabels(names, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel(unit)
        ax.set_title(f"{title}: what is used and what is dropped")
        _grid(ax)
    return _save(fig, out_dir, "census")


def draw(out_dir: Path, **kwargs) -> dict[str, Path]:
    """Every figure, keyed by the name the report refers to it by."""
    return {
        "census": census_chart(kwargs["census"],
                                kwargs["loss_census"], out_dir),
        "pit_loss": pit_loss(kwargs["losses"], kwargs["summary"],
                             kwargs["circuit_loss"], out_dir),
        "stop_count": stop_count(kwargs["stop_table"], 57, out_dir),
        "validation": validation(kwargs["entries"], out_dir),
        "distributions": distributions(kwargs["compared"], kwargs["focus"],
                                       out_dir),
        "hazard": hazard(kwargs["crossover"], kwargs["safety"].hazard,
                         out_dir),
        "selection": selection(kwargs["coverage"], kwargs["hinges"],
                               kwargs["demo"], kwargs["thresholds"], out_dir),
        "reconciliation": reconciliation(kwargs["reconciled"],
                                         kwargs["knots"], out_dir),
    }
