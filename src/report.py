"""One self-contained HTML file: charts inlined, no assets, no server."""

from __future__ import annotations

import base64
from pathlib import Path

CSS = """
body{font:15px/1.6 -apple-system,Segoe UI,Roboto,sans-serif;max-width:1120px;
margin:40px auto;padding:0 20px;color:#16181d}
h1{margin-bottom:4px}h2{margin-top:38px;border-bottom:1px solid #e3e6ea;
padding-bottom:6px}
.sub{color:#5b6472;margin-top:0}
.tiles{display:flex;gap:14px;flex-wrap:wrap;margin:22px 0}
.tile{flex:1;min-width:150px;background:#f6f8fa;border:1px solid #e3e6ea;
border-radius:9px;padding:14px 16px}
.tile .v{font-size:25px;font-weight:650;letter-spacing:-.5px}
.tile .k{font-size:11px;text-transform:uppercase;letter-spacing:.07em;
color:#5b6472;margin-top:3px}
table{border-collapse:collapse;width:100%;margin:14px 0;font-size:14px}
th,td{border:1px solid #e3e6ea;padding:7px 10px;text-align:right}
th:first-child,td:first-child{text-align:left}
th{background:#f6f8fa;font-weight:600}
td.good{color:#17803d}td.bad{color:#b4232c}
img{width:100%;margin:14px 0;border:1px solid #e3e6ea;border-radius:7px}
.note{background:#fff8e6;border-left:3px solid #d9a441;padding:11px 15px;
margin:18px 0;font-size:14px}
code{background:#f0f2f5;padding:1px 5px;border-radius:4px;font-size:13px}
"""


def _img(path: Path) -> str:
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{Path(path).stem}">'


def _plan(plan) -> str:
    return "-".join(str(stint) for stint in plan)


def build(*, census, summary, loss_census, circuit_loss, fitted, sigma,
          sc_stats, pooled, stop_table, interval, entries, ranked, focus,
          compared, reactive_gain, window_rows, crossover, coverage, hinges,
          demo, thresholds, reconciled, knots, seasons, figures,
          out_path: Path) -> Path:
    agree = sum(1 for row in ranked if row["verdict"] == "agrees")
    under = sum(1 for row in ranked if row["verdict"] == "too few stops")
    over = sum(1 for row in ranked if row["verdict"] == "too many stops")

    tiles = [
        ("green-flag pit loss", f"{summary['green']:.1f} s"),
        ("under a safety car", f"{summary['neutral']:.1f} s"),
        ("degradation", f"{fitted['age']:+.4f} s/lap"),
        ("circuits the model gets right", f"{agree}/{len(ranked)}"),
    ]
    tiles_html = "".join(
        f'<div class="tile"><div class="v">{v}</div><div class="k">{k}</div></div>'
        for k, v in tiles)

    loss_html = "".join(
        f"<tr><td>{row['circuit']}</td><td>{row['loss']:.2f}</td>"
        f"<td>{row['p25']:.2f} to {row['p75']:.2f}</td>"
        f"<td>{row['transit']:.2f}</td><td>{row['n']}</td></tr>"
        for row in circuit_loss)

    census_html = "".join(
        f"<tr><td>{key}</td><td>{value:,}</td>"
        f"<td>{value / census['total']:.2%}</td></tr>"
        for key, value in census.items() if key != "total")

    stop_html = "".join(
        f"<tr><td>{row['stops']}</td><td>{_plan(row['plan'])}</td>"
        f"<td>{row['time']:.2f}</td>"
        f'<td class="{"good" if row["gap"] == 0 else ""}">'
        f"{row['gap']:+.2f}</td></tr>" for row in stop_table)

    validation_html = "".join(
        f"<tr><td>{row['circuit']}</td><td>{row['laps']}</td>"
        f"<td>{row['degradation']:+.4f}</td>"
        f"<td>{row['low']:+.4f} to {row['high']:+.4f}</td>"
        f"<td>{row['loss']:.2f}</td><td>{row['k_star']:.2f}</td>"
        f"<td>{row['model']}</td><td>{row['ran']}</td>"
        f'<td class="{"good" if row["verdict"] == "agrees" else "bad"}">'
        f"{row['verdict']}</td></tr>" for row in ranked)

    refused = [row for row in entries if row["model"] is None]
    refused_html = "".join(
        f"<tr><td>{row['circuit']}</td><td>{row['degradation']:+.4f}</td>"
        f"<td>{row['low']:+.4f} to {row['high']:+.4f}</td>"
        f"<td>{row['ran']}</td></tr>" for row in refused)

    mc_html = "".join(
        f"<tr><td>{row['name']}</td><td>{_plan(row['plan'])}</td>"
        f"<td>{row['mean']:.2f}</td><td>{row['sd']:.2f}</td>"
        f"<td>{row['p05']:.1f} to {row['p95']:.1f}</td>"
        f'<td class="{"good" if row["gap_mean"] == 0 else ""}">'
        f"{row['gap_mean']:+.2f} &plusmn; {1.96 * row['gap_error']:.2f}</td>"
        f"<td>{row['win_share']:.1%}</td></tr>"
        for row in compared["rows"])

    hazard_html = "".join(
        f"<tr><td>{row['hazard']:.3f}</td>"
        f"<td>{row['neutralised_any']:.1%}</td><td>{row['best']}</td></tr>"
        for row in crossover)

    window_html = "".join(
        f"<tr><td>{row['window']}</td><td>{row['best']}</td>"
        f"<td>{row['mean']:.2f}</td></tr>" for row in window_rows)

    coverage_html = "".join(
        f"<tr><td>{row['age']}</td><td>{row['laps']:,}</td>"
        f"<td>{row['share']:.2%}</td></tr>" for row in coverage)

    hinge_html = "".join(
        f"<tr><td>{row['knot']}</td>"
        + (f'<td colspan="4">only {row["beyond"]} laps beyond the knot</td>'
           if not row["testable"] else
           f'<td class="bad">{row["extra"]:+.4f}</td>'
           f'<td>{row["low"]:+.4f} to {row["high"]:+.4f}</td>'
           f'<td>{row["p"]:.3f}</td>'
           f'<td>{row["beyond"]:,} ({row["share"]:.1%})</td>')
        + "</tr>" for row in hinges)

    demo_html = "".join(
        f"<tr><td>{'fixed plans' if name == 'fixed' else 'teams stop when tyres go off'}"
        f"</td><td>{demo['truth']:+.4f}</td>"
        f'<td class="{"good" if name == "fixed" else "bad"}">'
        f"{row['extra']:+.4f}</td>"
        f"<td>{row['low']:+.4f} to {row['high']:+.4f}</td>"
        f"<td>{row['extra'] / demo['truth']:.0%}</td>"
        f"<td>{row['max_age']}</td></tr>"
        for name, row in demo["arms"].items())

    threshold_html = "".join(
        f"<tr><td>{row['threshold']:.1f}</td><td>{row['extra']:+.4f}</td>"
        f"<td>{row['share']:.0%}</td></tr>" for row in thresholds)

    knot_head = "".join(f"<th>from age {knot}</th>" for knot in knots)
    reconcile_html = ""
    for row in reconciled:
        cells = ""
        for knot in knots:
            value = row["rates"][knot]
            if row["refused"]:
                cells += '<td class="bad">refused</td>'
            elif value is None:
                cells += "<td>no cliff reaches</td>"
            elif value == 0.0:
                cells += '<td class="good">already agrees</td>'
            else:
                cells += f"<td>{value:+.3f}</td>"
        reconcile_html += (f"<tr><td>{row['circuit']}</td>"
                           f"<td>{row['ran']}</td>{cells}</tr>")

    winners = sorted({row["best"] for row in crossover})

    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Race strategy Monte Carlo</title><style>{CSS}</style></head><body>
<h1>Race strategy Monte Carlo</h1>
<p class="sub">Pit strategy solved from {census['total']:,} F1 lap times and
{loss_census['stops']:,} pit stops,
{"-".join(str(s) for s in seasons)} &mdash; with every input measured from
the same season the answer is then tested against.</p>

<div class="tiles">{tiles_html}</div>

<div class="note"><b>The result in one line.</b> Fed degradation and pit
loss measured from real laps, the optimiser recommends fewer stops than the
2023 field actually made at {under} of {len(ranked)} circuits, and more at
{over}. An error with a direction is a missing term, not noise &mdash; and
the term it is missing is one the lap times cannot measure, because teams
stop before it bites.</div>

<h2>What a pit stop costs</h2>
{_img(figures['pit_loss'])}
<p>The feed records a pit-lane transit time, not a pit loss. The cost is
measured instead as the excess of the in-lap and the out-lap over what the
car's own fitted line says those laps should have taken:
<b>{summary['green']:.2f} s</b> under green flag across
{summary['green_n']} stops, interquartile
{summary['green_p25']:.2f} to {summary['green_p75']:.2f}. The same
measurement over the {summary['neutral_n']} stops taken while the race was
neutralised gives <b>{summary['neutral']:.2f} s</b>, a discount of
<b>{summary['discount']:.2f}&times;</b>. That discount is the only reason a
strategy model needs a simulation at all.</p>
<table><tr><th>circuit</th><th>pit loss</th><th>interquartile</th>
<th>transit in feed</th><th>stops</th></tr>{loss_html}</table>

<h2>Which laps and stops are used</h2>
{_img(figures['census'])}
<table><tr><th>rule</th><th>laps</th><th>share</th></tr>{census_html}</table>
<p>Single-car pace scatter is {sigma:.3f} s a lap, measured as the residual
spread of one car around its own line. The pooled residual from a fit with
car-race fixed effects is {fitted.sigma:.3f} s, which is larger because it
still carries differences between cars inside a race; a one-car simulation
has no business inheriting those.</p>

<h2>The race as arithmetic</h2>
{_img(figures['stop_count'])}
<p>Race time splits into a part no plan can change and a part that is
entirely the plan: the sum of squared stint lengths, and the stops. Two
things follow without simulating anything. Equal stints minimise the sum of
squares, so the best plan at any stop count is the balanced split. And
differentiating gives the best number of stints outright,
<code>k* = N&radic;(d/2L)</code>, which at the pooled numbers over
{pooled.laps} laps is <b>{pooled.laps * (pooled.degradation / (2 * pooled.loss)) ** 0.5:.3f}</b>.</p>
<table><tr><th>stops</th><th>plan</th><th>race time, s</th>
<th>behind the best</th></tr>{stop_html}</table>
<p>The exhaustive search over every partition of the distance finds exactly
what the algebra does, which is a test rather than a hope. Across the
degradation confidence interval the optimum moves between
{interval['range'][0]} and {interval['range'][1]} stints, so on the pooled
number the data
{"does" if interval['decided'] else "does <b>not</b>"} pick a stop
count on its own.</p>

<h2>Tested against what the field actually did</h2>
{_img(figures['validation'])}
<table><tr><th>circuit</th><th>laps</th><th>degradation</th>
<th>95% interval</th><th>pit loss</th><th>k*</th><th>model</th>
<th>field</th><th></th></tr>{validation_html}</table>
<p>Agrees at {agree}, recommends too few stops at {under}, too many at
{over}. The direction is the finding. Random error would land on both
sides.</p>
<p>Four circuits are refused outright rather than ranked, because their
measured degradation is not positive &mdash; on those laps tyres apparently
improve with age, and no stop count exploits that. Returning a strategy
anyway would be the optimiser lending false precision to a number that has
none.</p>
<table><tr><th>circuit refused</th><th>degradation</th><th>95% interval</th>
<th>field ran</th></tr>{refused_html}</table>

<h2>Is it safety car risk?</h2>
{_img(figures['distributions'])}
<p>{compared['draws']:,} simulated races at {focus['circuit']}, every plan
run through the <i>same</i> draws so the comparison is paired and the error
on a difference is far smaller than the error on either mean. The
simulation reproduces the observed frequency of neutralisation without
being told it: {compared['neutralised_any']:.1%} of simulated races contain
at least one, against {sc_stats['any']:.1%} of the real dry races.</p>
<table><tr><th>plan</th><th>stints</th><th>mean, s</th><th>sd</th>
<th>5th to 95th</th><th>behind best (paired, 95%)</th>
<th>races won</th></tr>{mc_html}</table>
<p>Reacting to a neutralisation &mdash; bringing the next stop forward when
one starts within the window &mdash; is worth
<b>{reactive_gain:+.2f} s</b> to the best plan.</p>
<table><tr><th>window, laps</th><th>best plan</th><th>mean, s</th></tr>
{window_html}</table>
{_img(figures['hazard'])}
<table><tr><th>per-lap hazard</th><th>P(any neutralisation)</th>
<th>best plan</th></tr>{hazard_html}</table>
<p>Across hazards from zero to {max(row['hazard'] for row in crossover):.2f}
&mdash; the last of which neutralises almost every race &mdash;
{len(winners)} plan ever wins: {', '.join(winners)}. Safety car risk
narrows the gap between one stop and two steadily, and never closes it. So
it is <b>not</b> the explanation for the field stopping more often.</p>

<h2>The penalty the lap times cannot see</h2>
{_img(figures['selection'])}
<p>The remaining candidate is the one every engineer names first: tyre
performance is not linear in age. Fitting an extra slope beyond a knot
tests it directly, and the answer comes back with the <i>wrong sign</i>,
significantly.</p>
<table><tr><th>knot</th><th>extra s/lap beyond it</th><th>95% interval</th>
<th>p</th><th>laps beyond</th></tr>{hinge_html}</table>
<table><tr><th>tyre age</th><th>laps reaching it</th><th>share</th></tr>
{coverage_html}</table>
<div class="note"><b>Why that result is unusable.</b> Teams choose when to
stop, and they stop before the tyre falls away. So a stint only reaches a
high age when that particular set happened to be holding on &mdash; which
is selection directly on the thing being measured. The laps that would
reveal a cliff are the laps strategy exists to avoid.</div>
<p>That argument can be tested where the truth is known. Below, every car
is given the <b>same</b> real cliff of {demo['truth']:+.4f} s per lap
beyond age {demo['knot']}, and the two arms differ only in how stints end.</p>
<table><tr><th>stints ended by</th><th>true extra slope</th>
<th>recovered</th><th>95% interval</th><th>share of truth</th>
<th>longest stint</th></tr>{demo_html}</table>
<p>With fixed plans the estimator recovers the cliff almost exactly. With a
stopping rule that retires the worst sets first it recovers a fraction of
it &mdash; and the more promptly teams are assumed to react, the less
survives.</p>
<table><tr><th>stop at, s/lap</th><th>recovered</th><th>share of truth</th>
</tr>{threshold_html}</table>

<h2>How big would it have to be?</h2>
{_img(figures['reconciliation'])}
<p>Since the cliff cannot be measured from these laps, the question is
inverted: what extra penalty would make the field's own choice the optimal
one? This is not a measurement of the cliff and is not offered as one. It
is a statement of the form <i>if a cliff is the only thing missing, it is
this big</i>.</p>
<table><tr><th>circuit</th><th>field ran</th>{knot_head}</tr>
{reconcile_html}</table>

<p class="sub">Generated by <code>run.py</code>. Lap times and pit stops
from the Jolpica mirror of the Ergast schema. The cleaning rules and the
fixed-effects estimator are carried over from the stint-degradation
project.</p>
</body></html>"""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
    return out_path
