"""Self-contained interactive HTML dashboard.

No CDN, no external fonts, no network calls — the data is inlined as JSON and
the charts are drawn as SVG by a few dozen lines of vanilla JS, so the file
opens straight from disk and survives being emailed around.
"""

from __future__ import annotations

import json

import pandas as pd

from . import config

CSS = """
*,*::before,*::after{box-sizing:border-box}
.viz-root{
  color-scheme:light;
  --surface-1:#fcfcfb; --plane:#f9f9f7;
  --text-primary:#0b0b0b; --text-secondary:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
  --series-1:#2a78d6; --series-2:#eb6834; --series-3:#1baf7a;
  --good:#0ca30c; --critical:#d03b3b;
}
@media (prefers-color-scheme:dark){
  :root:where(:not([data-theme="light"])) .viz-root{
    color-scheme:dark;
    --surface-1:#1a1a19; --plane:#0d0d0d;
    --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
  }
}
:root[data-theme="dark"] .viz-root{
  color-scheme:dark;
  --surface-1:#1a1a19; --plane:#0d0d0d;
  --text-primary:#fff; --text-secondary:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
  --series-1:#3987e5; --series-2:#d95926; --series-3:#199e70;
}
body{margin:0;background:var(--plane)}
.viz-root{
  font-family:system-ui,-apple-system,"Segoe UI",sans-serif;
  background:var(--plane); color:var(--text-primary);
  padding:32px 20px 64px; min-height:100vh;
}
.wrap{max-width:1120px;margin:0 auto}
header h1{font-size:clamp(22px,3.4vw,31px);line-height:1.15;margin:0 0 8px}
header p{color:var(--text-secondary);margin:0;max-width:70ch;line-height:1.55;font-size:14.5px}
.tag{display:inline-block;font-size:11.5px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--muted);margin-bottom:10px;font-weight:600}
section{margin-top:40px}
h2{font-size:17.5px;margin:0 0 4px}
.sub{color:var(--text-secondary);font-size:13.5px;margin:0 0 16px;line-height:1.5;max-width:78ch}
.card{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:20px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:14px}
.tile{background:var(--surface-1);border:1px solid var(--border);border-radius:12px;padding:16px 18px}
.tile .k{font-size:11.5px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:600}
.tile .v{font-size:30px;font-weight:700;margin:6px 0 2px;line-height:1}
.tile .n{font-size:12.5px;color:var(--text-secondary);line-height:1.4}
.controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;margin-bottom:14px}
select{font:inherit;font-size:13.5px;padding:7px 11px;border-radius:8px;
  border:1px solid var(--border);background:var(--surface-1);color:var(--text-primary)}
label{font-size:12.5px;color:var(--text-secondary);font-weight:600}
.legend{display:flex;flex-wrap:wrap;gap:16px;margin-top:12px;font-size:12.5px;color:var(--text-secondary)}
.legend i{display:inline-block;width:11px;height:11px;border-radius:3px;margin-right:6px;vertical-align:-1px}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;width:100%;font-size:13px;min-width:560px}
th,td{padding:8px 11px;text-align:right;border-bottom:1px solid var(--border);white-space:nowrap}
th{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.05em;
  font-weight:600;border-bottom:1px solid var(--axis)}
th:first-child,td:first-child{text-align:left}
td{font-variant-numeric:tabular-nums;color:var(--text-secondary)}
td:first-child{color:var(--text-primary);font-weight:600}
tbody tr:hover{background:color-mix(in srgb,var(--series-1) 7%,transparent)}
.bar{display:inline-block;height:9px;border-radius:4px;background:var(--series-1);vertical-align:0}
.win{color:var(--good);font-weight:700}
.lose{color:var(--critical);font-weight:700}
svg{display:block;width:100%;height:auto;overflow:visible}
.tip{position:fixed;pointer-events:none;opacity:0;transition:opacity .1s;
  background:var(--surface-1);border:1px solid var(--border);border-radius:9px;
  padding:9px 12px;font-size:12.5px;color:var(--text-primary);
  box-shadow:0 6px 22px rgba(0,0,0,.16);z-index:40;font-variant-numeric:tabular-nums}
.tip b{display:block;margin-bottom:4px;font-size:13px}
.tip span{color:var(--text-secondary)}
footer{margin-top:48px;padding-top:18px;border-top:1px solid var(--border);
  color:var(--muted);font-size:12.5px;line-height:1.6}
@media (max-width:640px){.viz-root{padding:22px 14px 48px}.tile .v{font-size:25px}}
"""

JS = """
const D = window.__MSP__;
const tip = document.getElementById('tip');
const fmt = n => '\\u20b9' + Math.round(n).toLocaleString('en-IN');

function showTip(e, html){
  tip.innerHTML = html; tip.style.opacity = 1;
  const r = tip.getBoundingClientRect();
  let x = e.clientX + 16, y = e.clientY - r.height - 12;
  if (x + r.width > innerWidth - 8) x = e.clientX - r.width - 16;
  if (y < 8) y = e.clientY + 18;
  tip.style.left = x + 'px'; tip.style.top = y + 'px';
}
const hideTip = () => tip.style.opacity = 0;

function el(tag, attrs, txt){
  const n = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const k in attrs) n.setAttribute(k, attrs[k]);
  if (txt != null) n.textContent = txt;
  return n;
}

function drawSeries(name){
  const s = D.series[name];
  const box = document.getElementById('chart');
  box.innerHTML = '';
  const W = 900, H = 400, m = {t: 24, r: 118, b: 44, l: 68};

  const pts = s.history.concat(s.forecast);
  const acts = s.actual;
  const all = pts.map(p => p.v).concat(acts.map(p => p.v));
  const years = pts.map(p => p.y);
  const yMin = Math.min(...all), yMax = Math.max(...all);
  const pad = (yMax - yMin) * 0.16 || yMax * 0.1;
  const lo = yMin - pad, hi = yMax + pad;
  const x = y => m.l + (y - years[0]) / (years[years.length-1] - years[0]) * (W - m.l - m.r);
  const yy = v => H - m.b - (v - lo) / (hi - lo) * (H - m.t - m.b);

  const svg = el('svg', {viewBox: `0 0 ${W} ${H}`, role: 'img',
    'aria-label': `MSP history and forecast for ${name}`});

  // forecast region
  const cut = x(s.history[s.history.length-1].y);
  svg.appendChild(el('rect', {x: cut, y: m.t, width: W - m.r - cut, height: H - m.t - m.b,
    fill: 'var(--grid)', opacity: .5}));
  svg.appendChild(el('text', {x: cut + 8, y: m.t + 14, fill: 'var(--muted)',
    'font-size': 11.5, 'font-weight': 600}, 'beyond training data'));

  // grid + y axis
  for (let i = 0; i <= 4; i++){
    const v = lo + (hi - lo) * i / 4;
    svg.appendChild(el('line', {x1: m.l, x2: W - m.r, y1: yy(v), y2: yy(v),
      stroke: 'var(--grid)', 'stroke-width': 1}));
    svg.appendChild(el('text', {x: m.l - 10, y: yy(v) + 4, 'text-anchor': 'end',
      fill: 'var(--muted)', 'font-size': 11.5}, fmt(v)));
  }
  svg.appendChild(el('line', {x1: m.l, x2: W - m.r, y1: H - m.b, y2: H - m.b,
    stroke: 'var(--axis)', 'stroke-width': 1}));
  years.forEach(y => svg.appendChild(el('text', {x: x(y), y: H - m.b + 19,
    'text-anchor': 'middle', fill: 'var(--muted)', 'font-size': 11.5}, y + '-' + String(y+1).slice(2))));

  const path = ps => ps.map((p, i) => (i ? 'L' : 'M') + x(p.y) + ' ' + yy(p.v)).join(' ');
  svg.appendChild(el('path', {d: path(s.history), fill: 'none',
    stroke: 'var(--text-primary)', 'stroke-width': 2, 'stroke-linecap': 'round'}));
  const bridge = [s.history[s.history.length-1]].concat(s.forecast);
  svg.appendChild(el('path', {d: path(bridge), fill: 'none', stroke: 'var(--series-1)',
    'stroke-width': 2, 'stroke-dasharray': '7 5', 'stroke-linecap': 'round'}));

  const mark = (p, fill, label) => {
    const c = el('circle', {cx: x(p.y), cy: yy(p.v), r: 5.5, fill,
      stroke: 'var(--surface-1)', 'stroke-width': 2});
    const hit = el('circle', {cx: x(p.y), cy: yy(p.v), r: 16, fill: 'transparent'});
    hit.addEventListener('pointermove', e => showTip(e,
      `<b>${name}</b><span>${label} \\u00b7 ${p.y}-${String(p.y+1).slice(2)}</span>${fmt(p.v)}`));
    hit.addEventListener('pointerleave', hideTip);
    svg.appendChild(c); svg.appendChild(hit);
  };
  s.history.forEach(p => mark(p, 'var(--text-primary)', 'Observed'));
  s.forecast.forEach(p => mark(p, 'var(--series-1)', 'Forecast \\u00b7 ' + s.model));
  acts.forEach(p => {
    const g = el('path', {d: `M${x(p.y)-6} ${yy(p.v)-6} L${x(p.y)+6} ${yy(p.v)+6}
      M${x(p.y)+6} ${yy(p.v)-6} L${x(p.y)-6} ${yy(p.v)+6}`.replace(/\\s+/g,' '),
      stroke: 'var(--series-2)', 'stroke-width': 3, 'stroke-linecap': 'round', fill: 'none'});
    const hit = el('circle', {cx: x(p.y), cy: yy(p.v), r: 16, fill: 'transparent'});
    hit.addEventListener('pointermove', e => showTip(e,
      `<b>${name}</b><span>Announced MSP \\u00b7 ${p.season}</span>${fmt(p.v)}`));
    hit.addEventListener('pointerleave', hideTip);
    svg.appendChild(g); svg.appendChild(hit);
  });

  const last = s.forecast[s.forecast.length-1];
  svg.appendChild(el('text', {x: x(last.y) + 12, y: yy(last.v) + 4,
    fill: 'var(--text-secondary)', 'font-size': 12, 'font-weight': 700}, fmt(last.v)));

  box.appendChild(svg);
  document.getElementById('pick-note').textContent =
    `Backtest-selected model: ${s.model}` +
    (s.mape != null ? ` \\u00b7 hold-out error ${s.mape.toFixed(1)}%` : ' \\u00b7 no hold-out actual published for this series');
}

const sel = document.getElementById('crop');
Object.keys(D.series).sort().forEach(k => sel.appendChild(new Option(k, k)));
sel.value = D.defaultSeries;
sel.addEventListener('change', () => drawSeries(sel.value));
drawSeries(sel.value);
"""


def _bar_cell(value: float, vmax: float) -> str:
    width = max(2.0, (value / vmax) * 100) if vmax else 2.0
    return (f'<span class="bar" style="width:{width:.1f}px"></span> '
            f'{value:.2f}')


def _model_table(holdout_scores: pd.DataFrame, backtest_scores: pd.DataFrame) -> str:
    bt = dict(zip(backtest_scores["Model"], backtest_scores["MAPE"]))
    vmax = holdout_scores["MAPE"].max()
    rows = []
    for r in holdout_scores.itertuples():
        cls = "win" if r.MASE < 1 else "lose"
        rows.append(
            f"<tr><td>{r.Model}</td>"
            f"<td>{bt.get(r.Model, float('nan')):.2f}%</td>"
            f"<td>{_bar_cell(r.MAPE, vmax)}%</td>"
            f"<td>₹{r.MAE:,.0f}</td>"
            f"<td>₹{r.Bias:,.0f}</td>"
            f'<td class="{cls}">{r.MASE:.2f}</td>'
            f"<td>{r.Within_5pct:.0f}%</td></tr>"
        )
    return (
        '<div class="scroll"><table><thead><tr><th>Model</th><th>Backtest MAPE</th>'
        "<th>Hold-out MAPE</th><th>Hold-out MAE</th><th>Bias</th><th>MASE</th>"
        "<th>Within ±5%</th></tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def _forecast_table(selected: pd.DataFrame, best: pd.DataFrame) -> str:
    merged = selected.merge(best[["Series", "MAPE"]], on="Series", how="left")
    cols = [c for c in config.FORECAST_LABELS if c in merged.columns]
    rows = []
    # to_dict keeps the "2022-23"-style column names, which itertuples mangles.
    for d in merged.sort_values("Series").to_dict("records"):
        cells = "".join(f"<td>₹{d[c]:,.0f}</td>" for c in cols)
        rows.append(
            f"<tr><td>{d['Series']}</td><td>{d['Best_Model']}</td>"
            f"<td>{d['MAPE']:.2f}%</td>{cells}</tr>"
        )
    head = "".join(f"<th>{c}</th>" for c in cols)
    return (
        '<div class="scroll"><table><thead><tr><th>Commodity (variety)</th><th>Model</th>'
        f"<th>Backtest MAPE</th>{head}</tr></thead><tbody>"
        + "".join(rows) + "</tbody></table></div>"
    )


def _season_table(by_season: pd.DataFrame, champion: str) -> str:
    sub = by_season[by_season["Model"].isin([champion, "Naive", "Selected"])]
    rows = "".join(
        f"<tr><td>{r.Model}</td><td>{r.Season_Label}</td><td>{r.MAPE:.2f}%</td>"
        f"<td>₹{r.MAE:,.0f}</td><td>{r.Within_5pct:.0f}%</td><td>{int(r.N)}</td></tr>"
        for r in sub.sort_values(["Season_Label", "MAPE"]).itertuples()
    )
    return (
        '<div class="scroll"><table><thead><tr><th>Model</th><th>Season</th><th>MAPE</th>'
        "<th>MAE</th><th>Within ±5%</th><th>Crops scored</th></tr></thead><tbody>"
        f"{rows}</tbody></table></div>"
    )


def render(data: dict, growth: pd.DataFrame, backtest_scores: pd.DataFrame,
           holdout_scores: pd.DataFrame, holdout_by_season: pd.DataFrame,
           holdout_merged: pd.DataFrame, selected: pd.DataFrame,
           best: pd.DataFrame, champion: str) -> str:
    matrix = data["matrix"]
    holdout = data["holdout"]

    sel_rows = selected.set_index("Series")
    best_map = dict(zip(best["Series"], best["Best_Model"]))
    sel_err = (
        holdout_merged[holdout_merged["Model"] == "Selected"]
        .groupby("Series")["APE"].mean().to_dict()
    )

    payload = {"series": {}, "defaultSeries": "Wheat"}
    for name in matrix.columns:
        history = [{"y": int(y), "v": float(v)} for y, v in matrix[name].items()]
        fc = sel_rows.loc[name]
        forecast_pts = [
            {"y": int(lbl.split("-")[0]), "v": float(fc[lbl])}
            for lbl in config.FORECAST_LABELS if lbl in sel_rows.columns
        ]
        act = holdout[holdout["Series"] == name]
        payload["series"][name] = {
            "history": history,
            "forecast": forecast_pts,
            "actual": [
                {"y": int(r.Year), "v": float(r.Actual_MSP), "season": r.Marketing_Season}
                for r in act.itertuples()
            ],
            "model": best_map.get(name, "—"),
            "mape": round(sel_err[name], 2) if name in sel_err else None,
        }

    top = holdout_scores.iloc[0]
    named = holdout_scores[holdout_scores["Model"] == champion].iloc[0]
    bt_top = backtest_scores.iloc[0]
    n_points = int(holdout_merged["Model"].value_counts().iloc[0])
    fastest = growth.iloc[0]

    tiles = [
        ("Series modelled", f"{matrix.shape[1]}",
         "commodity-variety pairs, kept unaveraged"),
        ("Hold-out forecasts scored", f"{n_points}",
         "against MSP the government actually announced"),
        ("Best hold-out error", f"{top['MAPE']:.2f}%",
         f"{top['Model']} — {top['Within_5pct']:.0f}% of forecasts inside ±5%"),
        ("Beats the naive baseline by", f"{(1 - named['MASE']) * 100:.0f}%",
         f"{champion} MASE {named['MASE']:.2f} (below 1.0 = better than carry-forward)"),
    ]
    tile_html = "".join(
        f'<div class="tile"><div class="k">{k}</div><div class="v">{v}</div>'
        f'<div class="n">{n}</div></div>' for k, v, n in tiles
    )

    body = f"""
<div class="viz-root"><div class="wrap">
<header>
  <div class="tag">Fundamentals of AI using Agriculture Data Set · ANNAM.AI, IIT Ropar</div>
  <h1>Forecasting India's Minimum Support Prices</h1>
  <p>Ten forecasting models trained on five years of official MSP data for
  {matrix.shape[1]} commodity-variety series, compared under one identical
  rolling-origin protocol, then tested against the prices the Government of
  India actually announced for the following two seasons.</p>
</header>

<section>
  <div class="tiles">{tile_html}</div>
</section>

<section>
  <h2>Per-crop forecast explorer</h2>
  <p class="sub">Black is observed MSP from the Rajya Sabha dataset. The dashed
  blue line is the forecast from the model this crop's own backtest selected.
  Orange crosses are the MSP later announced — they were never shown to any
  model. Hover any point for exact values.</p>
  <div class="card">
    <div class="controls">
      <label for="crop">Commodity</label>
      <select id="crop"></select>
    </div>
    <div id="chart"></div>
    <div class="legend">
      <span><i style="background:var(--text-primary)"></i>Observed (2017-18 → 2021-22)</span>
      <span><i style="background:var(--series-1)"></i>Forecast</span>
      <span><i style="background:var(--series-2)"></i>Announced MSP (hold-out)</span>
    </div>
    <p class="sub" id="pick-note" style="margin:12px 0 0"></p>
  </div>
</section>

<section>
  <h2>Model leaderboard</h2>
  <p class="sub">The two error columns disagree, and that disagreement is the
  main finding. {bt_top['Model']} wins the internal backtest
  ({bt_top['MAPE']:.2f}% MAPE) but places lower once scored against real
  announcements — with only three or four training points per fold it fits the
  revision noise rather than the trend. MASE below 1.0 means the model beats
  simply carrying last season's MSP forward.</p>
  <div class="card">{_model_table(holdout_scores, backtest_scores)}</div>
</section>

<section>
  <h2>Hold-out accuracy by season</h2>
  <p class="sub">Error grows with horizon, as it must: the second season is
  forecast from the same 2021-22 origin, one further revision away.</p>
  <div class="card">{_season_table(holdout_by_season, champion)}</div>
</section>

<section>
  <h2>Forecasts for every commodity</h2>
  <p class="sub">Each row uses the model chosen by that crop's own backtest.
  Fastest five-year grower in the dataset: {fastest['Series']} at
  {fastest['CAGR_pct']:.1f}% CAGR.</p>
  <div class="card">{_forecast_table(selected, best)}</div>
</section>

<footer>
  Source: Rajya Sabha Session 255, Unstarred Question AU-1460, published on
  data.gov.in under the Open Government Data licence. Hold-out actuals from PIB
  Cabinet MSP announcements (see <code>data/external/msp_actuals_holdout.csv</code>
  for the per-value citation). Generated by <code>run_pipeline.py</code>.
</footer>
</div></div>
<div class="tip" id="tip"></div>
"""

    html = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>MSP Forecasting Dashboard</title>"
        f"<style>{CSS}</style></head><body>{body}"
        f"<script>window.__MSP__={json.dumps(payload)};</script>"
        f"<script>{JS}</script></body></html>"
    )
    config.DASHBOARD.write_text(html, encoding="utf-8")
    return str(config.DASHBOARD)
