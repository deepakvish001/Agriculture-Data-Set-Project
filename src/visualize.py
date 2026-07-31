"""Publication-quality figures.

Colour roles follow one validated categorical palette (slots assigned in fixed
order, never cycled), recessive chrome, thin marks and direct labels so identity
never rests on colour alone.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import BoxStyle, FancyBboxPatch

from . import config

# --- design tokens ----------------------------------------------------------

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

SEASON_COLOR = {
    "Kharif Crops": SERIES[0],
    "Rabi Crops": SERIES[1],
    "Other Crops": SERIES[2],
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "font.size": 9.5,
        "legend.frameon": False,
        "figure.dpi": 200,
    }
)


def _chrome(ax, xgrid: bool = False):
    """Recessive grid and axes: two spines, hairline grid behind the marks."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1)
    ax.grid(axis="x" if xgrid else "y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def _rounded_barh(ax, y, width, height, color, radius=0.03):
    """Horizontal bar with softened data-ends, anchored at x=0."""
    r = min(radius * abs(width) if width else 0, height / 2)
    patch = FancyBboxPatch(
        (0, y - height / 2),
        max(abs(width) - r, 1e-9) * np.sign(width or 1),
        height,
        boxstyle=BoxStyle("Round", pad=0, rounding_size=r),
        linewidth=0,
        facecolor=color,
        zorder=3,
    )
    ax.add_patch(patch)


def _save(fig, name: str) -> str:
    path = config.FIGURES / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


# --- figures ----------------------------------------------------------------

def fig_trends(matrix: pd.DataFrame) -> str:
    """MSP trajectories for the headline crops, directly labelled."""
    fig, ax = plt.subplots(figsize=(9, 5))
    _chrome(ax)

    names = [n for n in (f"{c} ({v})" if v != "NA" else c for c, v in config.SHOWCASE)
             if n in matrix.columns]
    for i, name in enumerate(names):
        y = matrix[name]
        ax.plot(matrix.index, y, color=SERIES[i], linewidth=2, marker="o",
                markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                label=name, zorder=3)

    ax.set_title("MSP trajectory of headline crops, crop years 2017-18 to 2021-22")
    ax.set_xlabel("Crop year (start)")
    ax.set_ylabel("MSP (₹ per quintal)")
    ax.set_xticks(matrix.index)
    ax.set_xlim(matrix.index[0] - 0.2, matrix.index[-1] + 1.9)

    # Direct labels, nudged apart so crops that end at similar prices (Wheat and
    # Paddy sit ~₹75 apart) do not print on top of each other.
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.055
    ends = sorted(((matrix[n].iloc[-1], n, SERIES[i]) for i, n in enumerate(names)),
                  reverse=True)
    placed: list[float] = []
    for value, name, colour in ends:
        target = value
        if placed and placed[-1] - target < gap:
            target = placed[-1] - gap
        placed.append(target)
        ax.annotate(
            f"{name}  ₹{value:,.0f}",
            (matrix.index[-1], value), xytext=(11, target),
            textcoords=("offset points", "data"),
            color=INK_2, fontsize=9, va="center", fontweight="bold",
            annotation_clip=False,
        )
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.legend(loc="upper left", ncol=2, fontsize=9)
    return _save(fig, "fig1_msp_trends.png")


def fig_growth(growth: pd.DataFrame) -> str:
    """Five-year CAGR for every series, grouped by cropping season."""
    data = growth.sort_values("CAGR_pct")
    fig, ax = plt.subplots(figsize=(9, 9))
    _chrome(ax, xgrid=True)

    for i, row in enumerate(data.itertuples()):
        _rounded_barh(ax, i, row.CAGR_pct, 0.68, SEASON_COLOR.get(row.Season, MUTED))
        ax.text(row.CAGR_pct + 0.12, i, f"{row.CAGR_pct:.1f}%", va="center",
                fontsize=8.5, color=INK_2)

    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["Series"], fontsize=9)
    ax.set_xlim(0, data["CAGR_pct"].max() * 1.18)
    ax.set_ylim(-0.8, len(data) - 0.2)
    ax.set_xlabel("Compound annual growth rate (%)")
    ax.set_title("Every MSP series grew — but at very different rates (2017-18 → 2021-22)")

    handles = [plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=c, label=s)
               for s, c in SEASON_COLOR.items()]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    return _save(fig, "fig2_cagr_by_series.png")


def fig_leaderboard(backtest_scores: pd.DataFrame, holdout_scores: pd.DataFrame) -> str:
    """Backtest MAPE against true hold-out MAPE, per model."""
    merged = (
        backtest_scores[["Model", "MAPE"]].rename(columns={"MAPE": "Backtest"})
        .merge(holdout_scores[["Model", "MAPE"]].rename(columns={"MAPE": "Holdout"}), on="Model")
        .sort_values("Holdout")
    )
    fig, ax = plt.subplots(figsize=(9, 6))
    _chrome(ax, xgrid=True)

    h = 0.36
    for i, row in enumerate(merged.itertuples()):
        _rounded_barh(ax, i + h / 2 + 0.02, row.Backtest, h, SERIES[0])
        _rounded_barh(ax, i - h / 2 - 0.02, row.Holdout, h, SERIES[1])
        ax.text(row.Backtest + 0.12, i + h / 2 + 0.02, f"{row.Backtest:.1f}%",
                va="center", fontsize=8.5, color=INK_2)
        ax.text(row.Holdout + 0.12, i - h / 2 - 0.02, f"{row.Holdout:.1f}%",
                va="center", fontsize=8.5, color=INK_2)

    ax.set_yticks(range(len(merged)))
    ax.set_yticklabels(merged["Model"], fontsize=9.5)
    ax.set_xlim(0, max(merged["Backtest"].max(), merged["Holdout"].max()) * 1.16)
    ax.set_ylim(-0.8, len(merged) - 0.2)
    ax.set_xlabel("Mean absolute percentage error (%) — lower is better")
    ax.set_title("Backtest rank ≠ hold-out rank\nInternal walk-forward CV vs MSP the government actually announced")

    handles = [
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=SERIES[0],
                   label="Rolling-origin backtest (2020, 2021)"),
        plt.Line2D([], [], marker="s", linestyle="", markersize=9, color=SERIES[1],
                   label="Hold-out (announced 2022-23, 2023-24)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9)
    return _save(fig, "fig3_model_leaderboard.png")


def fig_parity(merged: pd.DataFrame, model: str, long: pd.DataFrame) -> str:
    """Forecast vs announced MSP for the winning model."""
    sub = merged[merged["Model"] == model].merge(
        long.drop_duplicates("Series")[["Series", "Season"]], on="Series", how="left"
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.6))
    _chrome(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8, zorder=0)

    lo = min(sub["Actual_MSP"].min(), sub["Forecast"].min()) * 0.88
    hi = max(sub["Actual_MSP"].max(), sub["Forecast"].max()) * 1.12
    ax.plot([lo, hi], [lo, hi], color=AXIS, linewidth=1.5, linestyle="--", zorder=1)
    ax.fill_between([lo, hi], [lo * 0.95, hi * 0.95], [lo * 1.05, hi * 1.05],
                    color=GRID, alpha=0.55, zorder=0, linewidth=0)

    for season, grp in sub.groupby("Season"):
        ax.scatter(grp["Actual_MSP"], grp["Forecast"], s=62,
                   color=SEASON_COLOR.get(season, MUTED), edgecolor=SURFACE,
                   linewidth=1.6, zorder=3, label=season)

    # Callouts step downward and hug the inside of the panel so the labels on
    # the tightly-clustered high-price points cannot overlap each other.
    worst = sub.nlargest(3, "APE").sort_values("Actual_MSP", ascending=False)
    midpoint = np.sqrt(lo * hi)
    for k, row in enumerate(worst.itertuples()):
        right_half = row.Actual_MSP > midpoint
        ax.annotate(
            f"{row.Series} ({row.APE:.1f}%)",
            (row.Actual_MSP, row.Forecast),
            xytext=(-11 if right_half else 11, -16 - 13 * k),
            textcoords="offset points", fontsize=8.5, color=INK_2,
            ha="right" if right_half else "left",
            arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=1,
                            shrinkA=0, shrinkB=4),
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [t for t in (1500, 2000, 3000, 4000, 6000, 8000, 12000) if lo <= t <= hi]
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
        axis.set_minor_locator(matplotlib.ticker.NullLocator())
        axis.set_major_formatter(lambda v, _: f"{v:,.0f}")
    ax.set_xlabel("MSP actually announced (₹/qtl, log scale)")
    ax.set_ylabel("Forecast (₹/qtl, log scale)")
    hit = (sub["APE"] <= 5).mean() * 100
    ax.set_title(f"{model} forecasts vs announced MSP\n{hit:.0f}% of {len(sub)} forecasts land inside the ±5% band")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, "fig4_holdout_parity.png")


def fig_forecast_panels(matrix: pd.DataFrame, forecasts: pd.DataFrame,
                        holdout: pd.DataFrame, model: str) -> str:
    """History, forecast and announced MSP for the headline crops."""
    names = [f"{c} ({v})" if v != "NA" else c for c, v in config.SHOWCASE]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))

    for ax, name in zip(axes.ravel(), names):
        _chrome(ax)
        hist = matrix[name]
        ax.plot(hist.index, hist.values, color=INK, linewidth=2, marker="o",
                markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.8,
                label="Observed (training)", zorder=4)

        fc = forecasts[(forecasts["Series"] == name) & (forecasts["Model"] == model)].sort_values("Year")
        bridge_x = [hist.index[-1]] + fc["Year"].tolist()
        bridge_y = [hist.values[-1]] + fc["Forecast"].tolist()
        ax.plot(bridge_x, bridge_y, color=SERIES[0], linewidth=2, linestyle="--",
                marker="s", markersize=6.5, markeredgecolor=SURFACE,
                markeredgewidth=1.8, label=f"Forecast ({model})", zorder=3)

        act = holdout[holdout["Series"] == name]
        if len(act):
            ax.scatter(act["Year"], act["Actual_MSP"], s=95, color=SERIES[1],
                       marker="X", edgecolor=SURFACE, linewidth=1.6, zorder=5,
                       label="Announced MSP (hold-out)")

        ax.axvspan(hist.index[-1] + 0.5, max(fc["Year"]) + 0.4, color=GRID,
                   alpha=0.45, zorder=0, linewidth=0)
        ax.set_title(name, fontsize=11)
        ax.set_xticks(list(hist.index) + fc["Year"].tolist())
        ax.tick_params(labelsize=8.5)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:,.0f}")

    axes[0, 0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Forecast vs reality: shaded region is beyond the training data",
                 fontsize=13, fontweight="bold", y=0.985)
    fig.supylabel("MSP (₹ per quintal)", fontsize=10, color=INK_2)
    fig.tight_layout(rect=[0.015, 0, 1, 0.96])
    return _save(fig, "fig5_forecast_panels.png")


def fig_bias(holdout_scores: pd.DataFrame) -> str:
    """Systematic over/under-forecasting on the hold-out seasons."""
    data = holdout_scores.sort_values("Bias")
    fig, ax = plt.subplots(figsize=(9, 5.4))
    _chrome(ax, xgrid=True)

    for i, row in enumerate(data.itertuples()):
        colour = CRITICAL if row.Bias < 0 else SERIES[0]
        _rounded_barh(ax, i, row.Bias, 0.66, colour)
        offset = 45 if row.Bias >= 0 else -45
        ax.text(row.Bias + offset, i, f"₹{row.Bias:,.0f}", va="center",
                ha="left" if row.Bias >= 0 else "right", fontsize=8.5, color=INK_2)

    ax.axvline(0, color=AXIS, linewidth=1.2, zorder=2)
    ax.set_yticks(range(len(data)))
    ax.set_yticklabels(data["Model"], fontsize=9.5)
    ax.set_ylim(-0.8, len(data) - 0.2)
    pad = max(abs(data["Bias"])) * 0.35
    ax.set_xlim(data["Bias"].min() - pad, data["Bias"].max() + pad)
    ax.set_xlabel("Mean forecast error on hold-out seasons (₹/quintal)")
    ax.set_title("Which models systematically under-forecast MSP?\nNegative bias means the forecast fell short of the announced price")
    return _save(fig, "fig6_holdout_bias.png")


def fig_horizon(horizon_scores: pd.DataFrame) -> str:
    """Error growth as the forecast reaches further ahead."""
    fig, ax = plt.subplots(figsize=(8.4, 5))
    _chrome(ax)

    top = (
        horizon_scores.groupby("Model")["MAPE"].mean().nsmallest(5).index.tolist()
    )
    for i, model in enumerate(top):
        sub = horizon_scores[horizon_scores["Model"] == model].sort_values("Horizon")
        ax.plot(sub["Horizon"], sub["MAPE"], color=SERIES[i % len(SERIES)], linewidth=2,
                marker="o", markersize=7, markeredgecolor=SURFACE, markeredgewidth=2,
                label=model, zorder=3)
        ax.annotate(model, (sub["Horizon"].iloc[-1], sub["MAPE"].iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", fontsize=9,
                    color=INK_2, va="center", fontweight="bold")

    ax.set_xticks(sorted(horizon_scores["Horizon"].unique()))
    ax.set_xlim(0.85, horizon_scores["Horizon"].max() + 0.75)
    ax.set_xlabel("Forecast horizon (seasons ahead)")
    ax.set_ylabel("MAPE (%)")
    ax.set_title("Accuracy decays with horizon — five best backtest models")
    ax.legend(loc="upper left", fontsize=9)
    return _save(fig, "fig7_horizon_error.png")


def build_all(data: dict, growth: pd.DataFrame, backtest_scores: pd.DataFrame,
              horizon_scores: pd.DataFrame, forecasts: pd.DataFrame,
              holdout_merged: pd.DataFrame, holdout_scores: pd.DataFrame,
              champion: str) -> dict:
    return {
        "trends": fig_trends(data["matrix"]),
        "growth": fig_growth(growth),
        "leaderboard": fig_leaderboard(backtest_scores, holdout_scores),
        "parity": fig_parity(holdout_merged, champion, data["long"]),
        "panels": fig_forecast_panels(data["matrix"], forecasts, data["holdout"], champion),
        "bias": fig_bias(holdout_scores),
        "horizon": fig_horizon(horizon_scores),
    }
