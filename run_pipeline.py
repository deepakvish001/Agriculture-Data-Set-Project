"""End-to-end pipeline: data -> backtest -> forecast -> hold-out -> figures.

    python run_pipeline.py

Writes every table to outputs/tables, every figure to outputs/figures, a
machine-readable summary to outputs/results_summary.json and a self-contained
dashboard to outputs/dashboard.html.
"""

from __future__ import annotations

import json

import pandas as pd

from src import backtest, config, dashboard, data_loader, forecast, visualize


def _write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(config.TABLES / name, index=False)
    print(f"  wrote outputs/tables/{name}  ({len(df)} rows)")


def main() -> dict:
    print("[1/6] Loading and reshaping the Rajya Sabha MSP dataset")
    data = data_loader.build()
    audit = data["audit"]
    print(f"  {audit['rows']} series, {audit['commodities']} commodities, "
          f"{audit['missing_cells']} missing cells, "
          f"{audit['monotonic_series']}/{audit['rows']} monotonically non-decreasing")

    print("[2/6] Descriptive growth statistics")
    growth = forecast.growth_table(data["matrix"], data["long"])
    _write(growth.round(2), "growth_statistics.csv")

    print("[3/6] Rolling-origin backtest (10 models, identical protocol)")
    folds = backtest.rolling_origin(data["matrix"])
    overall = backtest.score_overall(folds)
    by_horizon = backtest.score_by_horizon(folds)
    by_series = backtest.score_by_series(folds)
    best = backtest.best_model_per_series(by_series)
    _write(folds.round(2), "backtest_folds.csv")
    _write(overall.round(3), "backtest_scores_overall.csv")
    _write(by_horizon.round(3), "backtest_scores_by_horizon.csv")
    _write(by_series.round(3), "backtest_scores_by_series.csv")
    _write(best.round(3), "best_model_per_series.csv")
    print(f"  backtest winner: {overall.iloc[0]['Model']} "
          f"(MAPE {overall.iloc[0]['MAPE']:.2f}%, MASE {overall.iloc[0]['MASE']:.2f})")

    print("[4/6] Refitting on full history and forecasting 2022-23 → 2024-25")
    forecasts = forecast.forecast_all(data["matrix"])
    forecasts = forecast.add_selected_strategy(forecasts, best)
    selected = forecast.selected_forecast(forecasts, best)
    _write(forecasts.drop(columns=["Chosen_Model"], errors="ignore"), "forecasts_all_models.csv")
    _write(selected, "forecast_selected_per_series.csv")

    print("[5/6] Scoring against MSP the government actually announced")
    merged = forecast.evaluate_holdout(forecasts, data["holdout"])
    holdout_scores = forecast.score_holdout(merged)
    holdout_by_season = forecast.score_holdout(merged, by=["Season_Label"])
    _write(merged.round(2), "holdout_predictions.csv")
    _write(holdout_scores.round(3), "holdout_scores.csv")
    _write(holdout_by_season.round(3), "holdout_scores_by_season.csv")

    champion = holdout_scores.iloc[0]["Model"]
    if champion == "Selected":  # prefer a nameable model for the figures
        champion = holdout_scores[holdout_scores["Model"] != "Selected"].iloc[0]["Model"]
    print(f"  hold-out winner: {holdout_scores.iloc[0]['Model']} "
          f"(MAPE {holdout_scores.iloc[0]['MAPE']:.2f}%) | figures use {champion}")

    print("[6/6] Rendering figures and dashboard")
    figures = visualize.build_all(
        data, growth, overall, by_horizon, forecasts, merged, holdout_scores, champion
    )
    for name, path in figures.items():
        print(f"  {name}: {path}")

    summary = {
        "audit": audit,
        "backtest_winner": overall.iloc[0]["Model"],
        "backtest_scores": overall.round(3).to_dict("records"),
        "holdout_winner": holdout_scores.iloc[0]["Model"],
        "champion_named_model": champion,
        "holdout_scores": holdout_scores.round(3).to_dict("records"),
        "holdout_by_season": holdout_by_season.round(3).to_dict("records"),
        "horizon_scores": by_horizon.round(3).to_dict("records"),
        "best_model_counts": best["Best_Model"].value_counts().to_dict(),
        "growth_top5": growth.head(5).round(2).to_dict("records"),
        "growth_bottom5": growth.tail(5).round(2).to_dict("records"),
        "n_holdout_points": int(merged["Model"].value_counts().iloc[0]),
        "figures": figures,
    }
    (config.OUTPUTS / "results_summary.json").write_text(json.dumps(summary, indent=2))
    print("  wrote outputs/results_summary.json")

    dashboard.render(data, growth, overall, holdout_scores, holdout_by_season,
                     merged, selected, best, champion)
    print(f"  wrote {config.DASHBOARD}")

    print("\nPipeline complete.")
    return summary


if __name__ == "__main__":
    main()
