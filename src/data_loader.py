"""Load, validate and reshape the Rajya Sabha MSP dataset.

The public CSV is a wide table: one row per commodity-variety, one column per
crop year. Everything downstream wants a tidy long table keyed by
(series_id, year), so the reshaping happens here once.
"""

from __future__ import annotations

import pandas as pd

from . import config


def _series_id(commodity: str, variety: str) -> str:
    """Stable identifier for a commodity-variety pair."""
    variety = (variety or "NA").strip()
    if variety in ("NA", "", "General"):
        return commodity.strip()
    return f"{commodity.strip()} ({variety})"


def load_raw() -> pd.DataFrame:
    """Read the CSV exactly as published, with light column hygiene."""
    df = pd.read_csv(config.DATA_RAW)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={"Crops Session": "Season"})
    for col in ("Season", "Commodity", "Variety"):
        df[col] = df[col].astype(str).str.strip()
    return df


def audit(df: pd.DataFrame) -> dict:
    """Data-quality summary reported in the notebook and the write-up."""
    numeric = df[config.YEAR_COLS].apply(pd.to_numeric, errors="coerce")
    monotonic = numeric.diff(axis=1).iloc[:, 1:].ge(0).all(axis=1)
    return {
        "rows": len(df),
        "commodities": df["Commodity"].nunique(),
        "series": len(df),
        "seasons": df["Season"].value_counts().to_dict(),
        "missing_cells": int(numeric.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated(subset=["Commodity", "Variety"]).sum()),
        "non_numeric_cells": int(numeric.isna().sum().sum()),
        "variety_na_rows": int((df["Variety"] == "NA").sum()),
        "monotonic_series": int(monotonic.sum()),
        "non_monotonic_series": df.loc[~monotonic, "Commodity"].tolist(),
        "min_msp": float(numeric.min().min()),
        "max_msp": float(numeric.max().max()),
    }


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Type-cast prices and normalise the Variety column.

    'NA' in Variety is a genuine category (the commodity has a single grade),
    not a missing value, so it is kept rather than imputed or dropped.
    """
    out = df.copy()
    for col in config.YEAR_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out["Variety"] = out["Variety"].fillna("NA").replace({"": "NA"})
    out = out.dropna(subset=config.YEAR_COLS, how="all")
    out["Series"] = [
        _series_id(c, v) for c, v in zip(out["Commodity"], out["Variety"])
    ]
    return out.reset_index(drop=True)


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Wide -> tidy long, with both crop-year and marketing-season indices.

    ``Year`` is the crop year as published. ``Marketing_Year`` applies the
    season offset so Rabi rows line up with the season the MSP was actually
    announced for.
    """
    long = df.melt(
        id_vars=["Sl. No.", "Season", "Commodity", "Variety", "Series"],
        value_vars=config.YEAR_COLS,
        var_name="Season_Label",
        value_name="MSP",
    )
    long["Year"] = long["Season_Label"].str.split("-").str[0].astype(int)
    long["Offset"] = long["Season"].map(config.SEASON_OFFSET).fillna(0).astype(int)
    long["Marketing_Year"] = long["Year"] + long["Offset"]
    long["Marketing_Season"] = long["Marketing_Year"].map(
        lambda y: f"{y}-{str(y + 1)[-2:]}"
    )
    return long.sort_values(["Series", "Year"]).reset_index(drop=True)


def price_matrix(long: pd.DataFrame) -> pd.DataFrame:
    """Year x Series matrix of MSP values.

    Kept at commodity-*variety* granularity on purpose. Averaging Paddy Common
    with Paddy Grade 'A' (or Cotton Medium with Long Staple) invents a price
    the government never announced and makes the forecast impossible to score
    against a real MSP notification.
    """
    return long.pivot_table(
        index="Year", columns="Series", values="MSP", aggfunc="mean"
    ).sort_index()


def load_holdout() -> pd.DataFrame:
    """Announced MSP for seasons after the training window (ground truth)."""
    hold = pd.read_csv(config.DATA_EXTERNAL)
    hold["Variety"] = hold["Variety"].fillna("NA")
    hold["Series"] = [
        _series_id(c, v) for c, v in zip(hold["Commodity"], hold["Variety"])
    ]
    hold["Year"] = hold["Season_Label"].str.split("-").str[0].astype(int)
    return hold


def build() -> dict:
    """Run the whole data stage and return every artefact downstream needs."""
    raw = load_raw()
    report = audit(raw)
    cleaned = clean(raw)
    long = to_long(cleaned)
    return {
        "raw": raw,
        "audit": report,
        "clean": cleaned,
        "long": long,
        "matrix": price_matrix(long),
        "holdout": load_holdout(),
    }
