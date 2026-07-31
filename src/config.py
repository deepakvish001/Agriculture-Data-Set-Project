"""Central configuration: paths, constants and domain knowledge about the dataset."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_RAW = ROOT / "data" / "raw" / "RS_Session_255_AU_1460_1.csv"
DATA_EXTERNAL = ROOT / "data" / "external" / "msp_actuals_holdout.csv"

OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
DASHBOARD = OUTPUTS / "dashboard.html"

for _d in (OUTPUTS, FIGURES, TABLES):
    _d.mkdir(parents=True, exist_ok=True)

# Year columns exactly as they appear in the Rajya Sabha CSV.
YEAR_COLS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22"]

# Seasons we forecast beyond the training window.
FORECAST_LABELS = ["2022-23", "2023-24", "2024-25"]

# ---------------------------------------------------------------------------
# Season-label semantics.
#
# The CSV labels every column by *crop year*, but MSP is announced per
# *marketing season* and the two only coincide for Kharif. A Rabi crop sown in
# crop year Y is harvested and marketed in RMS Y+1, so the column labelled
# "2021-22" for Wheat holds the RMS 2022-23 figure (Rs 2,015), not RMS 2021-22
# (Rs 1,975). Copra is announced on a calendar-year basis.
#
# Getting this wrong silently shifts every Rabi series by one year and makes
# any comparison against announced MSP look wrong by a full annual revision.
# ---------------------------------------------------------------------------
SEASON_OFFSET = {
    "Kharif Crops": 0,   # crop year Y  -> KMS Y/Y+1
    "Rabi Crops": 1,     # crop year Y  -> RMS Y+1/Y+2
    "Other Crops": 0,    # copra/jute announced on the crop year itself
}

# Crops used for the headline trend figures.
SHOWCASE = [
    ("Paddy", "Common"),
    ("Wheat", "NA"),
    ("Cotton", "Medium Staple"),
    ("Arhar(Tur)", "NA"),
]

RANDOM_STATE = 42

# Minimum number of observations a series needs before a fold is scored.
MIN_TRAIN_POINTS = 3
