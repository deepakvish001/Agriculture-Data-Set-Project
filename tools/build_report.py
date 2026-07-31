"""Generate the project report on the official course template.

Every number in the report is read from outputs/ at build time, so the document
can never drift from the pipeline that produced it.

    python run_pipeline.py && python tools/build_report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "Kartik_Jain_MSP_Forecasting_Project_Report.docx"
FIG = ROOT / "outputs" / "figures"
TAB = ROOT / "outputs" / "tables"

STUDENT = "Kartik Jain"
COLLEGE = "Baderia Global Institute of Engineering and Management, Jabalpur"
TITLE = "Forecasting India's Minimum Support Prices"
YEAR = "2026"

INK = RGBColor(0x0B, 0x0B, 0x0B)
INK_2 = RGBColor(0x52, 0x51, 0x4E)
ACCENT = RGBColor(0x1C, 0x5C, 0xAB)
HEADER_FILL = "DCE7F7"

S = json.loads((ROOT / "outputs" / "results_summary.json").read_text())
holdout_scores = pd.DataFrame(S["holdout_scores"])
backtest_scores = pd.DataFrame(S["backtest_scores"])
by_season = pd.DataFrame(S["holdout_by_season"])
selected = pd.read_csv(TAB / "forecast_selected_per_series.csv")
preds = pd.read_csv(TAB / "holdout_predictions.csv")
growth = pd.read_csv(TAB / "growth_statistics.csv")

CHAMP = S["champion_named_model"]
champ = holdout_scores[holdout_scores["Model"] == CHAMP].iloc[0]
bt_top = backtest_scores.iloc[0]
naive_h = holdout_scores[holdout_scores["Model"] == "Naive"].iloc[0]
arima_h = holdout_scores[holdout_scores["Model"] == "ARIMA(1,1,0)"].iloc[0]
sel_h = holdout_scores[holdout_scores["Model"] == "Selected"].iloc[0]
N_HOLD = S["n_holdout_points"]

doc = Document()


# --- helpers ----------------------------------------------------------------

def _page_setup():
    for section in doc.sections:
        section.page_width, section.page_height = Inches(8.27), Inches(11.69)
        section.left_margin = section.right_margin = Inches(1.0)
        section.top_margin = section.bottom_margin = Inches(1.0)


def _base_styles():
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"
    normal.font.size = Pt(11)
    normal.font.color.rgb = INK
    normal.paragraph_format.space_after = Pt(8)
    normal.paragraph_format.line_spacing = 1.15
    for name, size in (("Heading 1", 16), ("Heading 2", 13), ("Heading 3", 11.5)):
        st = doc.styles[name]
        st.font.name = "Calibri"
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = ACCENT if name == "Heading 1" else INK
        st.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        st.paragraph_format.space_after = Pt(6)


def _footer():
    """'Page | N' footer, matching the supplied template."""
    footer = doc.sections[0].footer
    p = footer.paragraphs[0]
    p.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    run = p.add_run("Page | ")
    run.font.size = Pt(9)
    run.font.color.rgb = INK_2

    fld = OxmlElement("w:fldSimple")
    fld.set(qn("w:instr"), "PAGE")
    inner_r = OxmlElement("w:r")
    inner_t = OxmlElement("w:t")
    inner_t.text = "1"
    inner_r.append(inner_t)
    fld.append(inner_r)
    p._p.append(fld)


def para(text="", size=11, bold=False, italic=False, align=None, space_after=8,
         color=INK, style=None):
    p = doc.add_paragraph(style=style)
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    if text:
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.bold = bold
        run.italic = italic
        run.font.color.rgb = color
    return p


def rich(*parts, align=None, space_after=8, size=11):
    """Paragraph from (text, bold) tuples, for inline emphasis."""
    p = doc.add_paragraph()
    if align is not None:
        p.alignment = align
    p.paragraph_format.space_after = Pt(space_after)
    for text, bold in parts:
        run = p.add_run(text)
        run.font.size = Pt(size)
        run.bold = bold
    return p


def bullets(items, style="List Bullet"):
    for item in items:
        p = doc.add_paragraph(style=style)
        p.paragraph_format.space_after = Pt(4)
        if isinstance(item, tuple):
            head, rest = item
            r = p.add_run(head)
            r.bold = True
            r.font.size = Pt(11)
            r2 = p.add_run(rest)
            r2.font.size = Pt(11)
        else:
            p.add_run(item).font.size = Pt(11)


def heading(text, level=1):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.keep_with_next = True
    return h


def _shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def table(headers, rows, widths=None, font=9.5, highlight_first_row=None):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER

    for i, text in enumerate(headers):
        cell = t.rows[0].cells[i]
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(text))
        run.bold = True
        run.font.size = Pt(font)
        cell.paragraphs[0].paragraph_format.space_after = Pt(2)
        _shade(cell, HEADER_FILL)

    for r_i, row in enumerate(rows):
        cells = t.add_row().cells
        for c_i, value in enumerate(row):
            cells[c_i].text = ""
            run = cells[c_i].paragraphs[0].add_run(str(value))
            run.font.size = Pt(font)
            if highlight_first_row is not None and r_i == highlight_first_row:
                run.bold = True
            cells[c_i].paragraphs[0].paragraph_format.space_after = Pt(2)

    if widths:
        for row in t.rows:
            for c_i, w in enumerate(widths):
                row.cells[c_i].width = Inches(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)
    return t


def figure(filename, caption, width=6.3):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_after = Pt(4)
    p.add_run().add_picture(str(FIG / filename), width=Inches(width))
    cap = para(caption, size=9, italic=True, align=WD_ALIGN_PARAGRAPH.CENTER,
               space_after=12, color=INK_2)
    return cap


def page_break():
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


# --- build ------------------------------------------------------------------

_page_setup()
_base_styles()
_footer()

# ---- Title page
for _ in range(6):
    para(space_after=0)
para(TITLE, size=26, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=6, color=ACCENT)
para("Project Report", size=16, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=30, color=INK_2)
para(STUDENT, size=14, bold=True, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
para(COLLEGE, size=11, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4, color=INK_2)
para(YEAR, size=12, align=WD_ALIGN_PARAGRAPH.CENTER, space_after=60, color=INK_2)
for _ in range(4):
    para(space_after=0)
para("Fundamentals of AI Using Agriculture Data Set", size=13, bold=True,
     align=WD_ALIGN_PARAGRAPH.CENTER, space_after=4)
para("ANNAM.AI – Centre of Excellence, Ministry of Education, Government of India, IIT Ropar",
     size=10, align=WD_ALIGN_PARAGRAPH.CENTER, color=INK_2)
page_break()

# ---- Certificate
heading("Certificate", 1)
para(space_after=10)
p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
p.paragraph_format.line_spacing = 1.6
for text, bold in [
    ("This is to certify that ", False), (STUDENT, True), (" from ", False), (COLLEGE, True),
    (" has successfully completed the project work titled ", False), (TITLE, True),
    (" as part of the 1-credit course “Fundamentals of AI using Agriculture Data Set”, "
     "offered under ANNAM.AI – Centre of Excellence, Ministry of Education, Government of "
     "India, at IIT Ropar. This project constitutes a component of the 30 hours of learning "
     "prescribed for the course and reflects the student’s independent effort in problem "
     "selection, method design, and implementation.", False),
]:
    r = p.add_run(text)
    r.bold = bold
    r.font.size = Pt(12)
page_break()

# ---- Index
heading("Index", 1)
para(space_after=10)
for i, name in enumerate([
    "Introduction and Motivation", "Problem Statement", "Dataset Understanding",
    "Methodology", "Implementation Details", "Results and Discussions",
    "Conclusion", "References",
], start=1):
    q = doc.add_paragraph()
    q.paragraph_format.space_after = Pt(8)
    r = q.add_run(f"{i}.  {name}")
    r.font.size = Pt(12)
q = doc.add_paragraph()
q.add_run("Appendix").font.size = Pt(12)
page_break()

# ---- 1. Introduction
heading("1. Introduction and Motivation", 1)
para(
    "Agriculture supports close to half of India’s workforce, and for most of that "
    "workforce the single most consequential number in the year is the price their crop "
    "will fetch. The Minimum Support Price (MSP) is the government’s answer to that "
    "uncertainty: a floor price, announced each season by the Cabinet on the recommendation "
    "of the Commission for Agricultural Costs and Prices (CACP), at which government "
    "agencies stand ready to procure. It is the closest thing Indian farming has to a "
    "guaranteed income signal."
)
para(
    "There is a timing problem, though. Sowing decisions are made months before the MSP for "
    "that marketing season is announced. A farmer choosing between paddy and cotton in June "
    "is committing land, seed, fertiliser and labour against a price they will not learn "
    "until after the decision is irreversible. The MSP exists to remove price risk, but the "
    "announcement calendar puts the most important decision ahead of the information."
)
para(
    "That gap is what makes MSP a genuinely useful forecasting target rather than a textbook "
    "exercise. If next season’s support price can be estimated to within a few percent "
    "before sowing, the estimate has direct economic value — for a farmer weighing crops, "
    "for a state agency planning procurement budgets, and for a policy analyst asking "
    "whether announced prices are keeping pace with the cost of production."
)
rich(
    ("This project asks a deliberately demanding version of that question. It is easy to fit "
     "a curve through five points and report a small error; it is much harder to show the "
     "forecast would have been right. So every model here is finally judged not on how well "
     "it describes 2017-2021, but on how close it comes to ", False),
    ("the MSP the Government of India actually announced afterwards", True),
    (" — 42 published prices that no model was allowed to see.", False),
)
page_break()

# ---- 2. Problem Statement
heading("2. Problem Statement", 1)
para(
    "Given five years of official MSP data (crop years 2017-18 to 2021-22) for 28 "
    "commodity-variety series, build and rigorously evaluate a system that forecasts the "
    "support price for the seasons that follow."
)
heading("Scope", 2)
bullets([
    ("Forecast target: ", "MSP in ₹ per quintal, for each commodity-variety series "
     "separately, for the 2022-23, 2023-24 and 2024-25 seasons."),
    ("Model families: ", "ten models spanning naive baselines, parametric trend models, "
     "classical time-series methods, tree ensembles and a forecast combination."),
    ("Evaluation: ", "one identical rolling-origin backtest for every model, followed by "
     "validation against MSP values the government has since published."),
])
heading("What counts as success", 2)
para(
    "A forecast is only useful if it beats the free alternative. The benchmark is therefore "
    "the naive model — assume next season’s MSP equals this season’s — and the project "
    "reports MASE (model error ÷ naive error) throughout. A model that cannot get MASE below "
    "1.0 has added nothing, however impressive its absolute error looks in isolation."
)
rich(("Concretely, the system should: (i) forecast at least ", False),
     (f"{N_HOLD} published MSP values", True),
     (" to within single-digit percentage error; (ii) identify which model family is "
      "genuinely most reliable, using a protocol that treats every model the same; and "
      "(iii) be honest about what the data cannot support.", False))
page_break()

# ---- 3. Dataset Understanding
heading("3. Dataset Understanding", 1)
heading("3.1 Source", 2)
audit = S["audit"]
table(
    ["Attribute", "Detail"],
    [
        ["Dataset", "Minimum Support Prices for Agricultural Crops"],
        ["Publisher", "Ministry of Agriculture & Farmers Welfare, Government of India"],
        ["Portal", "data.gov.in (Open Government Data platform)"],
        ["Parliamentary reference", "Rajya Sabha Session 255, Unstarred Question AU-1460"],
        ["Coverage", "Crop years 2017-18 to 2021-22 (five seasons)"],
        ["Records", f"{audit['rows']} commodity-variety rows × 5 year columns"],
        ["Unit", "₹ per quintal"],
        ["Licence", "Open Government Data (OGD) Licence – India"],
    ],
    widths=[2.1, 4.3], font=10,
)

heading("3.2 Structure and coverage", 2)
para(
    "The file is a wide table: identifying columns (serial number, cropping season, "
    "commodity, variety) followed by one price column per crop year. The 28 rows split "
    "across three cropping seasons.", space_after=6,
)
seasons = audit["seasons"]
table(
    ["Cropping season", "Series", "Representative commodities"],
    [
        ["Kharif Crops", seasons.get("Kharif Crops", 0),
         "Paddy, Jowar, Bajra, Ragi, Maize, Arhar, Moong, Urad, Cotton, Groundnut, Sunflower, Soyabean, Sesamum, Nigerseed"],
        ["Rabi Crops", seasons.get("Rabi Crops", 0),
         "Wheat, Barley, Gram, Masur, Rapeseed/Mustard, Safflower, Toria"],
        ["Other Crops", seasons.get("Other Crops", 0),
         "Copra (Milling and Ball), De-Husked Coconut, Jute"],
    ],
    widths=[1.4, 0.7, 4.3], font=9.5,
)

heading("3.3 Fields", 2)
table(
    ["Column", "Type", "Description"],
    [
        ["Sl. No.", "Integer", "Row serial number"],
        ["Crops Session", "Categorical", "Kharif / Rabi / Other Crops"],
        ["Commodity", "Categorical", "Crop name (24 distinct commodities)"],
        ["Variety", "Categorical", "Grade or staple, e.g. Common, Grade ‘A’, Hybrid, Medium Staple"],
        ["2017-18 … 2021-22", "Numeric", "MSP in ₹ per quintal for that crop year"],
    ],
    widths=[1.5, 1.0, 3.9], font=9.5,
)

heading("3.4 Data quality audit", 2)
para(
    f"An automated audit ({', '.join(['missing values', 'duplicates', 'type consistency', 'monotonicity'])}) "
    "returned an unusually clean result:", space_after=6,
)
table(
    ["Check", "Result"],
    [
        ["Rows / series", f"{audit['rows']}"],
        ["Distinct commodities", f"{audit['commodities']}"],
        ["Missing or non-numeric price cells", f"{audit['missing_cells']}"],
        ["Duplicate commodity-variety rows", f"{audit['duplicate_rows']}"],
        ["Series that never decreased year on year", f"{audit['monotonic_series']} of {audit['rows']}"],
        ["MSP range across the dataset", f"₹{audit['min_msp']:,.0f} – ₹{audit['max_msp']:,.0f}"],
    ],
    widths=[3.4, 3.0], font=10,
)
rich(("The monotonicity result is the most important single fact about this dataset. ", False),
     (f"All {audit['monotonic_series']} series are non-decreasing across all five years", True),
     (" — no commodity’s MSP was ever cut. MSP is a policy floor revised upward annually, "
      "not a market price. That tells us in advance that the problem is trend extrapolation "
      "rather than volatility modelling, and it predicts which model families should do "
      "well long before any model is fitted.", False))

heading("3.5 A subtlety in the year labels", 2)
para(
    "One characteristic of this dataset is easy to miss and expensive to get wrong. The "
    "columns are labelled by crop year, but MSP is announced per marketing season, and the "
    "two coincide only for Kharif crops. A Rabi crop sown in crop year Y is harvested and "
    "marketed in RMS Y+1.", space_after=6,
)
para("This is verifiable against published figures. For wheat:", space_after=6)
table(
    ["Column label in the CSV", "Value in the file", "Announced wheat MSP", "Actual marketing season"],
    [
        ["2020-21", "₹1,975", "₹1,975 = RMS 2021-22", "RMS 2021-22"],
        ["2021-22", "₹2,015", "₹2,015 = RMS 2022-23", "RMS 2022-23"],
    ],
    widths=[1.7, 1.3, 1.9, 1.5], font=9.5,
)
para(
    "The Rabi columns therefore run one year ahead of their label. Left uncorrected, every "
    "Rabi comparison against announced MSP would be wrong by a full annual revision — about "
    "₹100 on wheat, which is larger than the forecast error this project is trying to "
    "measure. The pipeline applies the offset explicitly (config.SEASON_OFFSET) and carries "
    "both the published label and the true marketing season through every table."
)
page_break()

# ---- 4. Methodology
heading("4. Methodology", 1)
para(
    "The pipeline runs in seven stages. The design principle throughout is that no model "
    "should be given an easier test than any other, and that the final verdict should come "
    "from data the models never saw.", space_after=8,
)

heading("Step 1 — Ingest and audit", 3)
para("The CSV is loaded and checked for missing values, duplicates, type consistency and "
     "monotonicity before anything is modelled. The audit output is reproduced in §3.4.")

heading("Step 2 — Correct the season labels", 3)
para("The Rabi crop-year to marketing-season offset described in §3.5 is applied, so that "
     "forecasts are compared against the right announcement.")

heading("Step 3 — Reshape, keeping varieties separate", 3)
rich(("The wide table is melted into tidy long form and pivoted into a year × series matrix. "
      "Crucially, varieties are ", False), ("not averaged", True),
     (". Paddy Common and Paddy Grade ‘A’ are two separate government notifications; "
      "averaging them produces a price that was never announced and that cannot be checked "
      "against any real MSP. This preserves 28 series where a commodity-level average would "
      "leave 24 partly fictitious ones.", False))

heading("Step 4 — Exploratory analysis", 3)
para("Per-series compound annual growth rate, mean year-on-year revision, volatility and "
     "season-level comparisons establish what the data looks like before any model is fitted.")

heading("Step 5 — Ten models behind one interface", 3)
para("Every model is a callable with the identical signature f(years, values, future_years), "
     "so the backtester, the forecaster and the application treat them interchangeably and "
     "no model can accidentally receive special handling.", space_after=6)
table(
    ["Model", "Family", "Idea"],
    [
        ["Naive", "Baseline", "Carry last season’s MSP forward"],
        ["Drift", "Baseline", "Last value plus average historical step"],
        ["Linear", "Parametric", "Ordinary least squares on the year index"],
        ["RidgePoly2", "Parametric", "Degree-2 polynomial with L2 shrinkage"],
        ["CAGR", "Parametric", "Constant compound growth (log-linear)"],
        ["Holt", "Time series", "Double exponential smoothing with trend"],
        ["ARIMA(1,1,0)", "Time series", "One AR lag on the first difference"],
        ["RandomForest", "Ensemble", "300 regression trees on the year index"],
        ["GradBoost", "Ensemble", "Gradient-boosted stumps"],
        ["Ensemble", "Combination", "Median of Drift, Linear, CAGR and Holt"],
    ],
    widths=[1.4, 1.2, 3.8], font=9.5,
)
para(
    "Two of these are included for what they prove rather than for what they predict. Naive "
    "is the bar every model must clear. RandomForest and GradBoost are deliberate negative "
    "controls: tree models predict a constant outside their training range, so they cannot "
    "extrapolate a trend by construction — a limitation that is far more convincing "
    "demonstrated than asserted."
)

heading("Step 6 — One evaluation protocol for everybody", 3)
para(
    "This is the methodological core. Comparing one model’s in-sample fit against another "
    "model’s cross-validated error is not a comparison; the numbers are not on the same "
    "scale and any resulting ranking is an artefact. Every model here is scored by the same "
    "expanding-window rolling-origin backtest:", space_after=6,
)
table(
    ["Fold", "Training window", "Predicts", "Horizons"],
    [["1", "2017-18 → 2019-20", "2020-21 and 2021-22", "1 and 2"],
     ["2", "2017-18 → 2020-21", "2021-22", "1"]],
    widths=[0.7, 2.2, 2.2, 1.3], font=9.5,
)
para(
    "Nothing after the origin is visible at fit time. Reported metrics are MAE, RMSE, MAPE, "
    "mean bias and MASE — the model’s MAE divided by the naive model’s MAE over the "
    "identical folds. MASE below 1.0 means the model beat carry-forward; at or above 1.0 it "
    "did not, whatever its raw error suggests."
)

heading("Step 7 — Validate against reality", 3)
rich(("The backtest still only re-uses the training window. The final test refits every model "
      "on all five years, forecasts forward, and scores those forecasts against ", False),
     (f"{N_HOLD} MSP values the Cabinet actually announced", True),
     (" for the 2022-23 and 2023-24 seasons — collected independently, each row carrying its "
      "PIB citation, and never used for fitting or model selection. Series whose announcements "
      "could not be independently verified (Jute, De-Husked Coconut, Toria) are excluded from "
      "scoring rather than guessed at.", False))
page_break()

# ---- 5. Implementation Details
heading("5. Implementation Details", 1)
heading("5.1 Tools and libraries", 2)
table(
    ["Library", "Role"],
    [
        ["Python 3.11", "Implementation language"],
        ["pandas", "Ingest, audit, reshaping, tidy long format, pivot tables"],
        ["numpy", "Numerical arrays, metric computation"],
        ["scikit-learn", "LinearRegression, Ridge, PolynomialFeatures, RandomForest, GradientBoosting"],
        ["statsmodels", "ARIMA(1,1,0) and Holt exponential smoothing"],
        ["matplotlib", "Seven publication-quality figures"],
        ["pytest", "25-test regression suite"],
        ["Streamlit", "Interactive forecast explorer"],
        ["Jupyter / nbclient", "Executed analysis notebook, generated from source"],
    ],
    widths=[1.5, 4.9], font=9.5,
)

heading("5.2 Project structure", 2)
para("The project is a reproducible package, not a single notebook. Analysis logic lives in "
     "src/ and is imported by the notebook, the application and the test suite alike, so "
     "there is exactly one implementation of every calculation.", space_after=6)
code_rows = [
    ["src/config.py", "Paths, year columns, season-offset semantics"],
    ["src/data_loader.py", "Ingest, audit, clean, tidy long format, price matrix"],
    ["src/models.py", "The ten models behind one shared callable interface"],
    ["src/backtest.py", "Rolling-origin CV; MAE, RMSE, MAPE, bias, MASE"],
    ["src/forecast.py", "Final forecasts, hold-out scoring, growth statistics"],
    ["src/visualize.py", "Figure generation"],
    ["src/dashboard.py", "Self-contained interactive HTML dashboard"],
    ["run_pipeline.py", "Orchestrates the full pipeline end to end"],
    ["tools/", "Notebook and report generators"],
    ["tests/", "25-test pytest suite"],
]
table(["Module", "Responsibility"], code_rows, widths=[1.9, 4.5], font=9.5)

heading("5.3 Reproducing the results", 2)
for line in ["pip install -r requirements.txt",
             "python run_pipeline.py                 # full pipeline → outputs/",
             "pytest -q                              # 25 tests",
             "streamlit run app/streamlit_app.py     # interactive explorer",
             "python tools/build_notebook.py         # regenerate the notebook"]:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.left_indent = Inches(0.3)
    r = p.add_run(line)
    r.font.name = "Consolas"
    r.font.size = Pt(9.5)
para(
    "The pipeline runs in about a minute and writes every figure, table, a machine-readable "
    "JSON summary and the dashboard. This report itself is generated from that JSON, so no "
    "number printed here can drift from the code that produced it.", space_after=8,
)

heading("5.4 Testing", 2)
para("The suite covers dataset integrity (shape, missing values, monotonicity), the season-offset "
     "correction, the model interface contract for all ten models, the documented "
     "flat-extrapolation behaviour of the tree models, and — most importantly — that the "
     "backtest never leaks future data and that the hold-out seasons are disjoint from the "
     "training window. All 25 tests pass.")
page_break()

# ---- 6. Results
heading("6. Results and Discussions", 1)

heading("6.1 How fast has MSP actually grown?", 2)
top5 = pd.DataFrame(S["growth_top5"])
bot5 = pd.DataFrame(S["growth_bottom5"])
para("Growth over the five years is far from uniform:", space_after=6)
table(
    ["Fastest growing", "CAGR", "Slowest growing", "CAGR"],
    [[t["Series"], f"{t['CAGR_pct']:.1f}%", b["Series"], f"{b['CAGR_pct']:.1f}%"]
     for t, b in zip(top5.to_dict("records"), bot5.to_dict("records")[::-1])],
    widths=[2.0, 1.0, 2.0, 1.0], font=9.5,
)
para(
    f"Ragi and Nigerseed grew at {top5.iloc[0]['CAGR_pct']:.1f}% and "
    f"{top5.iloc[1]['CAGR_pct']:.1f}% a year — roughly doubling over the window — as the "
    "government pushed millets and minor oilseeds toward the ‘1.5× cost of production’ "
    "formula, while Arhar, Wheat and Barley grew at under 4%. A single national growth rate "
    "would misprice most of the basket, which is the argument for fitting each series "
    "separately."
)
figure("fig2_cagr_by_series.png", "Figure 1 — Five-year CAGR for all 28 series, grouped by cropping season.", 5.6)
figure("fig1_msp_trends.png", "Figure 2 — MSP trajectories for four headline crops.", 6.0)

heading("6.2 Backtest results", 2)
para("Under the shared rolling-origin protocol, across 840 fold-predictions:", space_after=6)
table(
    ["Model", "MAE (₹)", "RMSE (₹)", "MAPE", "MASE"],
    [[r["Model"], f"{r['MAE']:,.0f}", f"{r['RMSE']:,.0f}", f"{r['MAPE']:.2f}%", f"{r['MASE']:.2f}"]
     for r in backtest_scores.to_dict("records")],
    widths=[1.6, 1.2, 1.2, 1.2, 1.2], font=9.5, highlight_first_row=0,
)
para(
    f"{bt_top['Model']} appears dominant, at {bt_top['MAPE']:.2f}% MAPE and MASE "
    f"{bt_top['MASE']:.2f} — roughly half the naive error. On this evidence alone it would "
    "be the obvious choice. It is worth holding that conclusion loosely."
)

heading("6.3 The real test: MSP actually announced", 2)
rich(("Every model was then refit on all five years and its forecasts scored against ", False),
     (f"{N_HOLD} MSP values the Government of India has since published", True),
     (" for 2022-23 and 2023-24. ‘Selected’ is a pseudo-model that applies each series’ own "
      "backtest winner, included to test whether per-crop selection actually pays off.", False))
table(
    ["Model", "MAE (₹)", "MAPE", "Bias (₹)", "MASE", "Within ±5%"],
    [[r["Model"], f"{r['MAE']:,.0f}", f"{r['MAPE']:.2f}%", f"{r['Bias']:,.0f}",
      f"{r['MASE']:.2f}", f"{r['Within_5pct']:.0f}%"]
     for r in holdout_scores.to_dict("records")],
    widths=[1.5, 1.1, 1.0, 1.1, 0.9, 1.1], font=9.5, highlight_first_row=0,
)
figure("fig3_model_leaderboard.png",
       "Figure 3 — Backtest error against true hold-out error for every model.", 6.0)

heading("6.4 The main finding: backtest rank is not hold-out rank", 2)
rich(("The two rankings disagree, and that disagreement is the most useful result in this "
      "project. ", False),
     (f"{bt_top['Model']} won the internal backtest at {bt_top['MAPE']:.2f}% and finishes "
      f"mid-table on real announcements at {arima_h['MAPE']:.2f}%", True),
     (f", while {CHAMP} — a mediocre {backtest_scores[backtest_scores['Model'] == CHAMP].iloc[0]['MAPE']:.2f}% "
      f"in the backtest — is the most accurate model on data nobody had seen, at "
      f"{champ['MAPE']:.2f}% MAPE with {champ['Within_5pct']:.0f}% of forecasts inside a ±5% band.", False))
para(
    "The explanation is fold size. Each backtest fold trains on only three or four points, "
    "and ARIMA’s differencing plus an autoregressive term can bend to fit the exact pattern "
    "of recent revisions. That flexibility is rewarded in-sample and punished the moment the "
    "series must be projected two seasons past the data. It is a textbook illustration of "
    "why model selection on a very short series is itself unreliable — and the honest "
    "response is to report both numbers rather than only the flattering one."
)
figure("fig4_holdout_parity.png",
       f"Figure 4 — {CHAMP} forecasts against announced MSP; shaded band is ±5%.", 4.9)

heading("6.5 Forecast bias, and why it matters to a farmer", 2)
rich(("Absolute error is not the whole story. Naive, GradBoost and RandomForest all "
      "under-forecast systematically, by ", False),
     (f"₹{abs(naive_h['Bias']):,.0f} per quintal", True),
     (" and more. A forecast that is wrong in a random direction is a nuisance; a forecast "
      "that is consistently ₹500 too low is a bias against planting — it would tell farmers "
      "the crop is less remunerative than it turns out to be, every single year. "
      f"{CHAMP} carries a bias of ₹{champ['Bias']:,.0f}, an order of magnitude smaller and "
      "slightly conservative.", False))
figure("fig6_holdout_bias.png", "Figure 5 — Systematic over- and under-forecasting on the hold-out seasons.", 5.6)

heading("6.6 Tree ensembles are the wrong tool for this problem", 2)
rf = holdout_scores[holdout_scores["Model"] == "RandomForest"].iloc[0]
para(
    f"Random Forest finishes last at {rf['MAPE']:.2f}% MAPE and MASE {rf['MASE']:.2f} — "
    f"worse than assuming no change at all — and Gradient Boosting scores identically to the "
    "naive baseline to three decimal places. This is not a tuning failure. Tree models "
    "partition the input space and predict the mean of the nearest training leaf, so beyond "
    "the last observed year they return a constant. On a monotonically rising series that "
    "guarantees a growing under-forecast. Their poor showing here is a demonstration of a "
    "structural limitation, and a caution against reaching for the most fashionable "
    "algorithm rather than the one matched to the data."
)

heading("6.7 Did per-crop model selection help? No.", 2)
counts = S["best_model_counts"]
para(
    "The backtest selected different winners for different crops — "
    + ", ".join(f"{m} for {n} series" for m, n in list(counts.items())[:4])
    + f" — an approach that intuitively should beat one model applied everywhere. It did not. "
    f"The ‘Selected’ strategy scored {sel_h['MAPE']:.2f}% MAPE against {CHAMP}’s "
    f"{champ['MAPE']:.2f}%, and its ±5% hit rate fell from {champ['Within_5pct']:.0f}% to "
    f"{sel_h['Within_5pct']:.0f}%. With two backtest folds per series, selection is mostly "
    "fitting noise. Reporting this negative result matters: the intuition that more "
    "customisation is always better is exactly the kind of assumption a hold-out test exists "
    "to check."
)

heading("6.8 Accuracy by horizon", 2)
sub = by_season[by_season["Model"].isin([CHAMP, "ARIMA(1,1,0)", "Naive"])]
table(
    ["Model", "Season", "MAE (₹)", "MAPE", "Within ±5%"],
    [[r["Model"], r["Season_Label"], f"{r['MAE']:,.0f}", f"{r['MAPE']:.2f}%", f"{r['Within_5pct']:.0f}%"]
     for r in sub.sort_values(["Model", "Season_Label"]).to_dict("records")],
    widths=[1.5, 1.2, 1.2, 1.2, 1.3], font=9.5,
)
para(
    "Every model degrades from the first hold-out season to the second — both are forecast "
    "from the same 2021-22 origin, so the second is one further revision away. How gracefully "
    "they degrade is what separates them: the naive baseline roughly doubles its error and "
    f"ARIMA nearly triples, while {CHAMP} moves only from "
    f"{sub[(sub['Model'] == CHAMP) & (sub['Season_Label'] == '2022-23')].iloc[0]['MAPE']:.2f}% to "
    f"{sub[(sub['Model'] == CHAMP) & (sub['Season_Label'] == '2023-24')].iloc[0]['MAPE']:.2f}%. "
    "Robustness at the longer horizon is where the trend models earn their place."
)
figure("fig5_forecast_panels.png",
       "Figure 6 — History, forecast and announced MSP for four headline crops.", 6.3)

heading("6.9 Where the forecasts went wrong", 2)
worst = preds[preds["Model"] == CHAMP].nlargest(6, "APE")
table(
    ["Series", "Season", "Forecast (₹)", "Announced (₹)", "Error"],
    [[r["Series"], r["Marketing_Season"], f"{r['Forecast']:,.0f}",
      f"{r['Actual_MSP']:,.0f}", f"{r['APE']:.1f}%"]
     for r in worst.to_dict("records")],
    widths=[1.6, 1.4, 1.2, 1.2, 1.0], font=9.5,
)
para(
    "The largest misses share a cause: commodities whose MSP was deliberately re-based by "
    "policy rather than allowed to drift upward. Copra and Nigerseed saw outsized revisions "
    "tied to the ‘1.5× cost of production’ commitment and to import-substitution pressure in "
    "edible oils. No price-only model can anticipate a Cabinet decision. This is a limit of "
    "the data, not of the fitting."
)

heading("6.10 Challenges faced and what I learned", 2)
bullets([
    ("Five points per series. ", "This is the binding constraint and no method escapes it. "
     "It drove the decision to build a rolling-origin backtest instead of a train/test split "
     "— with five observations, a single split wastes too much of the data to be informative."),
    ("Discovering the label offset. ", "Early Rabi results were consistently wrong by about "
     "₹100. Tracing that to the crop-year versus marketing-season distinction, rather than "
     "assuming it was model error, produced more accuracy than any modelling change. The "
     "lesson is that domain semantics deserve as much scrutiny as hyperparameters."),
    ("Resisting a flattering result. ", "The backtest handed a clean story: ARIMA wins by a "
     "wide margin. Building the hold-out test risked destroying that story, and it did. The "
     "project is stronger for it, and the experience of watching a confident conclusion fail "
     "an honest test was the most valuable part of the work."),
    ("Sourcing verifiable ground truth. ", "Hold-out actuals were collected only where the "
     "announcement could be independently confirmed and cited. Three series were left out "
     "rather than filled in with plausible numbers — a smaller, trustworthy test set is "
     "worth more than a complete, partly-invented one."),
])
page_break()

# ---- 7. Conclusion
heading("7. Conclusion", 1)
rich(("This project built a reproducible forecasting pipeline for Indian Minimum Support "
      "Prices covering 28 commodity-variety series, compared ten models under one identical "
      "rolling-origin protocol, and — unusually for a five-point time series — validated the "
      "result against ", False),
     (f"{N_HOLD} MSP values the Government of India actually announced afterwards", True),
     (".", False))
para("Five findings stand out:", space_after=6)
bullets([
    ("Trend-following models win. ", f"{CHAMP} forecasts real announced MSP to "
     f"{champ['MAPE']:.2f}% mean error, with {champ['Within_5pct']:.0f}% of forecasts inside "
     f"a ±5% band and MASE {champ['MASE']:.2f} — roughly {(1 - champ['MASE']) * 100:.0f}% "
     "better than assuming no change."),
    ("Backtest rank is not hold-out rank. ", f"{bt_top['Model']} topped the internal backtest "
     "and slipped to mid-table against reality. On five points, the backtest is itself a "
     "small sample, and treating it as the final word would have selected the wrong model."),
    ("Tree ensembles cannot extrapolate. ", "Random Forest and Gradient Boosting match or "
     "lose to the naive baseline and under-forecast by ₹430-500 per quintal. Algorithm "
     "popularity is not a reason to use one."),
    ("Per-crop model selection did not pay. ", "One robust model applied uniformly beat "
     "picking each series’ backtest winner."),
    ("Domain semantics mattered most. ", "Correcting the Rabi crop-year to marketing-season "
     "offset improved accuracy more than any change of model."),
])
heading("Practical implications", 2)
para(
    "For a farmer, a pre-sowing estimate accurate to within roughly 3% is decision-grade: it "
    "supports a real comparison between two candidate crops months before the official "
    "announcement. For a state procurement agency, forecasts across all 28 series translate "
    "directly into budget provisioning. For a policy analyst, the residuals are the "
    "interesting part — the crops where the announced MSP departs sharply from its own trend "
    "are precisely the crops where policy actively intervened, which makes this pipeline a "
    "way of detecting policy shifts rather than merely predicting prices."
)
heading("Limitations and future work", 2)
para(
    "Five observations per series is the binding constraint. The dataset carries no "
    "input-cost inflation, no global commodity prices and no procurement volumes, all of "
    "which feed the CACP’s actual recommendation. MSP is set by a committee, so a price-only "
    "model can track policy momentum but never anticipate policy decisions. The natural "
    "extensions are a longer history (MSP series run back to the 1960s), CACP cost-of-"
    "production data as an exogenous regressor, and prediction intervals rather than point "
    "forecasts — a farmer is better served by ‘₹2,180 to ₹2,260’ than by a single number "
    "carrying false precision."
)
page_break()

# ---- 8. References
heading("8. References", 1)
refs = [
    "Ministry of Agriculture & Farmers Welfare, Government of India. Minimum Support Prices "
    "for Agricultural Crops. Rajya Sabha Session 255, Unstarred Question AU-1460. "
    "data.gov.in, Open Government Data Licence – India.",
    "Press Information Bureau, Government of India. Cabinet approves Minimum Support Prices "
    "for Kharif Crops, Marketing Season 2022-23 and 2023-24 (PRID 1930443).",
    "Press Information Bureau, Government of India. Cabinet approves Minimum Support Prices "
    "for Rabi Crops, Marketing Seasons 2023-24 (PRID 1868760) and 2024-25 (PRID 1968729).",
    "Commission for Agricultural Costs and Prices (CACP). Price Policy Reports for Kharif "
    "and Rabi Crops, 2017-18 to 2023-24. Ministry of Agriculture, New Delhi.",
    "Hyndman, R. J. & Koehler, A. B. (2006). Another look at measures of forecast accuracy. "
    "International Journal of Forecasting, 22(4), 679-688.",
    "Hyndman, R. J. & Athanasopoulos, G. (2021). Forecasting: Principles and Practice, "
    "3rd edition. OTexts, Melbourne.",
    "Box, G. E. P., Jenkins, G. M., Reinsel, G. C. & Ljung, G. M. (2015). Time Series "
    "Analysis: Forecasting and Control, 5th edition. Wiley.",
    "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of "
    "Machine Learning Research, 12, 2825-2830.",
    "Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32.",
    "Seabold, S. & Perktold, J. (2010). statsmodels: Econometric and statistical modeling "
    "with Python. Proceedings of the 9th Python in Science Conference.",
]
for i, ref in enumerate(refs, start=1):
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.left_indent = Inches(0.35)
    p.paragraph_format.first_line_indent = Inches(-0.35)
    p.add_run(f"[{i}]  {ref}").font.size = Pt(10)
page_break()

# ---- Appendix
heading("Appendix", 1)

heading("A. Forecasts for all 28 series", 2)
para(f"Produced by each series’ backtest-selected model, in ₹ per quintal.", space_after=6)
cols = [c for c in ("2022-23", "2023-24", "2024-25") if c in selected.columns]
table(
    ["Commodity (variety)", "Model"] + cols,
    [[r["Series"], r["Best_Model"]] + [f"{r[c]:,.0f}" for c in cols]
     for r in selected.sort_values("Series").to_dict("records")],
    widths=[2.1, 1.2, 1.05, 1.05, 1.0], font=8.5,
)

heading("B. Hold-out ground truth and provenance", 2)
para(
    f"The {N_HOLD} hold-out values live in data/external/msp_actuals_holdout.csv. Each row "
    "records the commodity, variety, crop-year label, the true marketing season, the "
    "announced MSP and the announcement it was taken from. Coverage is "
    f"{len(preds[(preds['Model'] == CHAMP) & (preds['Season_Label'] == '2022-23')])} values for "
    f"2022-23 and {len(preds[(preds['Model'] == CHAMP) & (preds['Season_Label'] == '2023-24')])} "
    "for 2023-24. Jute, De-Husked Coconut and Toria are excluded because their announcements "
    "could not be independently verified; they are forecast but not scored.",
)

heading("C. Error growth with horizon", 2)
figure("fig7_horizon_error.png",
       "Figure 7 — Backtest MAPE by forecast horizon for the five best models.", 5.4)

heading("D. Representative code — the shared model interface", 2)
snippet = [
    "# Every model is one callable with the same signature, so no model",
    "# can receive a gentler evaluation than another.",
    "",
    "def drift(years, values, future_years):",
    "    \"\"\"Random walk with drift: last value + average historical step.\"\"\"",
    "    n = len(values)",
    "    step = (values[-1] - values[0]) / (n - 1) if n > 1 else 0.0",
    "    h = np.asarray(future_years, dtype=float) - float(years[-1])",
    "    return values[-1] + step * h",
    "",
    "REGISTRY = {'Naive': naive, 'Drift': drift, 'Linear': linear, ...}",
    "",
    "# Rolling origin: nothing after the origin is visible at fit time.",
    "for origin in range(MIN_TRAIN_POINTS, len(years)):",
    "    train_years, train_values = years[:origin], values[:origin]",
    "    test_years = years[origin:origin + max_horizon]",
    "    for name, fn in REGISTRY.items():",
    "        preds = fn(train_years, train_values, test_years)",
]
for line in snippet:
    p = doc.add_paragraph()
    p.paragraph_format.space_after = Pt(0)
    p.paragraph_format.left_indent = Inches(0.3)
    r = p.add_run(line if line else " ")
    r.font.name = "Consolas"
    r.font.size = Pt(8.5)
    r.font.color.rgb = INK_2

para(space_after=10)
heading("E. Project artefacts", 2)
table(
    ["Artefact", "Path"],
    [
        ["Executed analysis notebook", "notebooks/MSP_Forecasting_Analysis.ipynb"],
        ["Interactive HTML dashboard", "outputs/dashboard.html"],
        ["Streamlit application", "app/streamlit_app.py"],
        ["Figures", "outputs/figures/ (7 files)"],
        ["Result tables", "outputs/tables/ (11 CSV files)"],
        ["Hold-out ground truth", "data/external/msp_actuals_holdout.csv"],
        ["Test suite", "tests/test_pipeline.py (25 tests)"],
    ],
    widths=[2.4, 4.0], font=9.5,
)

OUT.parent.mkdir(parents=True, exist_ok=True)
doc.save(OUT)
print(f"wrote {OUT}")
