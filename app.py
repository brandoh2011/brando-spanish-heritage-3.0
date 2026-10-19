from __future__ import annotations
import ast, io, math, re
from datetime import date, datetime
from typing import Optional

import numpy as np
import pandas as pd
import requests
import streamlit as st
import matplotlib.pyplot as plt
import sympy as sp
from sklearn.metrics import mean_absolute_error, r2_score
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

st.set_page_config(page_title="Hispanic Heritage Historical Data Explorer", page_icon="📈", layout="wide")

COUNTRIES = {
    "Mexico (MX)":"MEX", "Brazil (BR)":"BRA", "Argentina (AR)":"ARG", "Colombia (CO)":"COL",
    "Chile (CL)":"CHL", "Peru (PE)":"PER", "Venezuela (VE)":"VEN", "Ecuador (EC)":"ECU", "Uruguay (UY)":"URY"
}
INDICATORS = {
    "Life expectancy at birth": ("SP.DYN.LE00.IN", "years", "Expected years a newborn would live if current age-specific mortality rates remained constant."),
    "Unemployment, national estimate": ("SL.UEM.TOTL.NE.ZS", "% of labor force", "Share of the labor force without work but available for and seeking employment, using national estimates."),
    "GDP per capita": ("NY.GDP.PCAP.CD", "current US dollars per person", "GDP divided by midyear population. It is not average wealth or household income."),
    "Crude birth rate": ("SP.DYN.CBRT.IN", "births per 1,000 people per year", "Annual live births per 1,000 people; this is a rate, not a birth count."),
    "Net migration": ("SM.POP.NETM", "people", "Estimated net number of migrants over the period; it is not an emigration count."),
    "Intentional homicide rate": ("VC.IHR.PSRC.P5", "per 100,000 people", "Intentional homicide victims per 100,000 people."),
    "Consumer price inflation": ("FP.CPI.TOTL.ZG", "% annual change", "Annual percentage change in the consumer price index."),
    "Central government debt": ("GC.DOD.TOTL.GD.ZS", "% of GDP", "Central government debt as a percentage of GDP."),
    "Total reserves including gold": ("FI.RES.TOTL.CD", "current US dollars", "Total reserves including gold, valued in current US dollars."),
    "External balance on goods and services": ("NE.RSB.GNFS.ZS", "% of GDP", "External balance on goods and services as a share of GDP; not the full current account.")
}
CONTEXTS = {
    "Business": ("Polynomial trend", "year", "sales", "year", "USD/year", "Sales trend over time"),
    "Engineering": ("Quadratic motion", "time", "position", "seconds", "meters", "Position over time"),
    "Medical": ("Exponential decay", "time", "concentration", "hours", "mg/L", "Illustrative concentration decay; not medical advice"),
    "Sports science": ("Quadratic motion", "time", "speed", "seconds", "m/s", "Illustrative performance curve"),
    "Biology": ("Logistic growth", "time", "population", "days", "organisms", "Illustrative population growth"),
    "Bioengineering": ("Exponential growth", "time", "cell count", "hours", "cells", "Illustrative cell growth"),
    "Economic forecasting": ("Polynomial trend", "year", "indicator", "year", "index units", "Trend model; extrapolation is uncertain"),
    "Weather": ("Sinusoidal cycle", "time", "temperature", "hours", "°C", "Illustrative periodic temperature"),
    "Custom": ("Polynomial trend", "x", "f(x)", "x-units", "y-units", "Custom model")
}
FUNCTIONS = ["Polynomial", "Rational", "Exponential", "Logarithmic", "Trigonometric"]

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_world_bank(country_code: str, indicator_code: str, start_year: int, end_year: int):
    url = f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}"
    try:
        response = requests.get(url, params={"date": f"{start_year}:{end_year}", "format": "json", "per_page": 20000}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list) or len(payload) < 2 or payload[1] is None:
            return pd.DataFrame(columns=["year", "value"]), f"World Bank returned no observations (HTTP {response.status_code})."
        rows = [{"year": int(r["date"]), "value": r.get("value")} for r in payload[1] if r.get("date")]
        df = pd.DataFrame(rows)
        if df.empty:
            return pd.DataFrame(columns=["year", "value"]), "The API response contained no dated observations."
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"]).sort_values("year").reset_index(drop=True)
        return df, ("" if not df.empty else "The API response contained only missing values for this period.")
    except requests.Timeout:
        return pd.DataFrame(columns=["year", "value"]), "The World Bank request timed out. Try again later."
    except (requests.RequestException, ValueError, KeyError, TypeError) as exc:
        return pd.DataFrame(columns=["year", "value"]), f"Could not retrieve valid data: {type(exc).__name__}: {exc}"

def validate_data(df: pd.DataFrame, indicator_name: str):
    if df.empty:
        return False, "There are no rows to validate."
    if df["year"].isna().any() or not np.all(np.equal(df["year"], np.floor(df["year"]))):
        return False, "Years must be whole numbers."
    if df["year"].duplicated().any():
        return False, "Each year must appear only once. Remove duplicate years."
    if df["value"].isna().any() or not np.isfinite(df["value"].astype(float)).all():
        return False, "Values must be finite numbers, not blanks or infinity."
    if indicator_name == "Life expectancy at birth" and ((df.value < 0).any() or (df.value > 120).any()):
        return False, "Life expectancy should be between 0 and 120 years."
    if indicator_name in ("Unemployment, national estimate", "Consumer price inflation") and indicator_name.startswith("Unemployment") and ((df.value < 0).any() or (df.value > 100).any()):
        return False, "Unemployment percentages must be between 0 and 100."
    if indicator_name == "Crude birth rate" and (df.value < 0).any():
        return False, "Crude birth rate cannot be negative."
    if indicator_name == "Intentional homicide rate" and (df.value < 0).any():
        return False, "Homicide rates cannot be negative."
    return True, "Data passed basic validation. Check unusual values against the source."

def safe_expression(expr: str, x: sp.Symbol):
    expr = expr.strip().replace("^", "**")
    if not expr or len(expr) > 160:
        raise ValueError("Enter a function of at most 160 characters.")
    if not re.fullmatch(r"[0-9xX+\-*/().,\sA-Za-z_0-9*]+", expr):
        raise ValueError("Use standard math characters, x, numbers, and functions such as sin, cos, exp, log, sqrt.")
    expr = re.sub(r"\bX\b", "x", expr)
    allowed = {"x": x, "sin": sp.sin, "cos": sp.cos, "tan": sp.tan, "exp": sp.exp, "log": sp.log,
               "sqrt": sp.sqrt, "abs": sp.Abs, "pi": sp.pi, "E": sp.E}
    try:
        parsed = sp.sympify(expr, locals=allowed, evaluate=True)
    except Exception as exc:
        raise ValueError("Could not read that expression. Example: 2*x**2 - 3*x + 1") from exc
    if parsed.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ValueError("The expression contains an undefined constant.")
    if any(isinstance(node, (sp.Derivative, sp.Integral)) for node in sp.preorder_traversal(parsed)):
        raise ValueError("Derivatives and integrals are not accepted as input functions.")
    return parsed

def numeric_fn(expr, xvals):
    f = sp.lambdify(sp.Symbol("x"), expr, modules=["numpy"])
    with np.errstate(all="ignore"):
        out = f(np.asarray(xvals, dtype=float))
    arr = np.asarray(out)
    if arr.ndim == 0:
        arr = np.full(np.asarray(xvals).shape, float(arr))
    return np.asarray(arr, dtype=float)

def fmt_num(v, digits=6):
    try:
        if not np.isfinite(float(v)): return "undefined"
        return f"{float(v):.{digits}g}"
    except Exception: return str(v)

def make_pdf(title, metadata, equation, metrics, findings, preview, results, chart_bytes):
    buff = io.BytesIO(); doc = SimpleDocTemplate(buff, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet(); story = [Paragraph(title, styles["Title"]), Spacer(1, 8)]
    for line in metadata: story.append(Paragraph(str(line).replace("&", "&amp;"), styles["BodyText"]))
    story += [Spacer(1, 8), Paragraph("Model / function", styles["Heading2"]), Paragraph(str(equation).replace("&", "&amp;"), styles["BodyText"]), Paragraph("Metrics", styles["Heading2"])]
    for line in metrics: story.append(Paragraph(str(line).replace("&", "&amp;"), styles["BodyText"]))
    if chart_bytes:
        story += [Spacer(1, 8), RLImage(io.BytesIO(chart_bytes), width=500, height=280)]
    story.append(Paragraph("Findings and limitations", styles["Heading2"]))
    for line in findings: story.append(Paragraph(str(line).replace("&", "&amp;"), styles["BodyText"]))
    for heading, data in [("Data preview", preview), ("Calculated years", results)]:
        story.append(Paragraph(heading, styles["Heading2"]))
        if data is not None and len(data):
            clean = data.fillna("").astype(str).head(25)
            rows = [list(clean.columns)] + clean.values.tolist()
            table = Table(rows, repeatRows=1, colWidths=[min(150, 500/max(1,len(rows[0])))]*len(rows[0]))
            table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.lightgrey),("GRID",(0,0),(-1,-1),0.25,colors.grey),("FONTSIZE",(0,0),(-1,-1),7),("VALIGN",(0,0),(-1,-1),"TOP")]))
            story.append(table)
        else: story.append(Paragraph("No results available.", styles["BodyText"]))
    doc.build(story); return buff.getvalue()

st.title("Hispanic Heritage Historical Data Explorer")
st.caption("HISPANIC HERITAGE DATA LAB · Real observations, fitted models, and student-built simulations")
st.info("Brazil is included as a Latin American comparison country. World Bank coverage varies by indicator and country; missing values are not invented.")

now_year = date.today().year
with st.sidebar:
    st.header("Historical data")
    country_label = st.selectbox("Country", list(COUNTRIES), index=0)
    indicator_name = st.selectbox("Indicator", list(INDICATORS))
    compare_label = st.selectbox("Comparison country", ["None"] + [c for c in COUNTRIES if c != country_label], index=1)
    degree = st.slider("Polynomial degree", 3, 8, 3)
    sample_step = st.slider("Graph sample increment (years)", 1, 10, 2)
    horizon = st.slider("Forward display horizon (years)", 0, 50, 10)
    printer_mode = st.checkbox("Printer-friendly view", value=False)
    st.caption(f"Requested historical window: {now_year-70}–{now_year-1}")

country_code = COUNTRIES[country_label]
indicator_code, units, definition = INDICATORS[indicator_name]
start_year, end_year = now_year-70, now_year-1
source_df, fetch_message = fetch_world_bank(country_code, indicator_code, start_year, end_year)
source_df = source_df.copy()
if not source_df.empty: source_df["year"] = source_df["year"].astype(int)
if fetch_message: st.warning(fetch_message)
if not source_df.empty:
    st.caption(f"Source: World Bank API · Retrieved {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')} · Actual coverage {source_df.year.min()}–{source_df.year.max()} · {len(source_df)} observations")
st.markdown(f"**Definition:** {definition}  \n**Units:** {units}  \n**Indicator code:** `{indicator_code}`  \n**Source link:** [World Bank indicator API](https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json)")

if "editor_country" not in st.session_state: st.session_state.editor_country = country_label
if "editor_indicator" not in st.session_state: st.session_state.editor_indicator = indicator_name
if st.session_state.editor_country != country_label or st.session_state.editor_indicator != indicator_name:
    st.session_state.editor_country, st.session_state.editor_indicator = country_label, indicator_name
    st.session_state.edited_data = source_df.copy()
    st.session_state.data_modified = False
if "edited_data" not in st.session_state: st.session_state.edited_data = source_df.copy()
if "data_modified" not in st.session_state: st.session_state.data_modified = False
if st.button("Reset to source data"):
    st.session_state.edited_data = source_df.copy(); st.session_state.data_modified = False
    st.rerun()

st.subheader("Editable working data")
st.caption("Edit years or values below. This table is separate from the original API response. Changes are a user-modified scenario.")
editor_df = st.data_editor(st.session_state.edited_data, num_rows="dynamic", use_container_width=True, key=f"data_editor_{country_code}_{indicator_code}")
if not editor_df.equals(st.session_state.edited_data):
    st.session_state.edited_data = editor_df.copy(); st.session_state.data_modified = True
working = st.session_state.edited_data.copy()
if not working.empty:
    working["year"] = pd.to_numeric(working["year"], errors="coerce")
    working["value"] = pd.to_numeric(working["value"], errors="coerce")
valid, validation_msg = validate_data(working, indicator_name)
st.caption(("⚠ USER-MODIFIED SCENARIO · " if st.session_state.data_modified else "SOURCE-DATA WORKING COPY · ") + validation_msg)
if not valid: st.error(validation_msg)

# Fit centered polynomial when valid and sufficiently populated.
fit = None; fit_expr = None; fit_summary = {}; model_message = ""
if valid and len(working) >= degree + 1:
    ordered = working.sort_values("year").copy()
    years = ordered.year.to_numpy(dtype=float); values = ordered.value.to_numpy(dtype=float)
    base_year = int(np.round(np.mean(years)))
    t = years - base_year
    try:
        coeffs = np.polynomial.polynomial.polyfit(t, values, degree)
        pred = np.polynomial.polynomial.polyval(t, coeffs)
        fit = (base_year, coeffs, int(years.min()), int(years.max()))
        x_sym = sp.Symbol("x")
        fit_expr = sum(sp.Float(float(c), 12) * (x_sym-base_year)**i for i,c in enumerate(coeffs))
        fit_summary = {"R-squared (in-sample)": r2_score(values, pred) if len(values)>1 else np.nan,
                       "MAE (in-sample)": mean_absolute_error(values, pred), "Mean observed value": float(np.mean(values)),
                       "Observation count": len(values), "Base year": base_year, "Degree": degree,
                       "Historical range": f"{int(years.min())}–{int(years.max())}"}
    except Exception as exc: model_message = f"Model fit failed: {exc}"
else:
    model_message = f"Need at least {degree+1} valid, distinct years for a degree-{degree} polynomial. Current valid rows: {len(working) if valid else 0}."
if model_message: st.warning(model_message)

# Holdout: latest 20%, chronological; use a lower degree if training set cannot support selected degree.
holdout_text = "Holdout score unavailable: not enough observations for a chronological holdout."
if valid and len(working) >= max(degree+4, 10):
    ordered = working.sort_values("year"); cut = max(degree+1, int(len(ordered)*0.8))
    if cut < len(ordered):
        tr, te = ordered.iloc[:cut], ordered.iloc[cut:]
        try:
            hb = int(np.round(tr.year.mean())); hc = np.polynomial.polynomial.polyfit(tr.year.to_numpy()-hb, tr.value.to_numpy(), degree)
            hp = np.polynomial.polynomial.polyval(te.year.to_numpy()-hb, hc)
            holdout_text = f"Chronological holdout MAE: {mean_absolute_error(te.value, hp):.6g} ({len(te)} held-out observation(s); separate from in-sample metrics)."
        except Exception: pass

st.subheader("Historical fit")
if fit is not None:
    base_year, coeffs, min_year, max_year = fit
    equation = " + ".join([f"({c:.8g})·(year − {base_year})^{i}" for i,c in enumerate(coeffs)])
    st.code("t = year − " + str(base_year) + "\nŷ = " + equation, language="text")
    mcols = st.columns(4)
    for col, (key,val) in zip(mcols, fit_summary.items()): col.metric(key, fmt_num(val) if isinstance(val,(int,float,np.number)) else str(val))
    st.caption("All fit metrics are in-sample. Polynomial extrapolations can become implausible outside observed years; MAE describes typical in-sample error, not a validated prediction interval.")
    st.write(holdout_text)
    residuals = working.sort_values("year").copy()
    residuals["fitted"] = np.polynomial.polynomial.polyval(residuals.year.to_numpy()-base_year, coeffs)
    residuals["residual"] = residuals.value - residuals.fitted
    with st.expander("Residuals and fitted values"): st.dataframe(residuals, use_container_width=True)
else:
    equation = "No fitted polynomial available."

# Year prediction controls, session-persisted unless model key changes.
model_key = f"{country_code}|{indicator_code}|{degree}|{hash(tuple(map(tuple, working.fillna('').astype(str).values.tolist()))) if not working.empty else 0}"
if st.session_state.get("prediction_model_key") != model_key:
    st.session_state.prediction_model_key = model_key
    st.session_state.single_result = None; st.session_state.batch_results = pd.DataFrame()
st.subheader("Calculate fitted values")
p1,p2 = st.columns(2)
with p1:
    single_year = st.number_input("Single year (1950–2100)", min_value=1950, max_value=2100, value=2025, step=1)
    if st.button("Calculate Value", disabled=(fit is None)):
        val = float(np.polynomial.polynomial.polyval(single_year-base_year, coeffs))
        st.session_state.single_result = {"year": int(single_year), "estimate": val, "within": min_year <= single_year <= max_year}
    if st.session_state.get("single_result"):
        r = st.session_state.single_result
        st.metric(f"Fitted estimate for {r['year']}", f"{r['estimate']:.6g} {units}")
        st.caption(f"{'Within' if r['within'] else 'Outside'} fitted historical range ({min_year}–{max_year}). This is a model estimate, not necessarily an actual observation.")
with p2:
    batch_mode = st.radio("Batch years input", ["Year range", "Custom years"], horizontal=True)
    if batch_mode == "Year range":
        c1,c2,c3 = st.columns(3)
        with c1: batch_start = st.number_input("Start", min_value=1950, max_value=2100, value=max(1950, min_year if fit else 2000), step=1)
        with c2: batch_end = st.number_input("End", min_value=1950, max_value=2100, value=min(2100, max_year+5 if fit else 2025), step=1)
        with c3: batch_step = st.selectbox("Step", list(range(1,11)), index=0)
        years_requested = list(range(int(batch_start), int(batch_end)+1, int(batch_step))) if batch_start <= batch_end else []
        if batch_start > batch_end: st.error("Start year must be less than or equal to end year.")
    else:
        custom_years = st.text_input("Comma-separated years", "2000, 2010, 2020, 2030")
        years_requested = []
        for item in custom_years.split(","):
            item = item.strip()
            if not item: continue
            try:
                y = int(item)
                if 1950 <= y <= 2100: years_requested.append(y)
                else: st.warning(f"Ignored {item}: years must be 1950–2100.")
            except ValueError: st.warning(f"Ignored '{item}': enter whole years separated by commas.")
    if st.button("Calculate batch", disabled=(fit is None)):
        unique_years = sorted(set(years_requested))
        if not unique_years: st.session_state.batch_results = pd.DataFrame()
        else:
            est = np.polynomial.polynomial.polyval(np.array(unique_years)-base_year, coeffs)
            batch = pd.DataFrame({"year": unique_years, "estimate": est, "units": units})
            batch["range_classification"] = np.where(batch.year.between(min_year,max_year), "Within fitted range", "Outside fitted range")
            batch["actual_observation"] = batch.year.isin(working.year.astype(int))
            st.session_state.batch_results = batch
batch_results = st.session_state.get("batch_results", pd.DataFrame())
if not batch_results.empty: st.dataframe(batch_results, use_container_width=True)

st.subheader("Historical chart")
chart_buf = None
fig, ax = plt.subplots(figsize=(10,4.8))
if not working.empty and "year" in working and "value" in working:
    ax.scatter(working.year, working.value, label="Observed / edited data", color="#176B87", zorder=3)
if fit is not None:
    plot_end = min(2100, max_year+horizon)
    smooth_years = np.linspace(min_year, max(min_year+1,plot_end), 600)
    smooth_vals = np.polynomial.polynomial.polyval(smooth_years-base_year, coeffs)
    hist_mask = smooth_years <= max_year
    ax.plot(smooth_years[hist_mask], smooth_vals[hist_mask], label="Fitted curve (historical range)", color="#E07A2D")
    if horizon > 0 and plot_end > max_year: ax.plot(smooth_years[~hist_mask], smooth_vals[~hist_mask], "--", label="Extrapolated segment", color="#9B59B6")
    sampled = np.arange(min_year, max_year+1, sample_step)
    ax.scatter(sampled, np.polynomial.polynomial.polyval(sampled-base_year, coeffs), marker="x", label=f"Model markers every {sample_step} year(s)", color="#2E8B57")
if not batch_results.empty: ax.scatter(batch_results.year, batch_results.estimate, marker="D", s=45, label="Batch estimates", color="#B23A48")
if st.session_state.get("single_result"):
    sr=st.session_state.single_result; ax.scatter([sr["year"]],[sr["estimate"]],marker="*",s=130,label="Single-year estimate",color="#111111")
if fit is not None: ax.axvline(max_year, linestyle=":", color="grey", label="Historical boundary")
ax.set_xlabel("Year"); ax.set_ylabel(units); ax.set_title(f"{country_label}: {indicator_name}"); ax.grid(True, alpha=.25); ax.legend(loc="best", fontsize=8); fig.tight_layout()
st.pyplot(fig, use_container_width=True)
img_io=io.BytesIO(); fig.savefig(img_io, format="png", dpi=140, bbox_inches="tight"); chart_bytes=img_io.getvalue(); plt.close(fig)

if compare_label != "None":
    st.subheader("Country comparison")
    compare_code = COUNTRIES[compare_label]
    compare_df, compare_msg = fetch_world_bank(compare_code, indicator_code, start_year, end_year)
    if compare_msg: st.warning(f"{compare_label}: {compare_msg}")
    if not compare_df.empty:
        st.caption(f"Coverage differs by country: {compare_label} has {len(compare_df)} observations from {compare_df.year.min()}–{compare_df.year.max()}; only observed years are plotted.")
        fig2, ax2 = plt.subplots(figsize=(10,3.5)); ax2.plot(working.year, working.value, marker="o", label=country_label)
        ax2.plot(compare_df.year, compare_df.value, marker="s", label=compare_label); ax2.set_xlabel("Year"); ax2.set_ylabel(units); ax2.legend(); ax2.grid(alpha=.25); fig2.tight_layout(); st.pyplot(fig2, use_container_width=True); plt.close(fig2)

# Function builder and calculus analysis
st.divider(); st.header("Function builder and calculus lab")
course = st.selectbox("Mathematics pathway", ["Honors Algebra 2", "Precalculus", "Calculus"], index=2)
context = st.selectbox("Context", list(CONTEXTS))
fn_type = st.selectbox("Function family", FUNCTIONS)
def_ctx, default_x, default_y, default_xu, default_yu, default_desc = CONTEXTS[context]
with st.expander("Customize context and variables", expanded=True):
    description = st.text_input("Context description", default_desc)
    x_name = st.text_input("Independent variable name", default_x)
    y_name = st.text_input("Dependent variable name", default_y)
    x_units = st.text_input("Independent variable units", default_xu)
    y_units = st.text_input("Dependent variable units", default_yu)
    interval_left, interval_right = st.slider("Analysis interval", -20.0, 20.0, (-5.0, 5.0), step=0.5)
if interval_left >= interval_right: st.error("Analysis interval needs a left endpoint smaller than the right endpoint.")

# Suggestions; historical polynomial can be selected directly.
suggestions = {
    "Business": "0.5*x**2 + 2*x + 10", "Engineering": "-4.9*x**2 + 20*x + 0", "Medical": "100*exp(-0.25*x)",
    "Sports science": "-0.2*x**2 + 3*x + 5", "Biology": "1000/(1 + 9*exp(-0.5*x))", "Bioengineering": "20*exp(0.15*x)",
    "Economic forecasting": "0.1*x**2 + 2*x + 100", "Weather": "10*sin(2*pi*x/24) + 20", "Custom": "2*x**2 - 3*x + 1"
}
function_source = st.radio("Function source", ["Suggested function", "Enter my own function", "Use historical best fit"], horizontal=True)
if function_source == "Suggested function":
    suggested = suggestions[context]
    if fn_type == "Rational": suggested = "(x**2 + 1)/(x + 3)"
    elif fn_type == "Exponential": suggested = "2*exp(0.2*x)"
    elif fn_type == "Logarithmic": suggested = "3*log(x + 5)"
    elif fn_type == "Trigonometric": suggested = "2*sin(x) + 0.5*cos(2*x)"
    elif fn_type == "Polynomial" and context == "Biology": suggested = "1000/(1 + 9*exp(-0.5*x))"
    expression_text = st.text_input("Suggested expression (editable)", suggested)
elif function_source == "Enter my own function":
    expression_text = st.text_input("Enter f(x)", "2*x**2 - 3*x + 1", help="Use x, numbers, + - * / **, parentheses, sin, cos, tan, exp, log, sqrt, pi. Example: (x**2+1)/(x+3)")
else:
    if fit_expr is None:
        st.warning("Historical fit is unavailable; using a suggested function until enough valid historical data are available.")
        expression_text = st.text_input("Fallback function", suggestions[context])
    else:
        expression_text = st.text_input("Historical fitted function (x means calendar year)", str(sp.expand(fit_expr)).replace("**", "**"), help="This function uses calendar year as x. Change the analysis interval accordingly for historical years.")

param1, param2, param3 = st.columns(3)
with param1: use_parameters = st.checkbox("Build quadratic from parameters", value=False)
with param2: a_param = st.number_input("a", value=1.0, step=0.5, disabled=not use_parameters)
with param3: b_param = st.number_input("b", value=0.0, step=0.5, disabled=not use_parameters)
c_param = st.number_input("c parameter", value=0.0, step=0.5, disabled=not use_parameters)
if use_parameters: expression_text = f"({a_param})*x**2 + ({b_param})*x + ({c_param})"

x_symbol = sp.Symbol("x")
expr = None; d1 = None; d2 = None; function_error = None
try:
    expr = safe_expression(expression_text, x_symbol)
    d1 = sp.diff(expr, x_symbol); d2 = sp.diff(d1, x_symbol)
except Exception as exc: function_error = str(exc)
if function_error:
    st.error(f"Function input error: {function_error}")
else:
    st.latex(r"f(x)=" + sp.latex(expr))
    st.caption(f"Context: {description}. {x_name} is measured in {x_units}; {y_name} is measured in {y_units}.")
    if course == "Calculus":
        st.latex(r"f'(x)=" + sp.latex(d1)); st.latex(r"f''(x)=" + sp.latex(d2))
    elif course == "Precalculus":
        st.write("Precalculus focus: degree and coefficients shape the curve; turning points and concavity can be explored through changing average rates.")
    else:
        st.write("Algebra 2 focus: domain, range, end behavior, graphical trends, and average rates of change.")

    st.subheader("Interactive graph and rates")
    s1,s2,s3 = st.columns(3)
    with s1: show_f = st.checkbox("Show f(x)", True)
    with s2: show_d1 = st.checkbox("Show f'(x)", course=="Calculus")
    with s3: show_d2 = st.checkbox("Show f''(x)", False)
    current_x = st.slider("Current x", float(interval_left), float(interval_right), float((interval_left+interval_right)/2), step=max(0.01,(interval_right-interval_left)/200))
    sec_l, sec_r = st.slider("Average-rate interval endpoints", float(interval_left), float(interval_right), (float(interval_left), float(interval_right)), step=max(0.01,(interval_right-interval_left)/100))
    play_col1,play_col2,play_col3 = st.columns(3)
    with play_col1: play = st.checkbox("Auto playback", value=False)
    with play_col2: speed = st.select_slider("Playback speed", options=["Slow", "Normal", "Fast"], value="Normal")
    with play_col3:
        if st.button("Reset point"): st.session_state.current_x_reset = True
    if st.session_state.get("current_x_reset"):
        current_x = float((interval_left+interval_right)/2); st.session_state.current_x_reset = False
    # Streamlit reruns are used as a simple play control; no unbounded loop.
    if play:
        step_amount = (interval_right-interval_left) / {"Slow":100,"Normal":40,"Fast":15}[speed]
        current_x = interval_left + ((current_x-interval_left+step_amount) % (interval_right-interval_left))
        st.caption("Playback advances one step per app rerun. Move a control or rerun the app to advance; no background timer is used.")
    f0 = float(numeric_fn(expr,[current_x])[0]); d10 = float(numeric_fn(d1,[current_x])[0]); d20 = float(numeric_fn(d2,[current_x])[0])
    left_y = float(numeric_fn(expr,[sec_l])[0]); right_y = float(numeric_fn(expr,[sec_r])[0])
    avg_rate = (right_y-left_y)/(sec_r-sec_l) if sec_r != sec_l and np.isfinite([left_y,right_y]).all() else np.nan
    tangent_intercept = f0-d10*current_x
    xs = np.linspace(interval_left, interval_right, 900)
    ys = numeric_fn(expr,xs); dy = numeric_fn(d1,xs); ddy = numeric_fn(d2,xs)
    fig3, ax3 = plt.subplots(figsize=(10,5))
    if show_f: ax3.plot(xs, np.where(np.isfinite(ys),ys,np.nan), label="f(x)", linewidth=2)
    if show_d1: ax3.plot(xs, np.where(np.isfinite(dy),dy,np.nan), label="f'(x)")
    if show_d2: ax3.plot(xs, np.where(np.isfinite(ddy),ddy,np.nan), label="f''(x)")
    if np.isfinite(f0):
        ax3.scatter([current_x],[f0],s=55,label="Current point")
        if np.isfinite(d10): ax3.plot(xs, d10*(xs-current_x)+f0, "--", label="Tangent at current x")
    if np.isfinite(avg_rate) and sec_l != sec_r:
        sec_x = np.array([sec_l,sec_r]); sec_y = left_y+avg_rate*(sec_x-sec_l)
        ax3.plot(sec_x,sec_y,":",linewidth=2,label="Secant line")
        ax3.scatter(sec_x,[left_y,right_y],s=25)
    ax3.axvline(sec_l,color="grey",alpha=.3); ax3.axvline(sec_r,color="grey",alpha=.3)
    ax3.set_xlabel(f"{x_name} ({x_units})"); ax3.set_ylabel(f"{y_name} / derivative values (scale may differ)"); ax3.grid(alpha=.25); ax3.legend(loc="best",fontsize=8); fig3.tight_layout(); st.pyplot(fig3,use_container_width=True)
    st.write(f"At x = {current_x:g}: f(x) = {fmt_num(f0)} {y_units}; f'(x) = {fmt_num(d10)} {y_units}/{x_units}; f''(x) = {fmt_num(d20)} {y_units}/{x_units}².")
    st.write(f"Average rate from {sec_l:g} to {sec_r:g}: {fmt_num(avg_rate)} {y_units}/{x_units}.")
    if context in ("Engineering", "Sports science") and "position" in y_name.lower():
        st.write(f"Interpretation: the first derivative is velocity ({y_units}/{x_units}); the second derivative is acceleration ({y_units}/{x_units}²).")
    elif context in ("Business", "Economic forecasting"):
        st.write("Interpretation: f(x) is the modeled indicator; f'(x) is its instantaneous rate of change; f''(x) describes whether that rate is increasing or decreasing. These are model interpretations, not proof of a cause.")
    else:
        st.write(f"Interpretation: f(x) measures {y_name}; the first derivative describes its instantaneous change per {x_units}; the second derivative describes how that rate changes. Units follow the labels above.")

    if course == "Calculus":
        st.subheader("Calculus analysis (numerical classifications are approximate)")
        try:
            critical_expr = sp.solveset(sp.Eq(d1,0), x_symbol, domain=sp.Interval(float(interval_left),float(interval_right)))
            crit_candidates = list(critical_expr) if isinstance(critical_expr, sp.FiniteSet) else []
        except Exception: crit_candidates=[]
        # Add numerical roots from a sign-change grid, then classify with derivative signs.
        roots=[]
        for i in range(len(xs)-1):
            if not (np.isfinite(dy[i]) and np.isfinite(dy[i+1])): continue
            if dy[i] == 0: roots.append(xs[i])
            elif dy[i]*dy[i+1] < 0:
                lo,hi=xs[i],xs[i+1]
                for _ in range(35):
                    mid=(lo+hi)/2; fm=float(numeric_fn(d1,[mid])[0])
                    if not np.isfinite(fm): break
                    if float(numeric_fn(d1,[lo])[0])*fm <= 0: hi=mid
                    else: lo=mid
                roots.append((lo+hi)/2)
        roots += [float(sp.N(v)) for v in crit_candidates if v.is_real and not v.free_symbols]
        roots=sorted(set(round(float(r),5) for r in roots if interval_left <= float(r) <= interval_right))
        critical_rows=[]
        for r in roots:
            eps=max(1e-4,(interval_right-interval_left)/10000)
            before=float(numeric_fn(d1,[r-eps])[0]) if r-eps>=interval_left else np.nan
            after=float(numeric_fn(d1,[r+eps])[0]) if r+eps<=interval_right else np.nan
            cls="not classified as extremum"
            if before>0 and after<0: cls="local maximum (f' changes + to −)"
            elif before<0 and after>0: cls="local minimum (f' changes − to +)"
            critical_rows.append({"x (approx.)":r,"f(x)":float(numeric_fn(expr,[r])[0]),"classification":cls})
        if critical_rows: st.dataframe(pd.DataFrame(critical_rows),use_container_width=True)
        else: st.write("No critical points detected in this interval (zeros and derivative-undefined points are checked numerically; symbolic solutions may be incomplete).")
        inflections=[]
        for i in range(len(xs)-1):
            if not (np.isfinite(ddy[i]) and np.isfinite(ddy[i+1])): continue
            if ddy[i]*ddy[i+1] < 0:
                r=(xs[i]+xs[i+1])/2
                eps=max(1e-3,(interval_right-interval_left)/1000)
                before=float(numeric_fn(d2,[r-eps])[0]); after=float(numeric_fn(d2,[r+eps])[0])
                if np.isfinite(before) and np.isfinite(after) and before*after<0: inflections.append((r,float(numeric_fn(expr,[r])[0])))
        st.write("Inflection points (approximate, only where sampled concavity changes):")
        st.write(inflections if inflections else "No concavity-changing inflection point detected on this interval.")
        finite = np.isfinite(ys)
        if finite.any():
            ix=np.where(finite)[0]; imin,imax=ix[np.argmin(ys[finite])],ix[np.argmax(ys[finite])]
            st.write(f"Approximate absolute extrema on the selected closed interval, including endpoints when defined: min at x≈{xs[imin]:.5g}, f≈{ys[imin]:.5g}; max at x≈{xs[imax]:.5g}, f≈{ys[imax]:.5g}.")
        st.caption("Numerical root detection can miss even-multiplicity roots or closely spaced features. The interval endpoints are included for absolute-extrema checks. Undefined regions split the plotted curve; classifications are approximate.")
    elif course == "Precalculus":
        st.write("Changing average rates help reveal turning points and concavity. Polynomial degree sets the broad end behavior; coefficients change vertical scale, direction, and location of features.")
    else:
        st.write(f"Domain: determined by the expression (for example, log requires a positive argument and a rational denominator cannot be zero). Approximate range on this interval: {fmt_num(np.nanmin(ys))} to {fmt_num(np.nanmax(ys))} when finite values exist. End behavior depends on the function family and leading terms.")

    st.subheader("Contextual simulation")
    visual = st.selectbox("Visual simulation", ["Walking person with prosthetic leg", "Moving object / projectile", "Tank or concentration indicator", "Population / cell colony", "Thermometer / weather", "Business / economic indicator"])
    st.caption("The visual is a schematic, not a physical or medical validation. It uses the function value after bounded display scaling; changing the function changes the displayed indicator.")
    sim_val = f0 if np.isfinite(f0) else 0.0
    scale = max(1.0, abs(sim_val)); normalized = max(-1.0,min(1.0,sim_val/scale))
    if visual == "Walking person with prosthetic leg":
        angle = 25*math.sin(current_x) * (0.5+0.5*abs(normalized))
        st.markdown(f"**Walking-cycle schematic** · input x = {current_x:.2f}; model output = {fmt_num(f0)} {y_units}; leg-angle proxy = {angle:.1f}°. This is an illustrative motion mapping, not a prosthetic prescription.")
        st.progress(min(1.0,max(0.0,(math.sin(current_x)+1)/2)), text="Illustrative gait phase")
    elif visual == "Moving object / projectile":
        st.markdown(f"**Object simulation** · output = {fmt_num(f0)} {y_units}. Height/position proxy is scaled for display only.")
        st.progress(min(1.0,max(0.0,(normalized+1)/2)),text="Scaled position")
    elif visual == "Tank or concentration indicator":
        st.markdown(f"**Tank / concentration gauge** · raw output = {fmt_num(f0)} {y_units}; display gauge is normalized to 0–100% for readability, not a real dose or safety threshold.")
        st.progress(min(1.0,max(0.0,(normalized+1)/2)),text="Normalized indicator")
    elif visual == "Population / cell colony":
        st.markdown(f"**Colony indicator** · modeled output = {fmt_num(f0)} {y_units}; size indicator is illustrative and not a biological forecast.")
        st.progress(min(1.0,max(0.0,(normalized+1)/2)),text="Scaled colony indicator")
    elif visual == "Thermometer / weather":
        st.markdown(f"**Weather indicator** · modeled value = {fmt_num(f0)} {y_units}. Gauge maps the current output to a bounded display scale.")
        st.progress(min(1.0,max(0.0,(normalized+1)/2)),text="Scaled temperature indicator")
    else:
        st.markdown(f"**Business indicator** · modeled output = {fmt_num(f0)} {y_units}; slope = {fmt_num(d10)} {y_units}/{x_units}; curvature = {fmt_num(d20)} {y_units}/{x_units}².")
        st.progress(min(1.0,max(0.0,(normalized+1)/2)),text="Scaled business indicator")

# Exports use current working data and calculations.
st.divider(); st.header("Export current report")
metadata = [f"Country: {country_label}", f"Indicator: {indicator_name} ({indicator_code})", f"Units: {units}", f"Retrieved: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}", f"Working data status: {'user-modified scenario' if st.session_state.data_modified else 'source-data working copy'}", f"Actual data rows: {len(source_df)}; working rows: {len(working)}"]
summary_df = pd.DataFrame([{"country":country_label,"indicator":indicator_name,"indicator_code":indicator_code,"units":units,"source_url":f"https://api.worldbank.org/v2/country/{country_code}/indicator/{indicator_code}?format=json","retrieved_at":datetime.now().astimezone().isoformat(),"data_status":"user-modified scenario" if st.session_state.data_modified else "source-data working copy","degree":degree,"equation":equation,"fit_metrics":str(fit_summary),"holdout":holdout_text}])
export_working = working.copy()
if fit is not None:
    export_working=export_working.sort_values("year").copy(); export_working["fitted_value"]=np.polynomial.polynomial.polyval(export_working.year.to_numpy()-base_year,coeffs); export_working["residual"]=export_working.value-export_working.fitted_value
csv_bytes=export_working.to_csv(index=False).encode("utf-8")
xlsx_io=io.BytesIO()
with pd.ExcelWriter(xlsx_io,engine="xlsxwriter") as writer:
    summary_df.to_excel(writer,index=False,sheet_name="Summary"); export_working.to_excel(writer,index=False,sheet_name="Working data")
    (batch_results if not batch_results.empty else pd.DataFrame(columns=["year","estimate","units","range_classification","actual_observation"])).to_excel(writer,index=False,sheet_name="Calculated years")
pdf_preview=export_working.head(20)
pdf_results=batch_results if not batch_results.empty else pd.DataFrame()
findings=["Observed values come from the World Bank API unless the working data are marked as user-modified.","Polynomial fit metrics are in-sample; a chronological holdout score is shown separately when possible.","MAE and residuals describe model fit and are not a validated 95% prediction interval.","Extrapolation can be implausible, especially for high-degree polynomials. No predictions are silently clipped.","Missing World Bank observations are left missing; this app does not invent data."]
pdf_bytes=make_pdf("Hispanic Heritage Historical Data Explorer",metadata,equation,[f"{k}: {v}" for k,v in fit_summary.items()]+[holdout_text],findings,pdf_preview,pdf_results,chart_bytes)
d1,d2,d3=st.columns(3)
with d1: st.download_button("Download CSV",csv_bytes,file_name="historical_data.csv",mime="text/csv",use_container_width=True)
with d2: st.download_button("Download Excel workbook",xlsx_io.getvalue(),file_name="historical_data_explorer.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",use_container_width=True)
with d3: st.download_button("Download PDF report",pdf_bytes,file_name="historical_data_report.pdf",mime="application/pdf",use_container_width=True)

if printer_mode:
    st.markdown("---")
    st.subheader("Printer-friendly summary")
    st.write(f"{country_label} · {indicator_name} · {units}")
    st.write(f"Data coverage: {source_df.year.min()}–{source_df.year.max()} ({len(source_df)} observations)" if not source_df.empty else "No source observations retrieved.")
    st.write(f"Model: {equation}")
    st.write(f"Status: {'user-modified scenario' if st.session_state.data_modified else 'source-data working copy'}")

st.divider(); st.subheader("How the app works")
st.markdown("""
1. **Retrieve:** requests the selected World Bank indicator for the previous 70 complete calendar years and caches the response for one hour.
2. **Validate and edit:** the source response remains separate from the editable working table. Reset restores the latest source response. The editor checks years, duplicates, finite values, and basic indicator-specific limits.
3. **Fit:** a polynomial is fitted to centered time, `t = year − base year`, to improve numerical stability. The app requires at least degree + 1 valid distinct years.
4. **Evaluate:** single-year and batch predictions use the same fitted coefficients and centered transformation. Estimates outside observed coverage are labeled extrapolations.
5. **Analyze:** SymPy differentiates the selected function symbolically. Numerical graphs and classifications are approximations over the chosen interval; rational/logarithmic functions may have domain restrictions or discontinuities.
6. **Export:** CSV contains working data, Excel contains summary/working/calculated-year sheets, and PDF includes metadata, equation, metrics, chart, limitations, data preview, and calculated results.

**Limitations:** polynomial fit does not establish causation; high-degree fits can overfit; historical indicators may have gaps or revisions; national estimates and definitions can vary. Numerical critical-point/inflection detection is approximate. Context simulations are teaching illustrations, not medical, engineering, or financial advice. The current playback control advances on app reruns rather than running a continuous background timer.
""")
