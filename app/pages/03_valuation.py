"""
Page 3 -- DCF Valuation & Scenario Analysis
Three-scenario (bear/base/bull) cash flow model, sculpted debt DSCR time series,
revenue waterfall, and NPV/IRR comparison.

Populated by: src/financial_model.py
  -> data/processed/dcf_outputs_{base,bull,bear}.parquet
  -> data/processed/valuation_summary.csv
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import CAPACITY_MW, DATA_PROCESSED, PLANT_NAME

st.set_page_config(page_title="DCF Valuation", layout="wide")

SUMMARY_PATH = DATA_PROCESSED / "valuation_summary.csv"
DCF_PATHS = {s: DATA_PROCESSED / f"dcf_outputs_{s}.csv" for s in ("base", "bull", "bear")}

missing = [p.name for p in [SUMMARY_PATH, *DCF_PATHS.values()] if not p.exists()]
if missing:
    st.title("DCF Valuation -- Scenario Analysis")
    st.warning(
        f"Missing: {', '.join(missing)}. Run `python src/financial_model.py` first.",
        icon="⚠️",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Colour palette (consistent with the financial model narrative)
# ---------------------------------------------------------------------------
C = {
    "bear":       "#d62728",
    "base":       "#1f77b4",
    "bull":       "#2ca02c",
    "contracted": "#17becf",
    "merchant":   "#ff7f0e",
    "opex":       "#aec7e8",
    "covenant":   "#d62728",
    "default":    "#333333",
}

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_PATH)

@st.cache_data(ttl=3600)
def load_dcf(scenario: str) -> pd.DataFrame:
    return pd.read_csv(DCF_PATHS[scenario])

vs     = load_summary().set_index("scenario")
base   = load_dcf("base")
bull   = load_dcf("bull")
bear   = load_dcf("bear")
dfs    = {"base": base, "bull": bull, "bear": bear}

# Recompute DSCR time series for all scenarios against the base-case debt schedule
# Debt service = base CFADS / target_dscr (1.35x sculpted)
TARGET_DSCR = 1.35
ds_schedule = base["cfads"] / TARGET_DSCR

wacc       = float(vs.loc["base", "wacc"])
rf         = float(vs.loc["base", "risk_free_rate"])
debt_rate  = float(vs.loc["base", "debt_rate"])
debt_m     = float(vs.loc["base", "debt_principal_m"])
base_npv_m = float(vs.loc["base", "npv_unlevered_m"])

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("DCF Valuation -- Scenario Analysis")
st.caption(
    f"{PLANT_NAME}  |  399.7 MW  |  Model: 2025-2039  |  "
    f"WACC {wacc:.1%}  |  Debt rate {debt_rate:.1%}"
)

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
base_lev_irr = vs.loc["base", "irr_levered"]
bull_lev_irr = vs.loc["bull", "irr_levered"]

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("WACC (CAPM)",         f"{wacc:.1%}",
          help=f"rf={rf:.1%} + 0.55 x 5.5% ERP + 1.5% size premium")
c2.metric("Debt Rate (Baa)",     f"{debt_rate:.1%}")
c3.metric("Debt Principal",      f"${debt_m:.0f}m")
c4.metric("Base NPV (unlev.)",   f"${base_npv_m:.0f}m")
c5.metric("Base Levered IRR",    f"{base_lev_irr:.1%}" if np.isfinite(base_lev_irr) else "N/A",
          delta=f"vs bull {bull_lev_irr:.1%}" if np.isfinite(bull_lev_irr) else None)

st.markdown("---")

# ---------------------------------------------------------------------------
# Scenario comparison table
# ---------------------------------------------------------------------------
st.subheader("Scenario Comparison")

table_rows = []
for scenario in ("bear", "base", "bull"):
    r = vs.loc[scenario]
    lev_irr = r["irr_levered"]
    table_rows.append({
        "Scenario":          scenario.capitalize(),
        "Generation (GWh/yr)": f"{r['gen_gwh']:,.0f}",
        "Capture Rate":      f"{r['capture_rate']:.3f}",
        "Avg Rev ($/MWh)":   f"${r['avg_rev_mwh']:.1f}",
        "EBITDA Margin":     f"{r['ebitda_margin']:.0%}",
        "Unlevered NPV":     f"${r['npv_unlevered_m']:.0f}m",
        "Unlevered IRR":     f"{r['irr_unlevered']:.1%}",
        "Levered IRR":       f"{lev_irr:.1%}" if np.isfinite(lev_irr) else "N/A (DSCR<1)",
        "Min DSCR":          f"{r['dscr_min']:.2f}x",
    })

tbl_df = pd.DataFrame(table_rows).set_index("Scenario")

def _color_row(row):
    s = row.name.lower()
    bg = {"bear": "#fdecea", "base": "#e8f4fd", "bull": "#eafaf1"}
    return [f"background-color: {bg.get(s, 'white')}"] * len(row)

st.dataframe(
    tbl_df.style.apply(_color_row, axis=1),
    use_container_width=True,
    height=140,
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Chart 1: CFADS by scenario (line chart)
# ---------------------------------------------------------------------------
col_l, col_r = st.columns(2)

with col_l:
    st.subheader("CFADS by Scenario (2025-2039)")
    fig_cfads = go.Figure()
    for scenario, df in dfs.items():
        fig_cfads.add_trace(go.Scatter(
            x=df["year"], y=df["cfads"] / 1e6,
            mode="lines+markers",
            name=scenario.capitalize(),
            line=dict(color=C[scenario], width=2.5),
            marker=dict(size=6),
            hovertemplate=f"{scenario.capitalize()}: $%{{y:.1f}}m<extra></extra>",
        ))
    fig_cfads.add_trace(go.Scatter(
        x=base["year"], y=ds_schedule / 1e6,
        mode="lines", name="Debt Service",
        line=dict(color="#888", width=1.5, dash="dash"),
        hovertemplate="Debt service: $%{y:.1f}m<extra></extra>",
    ))
    fig_cfads.update_layout(
        template="plotly_white",
        yaxis_title="CFADS ($m)", xaxis_title="Year",
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", y=1.12),
        height=380, margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig_cfads, use_container_width=True)
    st.caption(
        "CFADS = EBITDA (no capex in remaining life). "
        "Dashed grey = base-case sculpted debt service. "
        "Step-up in 2028 reflects hedge expiry (77% of output moves from $23.50/MWh to merchant)."
    )

with col_r:
    st.subheader("NPV by Scenario")
    npvs = [float(vs.loc[s, "npv_unlevered_m"]) for s in ("bear","base","bull")]
    fig_npv = go.Figure(go.Bar(
        x=["Bear", "Base", "Bull"],
        y=npvs,
        marker_color=[C["bear"], C["base"], C["bull"]],
        text=[f"${v:.0f}m" for v in npvs],
        textposition="outside",
    ))
    fig_npv.add_hline(y=debt_m, line_dash="dot", line_color="#888",
                      annotation_text=f"Debt ${debt_m:.0f}m", annotation_position="right")
    fig_npv.update_layout(
        template="plotly_white",
        yaxis_title="Unlevered NPV ($m)",
        yaxis=dict(range=[min(npvs) * 0.8, max(npvs) * 1.25]),
        height=380, margin=dict(t=40, b=40, r=80),
        showlegend=False,
    )
    st.plotly_chart(fig_npv, use_container_width=True)
    st.caption(
        "Dotted line = debt principal ($110m). Bear NPV ($46m) < debt → "
        "equity is underwater under P90 + low capture assumptions."
    )

st.markdown("---")

# ---------------------------------------------------------------------------
# Chart 2: Revenue waterfall -- base case
# ---------------------------------------------------------------------------
st.subheader("Revenue Breakdown -- Base Case (2025-2039)")

fig_rev = go.Figure()
fig_rev.add_trace(go.Bar(
    x=base["year"], y=base["contracted_rev"] / 1e6,
    name="Contracted (hedge @ $23.50/MWh)",
    marker_color=C["contracted"],
    hovertemplate="Year: %{x}<br>Contracted: $%{y:.1f}m<extra></extra>",
))
fig_rev.add_trace(go.Bar(
    x=base["year"], y=base["merchant_rev"] / 1e6,
    name="Merchant (hub x capture)",
    marker_color=C["merchant"],
    hovertemplate="Year: %{x}<br>Merchant: $%{y:.1f}m<extra></extra>",
))
fig_rev.add_trace(go.Scatter(
    x=base["year"], y=base["total_opex"] / 1e6,
    mode="lines", name="Total OpEx",
    line=dict(color=C["default"], width=2, dash="dash"),
    hovertemplate="Year: %{x}<br>OpEx: $%{y:.1f}m<extra></extra>",
))
fig_rev.add_vrect(
    x0=2024.5, x1=2027.5,
    fillcolor="#17becf", opacity=0.06,
    annotation_text="Hedge active",
    annotation_position="top left",
)
fig_rev.update_layout(
    template="plotly_white",
    barmode="stack",
    yaxis_title="Revenue ($m)",
    xaxis_title="Year",
    xaxis=dict(dtick=1),
    legend=dict(orientation="h", y=1.12),
    height=380, margin=dict(t=10, b=40),
)
st.plotly_chart(fig_rev, use_container_width=True)
st.caption(
    "Contracted revenue (teal) steps to zero at 2027 hedge expiry. "
    "Post-2028 merchant revenue is larger in dollar terms because the merchant price "
    "($33+/MWh escalating) exceeds the hedge price ($23.50/MWh)."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Chart 3: DSCR time series
# ---------------------------------------------------------------------------
st.subheader("DSCR Time Series -- Base-Case Debt Schedule")

fig_dscr = go.Figure()
for scenario, df in dfs.items():
    dscr = df["cfads"].values / ds_schedule.values
    fig_dscr.add_trace(go.Scatter(
        x=df["year"], y=dscr,
        mode="lines+markers",
        name=scenario.capitalize(),
        line=dict(color=C[scenario], width=2.5),
        marker=dict(size=6),
        hovertemplate=f"{scenario.capitalize()} DSCR: %{{y:.2f}}x<extra></extra>",
    ))

# Covenant floor and default trigger
fig_dscr.add_hline(y=TARGET_DSCR, line_dash="dash", line_color=C["covenant"], line_width=2,
                   annotation_text="Covenant floor (1.35x)", annotation_position="right")
fig_dscr.add_hline(y=1.0, line_dash="dot", line_color="#333", line_width=1.5,
                   annotation_text="Default trigger (1.0x)", annotation_position="right")

fig_dscr.update_layout(
    template="plotly_white",
    yaxis_title="DSCR (x)",
    xaxis_title="Year",
    xaxis=dict(dtick=1),
    yaxis=dict(range=[0, 2.2]),
    legend=dict(orientation="h", y=1.12),
    height=380, margin=dict(t=10, b=40, r=160),
)
st.plotly_chart(fig_dscr, use_container_width=True)
st.caption(
    "Debt is sculpted to hold base-case DSCR flat at 1.35x through 2039. "
    "Bear scenario DSCR (0.40x) falls far below covenant and default trigger -- "
    "quantifying the severity of the P90 generation + low capture stress."
)

# ---------------------------------------------------------------------------
# Amortization table (base case)
# ---------------------------------------------------------------------------
with st.expander("Base-Case Annual Cash Flow Detail"):
    detail = base.copy()
    for col in ["contracted_rev","merchant_rev","total_rev","total_opex","ebitda","cfads"]:
        detail[col] = (detail[col] / 1e6).round(2)
    detail["ds_m"]    = (ds_schedule / 1e6).round(2)
    detail["ebitda_margin"] = (detail["ebitda_margin"] * 100).round(1)
    show = detail[["year","aep_mwh","contracted_rev","merchant_rev",
                   "total_rev","total_opex","ebitda","ebitda_margin",
                   "cfads","ds_m","hedge_active"]].set_index("year")
    show.columns = ["AEP (MWh)","Contracted ($m)","Merchant ($m)",
                    "Revenue ($m)","OpEx ($m)","EBITDA ($m)","EBITDA %",
                    "CFADS ($m)","Debt Svc ($m)","Hedge Active"]
    show["AEP (MWh)"] = show["AEP (MWh)"].map("{:,.0f}".format)
    st.dataframe(show, use_container_width=True)
