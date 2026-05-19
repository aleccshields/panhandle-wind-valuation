"""
Page 2 -- EIA Generation & Capacity Factors
10-year actual generation history (EIA Form 923), P50/P90 exceedance bands,
monthly capacity factor heatmap, and NREL WTK resource crosscheck.

Populated by:
  src/eia_pipeline.py  -> data/processed/eia_generation.parquet
  src/nrel_pipeline.py -> data/processed/nrel_wind_resource.parquet
                       -> data/processed/nrel_p50_p90_summary.csv
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import CAPACITY_MW, DATA_PROCESSED, PLANT_NAME


GEN_PATH  = DATA_PROCESSED / "eia_generation.csv"
NREL_PATH = DATA_PROCESSED / "nrel_wind_resource.csv"
P50_PATH  = DATA_PROCESSED / "nrel_p50_p90_summary.csv"

NAMEPLATE_MW = 399.7   # EIA 860 confirmed (Phases 1 + 2)
COLORS = {"bar": "#1f77b4", "p50": "#2ca02c", "p90": "#d62728", "nrel": "#9467bd"}

missing = [p.name for p in (GEN_PATH, P50_PATH) if not p.exists()]
if missing:
    st.title("Generation & Capacity Factors")
    st.warning(
        f"Missing processed data: {', '.join(missing)}. "
        "Run `python src/eia_pipeline.py` and `python src/nrel_pipeline.py` first.",
        icon="⚠️",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_eia() -> pd.DataFrame:
    return pd.read_csv(GEN_PATH)

@st.cache_data(ttl=3600)
def load_nrel_summary() -> pd.DataFrame:
    return pd.read_csv(P50_PATH, index_col="metric")

eia   = load_eia()
nrel  = load_nrel_summary()

p50_mwh   = float(nrel.loc["P50_AEP_MWh", "value"])
p90_mwh   = float(nrel.loc["P90_AEP_MWh", "value"])
p50_cf    = float(nrel.loc["P50_CF",      "value"])
p90_cf    = float(nrel.loc["P90_CF",      "value"])
ratio     = float(nrel.loc["P90_P50_ratio","value"])
eia_avg_cf = float(nrel.loc["EIA_10yr_avg_CF","value"])
nrel_p50_mwh = float(nrel.loc["NREL_P50_AEP_MWh","value"])

# Annual aggregates
annual = (
    eia.groupby("year")["net_gen_mwh"]
    .sum()
    .reset_index()
    .rename(columns={"net_gen_mwh": "annual_mwh"})
)
annual["cf"] = annual["annual_mwh"] / (NAMEPLATE_MW * 8760)

# Monthly aggregates across both plants
monthly = (
    eia.groupby(["year", "month"])
    .agg(net_gen=("net_gen_mwh", "sum"), hours=("hours_in_month", "first"))
    .reset_index()
)
monthly["cf"] = monthly["net_gen"] / (NAMEPLATE_MW * monthly["hours"])

# ---------------------------------------------------------------------------
# Page header
# ---------------------------------------------------------------------------
st.title("Generation & Capacity Factors")
st.caption(f"{PLANT_NAME}  |  {NAMEPLATE_MW} MW nameplate  |  EIA Form 923, 2015-2024")

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("10-yr Avg CF",       f"{eia_avg_cf:.1%}")
c2.metric("P50 AEP (empirical)",f"{p50_mwh/1e6:.3f} TWh/yr")
c3.metric("P90 AEP (empirical)",f"{p90_mwh/1e6:.3f} TWh/yr")
c4.metric("P90/P50 Ratio",      f"{ratio:.3f}")
c5.metric("NREL WTK P50 AEP",   f"{nrel_p50_mwh/1e6:.3f} TWh/yr")

st.markdown("---")

# ---------------------------------------------------------------------------
# Chart 1: Annual generation bar + P50/P90 reference lines
# ---------------------------------------------------------------------------
st.subheader("Annual Net Generation vs P50/P90 Thresholds")

bar_colors = [
    "#d62728" if mwh < p90_mwh else "#1f77b4"
    for mwh in annual["annual_mwh"]
]

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=annual["year"],
    y=annual["annual_mwh"] / 1e6,
    marker_color=bar_colors,
    name="Actual AEP",
    hovertemplate="Year: %{x}<br>AEP: %{y:.3f} TWh<br>CF: " +
                  annual["cf"].map(lambda v: f"{v:.1%}").tolist()[0] +
                  "<extra></extra>",
))
fig1.add_hline(
    y=p50_mwh / 1e6, line_dash="dash", line_color=COLORS["p50"], line_width=2,
    annotation_text=f"P50 ({p50_mwh/1e6:.3f} TWh)", annotation_position="right",
)
fig1.add_hline(
    y=p90_mwh / 1e6, line_dash="dash", line_color=COLORS["p90"], line_width=2,
    annotation_text=f"P90 ({p90_mwh/1e6:.3f} TWh)", annotation_position="right",
)
fig1.add_hline(
    y=nrel_p50_mwh / 1e6, line_dash="dot", line_color=COLORS["nrel"], line_width=1.5,
    annotation_text=f"NREL P50 ({nrel_p50_mwh/1e6:.3f} TWh)", annotation_position="right",
)

# Custom hover with CF
hover_text = [
    f"Year: {row.year}<br>AEP: {row.annual_mwh/1e6:.3f} TWh<br>CF: {row.cf:.1%}"
    for _, row in annual.iterrows()
]
fig1.update_traces(hovertext=hover_text, hoverinfo="text")

fig1.update_layout(
    template="plotly_white",
    yaxis_title="Annual Generation (TWh)",
    xaxis_title="Year",
    xaxis=dict(dtick=1),
    height=400, margin=dict(t=10, b=40, r=140),
    showlegend=False,
)
fig1.add_annotation(
    text="Red bars = below P90 threshold",
    xref="paper", yref="paper", x=0.01, y=0.97,
    showarrow=False, font=dict(size=11, color="#d62728"),
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Chart 2: Monthly CF heatmap
# ---------------------------------------------------------------------------
st.subheader("Monthly Capacity Factor Heatmap")

pivot = monthly.pivot(index="year", columns="month", values="cf")
month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]

text_vals = [
    [f"{v:.0%}" if not pd.isna(v) else "" for v in row]
    for row in pivot.values
]

fig2 = go.Figure(go.Heatmap(
    z=pivot.values * 100,
    x=[month_labels[m - 1] for m in pivot.columns],
    y=pivot.index.tolist(),
    colorscale="RdYlGn",
    zmin=10, zmax=70,
    text=text_vals,
    texttemplate="%{text}",
    textfont=dict(size=10),
    colorbar=dict(title="CF (%)", ticksuffix="%"),
))
fig2.update_layout(
    template="plotly_white",
    xaxis_title="Month", yaxis_title="Year",
    height=330, margin=dict(t=10, b=40),
)
st.plotly_chart(fig2, use_container_width=True)

st.caption(
    "Green = high capacity factor months (spring). "
    "Red = low CF months (summer heat reduces air density; ERCOT curtailment). "
    "Dark red = below P90 monthly equivalent."
)

st.markdown("---")

# ---------------------------------------------------------------------------
# Chart 3: Annual CF trend line
# ---------------------------------------------------------------------------
col_l, col_r = st.columns([3, 2])

with col_l:
    st.subheader("Annual Capacity Factor Trend")
    fig3 = go.Figure()
    fig3.add_trace(go.Scatter(
        x=annual["year"], y=annual["cf"] * 100,
        mode="lines+markers",
        line=dict(color=COLORS["bar"], width=2),
        marker=dict(size=8),
        name="Actual CF",
        hovertemplate="Year: %{x}<br>CF: %{y:.1f}%<extra></extra>",
    ))
    fig3.add_hline(y=p50_cf * 100, line_dash="dash", line_color=COLORS["p50"],
                   annotation_text=f"P50 CF ({p50_cf:.1%})", annotation_position="right")
    fig3.add_hline(y=p90_cf * 100, line_dash="dash", line_color=COLORS["p90"],
                   annotation_text=f"P90 CF ({p90_cf:.1%})", annotation_position="right")
    fig3.add_hline(y=eia_avg_cf * 100, line_dash="dot", line_color="#888",
                   annotation_text=f"10-yr avg ({eia_avg_cf:.1%})", annotation_position="right")
    fig3.update_layout(
        template="plotly_white",
        yaxis_title="Capacity Factor (%)",
        xaxis=dict(dtick=1), yaxis=dict(ticksuffix="%"),
        height=340, margin=dict(t=10, b=40, r=120),
        showlegend=False,
    )
    st.plotly_chart(fig3, use_container_width=True)

with col_r:
    st.subheader("EIA vs NREL Crosscheck")
    st.markdown(
        """
        | Source | AEP (TWh/yr) | CF |
        |--------|--------------|----|
        | EIA 923 P50 (empirical) | {:.3f} | {:.1%} |
        | EIA 923 P90 (empirical) | {:.3f} | {:.1%} |
        | NREL WTK P50 | {:.3f} | {:.1%} |
        | EIA 10-yr avg | {:.3f} | {:.1%} |
        """.format(
            p50_mwh / 1e6, p50_cf,
            p90_mwh / 1e6, p90_cf,
            nrel_p50_mwh / 1e6, nrel_p50_mwh / (NAMEPLATE_MW * 8760),
            p50_mwh * eia_avg_cf / p50_cf / 1e6, eia_avg_cf,
        )
    )
    delta_pct = (nrel_p50_mwh - p50_mwh) / p50_mwh
    st.metric(
        "NREL vs EIA P50 Delta",
        f"{delta_pct:+.1%}",
        help="Within ±5% confirms the NREL resource model is reliable for this site.",
    )
    st.markdown(
        f"""
        **P90 haircut:** {(1 - ratio):.1%} below P50
        **P90/P50 ratio:** {ratio:.3f}

        Lenders size debt on P90 cash flows because wind is the sole
        debt service source and P50 means shortfall 50% of the time.
        """
    )

# ---------------------------------------------------------------------------
# Annual generation table
# ---------------------------------------------------------------------------
st.markdown("---")
st.subheader("Annual Generation Detail")
disp = annual.copy()
disp["annual_gwh"] = disp["annual_mwh"] / 1e3
disp["cf_pct"]     = disp["cf"] * 100
disp["vs_p50"]     = (disp["annual_mwh"] - p50_mwh) / p50_mwh * 100
disp["vs_p90"]     = (disp["annual_mwh"] - p90_mwh) / p90_mwh * 100
disp_show = disp[["year","annual_gwh","cf_pct","vs_p50","vs_p90"]].set_index("year")
disp_show.columns = ["AEP (GWh)", "CF (%)", "vs P50 (%)", "vs P90 (%)"]
disp_show = disp_show.round(1)
st.dataframe(
    disp_show.style.applymap(
        lambda v: "color: #d62728" if isinstance(v, float) and v < 0 else "",
        subset=["vs P50 (%)", "vs P90 (%)"]
    ),
    use_container_width=True,
)
