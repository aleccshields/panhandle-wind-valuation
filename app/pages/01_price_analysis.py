"""
Page 1 -- ERCOT Price Analysis
LZ_NORTH vs HB_NORTH settlement prices, annual capture ratio trend,
and seasonal / diurnal price patterns.

Populated by: src/ercot_pipeline.py
  -> data/processed/ercot_rtm_prices.parquet
  -> data/processed/ercot_rtm_summary.csv
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import DATA_PROCESSED, ERCOT_HUB, ERCOT_ZONE

st.set_page_config(page_title="ERCOT Price Analysis", layout="wide")

PRICES_PATH  = DATA_PROCESSED / "ercot_rtm_prices.parquet"
SUMMARY_PATH = DATA_PROCESSED / "ercot_rtm_summary.csv"

COLORS = {"lz": "#1f77b4", "hb": "#ff7f0e", "capture": "#2ca02c"}

# ---------------------------------------------------------------------------
# Data availability gate
# ---------------------------------------------------------------------------
prices_ready  = PRICES_PATH.exists()
summary_ready = SUMMARY_PATH.exists()

st.title("ERCOT Price Analysis -- Capture Rate")
st.caption(f"Settlement zone: {ERCOT_ZONE}  |  Hub reference: {ERCOT_HUB}")

if not prices_ready:
    st.warning(
        "ERCOT RTM price data not yet loaded.",
        icon="⚠️",
    )
    st.markdown(
        """
        **How to populate this page:**

        ERCOT's historical RTM Settlement Point Prices are available from the
        [ERCOT Market Data portal](https://www.ercot.com/mp/data-products/data-product-details?id=NP6-788-ER).
        The portal is JavaScript-rendered and requires manual download.

        1. Navigate to the link above and download annual ZIP files for 2015-2024
        2. Rename each file to `ercot_rtm_YYYY.zip` (e.g. `ercot_rtm_2022.zip`)
        3. Place all ZIPs in `data/raw/`
        4. Run `python src/ercot_pipeline.py` from the repo root

        **What this page will show once data loads:**
        - Annual average LZ_NORTH vs HB_NORTH prices ($/MWh)
        - Capture ratio (LZ/HB) trend 2015-2024 — the dominant merchant revenue driver
        - Monthly price heatmap by year (seasonal pattern)
        - Negative-price hour count by year (curtailment risk)
        """
    )
    st.stop()

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600)
def load_summary() -> pd.DataFrame:
    return pd.read_csv(SUMMARY_PATH)

@st.cache_data(ttl=3600)
def load_prices() -> pd.DataFrame:
    return pd.read_parquet(PRICES_PATH)

summary = load_summary()
prices  = load_prices()

# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------
lz_col  = "lz_mean"
hb_col  = "hb_mean"
cap_col = "capture_ratio"

latest = summary.sort_values("year").iloc[-1]
avg_capture = summary[cap_col].mean()

c1, c2, c3, c4 = st.columns(4)
c1.metric(f"{ERCOT_ZONE} Avg (latest yr)", f"${latest[lz_col]:.1f}/MWh")
c2.metric(f"{ERCOT_HUB} Avg (latest yr)", f"${latest[hb_col]:.1f}/MWh")
c3.metric("Capture Ratio (latest yr)", f"{latest[cap_col]:.3f}")
c4.metric("10-yr Avg Capture", f"{avg_capture:.3f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Chart 1: Annual price comparison + capture ratio
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([3, 2])

with col_left:
    st.subheader("Annual Average Settlement Prices")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=summary["year"], y=summary[hb_col],
        name=ERCOT_HUB, marker_color=COLORS["hb"], opacity=0.8,
    ))
    fig.add_trace(go.Bar(
        x=summary["year"], y=summary[lz_col],
        name=ERCOT_ZONE, marker_color=COLORS["lz"], opacity=0.8,
    ))
    fig.update_layout(
        template="plotly_white", barmode="group",
        yaxis_title="$/MWh", xaxis_title="Year",
        legend=dict(orientation="h", y=1.1),
        height=380, margin=dict(t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader("Capture Ratio Trend")
    capture_vals = summary[cap_col].fillna(method="ffill")
    colors = ["#d62728" if v < 0.85 else "#2ca02c" for v in capture_vals]
    fig2 = go.Figure()
    fig2.add_trace(go.Bar(
        x=summary["year"], y=capture_vals,
        marker_color=colors, name="LZ/HB",
    ))
    fig2.add_hline(y=1.0, line_dash="dash", line_color="#888",
                   annotation_text="Parity (1.0)")
    fig2.add_hline(y=avg_capture, line_dash="dot", line_color=COLORS["capture"],
                   annotation_text=f"10-yr avg ({avg_capture:.3f})")
    fig2.update_layout(
        template="plotly_white",
        yaxis_title="LZ_NORTH / HB_NORTH",
        yaxis_range=[0.5, 1.1],
        xaxis_title="Year",
        height=380, margin=dict(t=20, b=40),
        showlegend=False,
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---------------------------------------------------------------------------
# Chart 2: Monthly seasonal heatmap
# ---------------------------------------------------------------------------
st.subheader("Monthly Average Price Heatmap -- LZ_NORTH ($/MWh)")

if "month" in prices.columns and "price" in prices.columns:
    lz_prices = prices[prices["settlement_point"] == ERCOT_ZONE].copy()
    monthly_avg = (
        lz_prices.groupby(["year", "month"])["price"]
        .mean()
        .reset_index()
    )
    pivot = monthly_avg.pivot(index="year", columns="month", values="price")
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    fig3 = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[month_labels[m-1] for m in pivot.columns],
        y=pivot.index.tolist(),
        colorscale="RdYlGn",
        text=[[f"${v:.1f}" if not pd.isna(v) else "" for v in row]
              for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="$/MWh"),
    ))
    fig3.update_layout(
        template="plotly_white",
        xaxis_title="Month", yaxis_title="Year",
        height=320, margin=dict(t=10, b=40),
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---------------------------------------------------------------------------
# Chart 3: Negative price hours
# ---------------------------------------------------------------------------
if "lz_neg_hours" in summary.columns:
    st.subheader("Negative Price Hours per Year -- LZ_NORTH")
    fig4 = go.Figure(go.Bar(
        x=summary["year"], y=summary["lz_neg_hours"],
        marker_color="#d62728", name="Negative hours",
    ))
    fig4.update_layout(
        template="plotly_white",
        yaxis_title="Hours", xaxis_title="Year",
        height=280, margin=dict(t=10, b=40),
        showlegend=False,
    )
    st.caption(
        "Negative price hours indicate periods when wind supply exceeded load + "
        "transmission capacity. Curtailment becomes economically rational below $0/MWh."
    )
    st.plotly_chart(fig4, use_container_width=True)

# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------
st.subheader("Annual Statistics Table")
display_cols = {
    "year": "Year",
    lz_col: f"{ERCOT_ZONE} Avg ($/MWh)",
    hb_col: f"{ERCOT_HUB} Avg ($/MWh)",
    cap_col: "Capture Ratio",
    "lz_neg_hours": "LZ Neg Hours",
    "lz_neg_pct": "LZ Neg %",
}
show_cols = [c for c in display_cols if c in summary.columns]
disp = summary[show_cols].rename(columns=display_cols).set_index("Year")
disp = disp.round(3)
st.dataframe(disp, use_container_width=True)
