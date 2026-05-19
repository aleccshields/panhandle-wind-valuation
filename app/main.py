"""
Panhandle Wind — Asset Valuation Engine
Main Streamlit application shell.

This module configures the page layout, renders the sidebar with data source
status indicators, and displays the landing page. Individual analysis pages
live in app/pages/ and are loaded automatically by Streamlit's multi-page
routing.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure src/ is importable regardless of working directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    CAPACITY_MW,
    COD_YEAR,
    ERCOT_ZONE,
    PLANT_NAME,
)


# ── Page configuration ─────────────────────────────────────────────────────────
# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("💨 Panhandle Wind")
    st.caption("Asset Valuation Engine")

    st.markdown("---")
    st.markdown(
        "A practitioner-grade valuation model for a ~400 MW wind portfolio "
        "in Carson County, TX, built entirely from public data."
    )

    st.markdown("---")
    st.subheader("Data Layer Status")

    # Status indicators — these will be updated dynamically by each pipeline
    # module once the processed parquet files exist on disk.
    processed_dir = Path(__file__).parent.parent / "data" / "processed"

    def _status(label: str, filename: str) -> None:
        """Render a green checkmark or red dot based on file existence."""
        exists = (processed_dir / filename).exists()
        icon = "✅" if exists else "🔴"
        st.markdown(f"{icon} {label}")

    _status("ERCOT Prices (LZ_NORTH)", "ercot_rtm_prices.csv")
    _status("EIA Generation (860/923)", "eia_generation.csv")
    _status("NREL Wind Resource", "nrel_wind_resource.csv")
    _status("Financial Model", "dcf_outputs_base.csv")

    st.markdown("---")
    st.caption("Built with public data · Not investment advice")


# ── Landing page ───────────────────────────────────────────────────────────────
st.title("Panhandle Wind — Asset Valuation Engine")

st.markdown(
    """
    ### The Financial Question

    > **Given Panhandle Wind's historical generation profile and merchant exposure,
    > what is the asset worth under various power price scenarios — and what is the
    > dominant value driver?**

    Use the sidebar to navigate between analysis layers. Each page builds on the one
    before it: raw price and generation data feed a capture rate analysis, which feeds
    the DCF model, which feeds the IC memo.
    """
)

st.markdown("---")

# ── Asset summary card ─────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Asset", PLANT_NAME)

with col2:
    st.metric("Capacity", f"{CAPACITY_MW} MW")

with col3:
    st.metric("COD", str(COD_YEAR))

with col4:
    st.metric("ERCOT Zone", ERCOT_ZONE)

st.markdown("---")

col_left, col_right = st.columns([2, 1])

with col_left:
    st.subheader("About This Project")
    st.markdown(
        """
        This application demonstrates energy finance analytical methods using
        **Panhandle Wind** (Phases 1 & 2) as a case study. The asset is a
        ~400 MW wind portfolio located in Carson County, Texas, operating in
        the ERCOT LZ_NORTH pricing zone.

        **Analysis layers:**
        1. **ERCOT Price Analysis** — Historical DAM and real-time settlement prices,
           wind-weighted capture rates vs. hub prices, seasonal and diurnal patterns
        2. **EIA Generation** — Reported net generation (MWh), implied capacity factors,
           comparison to NREL P50 baseline
        3. **DCF Valuation** — Unlevered free cash flow model, P50/P90 scenario
           outputs, IRR and NPV sensitivity tables, DSCR for debt sizing
        4. **IC Memo** — Claude-generated investment committee memo synthesizing
           the quantitative outputs into a structured recommendation
        """
    )

with col_right:
    st.subheader("Quick Reference")
    st.markdown(
        f"""
        | Parameter | Value |
        |-----------|-------|
        | Plant | {PLANT_NAME} |
        | Location | Carson County, TX |
        | Capacity | {CAPACITY_MW} MW |
        | COD | {COD_YEAR} |
        | ISO | ERCOT |
        | Zone | LZ_NORTH |
        | Hub | HB_NORTH |
        | Fuel | Wind |
        """
    )

st.info(
    "**Portfolio project** — All data sourced from public APIs and regulatory filings "
    "(ERCOT, EIA, NREL, SEC EDGAR). Run the pipeline scripts in `src/` to populate "
    "the data layers before exploring the analysis pages.",
    icon="ℹ️",
)
