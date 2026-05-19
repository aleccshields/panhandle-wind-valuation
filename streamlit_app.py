"""
Root-level Streamlit entry point for Streamlit Community Cloud.

Uses st.navigation() to explicitly register pages so they are found
regardless of where the page files live relative to this script.
"""

import sys
from pathlib import Path

import streamlit as st

# Ensure src/ and app/ are importable from every page
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Panhandle Wind -- Valuation Engine",
    page_icon=":wind_face:",
    layout="wide",
    initial_sidebar_state="expanded",
)

pg = st.navigation(
    [
        st.Page("app/main.py",                        title="Overview",             icon=":material/home:"),
        st.Page("app/pages/01_price_analysis.py",     title="ERCOT Price Analysis", icon=":material/price_change:"),
        st.Page("app/pages/02_generation.py",         title="Generation & CFs",     icon=":material/bolt:"),
        st.Page("app/pages/03_valuation.py",          title="DCF Valuation",        icon=":material/bar_chart:"),
        st.Page("app/pages/04_ic_memo.py",            title="IC Memo",              icon=":material/description:"),
    ]
)
pg.run()
