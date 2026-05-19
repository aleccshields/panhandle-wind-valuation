"""
Page 4 -- Investment Committee Memo
Renders the Claude-generated IC memo for Panhandle Wind. Memos are stored
as markdown files in outputs/ic_memo/ and committed to the repo as the key
qualitative portfolio artifact.
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import OUTPUTS_IC_MEMO, PLANT_NAME

st.set_page_config(page_title="IC Memo", layout="wide")

st.title("Investment Committee Memorandum")
st.caption(f"{PLANT_NAME}  |  Panhandle Wind Phases 1 & 2  |  Carson County, TX")

# ---------------------------------------------------------------------------
# Discover available memos
# ---------------------------------------------------------------------------
memo_files = sorted(
    OUTPUTS_IC_MEMO.glob("*.md"),
    key=lambda p: p.stat().st_mtime,
    reverse=True,
)

if not memo_files:
    st.warning(
        "No IC memos found in `outputs/ic_memo/`. "
        "Run `python src/financial_model.py` to generate the valuation outputs, "
        "then place a markdown memo in `outputs/ic_memo/`.",
        icon="⚠️",
    )
    st.stop()

# ---------------------------------------------------------------------------
# Memo selector (if multiple versions exist)
# ---------------------------------------------------------------------------
col_sel, col_meta = st.columns([3, 1])

with col_sel:
    if len(memo_files) > 1:
        selected = st.selectbox(
            "Memo version",
            options=memo_files,
            format_func=lambda p: p.stem.replace("_", " ").title(),
        )
    else:
        selected = memo_files[0]
        st.markdown(f"**Document:** `{selected.name}`")

with col_meta:
    import datetime
    mtime = datetime.datetime.fromtimestamp(selected.stat().st_mtime)
    st.metric("Last modified", mtime.strftime("%Y-%m-%d"))
    size_kb = selected.stat().st_size / 1024
    st.metric("Size", f"{size_kb:.1f} KB")

st.markdown("---")

# ---------------------------------------------------------------------------
# Render memo
# ---------------------------------------------------------------------------
memo_text = selected.read_text(encoding="utf-8")
st.markdown(memo_text)

# ---------------------------------------------------------------------------
# Download button
# ---------------------------------------------------------------------------
st.markdown("---")
st.download_button(
    label="Download memo as Markdown",
    data=memo_text.encode("utf-8"),
    file_name=selected.name,
    mime="text/markdown",
)
st.caption(
    "This memo synthesizes outputs from Layers 1-4 (ERCOT prices, EIA generation, "
    "NREL wind resource, DCF financial model) into a structured investment narrative."
)
