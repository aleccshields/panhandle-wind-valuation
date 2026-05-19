"""
Shared constants and helper utilities for the Panhandle Wind valuation project.

All pipeline modules import from here to ensure consistent asset identifiers,
directory paths, and formatting conventions across the codebase.
"""

from pathlib import Path


# ── Directory roots ────────────────────────────────────────────────────────────
# Paths are relative to the repo root; scripts should call ensure_dirs() on
# startup so the gitignored data folders exist before any I/O is attempted.
DATA_RAW = Path("data/raw")
DATA_PROCESSED = Path("data/processed")
OUTPUTS_IC_MEMO = Path("outputs/ic_memo")


# ── Asset identifiers ──────────────────────────────────────────────────────────
PLANT_NAME = "Panhandle Wind"
ERCOT_ZONE = "LZ_NORTH"       # Load Zone — settlement price for wind in the Panhandle
ERCOT_HUB = "HB_NORTH"        # Hub price — reference for capture rate calculation
COD_YEAR = 2014                # Commercial operation date
CAPACITY_MW = 400              # Nameplate capacity (Phases 1 + 2 combined)

# ERCOT price node used in DAM/RT settlement data downloads
ERCOT_SETTLEMENT_POINT = "LZ_NORTH"


# ── Financial model defaults ───────────────────────────────────────────────────
# These are starting assumptions; the Streamlit app exposes sliders to override.
DISCOUNT_RATE = 0.08           # Unlevered WACC (8%)
TERMINAL_GROWTH_RATE = 0.0     # No terminal growth for merchant wind (conservative)
OPEX_PER_MWH = 12.0            # $/MWh all-in O&M (typical utility-scale wind)
PROJECT_LIFE_YEARS = 30        # Standard wind project financing horizon


def ensure_dirs() -> None:
    """Create all data directories if they don't exist.

    Call this at the top of every pipeline script before reading or writing
    any files. The directories are gitignored, so they won't exist in a
    fresh clone.
    """
    for directory in (DATA_RAW, DATA_PROCESSED, OUTPUTS_IC_MEMO):
        directory.mkdir(parents=True, exist_ok=True)


def format_currency(value: float, unit: str = "MWh") -> str:
    """Format a float as a dollar-denominated string.

    Args:
        value: Numeric value to format.
        unit:  'MWh' produces '$XX.X/MWh'; 'M' produces '$XX.XM'.

    Examples:
        format_currency(34.567)     → '$34.6/MWh'
        format_currency(245.1, 'M') → '$245.1M'
    """
    if unit == "MWh":
        return f"${value:,.1f}/MWh"
    if unit == "M":
        return f"${value:,.1f}M"
    return f"${value:,.1f}"


def format_pct(value: float) -> str:
    """Format a decimal fraction as a percentage string.

    Args:
        value: Decimal fraction (e.g., 0.823 → '82.3%').
    """
    return f"{value * 100:.1f}%"
