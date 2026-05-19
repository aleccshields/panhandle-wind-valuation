"""
Layer 2 -- EIA Form 860 / 923 Generation Pipeline

Identifies Panhandle Wind's EIA plant IDs from Form 860 (capacity registry),
downloads Form 923 monthly net generation for those plants across 2015-2024,
computes capacity factors, and writes to data/processed/eia_generation.parquet.

Financial context
-----------------
EIA-reported generation is the ground truth for capacity factor analysis.
Capacity factor (CF) is the ratio of actual MWh to maximum possible MWh:

    CF = Net Generation (MWh) / (Nameplate MW * Hours in Period)

CF is the dominant revenue driver in the DCF:
    Revenue = CF * Capacity_MW * 8760 * Capture_Price ($/MWh)

A 1% CF change on a 400 MW plant at $30/MWh changes annual revenue by ~$1.1M.
The 10-year CF distribution vs. the NREL P50 baseline (Layer 3) shows whether
Panhandle Wind has performed to expectation -- critical for IC memo risk framing.

Data sources (direct HTTP download, no browser required):
    EIA Form 860: https://www.eia.gov/electricity/data/eia860/
    EIA Form 923: https://www.eia.gov/electricity/data/eia923/

Run from repo root:
    python src/eia_pipeline.py
"""

import logging
import re
import sys
import zipfile
from calendar import monthrange
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

# -- Repo root on sys.path so 'src.utils' is importable when run as a script ---
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    CAPACITY_MW,
    DATA_PROCESSED,
    DATA_RAW,
    PLANT_NAME,
    ensure_dirs,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# -- Config --------------------------------------------------------------------
YEARS = list(range(2015, 2025))       # 2015-2024 (post-COD)
EIA860_REFERENCE_YEAR = 2023          # use this year's 860 to identify plant IDs

OUTPUT_GENERATION = DATA_PROCESSED / "eia_generation.parquet"
OUTPUT_SUMMARY = DATA_PROCESSED / "eia_generation_summary.csv"

# Plant search terms -- search by name + geography rather than hard-coding IDs
# so the pipeline self-discovers if PEGI reregistered plants under a new entity.
SEARCH_NAME = "panhandle"
SEARCH_STATE = "TX"
SEARCH_COUNTY = "Carson"
WIND_FUEL_CODE = "WND"
WIND_PRIME_MOVER = "WT"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
}

# Month name -> integer lookup; includes both full names and 3-letter abbreviations.
# EIA 923 real files use full names ('Netgen\nJanuary'); synthetic/older files
# may use abbreviations ('Netgen Jan'). Both must match.
_MONTH_NUM: dict[str, int] = {
    "january": 1,  "february": 2,  "march": 3,     "april": 4,
    "may": 5,       "june": 6,      "july": 7,      "august": 8,
    "september": 9, "october": 10,  "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4,
    "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


# -- URL builders --------------------------------------------------------------

def _eia860_url(year: int) -> str:
    """EIA Form 860 annual ZIP.

    EIA keeps the current year at xls/ and moves prior years to archive/xls/.
    Both paths are stable static URLs requiring no authentication.
    """
    base = "https://www.eia.gov/electricity/data/eia860"
    # 2024 is the current release; all prior years live under archive/
    if year >= 2024:
        return f"{base}/xls/eia860{year}.zip"
    return f"{base}/archive/xls/eia860{year}.zip"


def _eia923_url(year: int) -> str:
    """EIA Form 923 annual ZIP.

    All finalized years (prior to the current release year) live under archive/.
    Released approximately 5-6 months after year-end.
    """
    base = "https://www.eia.gov/electricity/data/eia923"
    return f"{base}/archive/xls/f923_{year}.zip"


def _raw860_path(year: int) -> Path:
    return DATA_RAW / f"eia860_{year}.zip"


def _raw923_path(year: int) -> Path:
    return DATA_RAW / f"eia923_{year}.zip"


# -- Download helpers ----------------------------------------------------------

def _download_file(url: str, dest: Path, label: str) -> bool:
    """Download a file to dest with streaming and progress logging.

    Args:
        url:   Direct download URL.
        dest:  Local destination path in data/raw/.
        label: Human-readable label for log messages.

    Returns:
        True on success, False on any HTTP or I/O error.
    """
    log.info("Downloading %s ...", label)
    try:
        with requests.get(url, headers=_HEADERS, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    fh.write(chunk)
        size_mb = dest.stat().st_size / 1e6
        log.info("  Saved %s (%.1f MB)", dest.name, size_mb)
        return True
    except requests.RequestException as exc:
        log.warning("  Download failed for %s: %s", label, exc)
        if dest.exists():
            dest.unlink()
        return False


def _ensure_eia860(year: int) -> bool:
    """Download EIA 860 ZIP if not already cached. Returns True if available."""
    dest = _raw860_path(year)
    if dest.exists():
        log.info("EIA 860 %d already cached", year)
        return True
    return _download_file(_eia860_url(year), dest, f"EIA 860 ({year})")


def _ensure_eia923(year: int) -> bool:
    """Download EIA 923 ZIP if not already cached. Returns True if available."""
    dest = _raw923_path(year)
    if dest.exists():
        return True
    return _download_file(_eia923_url(year), dest, f"EIA 923 ({year})")


# -- EIA Form 860 parsing ------------------------------------------------------

def _find_wind_xlsx_in_zip(zip_path: Path) -> Optional[str]:
    """Locate the wind generator Excel workbook inside an EIA 860 ZIP.

    EIA 860 ZIPs contain multiple workbooks (one per technology/schedule).
    The wind file is named '3_3_Wind_Y{YYYY}.xlsx' in recent years.
    """
    # EIA has renumbered the wind schedule across years (3_2, 3_3, etc.).
    # Match any xlsx whose name contains 'wind' to stay resilient to renumbering.
    with zipfile.ZipFile(zip_path, "r") as zf:
        candidates = [
            name for name in zf.namelist()
            if re.search(r"wind", name, re.IGNORECASE) and name.endswith(".xlsx")
        ]
    if not candidates:
        return None
    # Prefer the file with 'Wind_Y' in the name (the generator-level schedule)
    for name in candidates:
        if re.search(r"wind_y\d{4}", name, re.IGNORECASE):
            return name
    return candidates[0]


def _normalize_860_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize EIA 860 wind sheet column names to a consistent schema.

    EIA has used both 'Plant Code' and 'Plant Id' across years; we map both
    to 'plant_id' so downstream code doesn't need to branch.
    """
    col_map = {}
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_").replace("(", "").replace(")", "")
        if key in ("plant_code", "plant_id", "plant_code_eia"):
            col_map[col] = "plant_id"
        elif key in ("plant_name",):
            col_map[col] = "plant_name"
        elif key in ("state",):
            col_map[col] = "state"
        elif key in ("county",):
            col_map[col] = "county"
        elif key in ("generator_id",):
            col_map[col] = "generator_id"
        elif key in ("nameplate_capacity_mw", "nameplate_capacity"):
            col_map[col] = "capacity_mw"
        elif key in ("status",):
            col_map[col] = "status"
        elif key in ("operating_year",):
            col_map[col] = "operating_year"
    return df.rename(columns=col_map)


def parse_eia860_wind(zip_path: Path) -> pd.DataFrame:
    """Parse the EIA 860 wind generator sheet and return all TX wind generators.

    The 860 gives us plant IDs, nameplate capacity, and geographic attributes.
    We use it to identify Panhandle Wind's EIA plant IDs and per-plant capacity,
    which feed into capacity factor calculation against the 923 generation data.

    Returns:
        DataFrame with columns [plant_id, plant_name, state, county,
        generator_id, capacity_mw, status, operating_year].
    """
    wind_file = _find_wind_xlsx_in_zip(zip_path)
    if wind_file is None:
        log.error("No wind workbook found in %s", zip_path.name)
        return pd.DataFrame()

    log.info("  Parsing %s from %s", wind_file, zip_path.name)

    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(wind_file) as f:
            # EIA 860 wind sheet: row 0 is a title row, row 1 is the column header
            df = pd.read_excel(f, sheet_name=0, header=1, dtype=str)

    df = _normalize_860_columns(df)

    required = {"plant_id", "plant_name", "state"}
    if not required.issubset(df.columns):
        log.warning(
            "EIA 860 wind sheet missing expected columns. Found: %s", list(df.columns)[:10]
        )
        return pd.DataFrame()

    # Filter to Texas -- reduces dataset from ~70k rows to ~4k
    tx = df[df["state"].str.strip().str.upper() == "TX"].copy()
    log.info("  TX wind generators: %d rows", len(tx))

    if "capacity_mw" in tx.columns:
        tx["capacity_mw"] = pd.to_numeric(tx["capacity_mw"], errors="coerce")
    if "plant_id" in tx.columns:
        tx["plant_id"] = pd.to_numeric(tx["plant_id"], errors="coerce").astype("Int64")
    if "operating_year" in tx.columns:
        tx["operating_year"] = pd.to_numeric(tx["operating_year"], errors="coerce").astype("Int64")

    return tx.dropna(subset=["plant_id"])


def identify_panhandle_plants(eia860_df: pd.DataFrame) -> tuple[list[int], pd.DataFrame]:
    """Search EIA 860 for Panhandle Wind plant IDs using name + geography.

    We match on plant name (contains 'panhandle'), state (TX), and county
    (Carson). Searching by name rather than hard-coding IDs makes the pipeline
    resilient to ownership restructurings (e.g., PEGI -> new operator).

    Returns:
        (plant_id_list, filtered_DataFrame) -- the DataFrame is the full
        generator-level inventory for the matched plants.
    """
    # Primary match: name contains 'panhandle' AND county is 'Carson'.
    # Using AND (not OR) is critical -- the Panhandle TX area has dozens of
    # wind farms; OR would pull in every Carson County plant (Majestic, Route 66,
    # Grandview, etc.) and inflate totals well beyond the 400 MW portfolio.
    # The two Pattern Panhandle plants (IDs 58242 + 58720, ~400 MW combined) are
    # the only wind farms that satisfy both conditions simultaneously.
    name_mask = eia860_df["plant_name"].str.lower().str.contains(SEARCH_NAME, na=False)
    state_mask = eia860_df["state"].str.strip().str.upper() == SEARCH_STATE

    if "county" in eia860_df.columns:
        county_mask = eia860_df["county"].str.lower().str.contains(
            SEARCH_COUNTY.lower(), na=False
        )
        mask = name_mask & county_mask & state_mask
    else:
        mask = name_mask & state_mask

    matched = eia860_df[mask].copy()

    # Fallback: if AND is too narrow (e.g. after PEGI ownership change renames plants),
    # widen to name-only match and warn so the operator can verify.
    if matched.empty:
        log.warning(
            "AND search returned no results -- falling back to name-only match. "
            "Verify plant list matches the target asset."
        )
        matched = eia860_df[name_mask & state_mask].copy()

    if matched.empty:
        log.warning(
            "No plants found matching name='%s', state='%s', county='%s'",
            SEARCH_NAME, SEARCH_STATE, SEARCH_COUNTY,
        )
        return [], pd.DataFrame()

    plant_ids = sorted(matched["plant_id"].dropna().unique().tolist())
    log.info("Identified %d Panhandle Wind plant(s): IDs %s", len(plant_ids), plant_ids)

    # Log the plant inventory for verification
    for pid in plant_ids:
        subset = matched[matched["plant_id"] == pid]
        name = subset["plant_name"].iloc[0] if not subset.empty else "?"
        total_mw = subset["capacity_mw"].sum() if "capacity_mw" in subset.columns else float("nan")
        gen_count = len(subset)
        log.info("  Plant %d: %s -- %d generators, %.1f MW nameplate", pid, name, gen_count, total_mw)

    return [int(p) for p in plant_ids], matched


# -- EIA Form 923 parsing ------------------------------------------------------

def _find_923_generation_sheet(xl: pd.ExcelFile) -> Optional[str]:
    """Identify the generation data sheet inside an EIA 923 workbook.

    EIA has used several sheet names for Schedule 2 generation data.
    We try the most common variants in order of prevalence.
    """
    candidates = [
        "Page 1 Generation and Fuel Data",
        "Page 1 Gen and Fuel Data",
        "Generation and Fuel Data",
        "Gen and Fuel Data",
    ]
    for candidate in candidates:
        if candidate in xl.sheet_names:
            return candidate

    # Fall back to first sheet containing 'generation' in the name
    for name in xl.sheet_names:
        if "generation" in name.lower() or "gen" in name.lower():
            return name

    log.warning("Could not identify generation sheet. Sheets: %s", xl.sheet_names)
    return None


def _find_month_columns(df: pd.DataFrame) -> dict[int, str]:
    """Scan DataFrame columns for monthly net generation values.

    EIA 923 real-data column names use full month names separated by a newline:
      'Netgen\\nJanuary', 'Netgen\\nFebruary', ...  (current format, all years verified)

    Older or synthetic files may use 3-letter abbreviations:
      'Netgen Jan', 'Netgen Feb', ...

    We normalize column names (replace \\n with space, lowercase) and then check:
      1. The column contains 'netgen' or 'net gen' or 'net generation' (excludes
         fuel quantity, MMBtu, and electric quantity columns that also carry month names)
      2. The column contains a known month name (full or abbreviated)

    Returns:
        {month_number: column_name} for each month found (1=Jan, 12=Dec).
        Stops at the first matching column per month so earlier (netgen) columns
        win over later (total/annual) columns if both match.
    """
    found: dict[int, str] = {}

    for col in df.columns:
        # Normalize: lowercase, replace embedded newlines and underscores with space
        col_norm = str(col).lower().replace("\n", " ").replace("_", " ").strip()

        # Must be a net generation column -- not fuel quantity, MMBtu, or total annual
        is_netgen = any(kw in col_norm for kw in ("netgen", "net gen", "net generation"))
        if not is_netgen:
            continue

        # Identify which month
        for month_str, num in _MONTH_NUM.items():
            # Use word boundary so 'mar' doesn't match inside 'march' for abbrevs,
            # but full-name keys ('march', 'january', etc.) match as substrings.
            if len(month_str) >= 4:
                # Full name: simple containment is safe (no ambiguity)
                if month_str in col_norm and num not in found:
                    found[num] = col
                    break
            else:
                # 3-letter abbreviation: require word boundary to avoid 'jun' in 'june'
                if re.search(rf"\b{month_str}\b", col_norm) and num not in found:
                    found[num] = col
                    break

    return found


def _load_923_sheet(xl_path: Path, year: int) -> Optional[pd.DataFrame]:
    """Load the EIA 923 generation sheet from an Excel file or a ZIP containing one.

    Handles:
      - ZIP files (EIA distributes 923 as ZIP containing a single .xlsx)
      - Bare .xlsx files (in case the user places the Excel directly in data/raw/)

    The sheet has 4-6 header rows of metadata before the actual column names.
    We try header positions 4 and 5 (0-indexed) to handle year-over-year variation.
    """
    # Resolve to actual Excel bytes -- handle ZIP wrapper transparently.
    # We read the inner xlsx into a BytesIO buffer before closing the ZIP so
    # openpyxl can seek freely without the outer ZipFile's handle expiring.
    from io import BytesIO

    if xl_path.suffix.lower() == ".zip":
        with zipfile.ZipFile(xl_path, "r") as zf:
            xlsx_members = [m for m in zf.namelist() if m.endswith(".xlsx")]
            if not xlsx_members:
                log.warning("No .xlsx found in %s", xl_path.name)
                return None
            # Pick the largest member (usually the full dataset, not a supplemental)
            member = max(xlsx_members, key=lambda m: zf.getinfo(m).file_size)
            log.info("  Reading %s from %s", member, xl_path.name)
            excel_bytes = BytesIO(zf.read(member))  # buffer fully into memory
        xl = pd.ExcelFile(excel_bytes, engine="openpyxl")
    else:
        xl = pd.ExcelFile(xl_path, engine="openpyxl")

    sheet_name = _find_923_generation_sheet(xl)
    if sheet_name is None:
        return None

    log.info("  Sheet: '%s'", sheet_name)

    # Try header rows 4 then 5 -- pick whichever gives us recognizable month columns
    for skiprows in (4, 5, 3):
        df = pd.read_excel(xl, sheet_name=sheet_name, header=skiprows, dtype=str)
        month_cols = _find_month_columns(df)
        if len(month_cols) >= 10:   # expect at least 10 of 12 months
            log.info("  Header found at row %d, %d month columns", skiprows, len(month_cols))
            return df

    log.warning("Could not locate month columns in %d 923 sheet", year)
    return None


def _normalize_923_plant_id(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize the plant ID column name across EIA 923 format variants."""
    for col in df.columns:
        key = str(col).strip().lower().replace(" ", "_")
        if key in ("plant_id", "plant_code", "plant_code_eia", "plantid", "plantcode"):
            return df.rename(columns={col: "plant_id"})
    return df


def parse_eia923_year(zip_path: Path, plant_ids: list[int], year: int) -> pd.DataFrame:
    """Extract monthly net generation for target plants from one EIA 923 file.

    Wind plants appear in 923 with fuel type = 'WND' and prime mover = 'WT'.
    We filter to just those rows to exclude any thermal co-generation at the
    same plant (unlikely for Panhandle Wind but defensive practice).

    EIA 923 is in 'wide' format (one row per plant/fuel/prime-mover, columns
    per month). We melt to 'long' format so downstream code handles a simple
    (plant_id, year, month, net_gen_mwh) grain.

    Returns:
        Long-format DataFrame with columns [plant_id, year, month, net_gen_mwh].
        Empty DataFrame if the file cannot be parsed or contains no target rows.
    """
    df = _load_923_sheet(zip_path, year)
    if df is None or df.empty:
        return pd.DataFrame()

    df = _normalize_923_plant_id(df)
    if "plant_id" not in df.columns:
        log.warning("No plant_id column in %d 923 data", year)
        return pd.DataFrame()

    df["plant_id"] = pd.to_numeric(df["plant_id"], errors="coerce")
    df = df.dropna(subset=["plant_id"])
    df["plant_id"] = df["plant_id"].astype(int)

    # Filter to target plants before the expensive melt
    df = df[df["plant_id"].isin(plant_ids)].copy()

    if df.empty:
        log.warning("Plant IDs %s not found in %d 923 data", plant_ids, year)
        return pd.DataFrame()

    # Normalize 'Plant Name' column (923 uses this exact casing)
    if "Plant Name" in df.columns:
        df = df.rename(columns={"Plant Name": "plant_name"})

    # Filter to wind fuel type -- the real column name is 'Reported\nFuel Type Code'.
    # We match any column where the normalized name contains 'fuel' and 'type'.
    for col in df.columns:
        col_norm = col.lower().replace("\n", " ")
        if "fuel" in col_norm and "type" in col_norm:
            df = df[df[col].str.strip().str.upper() == WIND_FUEL_CODE]
            log.info(
                "    Fuel type filter ('%s' == '%s'): %d rows remain",
                col.replace("\n", "\\n"), WIND_FUEL_CODE, len(df),
            )
            break

    if df.empty:
        log.warning("No WND rows for plant IDs %s in year %d", plant_ids, year)
        return pd.DataFrame()

    month_cols = _find_month_columns(df)
    if not month_cols:
        log.warning("No month columns found for year %d", year)
        return pd.DataFrame()

    log.info("    Month columns found: %d (%s)", len(month_cols), list(month_cols.keys()))

    # Melt wide -> long so each row is one (plant, month) observation
    id_cols = ["plant_id"]
    if "plant_name" in df.columns:
        id_cols.append("plant_name")

    value_cols = [month_cols[m] for m in sorted(month_cols.keys())]
    month_nums = sorted(month_cols.keys())

    long = df[id_cols + value_cols].copy()
    long.columns = id_cols + [str(m) for m in month_nums]

    melted = long.melt(id_vars=id_cols, var_name="month", value_name="net_gen_mwh")
    melted["month"] = melted["month"].astype(int)
    melted["net_gen_mwh"] = pd.to_numeric(melted["net_gen_mwh"], errors="coerce")
    melted["year"] = year

    # EIA reports zero generation as 0 and missing data as blank/null.
    # Negative values can appear for net-metered or behind-the-meter adjustments;
    # for a utility-scale wind farm they indicate a data anomaly -- keep but flag.
    n_neg = (melted["net_gen_mwh"] < 0).sum()
    if n_neg:
        log.warning("  %d negative net-generation rows in year %d -- verify source data", n_neg, year)

    return melted.dropna(subset=["net_gen_mwh"]).reset_index(drop=True)


# -- Capacity factor and enrichment --------------------------------------------

def _hours_in_month(year: int, month: int) -> int:
    """Return the number of hours in a calendar month (accounts for leap years)."""
    _, days = monthrange(year, month)
    return days * 24


def compute_capacity_factors(
    gen_df: pd.DataFrame,
    plant_capacity: dict[int, float],
    total_portfolio_mw: float,
) -> pd.DataFrame:
    """Add per-plant and portfolio-level capacity factors to the generation DataFrame.

    Capacity factor = Net Generation (MWh) / (Nameplate MW * Hours in Month)

    We compute two CF columns:
      - cf_plant: CF using the plant's own EIA 860 nameplate capacity
      - cf_portfolio: CF using total portfolio MW (400 MW) -- what the IC memo uses

    The portfolio CF is more useful for valuation because it reflects the full
    investment basis. Plant-level CF is useful for engineering benchmarking.

    Args:
        gen_df:           Long-format generation DataFrame [plant_id, year, month, net_gen_mwh].
        plant_capacity:   {plant_id: nameplate_mw} from EIA 860.
        total_portfolio_mw: Sum of all plant capacities (or the utils.CAPACITY_MW constant).

    Returns:
        gen_df with added columns [hours_in_month, cf_plant, cf_portfolio].
    """
    df = gen_df.copy()
    df["hours_in_month"] = df.apply(
        lambda r: _hours_in_month(int(r["year"]), int(r["month"])), axis=1
    )

    # Map each plant ID to its 860-reported capacity
    df["plant_capacity_mw"] = df["plant_id"].map(plant_capacity).fillna(float("nan"))
    df["max_gen_plant_mwh"] = df["plant_capacity_mw"] * df["hours_in_month"]
    df["cf_plant"] = df["net_gen_mwh"] / df["max_gen_plant_mwh"]

    # Portfolio CF uses the total contracted capacity as denominator
    df["max_gen_portfolio_mwh"] = total_portfolio_mw * df["hours_in_month"]
    df["cf_portfolio"] = df["net_gen_mwh"] / df["max_gen_portfolio_mwh"]

    return df


# -- Summary and output --------------------------------------------------------

def build_annual_summary(gen_df: pd.DataFrame) -> pd.DataFrame:
    """Compute annual generation and capacity factor statistics.

    Aggregates across both plants to give portfolio-level figures. The annual
    capacity factor is the primary output -- it feeds the P50/P90 comparison
    in Layer 3 and the revenue assumption in the Layer 4 DCF.

    Returns:
        Annual summary DataFrame [year, net_gen_gwh, cf_portfolio, peak_month,
        trough_month, months_above_40pct, months_negative].
    """
    # Sum generation across all plants within each year/month
    monthly = (
        gen_df.groupby(["year", "month"])
        .agg(
            net_gen_mwh=("net_gen_mwh", "sum"),
            hours_in_month=("hours_in_month", "first"),
        )
        .reset_index()
    )

    total_mw = CAPACITY_MW
    monthly["cf_portfolio"] = monthly["net_gen_mwh"] / (total_mw * monthly["hours_in_month"])

    rows = []
    for year, grp in monthly.groupby("year"):
        annual_mwh = grp["net_gen_mwh"].sum()
        annual_hours = 8760 if year % 4 != 0 else 8784
        annual_cf = annual_mwh / (total_mw * annual_hours)
        peak_month = int(grp.loc[grp["net_gen_mwh"].idxmax(), "month"])
        trough_month = int(grp.loc[grp["net_gen_mwh"].idxmin(), "month"])

        rows.append(
            {
                "year": int(year),
                "net_gen_gwh": annual_mwh / 1000,
                "cf_portfolio": annual_cf,
                "peak_month": peak_month,
                "trough_month": trough_month,
                "months_above_40pct": int((grp["cf_portfolio"] >= 0.40).sum()),
                "months_negative": int((grp["net_gen_mwh"] < 0).sum()),
            }
        )

    return pd.DataFrame(rows)


def _print_plant_inventory(plants_df: pd.DataFrame) -> None:
    """Print the EIA 860 plant inventory for Panhandle Wind."""
    print("\nPanhandle Wind -- EIA 860 Plant Inventory")
    print("-" * 60)
    for pid, grp in plants_df.groupby("plant_id"):
        name = grp["plant_name"].iloc[0]
        mw = grp["capacity_mw"].sum() if "capacity_mw" in grp.columns else float("nan")
        n_gens = len(grp)
        county = grp["county"].iloc[0] if "county" in grp.columns else "?"
        print(f"  Plant {pid}: {name}")
        print(f"    County: {county}, TX  |  Generators: {n_gens}  |  Nameplate: {mw:.1f} MW")
    print()


def _print_summary_table(summary: pd.DataFrame) -> None:
    """Print the annual generation summary table to stdout.

    The capacity factor column is the headline output -- deviations from the
    NREL P50 (~38-40% for Carson County TX) flag underperformance risk.
    """
    _MONTH = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }

    header = (
        f"{'Year':<6} | {'Net Gen (GWh)':>14} | {'Annual CF':>10} | "
        f"{'Peak Month':>11} | {'Trough Month':>13} | {'Mo >= 40%':>10}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for _, row in summary.iterrows():
        peak = _MONTH.get(int(row["peak_month"]), "?")
        trough = _MONTH.get(int(row["trough_month"]), "?")
        print(
            f"{int(row['year']):<6} | {row['net_gen_gwh']:>12.1f}  | "
            f"{row['cf_portfolio']:>9.1%} | {peak:>11} | {trough:>13} | "
            f"{int(row['months_above_40pct']):>10}"
        )

    print(sep)
    print(
        "\n  CF benchmark: NREL P50 for Carson County TX is ~38-41%."
        " Values below 35% or above 45% warrant investigation.\n"
    )


# -- Pipeline orchestration ----------------------------------------------------

def fetch_eia_data(years: list[int]) -> tuple[list[int], list[int]]:
    """Download EIA 860 and 923 files for all requested years.

    Uses EIA's stable direct-download URLs (unlike ERCOT's JS-rendered portal,
    EIA data is accessible via plain HTTP GET with no authentication).

    Returns:
        (years_860_ok, years_923_ok) -- lists of years successfully downloaded
        or already cached.
    """
    # EIA 860: we only need one reference year to discover plant IDs
    years_860 = [EIA860_REFERENCE_YEAR]
    ok_860 = [y for y in years_860 if _ensure_eia860(y)]

    # EIA 923: download all requested years
    ok_923 = []
    for year in years:
        if _ensure_eia923(year):
            ok_923.append(year)
        else:
            log.warning("EIA 923 %d unavailable -- may not yet be published", year)

    return ok_860, ok_923


def main() -> None:
    """Run the full EIA generation pipeline end-to-end.

    Steps:
        1. Ensure data directories exist
        2. Download EIA 860 (plant registry) and 923 (monthly generation)
        3. Parse EIA 860 to identify Panhandle Wind's plant IDs
        4. Parse EIA 923 for each year using those plant IDs
        5. Compute capacity factors at plant and portfolio level
        6. Write Parquet and summary CSV
        7. Print the annual summary table
    """
    ensure_dirs()
    log.info("=" * 60)
    log.info("EIA Form 860 / 923 Generation Pipeline")
    log.info("Plant: %s  |  Years: %d-%d", PLANT_NAME, YEARS[0], YEARS[-1])
    log.info("=" * 60)

    # -- Step 1: Download ------------------------------------------------------
    log.info("Fetching EIA data...")
    ok_860, ok_923 = fetch_eia_data(YEARS)

    if not ok_860:
        log.error(
            "Could not obtain EIA 860 for year %d. "
            "Download manually from https://www.eia.gov/electricity/data/eia860/ "
            "and save to data/raw/eia860_%d.zip",
            EIA860_REFERENCE_YEAR,
            EIA860_REFERENCE_YEAR,
        )
        sys.exit(1)

    if not ok_923:
        log.error("No EIA 923 files available. Check https://www.eia.gov/electricity/data/eia923/")
        sys.exit(1)

    # -- Step 2: Parse EIA 860 to identify plants ------------------------------
    log.info("Parsing EIA 860 plant registry...")
    eia860_df = parse_eia860_wind(_raw860_path(EIA860_REFERENCE_YEAR))

    if eia860_df.empty:
        log.error("EIA 860 parsing returned empty DataFrame")
        sys.exit(1)

    plant_ids, plants_df = identify_panhandle_plants(eia860_df)

    if not plant_ids:
        log.error(
            "Could not identify Panhandle Wind plants. "
            "Check search terms: name='%s', state='%s', county='%s'",
            SEARCH_NAME, SEARCH_STATE, SEARCH_COUNTY,
        )
        sys.exit(1)

    _print_plant_inventory(plants_df)

    # Build plant_id -> nameplate capacity mapping for CF calculation
    if "capacity_mw" in plants_df.columns:
        plant_capacity: dict[int, float] = (
            plants_df.groupby("plant_id")["capacity_mw"]
            .sum()
            .to_dict()
        )
    else:
        # Fall back to splitting portfolio capacity equally across plants
        plant_capacity = {pid: CAPACITY_MW / len(plant_ids) for pid in plant_ids}
        log.warning("No capacity_mw column in 860 data -- split evenly across plants")

    total_mw = sum(plant_capacity.values())
    log.info("Total identified capacity: %.1f MW (vs %.0f MW constant)", total_mw, CAPACITY_MW)

    # -- Step 3: Parse EIA 923 for each year -----------------------------------
    log.info("Parsing EIA 923 monthly generation data...")
    year_frames: list[pd.DataFrame] = []

    for year in ok_923:
        log.info("  Processing year %d...", year)
        frame = parse_eia923_year(_raw923_path(year), plant_ids, year)
        if not frame.empty:
            year_frames.append(frame)
            mwh = frame["net_gen_mwh"].sum()
            log.info("    %d: %.0f MWh across %d plant-month records", year, mwh, len(frame))
        else:
            log.warning("    %d: no usable generation data found", year)

    if not year_frames:
        log.error("No generation data parsed from any year. Check EIA 923 files.")
        sys.exit(1)

    gen_df = pd.concat(year_frames, ignore_index=True)

    # Merge plant name from 860 for readability
    if "plant_name" not in gen_df.columns and not plants_df.empty:
        name_map = (
            plants_df[["plant_id", "plant_name"]]
            .drop_duplicates("plant_id")
            .set_index("plant_id")["plant_name"]
            .to_dict()
        )
        gen_df["plant_name"] = gen_df["plant_id"].map(name_map)

    # Merge county/state from 860
    if not plants_df.empty and "county" in plants_df.columns:
        geo = (
            plants_df[["plant_id", "county", "state"]]
            .drop_duplicates("plant_id")
        )
        gen_df = gen_df.merge(geo, on="plant_id", how="left")

    # -- Step 4: Compute capacity factors --------------------------------------
    log.info("Computing capacity factors...")
    gen_df = compute_capacity_factors(gen_df, plant_capacity, CAPACITY_MW)

    # -- Step 5: Write outputs -------------------------------------------------
    gen_df.to_parquet(OUTPUT_GENERATION, index=False)
    log.info("Wrote %s", OUTPUT_GENERATION)

    summary_df = build_annual_summary(gen_df)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, float_format="%.4f")
    log.info("Wrote %s", OUTPUT_SUMMARY)

    # -- Step 6: Print summary -------------------------------------------------
    _print_summary_table(summary_df)

    date_range = f"{int(gen_df['year'].min())} - {int(gen_df['year'].max())}"
    total_gwh = gen_df["net_gen_mwh"].sum() / 1000
    avg_cf = summary_df["cf_portfolio"].mean()

    print(f"[OK] eia_generation.parquet written -- {len(gen_df):,} rows, {date_range}")
    print(f"     Total generation: {total_gwh:,.0f} GWh  |  Avg annual CF: {avg_cf:.1%}")
    print(f"[OK] eia_generation_summary.csv written")
    print()
    print("Next: Run src/nrel_pipeline.py to build Layer 3")


if __name__ == "__main__":
    main()
