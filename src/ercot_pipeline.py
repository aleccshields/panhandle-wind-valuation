"""
Layer 1 -- ERCOT Real-Time Market Settlement Point Price Pipeline

Downloads (or reads from data/raw/) historical RTM Settlement Point Prices for
LZ_NORTH and HB_NORTH, normalizes column names and timestamps across annual
format variations, and writes a clean hourly price series to data/processed/.

Financial context
-----------------
LZ_NORTH is the settlement price received by wind generators in the Panhandle
region. HB_NORTH is the liquid North Hub reference price. Wind plants generate
most during high-wind periods, which often coincide with low-price hours --
depressing realized revenue below the flat hub average. The LZ/HB price ratio
is therefore the central metric for merchant wind valuation: a ratio of 0.85
means the plant captures only 85 cents of every hub dollar.

This ratio, computed annually here, feeds directly into the DCF model in Layer 4.

Run from repo root:
    python src/ercot_pipeline.py
"""

import logging
import sys
import zipfile
from io import StringIO
from pathlib import Path

import pandas as pd
import requests

# -- Repo root on sys.path so 'src.utils' is importable when run as a script ---
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import DATA_PROCESSED, DATA_RAW, ERCOT_HUB, ERCOT_ZONE, ensure_dirs

# -- Logging (all progress to stderr via logger; summary table prints to stdout) -
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stderr,
)
log = logging.getLogger(__name__)

# -- Pipeline config ------------------------------------------------------------
YEARS = list(range(2015, 2025))
SETTLEMENT_POINTS = frozenset({ERCOT_ZONE, ERCOT_HUB})  # {'LZ_NORTH', 'HB_NORTH'}

OUTPUT_PRICES = DATA_PROCESSED / "ercot_rtm_prices.parquet"
OUTPUT_SUMMARY = DATA_PROCESSED / "ercot_rtm_summary.csv"

# ERCOT's public data product page for Historical RTM Settlement Point Prices.
# The page is JavaScript-rendered, so link extraction requires a headless browser.
# We attempt a plain HTTP fetch first; on failure we print manual instructions.
ERCOT_DATA_PAGE = "https://www.ercot.com/mp/data-products/data-product-details?id=NP6-788-ER"
ERCOT_DOWNLOAD_BASE = "https://www.ercot.com"

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

# Column name normalization map -- ERCOT has changed its CSV headers several times
# across the 2015-2024 range. Every known variant maps to our standard schema.
_COLUMN_MAP: dict[str, str] = {
    # Delivery date
    "deliverydate": "delivery_date",
    "delivery_date": "delivery_date",
    # Delivery hour (ERCOT uses 1-24 hour-ending convention, not 0-23)
    "deliveryhour": "delivery_hour",
    "delivery_hour": "delivery_hour",
    "he": "delivery_hour",
    "hourending": "delivery_hour",
    "hour_ending": "delivery_hour",
    # 15-minute interval within each hour (1-4); absent in hourly files
    "deliveryinterval": "delivery_interval",
    "delivery_interval": "delivery_interval",
    "interval": "delivery_interval",
    # Settlement point name
    "settlementpointname": "settlement_point",
    "settlement_point_name": "settlement_point",
    "settlementpoint": "settlement_point",
    "settlement_point": "settlement_point",
    "spname": "settlement_point",
    "spp": "settlement_point",
    # Settlement point price ($/MWh)
    "settlementpointprice": "price",
    "settlement_point_price": "price",
    "spp_price": "price",
    "price": "price",
}


# -- Download helpers -----------------------------------------------------------

def _raw_zip_path(year: int) -> Path:
    """Return the expected local path for a year's raw ZIP file."""
    return DATA_RAW / f"ercot_rtm_{year}.zip"


def _try_scrape_download_links(session: requests.Session) -> dict[int, str]:
    """Attempt to extract annual ZIP download URLs from ERCOT's data product page.

    ERCOT's page is JavaScript-rendered, so this will usually return an empty dict.
    It is included as a best-effort attempt before falling back to manual instructions.

    Returns:
        Mapping of {year: absolute_download_url} for any years found in the HTML.
    """
    try:
        resp = session.get(ERCOT_DATA_PAGE, headers=_BROWSER_HEADERS, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.debug("Could not reach ERCOT data page: %s", exc)
        return {}

    # Look for links containing annual ZIP patterns in the raw HTML.
    # ERCOT embed links as /misdownload/servlets/mirDownload?doclookupId=XXXXX
    # or /files/docs/YYYY/.../filename.zip
    import re

    found: dict[int, str] = {}
    for year in YEARS:
        pattern = re.compile(
            rf'href="([^"]*(?:rtm.*spp.*{year}|{year}.*rtm.*spp)[^"]*\.zip)"',
            re.IGNORECASE,
        )
        match = pattern.search(resp.text)
        if match:
            href = match.group(1)
            url = href if href.startswith("http") else ERCOT_DOWNLOAD_BASE + href
            found[year] = url
            log.info("Found download URL for %d", year)

    return found


def _download_zip(year: int, url: str, session: requests.Session) -> bool:
    """Download a ZIP file from ERCOT and save it to data/raw/.

    Args:
        year: Calendar year being downloaded.
        url:  Direct download URL.

    Returns:
        True on success, False on any HTTP or I/O error.
    """
    dest = _raw_zip_path(year)
    log.info("Downloading %d from %s", year, url)
    try:
        with session.get(url, headers=_BROWSER_HEADERS, stream=True, timeout=120) as resp:
            resp.raise_for_status()
            with dest.open("wb") as fh:
                for chunk in resp.iter_content(chunk_size=1 << 20):  # 1 MB chunks
                    fh.write(chunk)
        log.info("Saved %s (%.1f MB)", dest.name, dest.stat().st_size / 1e6)
        return True
    except requests.RequestException as exc:
        log.warning("Download failed for %d: %s", year, exc)
        if dest.exists():
            dest.unlink()
        return False


def _print_download_instructions(missing_years: list[int]) -> None:
    """Print step-by-step manual download instructions for missing years.

    ERCOT's portal requires a browser to render JavaScript download links.
    This message gives the user the exact steps to fetch the files manually.
    """
    bar = "=" * 68
    print(f"\n{bar}", flush=True)
    print("  ERCOT DATA -- MANUAL DOWNLOAD REQUIRED", flush=True)
    print(bar, flush=True)
    print(f"\n  Missing years: {', '.join(str(y) for y in missing_years)}\n", flush=True)
    print("  ERCOT's download portal requires a browser (JavaScript-rendered).", flush=True)
    print("  Visit the URL below and download the annual ZIP for each missing year:\n", flush=True)
    print(f"  {ERCOT_DATA_PAGE}\n", flush=True)
    print("  Look for: 'Historical RTM Settlement Point Prices' -- download one ZIP", flush=True)
    print("  per year and save each file with this exact name:\n", flush=True)
    for year in missing_years:
        print(f"    data/raw/ercot_rtm_{year}.zip", flush=True)
    print(
        "\n  Note: Files are typically 100-500 MB each. After downloading,",
        flush=True,
    )
    print("  re-run:  python src/ercot_pipeline.py\n", flush=True)
    print(bar + "\n", flush=True)



# -- Parsing helpers ------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize ERCOT CSV column names to our standard schema.

    Strips whitespace, lowercases, and removes underscores/spaces before
    looking up in _COLUMN_MAP so that all known ERCOT naming variants collapse
    to the same canonical name.
    """
    renamed = {}
    for col in df.columns:
        key = col.strip().lower().replace(" ", "").replace("_", "")
        if key in _COLUMN_MAP:
            renamed[col] = _COLUMN_MAP[key]
    return df.rename(columns=renamed)


def _parse_date_column(series: pd.Series) -> pd.Series:
    """Parse ERCOT delivery dates, handling MM/DD/YYYY and YYYY-MM-DD formats."""
    # Try the most common historical format first (MM/DD/YYYY), then ISO.
    try:
        return pd.to_datetime(series, format="%m/%d/%Y")
    except ValueError:
        pass
    try:
        return pd.to_datetime(series, format="%Y-%m-%d")
    except ValueError:
        pass
    # Fall back to pandas' inference -- slower but handles edge cases.
    return pd.to_datetime(series, infer_datetime_format=True)


def _parse_single_csv(raw_text: str, year: int, filename: str) -> pd.DataFrame:
    """Parse one ERCOT RTM SPP CSV into a standardized DataFrame.

    ERCOT CSVs occasionally include metadata rows above the header. We detect
    the header row by scanning for the string "Delivery" (present in every
    known column name variant) and skip everything above it.

    Args:
        raw_text: Full text content of the CSV file.
        year:     Calendar year, used for debug logging only.
        filename: Source filename, used for debug logging only.

    Returns:
        DataFrame with columns [delivery_date, delivery_hour, delivery_interval?,
        settlement_point, price], filtered to SETTLEMENT_POINTS rows only.
    """
    # Find the true header row -- skip any leading metadata lines.
    lines = raw_text.splitlines()
    header_idx = 0
    for i, line in enumerate(lines):
        lower = line.lower()
        if "delivery" in lower or "settlement" in lower or "date" in lower:
            header_idx = i
            break

    csv_body = "\n".join(lines[header_idx:])
    df = pd.read_csv(StringIO(csv_body), low_memory=False)

    if df.empty:
        log.warning("Empty CSV: %s (year %d)", filename, year)
        return pd.DataFrame()

    df = _normalize_columns(df)

    required = {"delivery_date", "delivery_hour", "settlement_point", "price"}
    missing_cols = required - set(df.columns)
    if missing_cols:
        log.warning(
            "Missing columns %s in %s (year %d) -- skipping file",
            missing_cols,
            filename,
            year,
        )
        return pd.DataFrame()

    # Filter to the two nodes we care about before any heavy processing.
    # This reduces memory use by ~99% since ERCOT tracks hundreds of points.
    df = df[df["settlement_point"].isin(SETTLEMENT_POINTS)].copy()

    if df.empty:
        log.warning(
            "No LZ_NORTH or HB_NORTH rows found in %s (year %d)", filename, year
        )
        return pd.DataFrame()

    df["delivery_date"] = _parse_date_column(df["delivery_date"])
    df["delivery_hour"] = pd.to_numeric(df["delivery_hour"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["delivery_date", "delivery_hour", "price"])

    # delivery_interval may be absent in hourly-resolution files
    if "delivery_interval" not in df.columns:
        df["delivery_interval"] = 1

    df["delivery_interval"] = pd.to_numeric(
        df["delivery_interval"], errors="coerce"
    ).fillna(1).astype(int)

    return df


def _parse_zip(zip_path: Path, year: int) -> pd.DataFrame:
    """Open an ERCOT annual ZIP and parse all CSVs inside it.

    ERCOT packages data in two ways depending on the year:
      * One large annual CSV per ZIP (older years)
      * Twelve monthly CSVs per ZIP (some years)

    Both are handled here by concatenating all CSV members found.

    Args:
        zip_path: Path to the annual ZIP file in data/raw/.
        year:     Calendar year, used for filtering and logging.

    Returns:
        Concatenated, normalized DataFrame for all CSVs in the ZIP.
    """
    frames: list[pd.DataFrame] = []

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not csv_members:
                log.warning("No CSVs found in %s", zip_path.name)
                return pd.DataFrame()

            log.info(
                "  Parsing %d CSV file(s) from %s", len(csv_members), zip_path.name
            )
            for member in sorted(csv_members):
                try:
                    raw_bytes = zf.read(member)
                    # ERCOT files are occasionally encoded in latin-1
                    try:
                        raw_text = raw_bytes.decode("utf-8")
                    except UnicodeDecodeError:
                        raw_text = raw_bytes.decode("latin-1")

                    frame = _parse_single_csv(raw_text, year, member)
                    if not frame.empty:
                        frames.append(frame)
                        log.debug(
                            "    %s -> %d rows (after SP filter)", member, len(frame)
                        )
                except Exception as exc:
                    log.warning("Failed to parse %s in %s: %s", member, zip_path.name, exc)

    except zipfile.BadZipFile:
        log.error("Corrupt ZIP file: %s", zip_path.name)
        return pd.DataFrame()

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def _build_hourly_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Convert raw ERCOT data into the final hourly output schema.

    ERCOT publishes 15-minute interval prices. We average the four intervals
    within each hour to produce an hourly price series. For files that are
    already hourly (delivery_interval is uniformly 1), this is a no-op mean.

    ERCOT uses an hour-ending convention (hour 1 = 00:00-01:00 CPT), so we
    subtract 1 to get hour-starting timestamps aligned with standard practice
    in energy finance models.

    Output columns:
        datetime         -- pandas Timestamp, hour-starting, no tz
        settlement_point -- str: 'LZ_NORTH' or 'HB_NORTH'
        price            -- float: $/MWh (mean of 15-min intervals within hour)
        year             -- int
        month            -- int
        hour             -- int (0-23, hour-starting)
    """
    # Average across 15-min intervals -> one row per (date, hour, settlement_point)
    hourly = (
        raw_df.groupby(["delivery_date", "delivery_hour", "settlement_point"], sort=False)
        ["price"]
        .mean()
        .reset_index()
    )

    # Convert ERCOT hour-ending (1-24) to hour-starting (0-23)
    hour_starting = (hourly["delivery_hour"] - 1).astype(int)

    hourly["datetime"] = hourly["delivery_date"] + pd.to_timedelta(
        hour_starting, unit="h"
    )

    hourly["year"] = hourly["datetime"].dt.year
    hourly["month"] = hourly["datetime"].dt.month
    hourly["hour"] = hourly["datetime"].dt.hour

    return hourly[
        ["datetime", "settlement_point", "price", "year", "month", "hour"]
    ].sort_values("datetime").reset_index(drop=True)


# -- Summary computation --------------------------------------------------------

def _compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the annual per-settlement-point statistics table.

    The LZ_NORTH / HB_NORTH price ratio (capture proxy) is the key output.
    It measures how much of the hub price wind generators actually realize,
    penalized by the correlation between high output and low prices.

    A ratio below 1.0 implies wind-hour price suppression -- the dominant
    risk factor for merchant wind DCF valuations in ERCOT. This ratio is
    used directly as the capture rate assumption in src/financial_model.py.
    """
    stats = (
        df.groupby(["year", "settlement_point"])["price"]
        .agg(
            mean_price="mean",
            std_price="std",
            min_price="min",
            max_price="max",
            count="count",
        )
        .reset_index()
    )

    # Negative price hours matter for wind because curtailment becomes
    # economically rational below zero (avoid paying the market to take power).
    neg = (
        df[df["price"] < 0]
        .groupby(["year", "settlement_point"])
        .size()
        .reset_index(name="neg_hours")
    )
    stats = stats.merge(neg, on=["year", "settlement_point"], how="left")
    stats["neg_hours"] = stats["neg_hours"].fillna(0).astype(int)
    stats["neg_pct"] = stats["neg_hours"] / stats["count"] * 100

    # Pivot so each year has one row with both LZ and HB columns
    lz = stats[stats["settlement_point"] == ERCOT_ZONE].set_index("year")
    hb = stats[stats["settlement_point"] == ERCOT_HUB].set_index("year")

    summary_rows = []
    for year in sorted(df["year"].unique()):
        row: dict = {"year": year}
        if year in lz.index:
            lz_row = lz.loc[year]
            row["lz_mean"] = lz_row["mean_price"]
            row["lz_std"] = lz_row["std_price"]
            row["lz_min"] = lz_row["min_price"]
            row["lz_max"] = lz_row["max_price"]
            row["lz_neg_hours"] = lz_row["neg_hours"]
            row["lz_neg_pct"] = lz_row["neg_pct"]
        if year in hb.index:
            hb_row = hb.loc[year]
            row["hb_mean"] = hb_row["mean_price"]
            row["hb_std"] = hb_row["std_price"]
            row["hb_min"] = hb_row["min_price"]
            row["hb_max"] = hb_row["max_price"]
            row["hb_neg_hours"] = hb_row["neg_hours"]
            row["hb_neg_pct"] = hb_row["neg_pct"]

        # Capture proxy: LZ mean / HB mean -- core DCF input
        if "lz_mean" in row and "hb_mean" in row and row["hb_mean"] != 0:
            row["capture_ratio"] = row["lz_mean"] / row["hb_mean"]
        else:
            row["capture_ratio"] = None

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def _print_summary_table(summary: pd.DataFrame) -> None:
    """Print the annual summary in a format readable by energy finance practitioners.

    The capture ratio column is the headline output -- values below 0.90 indicate
    significant wind-hour price suppression and materially reduce DCF valuations
    versus a flat-price assumption.
    """
    header = (
        f"{'Year':<6} | {'LZ_NORTH Avg':>13} | {'HB_NORTH Avg':>13} | "
        f"{'Capture Ratio':>14} | {'LZ Neg Hrs':>12} | {'HB StdDev':>10}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for _, row in summary.iterrows():
        lz_avg = f"${row['lz_mean']:.1f}/MWh" if pd.notna(row.get("lz_mean")) else "N/A"
        hb_avg = f"${row['hb_mean']:.1f}/MWh" if pd.notna(row.get("hb_mean")) else "N/A"
        capture = f"{row['capture_ratio']:.3f}" if pd.notna(row.get("capture_ratio")) else "N/A"
        neg_hrs = (
            f"{int(row['lz_neg_hours'])} ({row['lz_neg_pct']:.1f}%)"
            if pd.notna(row.get("lz_neg_hours"))
            else "N/A"
        )
        hb_std = f"${row['hb_std']:.1f}" if pd.notna(row.get("hb_std")) else "N/A"

        print(
            f"{int(row['year']):<6} | {lz_avg:>13} | {hb_avg:>13} | "
            f"{capture:>14} | {neg_hrs:>12} | {hb_std:>10}"
        )

    print(sep)
    print(
        "\n  Capture ratio < 1.0 means wind generators realize less than hub average.\n"
        "  This discount is the primary merchant risk for Panhandle Wind in the DCF.\n"
    )


# -- Pipeline orchestration -----------------------------------------------------

def fetch_data(years: list[int]) -> list[int]:
    """Attempt programmatic download of missing ERCOT ZIP files.

    Returns the list of years for which no local ZIP is available after the
    download attempt (these require manual download per printed instructions).
    """
    missing = [y for y in years if not _raw_zip_path(y).exists()]
    if not missing:
        log.info("All ZIP files already present in data/raw/ -- skipping download step")
        return []

    log.info(
        "Attempting to locate download links for %d missing year(s): %s",
        len(missing),
        missing,
    )

    session = requests.Session()
    links = _try_scrape_download_links(session)

    still_missing = []
    for year in missing:
        if year in links:
            success = _download_zip(year, links[year], session)
            if not success:
                still_missing.append(year)
        else:
            still_missing.append(year)

    return still_missing


def parse_all_years(years: list[int]) -> pd.DataFrame:
    """Parse all available annual ZIP files and return a combined DataFrame.

    Years whose ZIP is absent in data/raw/ are skipped with a warning.

    Returns:
        Combined raw DataFrame before hourly aggregation, or empty DataFrame
        if no files could be parsed.
    """
    frames: list[pd.DataFrame] = []

    for year in years:
        zip_path = _raw_zip_path(year)
        if not zip_path.exists():
            log.warning("Skipping %d -- ZIP not found at %s", year, zip_path)
            continue

        log.info("Processing %d (%s)...", year, zip_path.name)
        frame = _parse_zip(zip_path, year)

        if frame.empty:
            log.warning("No usable data extracted from %d", year)
            continue

        log.info(
            "  %d: %d raw rows across %s settlement points",
            year,
            len(frame),
            list(frame["settlement_point"].unique()),
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    log.info(
        "Combined raw dataset: %d rows, %d years",
        len(combined),
        combined["delivery_date"].dt.year.nunique() if not combined.empty else 0,
    )
    return combined


def main() -> None:
    """Run the full ERCOT RTM price pipeline end-to-end.

    Steps:
        1. Ensure data directories exist
        2. Attempt to download any missing ZIP files from ERCOT
        3. Print manual download instructions for years that couldn't be fetched
        4. Parse all available ZIPs into a normalized raw DataFrame
        5. Aggregate to hourly resolution and apply output schema
        6. Write Parquet and CSV outputs
        7. Print the annual summary table
    """
    ensure_dirs()
    log.info("=" * 60)
    log.info("ERCOT RTM Settlement Point Price Pipeline")
    log.info("Nodes: %s, %s | Years: %d-%d", ERCOT_ZONE, ERCOT_HUB, YEARS[0], YEARS[-1])
    log.info("=" * 60)

    # -- Step 1: Download -------------------------------------------------------
    still_missing = fetch_data(YEARS)
    if still_missing:
        _print_download_instructions(still_missing)

    # -- Step 2: Parse ---------------------------------------------------------
    available_years = [y for y in YEARS if _raw_zip_path(y).exists()]
    if not available_years:
        log.error(
            "No ZIP files found in data/raw/. Download ERCOT data and re-run."
        )
        sys.exit(1)

    raw_df = parse_all_years(available_years)
    if raw_df.empty:
        log.error("No data parsed from available ZIPs. Check file contents.")
        sys.exit(1)

    # -- Step 3: Build hourly output --------------------------------------------
    log.info("Aggregating 15-min intervals to hourly resolution...")
    hourly_df = _build_hourly_dataframe(raw_df)
    log.info("Hourly dataset: %d rows", len(hourly_df))

    # -- Step 4: Write Parquet -------------------------------------------------
    hourly_df.to_parquet(OUTPUT_PRICES, index=False)
    date_min = hourly_df["datetime"].min().date()
    date_max = hourly_df["datetime"].max().date()
    log.info("Wrote %s", OUTPUT_PRICES)

    # -- Step 5: Compute and write summary --------------------------------------
    log.info("Computing annual summary statistics...")
    summary_df = _compute_summary(hourly_df)
    summary_df.to_csv(OUTPUT_SUMMARY, index=False, float_format="%.4f")
    log.info("Wrote %s", OUTPUT_SUMMARY)

    # -- Step 6: Print summary to stdout ---------------------------------------
    _print_summary_table(summary_df)

    # -- Completion checklist ---------------------------------------------------
    print(f"+ ercot_rtm_prices.parquet written -- {len(hourly_df):,} rows, {date_min} to {date_max}")
    print(f"+ ercot_rtm_summary.csv written")
    print()
    print("Next: Run src/eia_pipeline.py to build Layer 2")


if __name__ == "__main__":
    main()
