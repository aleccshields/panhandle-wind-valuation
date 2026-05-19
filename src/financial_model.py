"""
Layer 4 -- Panhandle Wind Financial Model
DCF valuation with sculpted project debt, CAPM discount rate, and
three-scenario (bear / base / bull) analysis.

Run from repo root:
    python src/financial_model.py
"""

import logging
import sys
from io import StringIO
from pathlib import Path

import numpy as np
import numpy_financial as npf
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.utils import DATA_PROCESSED, ensure_dirs

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ==============================================================================
# ASSUMPTIONS -- single source of truth for all model parameters.
# None values are populated at runtime from data files or FRED API.
# No magic numbers anywhere else in this file.
# ==============================================================================

ASSUMPTIONS: dict = {
    # -- Asset --
    "capacity_mw":       399.7,   # EIA 860 confirmed nameplate (Phases 1 + 2)
    "cod_year":          2014,    # Commercial operation date
    "asset_life_years":  25,      # Total useful life; end year = cod + life = 2039
    "model_start_year":  2025,    # First year of forward cash flows
    "degradation_pct":   0.005,   # 0.5%/yr turbine output degradation (industry standard)

    # -- Revenue: contracted tranche --
    # Source: PEGI 10-K. Citibank affiliate hedge on ~77% of output through ~2027.
    # Modeling the step-down explicitly is both accurate (it is disclosed) and
    # analytically superior to assuming expiry -- it captures the revenue cliff.
    "hedge_price_mwh":   23.50,   # $/MWh fixed price under hedge
    "hedge_pct":         0.77,    # fraction of output hedged
    "hedge_expiry_year": 2027,    # last year hedge applies; 2028+ is full merchant

    # -- Revenue: merchant tranche --
    # Populated from ERCOT Layer 1 data; constants applied if file is absent.
    "hub_price_2024":            None,  # $/MWh HB_NORTH 2024 annual average
    "capture_rate_base":         None,  # LZ_NORTH / HB_NORTH ratio (Layer 1)
    "capture_rate_bull":         None,  # base + 0.03 (storage/load growth absorbs wind)
    "capture_rate_bear":         None,  # base - 0.05 (solar cannibalisation of midday)
    "merchant_price_escalator":  0.02,  # 2%/yr real price escalation post-hedge

    # -- Costs (base-year; escalated forward at om_escalator) --
    "om_fixed_mw_yr":    35_000,  # $/MW/yr fixed O&M, ERCOT market standard
    "om_variable_mwh":    3.50,   # $/MWh variable O&M
    "land_lease_mw_yr":   6_000,  # $/MW/yr land lease, Carson County TX
    "insurance_mw_yr":    3_500,  # $/MW/yr property / casualty insurance
    "om_escalator":       0.025,  # 2.5%/yr cost escalation (PPI-based)

    # -- Discount rate: CAPM build-up for unlevered equity --
    # WACC_unlevered = rf + beta_unlevered * ERP + size_premium
    #
    #   rf              -- 10yr UST; appropriate tenor for a 15-year remaining-life asset
    #   beta_unlevered  -- 0.55, Damodaran utility-scale wind sector (unlevered)
    #   ERP             -- 5.5%, Damodaran implied US equity risk premium (stable)
    #   size_premium    -- 1.5%, Duff & Phelps small-cap / illiquidity premium;
    #                      appropriate for a single-asset, non-publicly-traded project
    "risk_free_rate":      None,  # populated from FRED GS10 at runtime
    "equity_risk_premium": 0.055, # Damodaran US ERP
    "beta_unlevered":      0.55,  # utility-scale wind, unlevered (Damodaran sector)
    "size_premium":        0.015, # small-cap / illiquidity premium
    "wacc_unlevered":      None,  # computed: rf + beta*erp + size_premium

    # -- Debt (project finance, sized on P50 CFADS with DSCR constraint) --
    "target_dscr":        1.35,   # minimum DSCR floor; 1.35x is IG market standard
    "debt_tenor_years":   15,     # covers full remaining asset life (2025-2039)
    "debt_rate":          None,   # populated from FRED BAA yield at runtime
    "debt_spread_bps":    175,    # bps over 10yr UST (IG project finance proxy)
    "debt_sculpting":     True,   # sculpt principal to maintain flat DSCR

    # -- Generation scenarios -- populated from nrel_p50_p90_summary.csv --
    "p50_aep_mwh": None,  # base and bull scenario AEP
    "p90_aep_mwh": None,  # bear scenario AEP (P90 is conservative input for lenders)
}

# FRED public API -- no key required
_FRED_GS10 = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=GS10"
_FRED_BAA  = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=BAA"

# Fallback constants applied when ERCOT Layer 1 file is absent
_HUB_PRICE_FALLBACK    = 33.0  # $/MWh, approximate 2024 HB_NORTH annual average
_CAPTURE_RATE_FALLBACK = 0.80  # LZ_NORTH/HB_NORTH, midpoint of 2022-2024 trend

# File paths
_ERCOT_SUMMARY = DATA_PROCESSED / "ercot_rtm_summary.csv"
_NREL_SUMMARY  = DATA_PROCESSED / "nrel_p50_p90_summary.csv"


# ==============================================================================
# STEP 1 -- CAPM INPUTS FROM FRED
# ==============================================================================

def fetch_capm_inputs() -> dict:
    """Fetch the current risk-free rate and project debt rate from FRED.

    Discount rate build-up (CAPM, unlevered):
        WACC = rf + beta * ERP + size_premium

    The risk-free rate anchors the CAPM; the 10-year UST matches the 15-year
    asset horizon reasonably well. The Moody's Baa index is used as a debt rate
    proxy because investment-grade project finance spreads (150-200 bps over
    UST) embed in the Baa rate without requiring a proprietary terminal.

    Falls back to hardcoded constants (rf=4.3%, debt_rate=6.2%) if FRED is
    unreachable.

    Returns:
        dict: risk_free_rate (decimal), debt_rate (decimal)
    """
    def _latest(url: str, name: str) -> float | None:
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            # Read without parse_dates -- pandas 2.2 changed inference rules
            df = pd.read_csv(StringIO(resp.text))
            df = df[df.iloc[:, 1] != "."].copy()   # FRED uses "." for missing
            df.iloc[:, 1] = pd.to_numeric(df.iloc[:, 1], errors="coerce")
            df = df.dropna(subset=[df.columns[1]])
            val = df.iloc[-1, 1]
            return float(val) / 100.0 if pd.notna(val) else None
        except Exception as exc:
            log.warning("FRED fetch failed (%s): %s", name, exc)
            return None

    rf  = _latest(_FRED_GS10, "GS10")
    baa = _latest(_FRED_BAA,  "BAA")

    if rf is not None and baa is not None:
        source = "FRED (live)"
        log.info("FRED live: GS10 rf=%.2f%%  BAA=%.2f%%", rf * 100, baa * 100)
    else:
        rf, baa = 0.043, 0.062
        source = "hardcoded fallback"
        log.warning("FRED unreachable -- fallback: rf=%.2f%%  debt_rate=%.2f%%",
                    rf * 100, baa * 100)

    print(f"  Rate source: {source}  |  rf={rf:.2%}  debt_rate={baa:.2%}")
    return {"risk_free_rate": rf, "debt_rate": baa}


def _populate_assumptions() -> None:
    """Fill None fields in ASSUMPTIONS from data files and FRED.

    Order: CAPM rates -> NREL AEP -> ERCOT price/capture.
    Mutates the module-level ASSUMPTIONS dict in place.
    """
    capm = fetch_capm_inputs()
    ASSUMPTIONS["risk_free_rate"] = capm["risk_free_rate"]
    ASSUMPTIONS["debt_rate"]      = capm["debt_rate"]

    rf   = ASSUMPTIONS["risk_free_rate"]
    beta = ASSUMPTIONS["beta_unlevered"]
    erp  = ASSUMPTIONS["equity_risk_premium"]
    sp   = ASSUMPTIONS["size_premium"]
    # Each CAPM component compensates for a distinct risk:
    #   rf      -- time value of money (riskless base)
    #   beta*erp -- systematic market risk (correlated with broad economy)
    #   sp       -- illiquidity / size premium (no public-market exit path)
    ASSUMPTIONS["wacc_unlevered"] = rf + beta * erp + sp

    # P50 / P90 from NREL Layer 3
    nrel = pd.read_csv(_NREL_SUMMARY, index_col="metric")
    ASSUMPTIONS["p50_aep_mwh"] = float(nrel.loc["P50_AEP_MWh", "value"])
    ASSUMPTIONS["p90_aep_mwh"] = float(nrel.loc["P90_AEP_MWh", "value"])

    # Hub price and capture rate from ERCOT Layer 1
    if _ERCOT_SUMMARY.exists():
        ercot = pd.read_csv(_ERCOT_SUMMARY)
        row24 = ercot[ercot["year"] == 2024]
        if not row24.empty:
            if "hb_mean" in ercot.columns:
                ASSUMPTIONS["hub_price_2024"] = float(row24["hb_mean"].iloc[0])
            if "capture_ratio" in ercot.columns:
                ASSUMPTIONS["capture_rate_base"] = float(row24["capture_ratio"].iloc[0])

    if ASSUMPTIONS["hub_price_2024"] is None:
        ASSUMPTIONS["hub_price_2024"] = _HUB_PRICE_FALLBACK
        log.info("ERCOT Layer 1 absent -- hub_price_2024 fallback: $%.1f/MWh",
                 _HUB_PRICE_FALLBACK)
    if ASSUMPTIONS["capture_rate_base"] is None:
        ASSUMPTIONS["capture_rate_base"] = _CAPTURE_RATE_FALLBACK
        log.info("ERCOT Layer 1 absent -- capture_rate_base fallback: %.3f",
                 _CAPTURE_RATE_FALLBACK)

    ASSUMPTIONS["capture_rate_bull"] = ASSUMPTIONS["capture_rate_base"] + 0.03
    ASSUMPTIONS["capture_rate_bear"] = ASSUMPTIONS["capture_rate_base"] - 0.05


# ==============================================================================
# STEP 2 -- ANNUAL CASH FLOW MODEL (2025-2039)
# ==============================================================================

def build_cash_flows(scenario: str) -> pd.DataFrame:
    """Build 15-year annual cash flow model for one scenario.

    Revenue is split between a contracted tranche (Citibank affiliate hedge,
    77% at $23.50/MWh through 2027 per PEGI 10-K) and a merchant tranche
    (hub * capture rate). After hedge expiry in 2027, all revenue is merchant.

    Modeling the hedge through its disclosed expiry -- rather than assuming it
    has already lapsed -- is both more accurate and more defensible; it captures
    the real revenue cliff that any acquirer would need to underwrite.

    Costs escalate at 2.5%/yr (PPI-based); generation degrades at 0.5%/yr.
    CFADS = EBITDA (no capex; remaining-life major maintenance in fixed O&M).

    Args:
        scenario: "base", "bull", or "bear"

    Returns:
        DataFrame with one row per year.
    """
    assert scenario in ("base", "bull", "bear"), f"Unknown scenario: {scenario}"
    a = ASSUMPTIONS

    # P50 for base/bull; P90 for bear (more conservative for downside stress)
    aep_base = a["p50_aep_mwh"] if scenario != "bear" else a["p90_aep_mwh"]

    # Capture rate by scenario:
    #   bull -- storage and load growth absorbs wind-hour oversupply, ratio improves
    #   bear -- continued solar cannibalisation compresses midday LZ prices further
    capture = {
        "base": a["capture_rate_base"],
        "bull": a["capture_rate_bull"],
        "bear": a["capture_rate_bear"],
    }[scenario]

    model_end = a["cod_year"] + a["asset_life_years"]          # 2014 + 25 = 2039
    years = list(range(a["model_start_year"], model_end + 1))  # 2025..2039

    rows = []
    for t, year in enumerate(years):
        # t=0 in 2025 means the first year carries no degradation haircut yet
        aep = aep_base * (1.0 - a["degradation_pct"]) ** t

        # Hub price escalates one step per year beyond the 2024 base
        hub_price = (
            a["hub_price_2024"]
            * (1.0 + a["merchant_price_escalator"]) ** (year - 2024)
        )

        hedge_active = (year <= a["hedge_expiry_year"])

        if hedge_active:
            contracted_rev = aep * a["hedge_pct"] * a["hedge_price_mwh"]
            merchant_rev   = aep * (1.0 - a["hedge_pct"]) * hub_price * capture
        else:
            contracted_rev = 0.0
            merchant_rev   = aep * hub_price * capture
        total_rev = contracted_rev + merchant_rev

        esc         = (1.0 + a["om_escalator"]) ** t
        fixed_om    = a["om_fixed_mw_yr"]   * a["capacity_mw"] * esc
        variable_om = a["om_variable_mwh"]  * aep              * esc
        land_lease  = a["land_lease_mw_yr"] * a["capacity_mw"] * esc
        insurance   = a["insurance_mw_yr"]  * a["capacity_mw"] * esc
        total_opex  = fixed_om + variable_om + land_lease + insurance

        ebitda        = total_rev - total_opex
        ebitda_margin = ebitda / total_rev if total_rev > 0.0 else 0.0
        cfads         = ebitda

        rows.append({
            "year":           year,
            "scenario":       scenario,
            "aep_mwh":        aep,
            "hub_price":      hub_price,
            "capture_rate":   capture,
            "hedge_active":   hedge_active,
            "contracted_rev": contracted_rev,
            "merchant_rev":   merchant_rev,
            "total_rev":      total_rev,
            "fixed_om":       fixed_om,
            "variable_om":    variable_om,
            "land_lease":     land_lease,
            "insurance":      insurance,
            "total_opex":     total_opex,
            "ebitda":         ebitda,
            "ebitda_margin":  ebitda_margin,
            "cfads":          cfads,
        })

    return pd.DataFrame(rows)


# ==============================================================================
# STEP 3 -- DEBT SIZING (SCULPTED)
# ==============================================================================

def size_debt(cf_df: pd.DataFrame) -> dict:
    """Size project debt using cash flow sculpting against a DSCR floor.

    Sculpted debt is the standard project finance structure for wind assets.
    Principal is scheduled year-by-year so that:
        debt_service_t = CFADS_t / DSCR_target

    This maximises debt capacity while holding DSCR flat at the covenant floor.

    Lenders size on the BASE CASE (P50) CFADS -- not the bull -- because P50
    means 50% of years will fall short of the projection. The 1.35x buffer
    provides a margin of safety in those underperformance years. Conservative
    structures size on P90 CFADS instead; that approach is implicit in passing
    the bear-scenario DataFrame as input.

    Total debt principal = PV of the sculpted DS stream at the debt rate.
    This is exactly the price of a bond with unequal coupon payments -- the
    amount that makes NPV(debt_service stream at debt_rate) = 0.

    Args:
        cf_df: Base-case DataFrame from build_cash_flows("base").

    Returns:
        dict: debt_principal, annual_ds (Series), dscr_by_year (Series),
              amortization (DataFrame)
    """
    a          = ASSUMPTIONS
    r          = a["debt_rate"]
    dscr_floor = a["target_dscr"]
    tenor      = a["debt_tenor_years"]

    cfads      = cf_df.set_index("year")["cfads"]
    debt_years = cfads.index[:tenor]   # 2025-2039

    # Sculpted DS: maximum affordable payment each year at the DSCR floor
    max_ds = cfads.loc[debt_years] / dscr_floor

    # Debt principal = PV of sculpted DS stream at the debt rate
    discount_factors = np.array([1.0 / (1.0 + r) ** (i + 1) for i in range(len(max_ds))])
    debt_principal   = float((max_ds.values * discount_factors).sum())

    # Amortization: interest on opening balance, principal = DS - interest
    balance    = debt_principal
    amort_rows = []
    for i, year in enumerate(debt_years):
        ds        = float(max_ds.loc[year])
        interest  = balance * r
        principal = ds - interest
        balance   = max(balance - principal, 0.0)
        dscr      = float(cfads.loc[year]) / ds
        amort_rows.append({
            "year":      year,
            "ds":        ds,
            "interest":  interest,
            "principal": principal,
            "balance":   balance,
            "dscr":      dscr,
        })

    amort = pd.DataFrame(amort_rows)

    return {
        "debt_principal": debt_principal,
        "annual_ds":      pd.Series(max_ds.values, index=debt_years, name="annual_ds"),
        "dscr_by_year":   amort.set_index("year")["dscr"],
        "amortization":   amort,
    }


# ==============================================================================
# STEP 4 -- RETURNS (NPV, IRR, DSCR)
# ==============================================================================

def compute_returns(cf_df: pd.DataFrame, debt: dict,
                    base_npv_unlevered: float | None = None) -> dict:
    """Compute unlevered and levered returns for one scenario.

    Unlevered returns assume an all-equity capital structure:
      - npv_unlevered: PV of CFADS at WACC -- the enterprise value of the asset
      - irr_unlevered: IRR where CF_0 = -npv_unlevered (buying at fair value).
        By construction, IRR_unlevered = WACC -- this is the self-consistency
        check that confirms the model is internally correct.

    Levered returns reflect the equity investor's perspective:
      - Equity CF = CFADS - annual debt service
      - Entry equity = base_npv_unlevered - debt_principal.  The entry price is
        fixed at the BASE CASE fair value across all scenarios because the investor
        commits equity before knowing which scenario materialises. This ensures
        IRR_bull > IRR_base > IRR_bear, which is the analytically correct ordering
        for a scenario stress test against a single acquisition price.
      - irr_levered: IRR on equity cash flows from the equity entry cost.
        IRR_levered > IRR_unlevered when debt_rate < WACC (positive leverage).

    DSCR for each year = this scenario's CFADS / fixed base-case debt service.
    In the bear scenario, DSCR falls well below 1.35x, quantifying the downside
    risk the lender is underwriting. A DSCR below 1.0 is a default trigger.

    Equity discount rate = WACC + 200 bps (standard infrastructure PE premium
    over the unlevered asset WACC to reflect residual equity risk after leverage).

    Args:
        cf_df:              Annual cash flow DataFrame from build_cash_flows().
        debt:               Dict from size_debt().
        base_npv_unlevered: Entry EV to use for equity sizing (None -> use this
                            scenario's own NPV; pass base case value for bull/bear).

    Returns:
        dict: npv_unlevered, irr_unlevered, npv_levered, irr_levered,
              dscr_min, dscr_avg, equity_cf (Series)
    """
    a    = ASSUMPTIONS
    wacc = a["wacc_unlevered"]
    r_eq = wacc + 0.02   # equity hurdle = WACC + 200 bps leverage premium

    cfads = cf_df.set_index("year")["cfads"]
    n     = len(cfads)

    # Unlevered NPV: PV of CFADS discounted at WACC
    npv_unlevered = float(
        sum(cfads.iloc[t] / (1.0 + wacc) ** (t + 1) for t in range(n))
    )

    # Unlevered IRR: entry price is fixed at the BASE CASE fair value across all
    # scenarios. Without this, CF_0 = -npv_unlevered for each scenario's own NPV,
    # and IRR = WACC identically for every scenario (a mathematical tautology --
    # the IRR of [-PV_at_r, CF_1..N] always equals r). By anchoring CF_0 to the
    # base case acquisition price, bull/bear scenarios yield IRR > / < WACC,
    # producing the expected 200-400 bps spread across scenarios.
    entry_ev_unlev = base_npv_unlevered if base_npv_unlevered is not None else npv_unlevered
    irr_cfs        = np.concatenate([[-entry_ev_unlev], cfads.values])
    irr_unlevered  = float(npf.irr(irr_cfs))

    # Equity cash flows: CFADS less scheduled debt service
    annual_ds = debt["annual_ds"].reindex(cfads.index, fill_value=0.0)
    equity_cf = cfads - annual_ds

    # Levered NPV: PV of equity cash flows at equity discount rate
    npv_levered = float(
        sum(equity_cf.iloc[t] / (1.0 + r_eq) ** (t + 1) for t in range(n))
    )

    # Levered IRR: fixed entry price across all scenarios (base case EV)
    entry_ev     = base_npv_unlevered if base_npv_unlevered is not None else npv_unlevered
    equity_entry = max(entry_ev - debt["debt_principal"], 1.0)
    irr_lev_cfs  = np.concatenate([[-equity_entry], equity_cf.values])
    raw_irr_lev  = npf.irr(irr_lev_cfs)
    # Bear equity CFs may be entirely negative (DSCR < 1); IRR is undefined there
    irr_levered  = float(raw_irr_lev) if np.isfinite(raw_irr_lev) else float("nan")

    # DSCR: this scenario's CFADS against the fixed base-case debt service
    ds_years    = debt["annual_ds"].index
    dscr_series = cfads.reindex(ds_years) / debt["annual_ds"]
    dscr_min    = float(dscr_series.min())
    dscr_avg    = float(dscr_series.mean())

    return {
        "npv_unlevered":  npv_unlevered,
        "irr_unlevered":  irr_unlevered,
        "npv_levered":    npv_levered,
        "irr_levered":    irr_levered,
        "dscr_min":       dscr_min,
        "dscr_avg":       dscr_avg,
        "equity_cf":      equity_cf,
    }


# ==============================================================================
# STEP 5 -- SCENARIO SUMMARY TABLE
# ==============================================================================

def _scenario_kpis(results: dict, scenario: str) -> dict:
    """Extract display KPIs from a completed scenario result."""
    cf  = results[scenario]["cf_df"]
    ret = results[scenario]["returns"]
    return {
        "gen_gwh":         cf["aep_mwh"].mean() / 1e3,
        "capture_rate":    cf["capture_rate"].iloc[0],
        "avg_rev_mwh":     cf["total_rev"].sum() / cf["aep_mwh"].sum(),
        "ebitda_margin":   cf["ebitda"].sum() / cf["total_rev"].sum(),
        "npv_unlevered_m": ret["npv_unlevered"] / 1e6,
        "irr_unlevered":   ret["irr_unlevered"],
        "irr_levered":     ret["irr_levered"],
        "dscr_min":        ret["dscr_min"],
        "dscr_avg":        ret["dscr_avg"],
    }


def print_summary(results: dict, debt: dict) -> dict:
    """Print the Panhandle Wind scenario comparison table and return KPI dict."""
    a    = ASSUMPTIONS
    wacc = a["wacc_unlevered"]
    rf   = a["risk_free_rate"]
    beta = a["beta_unlevered"]
    erp  = a["equity_risk_premium"]
    sp   = a["size_premium"]
    dr   = a["debt_rate"]
    dp_m = debt["debt_principal"] / 1e6

    base_cf      = results["base"]["cf_df"]
    avg_cfads_m  = base_cf["cfads"].mean() / 1e6
    debt_cfads_x = dp_m / avg_cfads_m

    print()
    print("Panhandle Wind -- Valuation Summary (as of 2025)")
    print("=" * 62)
    print(
        f"Discount Rate (WACC):  {wacc:.1%}  "
        f"[CAPM: rf={rf:.1%} + b({beta})*ERP({erp:.1%}) + size({sp:.1%})]"
    )
    print(f"Debt Rate:             {dr:.1%}  [Moody's Baa: {dr:.1%}]")
    print(f"Debt Principal:        ${dp_m:.0f}m")
    print(f"Debt / avg CFADS:      {debt_cfads_x:.1f}x")
    print()

    kpis = {s: _scenario_kpis(results, s) for s in ("bear", "base", "bull")}

    W = 12
    print(f"{'':28}  {'Bear':>{W}}  {'Base':>{W}}  {'Bull':>{W}}")
    print("-" * (28 + 2 + (W + 2) * 3))

    def row(label, bev, bsv, buv):
        print(f"{label:<28}  {bev:>{W}}  {bsv:>{W}}  {buv:>{W}}")

    row("Generation (avg GWh/yr)",
        f"{kpis['bear']['gen_gwh']:,.0f} GWh",
        f"{kpis['base']['gen_gwh']:,.0f} GWh",
        f"{kpis['bull']['gen_gwh']:,.0f} GWh")
    row("Capture Rate",
        f"{kpis['bear']['capture_rate']:.3f}",
        f"{kpis['base']['capture_rate']:.3f}",
        f"{kpis['bull']['capture_rate']:.3f}")
    row("Avg Revenue ($/MWh)",
        f"${kpis['bear']['avg_rev_mwh']:.1f}",
        f"${kpis['base']['avg_rev_mwh']:.1f}",
        f"${kpis['bull']['avg_rev_mwh']:.1f}")
    row("EBITDA Margin",
        f"{kpis['bear']['ebitda_margin']:.0%}",
        f"{kpis['base']['ebitda_margin']:.0%}",
        f"{kpis['bull']['ebitda_margin']:.0%}")
    row("Unlevered NPV",
        f"${kpis['bear']['npv_unlevered_m']:.0f}m",
        f"${kpis['base']['npv_unlevered_m']:.0f}m",
        f"${kpis['bull']['npv_unlevered_m']:.0f}m")
    row("Unlevered IRR",
        f"{kpis['bear']['irr_unlevered']:.1%}",
        f"{kpis['base']['irr_unlevered']:.1%}",
        f"{kpis['bull']['irr_unlevered']:.1%}")
    def _irr(val: float) -> str:
        return f"{val:.1%}" if np.isfinite(val) else "N/A"

    row("Levered IRR",
        _irr(kpis["bear"]["irr_levered"]),
        _irr(kpis["base"]["irr_levered"]),
        _irr(kpis["bull"]["irr_levered"]))
    row("Min DSCR",
        f"{kpis['bear']['dscr_min']:.2f}x",
        f"{kpis['base']['dscr_min']:.2f}x",
        f"{kpis['bull']['dscr_min']:.2f}x")

    print("=" * 62)
    print()

    return kpis


# ==============================================================================
# STEP 6 -- WRITE OUTPUTS
# ==============================================================================

def write_outputs(results: dict, debt: dict, kpis: dict) -> None:
    """Write all DCF outputs to data/processed/."""
    for scenario in ("base", "bull", "bear"):
        path = DATA_PROCESSED / f"dcf_outputs_{scenario}.parquet"
        results[scenario]["cf_df"].to_parquet(path, index=False)
        log.info("Wrote %s", path)

    summary_rows = []
    for scenario in ("base", "bull", "bear"):
        k   = kpis[scenario]
        ret = results[scenario]["returns"]
        summary_rows.append({
            "scenario":         scenario,
            "gen_gwh":          k["gen_gwh"],
            "capture_rate":     k["capture_rate"],
            "avg_rev_mwh":      k["avg_rev_mwh"],
            "ebitda_margin":    k["ebitda_margin"],
            "npv_unlevered_m":  k["npv_unlevered_m"],
            "irr_unlevered":    k["irr_unlevered"],
            "irr_levered":      k["irr_levered"],
            "dscr_min":         k["dscr_min"],
            "dscr_avg":         ret["dscr_avg"],
            "debt_principal_m": debt["debt_principal"] / 1e6,
            "wacc":             ASSUMPTIONS["wacc_unlevered"],
            "debt_rate":        ASSUMPTIONS["debt_rate"],
            "risk_free_rate":   ASSUMPTIONS["risk_free_rate"],
        })
    pd.DataFrame(summary_rows).to_csv(
        DATA_PROCESSED / "valuation_summary.csv", index=False
    )
    log.info("Wrote valuation_summary.csv")


# ==============================================================================
# MAIN
# ==============================================================================

def main() -> None:
    """Run the complete Panhandle Wind financial model."""
    ensure_dirs()
    log.info("=" * 60)
    log.info("Panhandle Wind -- Financial Model (Layer 4)")
    log.info("Model: 2025-2039  |  %.1f MW  |  ERCOT LZ_NORTH",
             ASSUMPTIONS["capacity_mw"])
    log.info("=" * 60)

    log.info("Populating assumptions from FRED and data files...")
    _populate_assumptions()
    log.info(
        "WACC: %.2f%%  debt_rate: %.2f%%  P50: %.0f MWh  P90: %.0f MWh",
        ASSUMPTIONS["wacc_unlevered"] * 100,
        ASSUMPTIONS["debt_rate"] * 100,
        ASSUMPTIONS["p50_aep_mwh"],
        ASSUMPTIONS["p90_aep_mwh"],
    )

    log.info("Building cash flows (base / bull / bear)...")
    results: dict = {}
    for scenario in ("base", "bull", "bear"):
        cf = build_cash_flows(scenario)
        results[scenario] = {"cf_df": cf}
        log.info(
            "  %-4s  avg CFADS $%.1fm/yr  EBITDA margin %.0f%%",
            scenario,
            cf["cfads"].mean() / 1e6,
            cf["ebitda_margin"].mean() * 100,
        )

    log.info("Sizing debt on base case P50 CFADS (DSCR floor %.2fx)...",
             ASSUMPTIONS["target_dscr"])
    debt = size_debt(results["base"]["cf_df"])
    log.info(
        "  Debt principal: $%.1fm  avg annual DS: $%.1fm  tenor: %d yrs",
        debt["debt_principal"] / 1e6,
        debt["annual_ds"].mean() / 1e6,
        ASSUMPTIONS["debt_tenor_years"],
    )

    log.info("Computing returns...")
    # Compute base case first to get the fixed entry EV for levered IRR
    base_ret = compute_returns(results["base"]["cf_df"], debt)
    results["base"]["returns"] = base_ret
    base_npv = base_ret["npv_unlevered"]

    for scenario in ("bull", "bear"):
        ret = compute_returns(results[scenario]["cf_df"], debt,
                              base_npv_unlevered=base_npv)
        results[scenario]["returns"] = ret

    for scenario in ("base", "bull", "bear"):
        ret = results[scenario]["returns"]
        irr_lev_str = (
            f"{ret['irr_levered'] * 100:.1f}%"
            if np.isfinite(ret["irr_levered"])
            else "N/A (DSCR<1)"
        )
        log.info(
            "  %-4s  NPV $%.0fm  IRR_unlev %.1f%%  IRR_lev %s  min DSCR %.2fx",
            scenario,
            ret["npv_unlevered"] / 1e6,
            ret["irr_unlevered"] * 100,
            irr_lev_str,
            ret["dscr_min"],
        )

    kpis = print_summary(results, debt)
    write_outputs(results, debt, kpis)

    print("[OK] dcf_outputs_{base,bull,bear}.parquet written")
    print("[OK] valuation_summary.csv written")
    print()
    print("Next: streamlit run streamlit_app.py to launch the portfolio dashboard")


if __name__ == "__main__":
    main()
