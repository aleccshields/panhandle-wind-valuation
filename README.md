# Panhandle Wind — Asset Valuation Engine

A practitioner-grade valuation model for Panhandle Wind (Phases 1 & 2, ~400 MW, Carson County TX, ERCOT LZ_NORTH), built entirely from public data sources.

---

## The Financial Question

> **Given Panhandle Wind's historical generation profile and merchant exposure, what is the asset worth under various power price scenarios — and what is the dominant value driver?**

Panhandle Wind sells into the ERCOT real-time market (merchant) and through a PPA. The spread between the wind-weighted capture price and the hub price determines realized revenue. This model quantifies that spread historically, stress-tests it forward under P50/P90 resource scenarios, and produces a DCF valuation range with sensitivity to power prices, capacity factor, and discount rate.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                             │
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │  ERCOT API   │  │  EIA 860/923 │  │  NREL Wind Toolkit   │  │
│  │  DAM + RT    │  │  Generation  │  │  Hourly Resource     │  │
│  │  Settlement  │  │  & Capacity  │  │  (NSRDB)             │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
│         │                 │                      │              │
└─────────┼─────────────────┼──────────────────────┼─────────────┘
          │                 │                      │
          ▼                 ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                     src/ PIPELINE LAYERS                        │
│                                                                 │
│   ercot_pipeline.py    eia_pipeline.py    nrel_pipeline.py      │
│   (Layer 1)            (Layer 2)          (Layer 3)             │
│                                                                 │
│            Outputs: data/processed/*.parquet                    │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│               src/financial_model.py  (Layer 4)                 │
│                                                                 │
│   • Capture rate calculation (wind-weighted vs. hub price)      │
│   • DCF valuation (unlevered FCF → NPV → IRR)                   │
│   • P50/P90 resource exceedance scenarios                       │
│   • DSCR analysis for debt sizing                               │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    app/ STREAMLIT DASHBOARD                     │
│                                                                 │
│   01_price_analysis.py   →  Capture rate vs. hub price          │
│   02_generation.py       →  Capacity factors, P50/P90 bands     │
│   03_valuation.py        →  DCF outputs, tornado chart          │
│   04_ic_memo.py          →  Claude-generated IC memo viewer     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Data Sources

| Layer | Source | Dataset | URL | Access |
|-------|--------|---------|-----|--------|
| 1 | ERCOT | DAM & RT Settlement Point Prices (LZ_NORTH, HB_NORTH) | https://www.ercot.com/mp/data-products/data-product-details?id=NP6-788-ER | Public download (ZIP) |
| 2 | EIA Form 860 | Wind plant capacity & ownership | https://www.eia.gov/electricity/data/eia860/ | Public Excel download |
| 2 | EIA Form 923 | Monthly net generation (MWh) | https://www.eia.gov/electricity/data/eia923/ | Public Excel download |
| 3 | NREL Wind Toolkit | Hourly wind speed & power profiles | https://developer.nrel.gov/docs/wind/wind-toolkit/ | Free API key (NREL) |
| 4 | SEC EDGAR | PEGI 10-K (plant-level financials) | https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=PEGI | Public EDGAR search |

---

## How to Run Locally

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The app will open at `http://localhost:8501`. Data pipeline scripts (`src/ercot_pipeline.py`, etc.) must be run separately to populate `data/processed/` before all dashboard pages become active.

---

## Key Concepts

**Capture Rate**
The ratio of a generator's wind-weighted average realized price to the hub (reference) price for the same period. A capture rate below 100% means the plant generates more during low-price hours than high-price hours — common for wind, which produces when wind blows, not when prices spike. Example: if hub average is $40/MWh and the wind-weighted price is $32/MWh, the capture rate is 80%.

**P50 / P90**
Probabilistic energy production estimates based on long-run wind resource analysis. P50 is the median expected annual generation — exceeded in 50% of years. P90 is the downside case exceeded in 90% of years (i.e., a one-in-ten bad year). Lenders typically size debt to P90; equity underwriters use P50.

**Merchant Exposure**
The portion of a plant's revenue that is unhedged by a long-term contract (PPA or CfD) and therefore exposed to spot market price volatility. A fully contracted plant has zero merchant exposure; a fully merchant plant's revenue tracks the real-time price hour by hour. Panhandle Wind has a mix of contracted and merchant revenue.

**DSCR (Debt Service Coverage Ratio)**
Annual operating cash flow divided by scheduled principal + interest payments. A DSCR of 1.0x means cash flow exactly covers debt service; project finance lenders typically require a minimum DSCR of 1.25x–1.40x. The P90 DSCR is the standard stress-test metric for wind project debt sizing.

---

## Project Notes

Built with public data as a portfolio demonstration of energy finance analytical methods. Not investment advice. All valuations are illustrative.
