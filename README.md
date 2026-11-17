# Stochastic ALM & IRRBB Stress-Testing Engine
**Monte Carlo Balance Sheet Simulation (CIR Jump-Diffusion) — First Republic Bank Case Study**

A vectorized Asset-Liability Management (ALM) and Interest Rate Risk in the Banking Book (IRRBB) simulation engine built in Python. Starting from bank Call Report balance sheet data, the model simulates stochastic interest rate trajectories and evaluates the joint impact on short-term accounting earnings (**NII**), long-term economic capital (**EVE**), regulatory solvency (**CET1 Ratio**), and liquidity resilience (**LCR**).

---

## Methodology & Simulation Pipeline

For each Monte Carlo simulation path $i = 1, \dots, N$ (default $N = 100,000$) over a 24-month horizon ($t = 1, \dots, 24$), the engine executes four sequential stages:

1. **Balance Sheet Ingestion (`load_balance_sheet_from_csv`)**  
   Parses official FDIC Call Report line items (assets, liabilities, insured/uninsured deposits, and equity) from a CSV snapshot.
2. **Stochastic Interest Rate Engine (`simulate_cir_and_plot`)**  
   Generates the $i$-th Fed Funds rate trajectory $r_{i,t}$ using a **Cox-Ingersoll-Ross (CIR)** mean-reverting process augmented with **Jump-Diffusion** shocks calibrated to discrete FOMC tightening steps ($r_0 = 0.25\%$, $\bar{r} = 4.25\%$, $a = 0.50$, $\sigma = 8\%$):
   $$dr_{i,t} = a(\bar{r} - r_{i,t})dt + \sigma\sqrt{r_{i,t}}\,dW_{i,t} + J_{i,t}$$
3. **Dynamic Earnings, Capital & Liquidity (`simulate_alm_dynamics`)**  
   Maps each rate path $r_{i,t}$ into balance sheet cash flows using granular asset/liability pass-through betas (`ALM_PARAMS`) and an endogenous **uninsured deposit run trigger** when rates breach the $3.5\%$ threshold:
   * **Monthly Net Interest Income ($\text{NII}_{i,t}$):** Interest income on loans, securities, and cash reserves plus Pay-Fixed/Receive-Floating Interest Rate Swap (IRS) net cash flows, minus funding costs.
   * **Dynamic Solvency ($\text{CET1}_{i,t}$):** Tracks cumulative retained earnings/losses against Risk-Weighted Assets (RWA) calibrated to the Call Report.
   * **Dynamic Liquidity ($\text{LCR}_{i,t}$):** Tracks High-Quality Liquid Assets (HQLA) over 30-day stressed net cash outflows as uninsured deposits run off.
4. **Economic Value of Equity (`calculate_and_plot_eve`)**  
   Computes the cross-sectional distribution of $\text{EVE}_{i,T}$ by discounting asset and liability duration gaps ($D_A \approx 7.0\text{ yrs}$ vs. $D_L \approx 0.7\text{ yrs}$), net of the duration offset provided by the IRS hedge.

---

## Empirical Validation: First Republic Bank (FRB)

The engine is validated on First Republic Bank's balance sheet evolution (`simulazione_ottima.csv`), bridging historical regulatory reporting with management's pro-forma restructuring across sequential time snapshots:

* **Historical Baseline (`COL_IDX = 2`, `IRS_NOTIONAL = 0.0` — Q4 2021 FDIC Call Report):** Captures First Republic Bank's pre-crisis structure ($181.09B total assets, $79.21B fixed-rate 1–4 family mortgages, and $116.70B uninsured deposits). Illustrates the *IRRBB Paradox*—short-term NII remains positive in **85.6%** of scenarios at 12 months due to sticky deposit repricing, while **95.5%** of paths result in economic capital insolvency ($\text{EVE} \le 0$).
* **Interim Trajectory (`COL_IDX = 3..7`):** Sequential month-by-month evolution of balance sheet items under rising interest rates.
* **Pro-Forma Hedged Restructuring (`COL_IDX = 8`, `IRS_NOTIONAL = 70000.0`):** Target balance sheet engineered by the CEO/ALCO team combined with a **$70B Pay-Fixed IRS** overlay to neutralize duration mismatch, reinforce liquidity buffers, and prevent deposit flight.

### Stress Test Results (12-Month Horizon — Unhedged Baseline)

| Earnings View: 12-Month NII Distribution | Economic Value View: 12-Month EVE Distribution |
| :---: | :---: |
| ![NII Distribution](results/baseline/4_nii_histogram.png) | ![EVE Distribution](results/baseline/5_eve_distribution.png) |

---

## Project Architecture & Quickstart

```text
├── alm_engine.py          # Vectorized CIR engine, NII/EVE/CET1/LCR calculators & plotting
├── main.py                # Scenario configuration and execution script
├── simulazione_ottima.csv # FDIC Call Report historical & pro-forma snapshots
├── results/               # Generated plots and executive summary CSVs
└── requirements.txt       # Python dependencies
