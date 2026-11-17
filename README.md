# irrbb-stochastic-stress-testing
**Stochastic ALM & IRRBB Stress-Testing Engine (CIR Jump-Diffusion) — First Republic Bank Case Study**

A vectorized Monte Carlo Asset-Liability Management (ALM) and Interest Rate Risk in the Banking Book (IRRBB) simulation engine built in Python. Starting from bank Call Report balance sheet data, the model simulates stochastic interest rate trajectories and evaluates the joint impact on short-term accounting earnings (**NII**), long-term economic capital (**EVE**), regulatory solvency (**CET1 Ratio**), and liquidity resilience (**LCR**).

---

## 1. End-to-End Simulation Pipeline

For each Monte Carlo simulation path $i = 1, \dots, N$ (default $N = 100,000$) over a 24-month horizon ($t = 1, \dots, 24$), the engine executes the following pipeline:

1. **Balance Sheet Ingestion (`load_balance_sheet_from_csv`)**  
   Parses official FDIC Call Report line items (assets, liabilities, insured/uninsured deposits, and equity) from a CSV snapshot[cite: 14].
2. **Stochastic Rate Engine (`simulate_cir_and_plot`)**  
   Generates the $i$-th Fed Funds rate trajectory $r_{i,t}$ using a **Cox-Ingersoll-Ross (CIR)** mean-reverting process augmented with **Jump-Diffusion** shocks calibrated to discrete FOMC tightening steps ($r_0 = 0.25\%$, $\bar{r} = 4.25\%$, $a = 0.50$, $\sigma = 8\%$)[cite: 6, 14]:
   $$dr_{i,t} = a(\bar{r} - r_{i,t})dt + \sigma\sqrt{r_{i,t}}\,dW_{i,t} + J_{i,t}$$
3. **Dynamic Earnings, Solvency & Liquidity (`calculate_nii`)**  
   Maps each rate path $r_{i,t}$ into balance sheet cash flows using granular asset/liability pass-through betas (`ALM_PARAMS`) and an endogenous **uninsured deposit run trigger** when rates breach the $3.5\%$ threshold[cite: 6, 14]:
   * **Monthly Net Interest Income ($\text{NII}_{i,t}$):** Interest income on loans, securities, and cash reserves plus Pay-Fixed/Receive-Floating Interest Rate Swap (IRS) net cash flows, minus funding costs[cite: 14].
   * **Dynamic Solvency ($\text{CET1}_{i,t}$):** Tracks cumulative retained earnings/losses against Risk-Weighted Assets (RWA) calibrated to the Call Report[cite: 14].
   * **Dynamic Liquidity ($\text{LCR}_{i,t}$):** Tracks High-Quality Liquid Assets (HQLA) over 30-day stressed net cash outflows as uninsured deposits run off[cite: 14].
4. **Economic Value of Equity (`calculate_and_plot_eve`)**  
   Computes the cross-sectional distribution of $\text{EVE}_{i,T}$ by discounting asset and liability duration gaps ($D_A \approx 7.0\text{ yrs}$ vs. $D_L \approx 0.7\text{ yrs}$), net of the duration offset provided by the IRS hedge[cite: 7, 14].

---

## 2. Case Study Application: First Republic Bank (FRB)

The engine is validated on First Republic Bank's Q4 2021 balance sheet ($181.1B total assets, 74.7% uninsured deposits, concentrated in long-duration fixed-rate jumbo mortgages)[cite: 3, 4].

* **Baseline Scenario (`COL_IDX = 2`, `IRS_NOTIONAL = 0.0`):** Captures the *IRRBB Paradox*—short-term NII remains positive in **85.6%** of scenarios at 12 months due to sticky deposit repricing, while **95.5%** of paths result in economic capital insolvency ($\text{EVE} \le 0$)[cite: 10, 13].
* **Hedged Pro-Forma Scenario (`COL_IDX = 8`, `IRS_NOTIONAL = 70000.0`):** Evaluates a balance sheet restructuring combined with a **$70B Pay-Fixed IRS** overlay to neutralize duration mismatch and protect capital.

### Visual Outputs (Baseline Stress Test)

| 12-Month NII Distribution (Earnings View) | 12-Month EVE Distribution (Economic Value View) |
| :---: | :---: |
| ![NII Distribution](results/baseline/4_nii_histogram.png) | ![EVE Distribution](results/baseline/5_eve_distribution.png) |

---

## 3. Repository Structure & Usage

```text
├── alm_engine.py          # Vectorized CIR engine, NII/EVE/CET1/LCR calculators & plotting
├── main.py                # Scenario configuration and execution script
├── simulazione_ottima.csv # FDIC Call Report input snapshots
├── results/               # Generated plots and executive summary CSVs
└── requirements.txt       # Python dependencies
