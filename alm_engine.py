import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Granular ALM Parameters (Initial Yield/Cost and Pass-Through Betas)
ALM_PARAMS = {
    # --- ASSETS ---
    'CHBAL':    {'yield': 0.0025, 'beta': 1.00},  # Cash & Fed balances (100% rate-sensitive)
    'SCAF':     {'yield': 0.0160, 'beta': 0.30},  # Available-for-Sale (AFS) securities
    'SCHA':     {'yield': 0.0185, 'beta': 0.10},  # Held-to-Maturity (HTM) securities (long duration)
    'LNRERES':  {'yield': 0.0290, 'beta': 0.10},  # 1-4 Family residential mortgages (mostly fixed-rate)
    'LNREMULT': {'yield': 0.0320, 'beta': 0.25},  # Multifamily real estate mortgages
    'LNOTHER':  {'yield': 0.0350, 'beta': 0.65},  # Other loans / HNW credit lines (mostly floating)

    # --- LIABILITIES ---
    'DEPNI':    {'cost': 0.0000, 'beta': 0.00},   # Noninterest-bearing deposits (0% rate, high run hazard)
    'DEPINT':   {'cost': 0.0020, 'beta': 0.75},   # Interest-bearing demand / MMDA deposits
    'NTRTIME':  {'cost': 0.0120, 'beta': 0.50},   # Time deposits / Certificates of Deposit (CDs)
    'OTHBRF':   {'cost': 0.0050, 'beta': 1.00},   # FHLB advances & wholesale borrowings (100% floating)
}


# -----------------------------------------------------------------------------------------
# 1. LOAD BALANCE SHEET FROM CALL REPORT CSV
# -----------------------------------------------------------------------------------------
def load_balance_sheet_from_csv(filepath: str = 'simulazione_ottima.csv', date_col_idx: int = 8) -> dict:
    """
    Parses the FRB FDIC Call Report CSV and extracts balance sheet items in $ Millions
    for the specified column (e.g., date_col_idx=2 -> Q4 2021 Status Quo; date_col_idx=8 -> Pro-Forma Hedged).
    """
    # Smart fallback: checks current directory first, then 'data/' folder
    if not os.path.exists(filepath) and os.path.exists(os.path.join('data', filepath)):
        filepath = os.path.join('data', filepath)

    df_raw = pd.read_csv(filepath, skiprows=2)
    df_data = df_raw.iloc[1:21, [0, 1, date_col_idx]].dropna()
    df_data.columns = ['Field', 'Description', 'Value']

    def to_millions(val_str) -> float:
        return float(str(val_str).replace(',', '')) / 1000.0

    bs_dict = {}
    for _, row in df_data.iterrows():
        try:
            bs_dict[row['Field'].strip()] = to_millions(row['Value'])
        except ValueError:
            pass

    print(f"[+] Balance sheet successfully loaded from '{filepath}' for date: {df_raw.columns[date_col_idx]}")
    return bs_dict


# -----------------------------------------------------------------------------------------
# 2. CIR INTEREST RATE SIMULATION WITH JUMP-DIFFUSION (FOMC RATE HIKES)
# -----------------------------------------------------------------------------------------
def simulate_cir_and_plot(
    r0: float = 0.0025,
    a: float = 0.50,
    r_bar: float = 0.0425,
    sigma: float = 0.08,
    T: int = 2,
    steps: int = 24,
    n_sims: int = 100000,
    save_plot: bool = True,
    filename: str = "1_rates_spaghetti_plot.png"
) -> np.ndarray:
    """
    Cox-Ingersoll-Ross (CIR) stochastic interest rate simulation augmented with
    Jump-Diffusion shocks calibrated to FOMC tightening cycles (2022-2023 stress scenario).
    """
    dt = T / steps
    rates = np.zeros((n_sims, steps + 1))
    rates[:, 0] = r0

    for t in range(1, steps + 1):
        r_prev = np.maximum(rates[:, t - 1], 0.0)
        dr = a * (r_bar - r_prev) * dt + sigma * np.sqrt(r_prev * dt) * np.random.normal(size=n_sims)

        jump = 0.0
        if t in [3, 6, 9]:
            jump = np.random.choice([0.0050, 0.0075, 0.0100], size=n_sims, p=[0.2, 0.6, 0.2])

        rates[:, t] = np.maximum(0.001, r_prev + dr + jump)

    if save_plot:
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(10, 5))
        time_axis = np.linspace(0, T * 12, steps + 1)

        plt.plot(time_axis, rates[:200, :].T * 100, color='#1f77b4', alpha=0.18)
        plt.plot(time_axis, np.mean(rates, axis=0) * 100, color='black', linewidth=2.5, label='Expected Average Rate')
        plt.plot(time_axis, np.percentile(rates, 95, axis=0) * 100, color='red', linestyle='--', linewidth=2,
                 label='Adverse Scenario 95th Pct (Stress Test)')

        plt.title("Fed Rates Monte Carlo Simulation (CIR Jump-Diffusion Stress Scenario)", fontsize=12, fontweight='bold')
        plt.xlabel("Months (0 = Early 2022)", fontsize=10)
        plt.ylabel("Fed Funds Rate (%)", fontsize=10)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] Rate Spaghetti Plot (CIR Stress) saved: {filename}")

    return rates[:, 1:]


# -----------------------------------------------------------------------------------------
# 3. VECTORIZED DYNAMIC NII, CET1 RATIO & LCR ENGINE
# -----------------------------------------------------------------------------------------
def simulate_alm_dynamics(
    rates_matrix: np.ndarray,
    bs_dict: dict,
    irs_notional: float = 70000.0,
    irs_fixed_rate: float = 0.0180,
    mortgage_yield: float = 0.0300,
    securities_yield: float = 0.0180,
    base_cost_of_funds: float = 0.0015,
    deposit_beta: float = 0.70,
    save_plot: bool = True,
    filename: str = "2_nii_spaghetti_plot.png"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes dynamic Net Interest Income (NII), CET1 Ratio, and Liquidity Coverage Ratio (LCR)
    across all Monte Carlo paths using fully vectorized NumPy operations.
    Automatically applies granular ALM_PARAMS if sub-line items are present in the balance sheet.
    """
    mortgages_amount = bs_dict['LNLSGR']
    res_mortgages = bs_dict.get('LNRERES', mortgages_amount * 0.58)
    other_loans = mortgages_amount - res_mortgages
    securities_amount = bs_dict['SC']
    cash_buffer_initial = bs_dict['CHBAL']
    liabilities_amount = bs_dict['LIAB']
    initial_equity = bs_dict['EQTOT']

    uninsured_dep_init = bs_dict.get('DEPUNINS', 116696.7)
    insured_dep_init = bs_dict.get('DEPINS', 40617.4)

    # Risk-Weighted Assets (RWA) calibrated to FRB Call Report (including off-balance sheet commitments -> ~9.7% CET1 at t=0)
    rwa = (0.50 * res_mortgages + 1.00 * other_loans + 0.20 * securities_amount) * 1.64

    n_sims, steps = rates_matrix.shape
    nii_matrix = np.zeros((n_sims, steps))
    cet1_matrix = np.zeros((n_sims, steps))
    lcr_matrix = np.zeros((n_sims, steps))

    r0 = rates_matrix[:, 0]
    current_cash = np.full(n_sims, cash_buffer_initial, dtype=float)
    curr_uninsured = np.full(n_sims, uninsured_dep_init, dtype=float)
    cum_nii = np.zeros(n_sims, dtype=float)

    # Check if granular Call Report sub-items are available to apply ALM_PARAMS
    granular_assets = ['SCAF', 'SCHA', 'LNRERES', 'LNREMULT', 'LNOTHER']
    granular_liabs = ['DEPNI', 'DEPINT', 'NTRTIME', 'OTHBRF']
    use_granular = all(k in bs_dict for k in granular_assets + granular_liabs)

    for t in range(steps):
        r_t = rates_matrix[:, t]
        delta_r = np.maximum(0.0, r_t - r0)

        # 1. Endogenous Uninsured Deposit Outflow (Bank Run trigger when Fed Funds Rate > 3.5%)
        run_rate = np.where(r_t > 0.035, 0.04, 0.005)
        outflow = curr_uninsured * run_rate
        curr_uninsured = np.maximum(0.0, curr_uninsured - outflow)
        current_cash = np.maximum(0.0, current_cash - outflow)

        # 2. Interest Income & Expense (Granular via ALM_PARAMS or Aggregated Fallback)
        if use_granular:
            ia_earning_assets = sum(
                bs_dict[k] * (ALM_PARAMS[k]['yield'] + ALM_PARAMS[k]['beta'] * delta_r)
                for k in granular_assets
            ) / 12.0
            total_interest_expense = sum(
                bs_dict[k] * (ALM_PARAMS[k]['cost'] + ALM_PARAMS[k]['beta'] * delta_r)
                for k in granular_liabs
            ) / 12.0
        else:
            ia_earning_assets = (mortgages_amount * mortgage_yield + securities_amount * securities_yield) / 12.0
            funding_cost_annual = np.maximum(base_cost_of_funds, base_cost_of_funds + deposit_beta * (r_t - r0))
            total_interest_expense = liabilities_amount * (funding_cost_annual / 12.0)

        income_cash = current_cash * (r_t / 12.0)
        total_interest_income = ia_earning_assets + income_cash

        # Net Payoff from Pay-Fixed / Receive-Floating Interest Rate Swap (IRS)
        irs_payoff = irs_notional * ((r_t - irs_fixed_rate) / 12.0)

        nii_t = total_interest_income + irs_payoff - total_interest_expense
        nii_matrix[:, t] = nii_t
        cum_nii += nii_t

        # Update Liquidity Buffer
        current_cash = np.maximum(0.0, current_cash + nii_t)

        # 3. Dynamic CET1 Ratio (%)
        cet1_matrix[:, t] = ((initial_equity + cum_nii) / rwa) * 100.0

        # 4. Dynamic Liquidity Coverage Ratio (LCR %) - HQLA / 30-Day Net Cash Outflows
        hqla = current_cash + 0.85 * securities_amount
        net_outflows_30d = 0.40 * curr_uninsured + 0.05 * insured_dep_init
        lcr_matrix[:, t] = (hqla / net_outflows_30d) * 100.0

    if save_plot:
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(11, 6))
        time_axis = np.arange(1, steps + 1)

        for i in range(min(120, n_sims)):
            for t in range(steps - 1):
                x_seg = [time_axis[t], time_axis[t + 1]]
                y_seg = [nii_matrix[i, t], nii_matrix[i, t + 1]]
                color = '#d62728' if nii_matrix[i, t + 1] <= 0 else '#38b6ff'
                alpha = 0.35 if nii_matrix[i, t + 1] <= 0 else 0.30
                plt.plot(x_seg, y_seg, color=color, alpha=alpha, linewidth=1.1)

        plt.axhline(0, color='darkred', linestyle='--', linewidth=2)
        plt.plot(time_axis, np.mean(nii_matrix, axis=0), color='black', linewidth=2.5, label='Expected Average NII')
        plt.title(f"NII Spaghetti Plot (IRS Notional = ${irs_notional/1000:.0f}B)", fontsize=12, fontweight='bold')
        plt.ylabel("Monthly NII ($ Millions)", fontsize=10)
        plt.xlabel("Months (0 = Early 2022)", fontsize=10)
        plt.legend(loc='lower left')
        plt.grid(True, alpha=0.3)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] NII Spaghetti Plot saved: {filename}")

    return nii_matrix, cet1_matrix, lcr_matrix


# -----------------------------------------------------------------------------------------
# 4. NII DISTRIBUTION HISTOGRAM & 95% VaR THRESHOLD
# -----------------------------------------------------------------------------------------
def plot_nii_histogram(
    nii_matrix: np.ndarray,
    month_target: int = 12,
    threshold: float = 0.0,
    save_plot: bool = True,
    filename: str = "4_nii_histogram.png"
) -> float:
    """
    Plots the cross-sectional NII distribution at a target horizon with Break-Even
    and 95% Value-at-Risk (VaR) markers.
    """
    nii_at_target = nii_matrix[:, month_target - 1]
    loss_prob = float(np.mean(nii_at_target <= threshold) * 100.0)

    if save_plot:
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        _, bins, patches = plt.hist(nii_at_target, bins=40, edgecolor='black', alpha=0.7)

        for i in range(len(patches)):
            bin_center = (bins[i] + bins[i + 1]) / 2.0
            if bin_center <= threshold:
                patches[i].set_facecolor('#d62728')
                patches[i].set_alpha(0.75)
            else:
                patches[i].set_facecolor('#38b6ff')
                patches[i].set_alpha(0.70)

        plt.axvline(threshold, color='darkred', linestyle='--', linewidth=2.5, label=f'Break-even (NII = {threshold:.0f})')
        var_5pct = np.percentile(nii_at_target, 5)
        plt.axvline(var_5pct, color='navy', linestyle=':', linewidth=2, label=f'NII VaR 95% ({var_5pct:.1f} $M)')

        plt.title(f"NII Distribution at {month_target} Months (Probability of Loss: {loss_prob:.1f}%)", fontsize=12, fontweight='bold')
        plt.xlabel("Monthly Net Interest Income ($ Millions)", fontsize=10)
        plt.ylabel("Frequency (Number of Scenarios)", fontsize=10)
        plt.legend(loc='upper left')
        plt.grid(True, alpha=0.3)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] NII Histogram saved: {filename}")

    return loss_prob


# -----------------------------------------------------------------------------------------
# 5. ECONOMIC VALUE OF EQUITY (EVE) & DURATION GAP DISTRIBUTION
# -----------------------------------------------------------------------------------------
def calculate_and_plot_eve(
    rates_matrix: np.ndarray,
    bs_dict: dict,
    irs_notional: float = 70000.0,
    irs_duration: float = 6.5,
    duration_loans: float = 7.0,
    duration_securities: float = 4.5,
    duration_liabilities: float = 0.7,
    month_target: int = 12,
    save_plot: bool = True,
    filename: str = "5_eve_distribution.png"
) -> tuple[np.ndarray, float]:
    """
    Computes the Economic Value of Equity (EVE) distribution accounting for asset/liability
    duration mismatch and IRS notional duration offset.
    """
    loans_amount = bs_dict['LNLSGR']
    securities_amount = bs_dict['SC']
    liabilities_amount = bs_dict['LIAB']
    initial_equity = bs_dict['EQTOT']

    r_initial = rates_matrix[:, 0]
    r_target = rates_matrix[:, month_target - 1]
    delta_r = r_target - r_initial

    delta_pv_loans = - (loans_amount * duration_loans - irs_notional * irs_duration) * delta_r
    delta_pv_securities = - securities_amount * duration_securities * delta_r
    total_delta_pv_assets = delta_pv_loans + delta_pv_securities
    delta_pv_liabilities = - liabilities_amount * duration_liabilities * delta_r

    eve_simulated = (initial_equity + (total_delta_pv_assets - delta_pv_liabilities)) / 1000.0
    initial_equity_b = initial_equity / 1000.0
    insolvency_prob = float(np.mean(eve_simulated <= 0.0) * 100.0)

    if save_plot:
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(10, 6))
        _, bins, patches = plt.hist(eve_simulated, bins=40, edgecolor='black', alpha=0.7)

        for i in range(len(patches)):
            bin_center = (bins[i] + bins[i + 1]) / 2.0
            if bin_center <= 0:
                patches[i].set_facecolor('#d62728')
                patches[i].set_alpha(0.75)
            else:
                patches[i].set_facecolor('#38b6ff')
                patches[i].set_alpha(0.70)

        plt.axvline(0, color='darkred', linestyle='--', linewidth=2.5, label='Capital Insolvency (EVE = 0)')
        plt.axvline(initial_equity_b, color='black', linestyle=':', linewidth=2, label=f'Initial Equity (${initial_equity_b:.1f}B)')

        plt.title(f"EVE Distribution at {month_target} Months (Probability EVE <= 0: {insolvency_prob:.1f}%)", fontsize=11, fontweight='bold')
        plt.xlabel("Economic Value of Equity - EVE ($ Billions)", fontsize=10)
        plt.ylabel("Frequency (Number of Scenarios)", fontsize=10)
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)

        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] EVE Distribution Plot saved: {filename}")

    return eve_simulated, insolvency_prob


# -----------------------------------------------------------------------------------------
# 6. ALCO HEDGING SENSITIVITY ANALYSIS (IRS NOTIONAL OPTIMIZATION FOR CFO)
# -----------------------------------------------------------------------------------------
def run_irs_sensitivity_analysis(
    rates_matrix: np.ndarray,
    bs_dict: dict,
    notional_grid_billions: list[float] = None,
    month_target: int = 12,
    save_plot: bool = True,
    filename: str = "6_irs_hedging_frontier.png"
) -> pd.DataFrame:
    """
    Evaluates the trade-off between Short-Term Earnings (NII) and Economic Capital Solvency (EVE)
    across a grid of Pay-Fixed IRS notionals to support CFO/ALCO hedge sizing.
    """
    if notional_grid_billions is None:
        notional_grid_billions = [0, 15, 30, 45, 60, 70, 85, 100]

    results = []
    for irs_b in notional_grid_billions:
        irs_m = irs_b * 1000.0
        nii_mat, _, _ = simulate_alm_dynamics(rates_matrix, bs_dict, irs_notional=irs_m, save_plot=False)
        _, eve_insolv_prob = calculate_and_plot_eve(rates_matrix, bs_dict, irs_notional=irs_m, month_target=month_target, save_plot=False)

        nii_m = nii_mat[:, month_target - 1]
        results.append({
            "IRS_Notional_$B": irs_b,
            "Mean_NII_$M": round(float(np.mean(nii_m)), 1),
            "NII_Loss_Prob_%": round(float(np.mean(nii_m <= 0) * 100.0), 1),
            "EVE_Insolvency_Prob_%": round(eve_insolv_prob, 1)
        })

    df_sens = pd.DataFrame(results)

    if save_plot:
        out_dir = os.path.dirname(filename)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        fig, ax1 = plt.subplots(figsize=(10, 5))

        color1 = '#1f77b4'
        ax1.set_xlabel('IRS Pay-Fixed Notional ($ Billions)', fontsize=10)
        ax1.set_ylabel('12M Expected Monthly NII ($M)', color=color1, fontsize=10)
        ax1.plot(df_sens["IRS_Notional_$B"], df_sens["Mean_NII_$M"], color=color1, marker='o', linewidth=2.5, label='Mean NII ($M)')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        color2 = '#d62728'
        ax2.set_ylabel('12M EVE Insolvency Probability (%)', color=color2, fontsize=10)
        ax2.plot(df_sens["IRS_Notional_$B"], df_sens["EVE_Insolvency_Prob_%"], color=color2, marker='s', linestyle='--', linewidth=2.5, label='EVE Insolvency Risk (%)')
        ax2.tick_params(axis='y', labelcolor=color2)

        plt.title("ALCO Hedging Optimization: Earnings (NII) vs. Capital Risk (EVE)", fontsize=12, fontweight='bold')
        fig.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"[+] ALCO Hedging Optimization Plot saved: {filename}")

    return df_sens


# -----------------------------------------------------------------------------------------
# 7. EXECUTIVE SUMMARY TABLES & CSV EXPORT
# -----------------------------------------------------------------------------------------
def print_simulation_summary(rates_matrix, nii_matrix, cet1_matrix=None, lcr_matrix=None, months=None) -> pd.DataFrame:
    if months is None:
        months = [6, 12, 18, 24]
    summary_data = []

    for m in months:
        idx = m - 1
        rates_m = rates_matrix[:, idx] * 100
        nii_m = nii_matrix[:, idx]
        loss_prob = np.mean(nii_m <= 0) * 100

        row = {
            "Horizon": f"t = {m}m",
            "Mean Rate (%)": f"{np.mean(rates_m):.2f}%",
            "P95 Rate (%)": f"{np.percentile(rates_m, 95):.2f}%",
            "Mean NII ($M)": f"{np.mean(nii_m):,.1f}",
            "NII VaR95 ($M)": f"{np.percentile(nii_m, 5):,.1f}",
            "Prob. NII <= 0": f"{loss_prob:.1f}%"
        }
        if cet1_matrix is not None:
            row["Mean CET1 (%)"] = f"{np.mean(cet1_matrix[:, idx]):.2f}%"
        if lcr_matrix is not None:
            row["Mean LCR (%)"] = f"{np.mean(lcr_matrix[:, idx]):.1f}%"
            row["Prob. LCR < 100%"] = f"{np.mean(lcr_matrix[:, idx] < 100) * 100:.1f}%"

        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)
    print("\n" + "=" * 95)
    print("      EXECUTIVE ALM SIMULATION SUMMARY (NII, CET1, LCR - FRB)      ")
    print("=" * 95)
    print(df_summary.to_string(index=False))
    print("=" * 95 + "\n")
    return df_summary


def print_eve_summary_table(
    rates_matrix, bs_dict,
    irs_notional=70000.0, irs_duration=6.5,
    duration_loans=7.0, duration_securities=4.5, duration_liabilities=0.7,
    months=None
) -> pd.DataFrame:
    if months is None:
        months = [6, 12, 18, 24]
    loans = bs_dict['LNLSGR']
    securities = bs_dict['SC']
    liabilities = bs_dict['LIAB']
    initial_equity_b = bs_dict['EQTOT'] / 1000.0

    r0 = rates_matrix[:, 0]
    summary_data = []

    for m in months:
        r_m = rates_matrix[:, m - 1]
        delta_r = r_m - r0

        delta_pv_assets = - (loans * duration_loans + securities * duration_securities - irs_notional * irs_duration) * delta_r
        delta_pv_liabilities = - liabilities * duration_liabilities * delta_r

        eve_m = initial_equity_b + (delta_pv_assets - delta_pv_liabilities) / 1000.0
        delta_eve_m = eve_m - initial_equity_b
        insolvency_prob = np.mean(eve_m <= 0) * 100

        summary_data.append({
            "Horizon": f"t = {m}m",
            "Fed Rate (%)": f"{np.mean(r_m)*100:.2f}%",
            "Mean EVE ($B)": f"${np.mean(eve_m):.2f}B",
            "VaR 95% EVE ($B)": f"${np.percentile(eve_m, 5):.2f}B",
            "Delta EVE ($B)": f"${np.mean(delta_eve_m):.2f}B",
            "Insolvency Risk": f"{insolvency_prob:.1f}%"
        })

    df_summary = pd.DataFrame(summary_data)
    print("\n" + "=" * 80)
    print("      ECONOMIC VALUE OF EQUITY (EVE) STRESS SUMMARY - FRB      ")
    print("=" * 80)
    print(df_summary.to_string(index=False))
    print("=" * 80 + "\n")
    return df_summary


def export_alm_summary_to_csv(
    rates_matrix, nii_matrix, bs_dict,
    cet1_matrix=None, lcr_matrix=None,
    irs_notional=70000.0, irs_duration=6.5,
    duration_loans=7.0, duration_securities=4.5, duration_liabilities=0.7,
    months=None,
    output_filename="alm_executive_summary.csv"
) -> pd.DataFrame:
    if months is None:
        months = [6, 12, 18, 24]
    out_dir = os.path.dirname(output_filename)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    loans = bs_dict['LNLSGR']
    securities = bs_dict['SC']
    liabilities = bs_dict['LIAB']
    initial_equity_b = bs_dict['EQTOT'] / 1000.0

    r0 = rates_matrix[:, 0]
    summary_list = []

    for m in months:
        idx = m - 1
        r_m = rates_matrix[:, idx]
        nii_m = nii_matrix[:, idx]

        delta_r = r_m - r0
        delta_pv_assets = - (loans * duration_loans + securities * duration_securities - irs_notional * irs_duration) * delta_r
        delta_pv_liabilities = - liabilities * duration_liabilities * delta_r

        eve_m = initial_equity_b + (delta_pv_assets - delta_pv_liabilities) / 1000.0
        delta_eve_m = eve_m - initial_equity_b

        row = {
            "Horizon": f"t = {m}m",
            "Mean_Fed_Rate_%": round(np.mean(r_m) * 100, 2),
            "P95_Fed_Rate_%": round(np.percentile(r_m, 95) * 100, 2),
            "Mean_NII_$M": round(np.mean(nii_m), 1),
            "VaR95_NII_$M": round(np.percentile(nii_m, 5), 1),
            "NII_Loss_Prob_%": round(np.mean(nii_m <= 0) * 100, 1),
            "Mean_EVE_$B": round(np.mean(eve_m), 2),
            "VaR95_EVE_$B": round(np.percentile(eve_m, 5), 2),
            "Mean_Delta_EVE_$B": round(np.mean(delta_eve_m), 2),
            "EVE_Insolvency_Prob_%": round(np.mean(eve_m <= 0) * 100, 1)
        }
        if cet1_matrix is not None:
            row["Mean_CET1_%"] = round(np.mean(cet1_matrix[:, idx]), 2)
        if lcr_matrix is not None:
            row["Mean_LCR_%"] = round(np.mean(lcr_matrix[:, idx]), 1)
            row["LCR_Breach_Prob_%"] = round(np.mean(lcr_matrix[:, idx] < 100) * 100, 1)

        summary_list.append(row)

    df_summary = pd.DataFrame(summary_list)
    df_summary.to_csv(output_filename, index=False)
    print(f"[+] Executive summary table successfully saved to '{output_filename}'")
    return df_summary