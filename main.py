import numpy as np
from alm_engine import (
    load_balance_sheet_from_csv,
    simulate_cir_and_plot,
    simulate_alm_dynamics,
    plot_nii_histogram,
    calculate_and_plot_eve,
    print_simulation_summary,
    print_eve_summary_table,
    export_alm_summary_to_csv
)

# Scenario configuration
CSV_FILE = 'simulazione_ottima.csv'
COL_IDX = 8             # 2: Dec 31, 2021 (baseline), 8: hedged pro-forma
IRS_NOTIONAL = 70000.0      # 0.0 for baseline, 70000.0 ($70B) for hedged scenario

np.random.seed(42)

# Load balance sheet data
bs_dict = load_balance_sheet_from_csv(CSV_FILE, date_col_idx=COL_IDX)

# Simulate CIR interest rate paths
rates_matrix = simulate_cir_and_plot(
    r0=0.0025, 
    n_sims=1000000, 
    save_plot=True, 
    filename="1_rates_spaghetti_plot.png"
)

# Compute NII, CET1, and LCR trajectories
nii_matrix, cet1_matrix, lcr_matrix = simulate_alm_dynamics(
    rates_matrix, 
    bs_dict, 
    irs_notional=IRS_NOTIONAL, 
    save_plot=True, 
    filename="2_nii_spaghetti_plot.png"
)

# Plot 12-month NII distribution
loss_prob = plot_nii_histogram(
    nii_matrix, 
    month_target=12, 
    save_plot=True, 
    filename="4_nii_histogram.png"
)

# Compute and plot 12-month EVE distribution
eve_sim, eve_loss_prob = calculate_and_plot_eve(
    rates_matrix, 
    bs_dict, 
    irs_notional=IRS_NOTIONAL, 
    month_target=12, 
    save_plot=True, 
    filename="5_eve_distribution.png"
)

# Print summary tables
print("\nGenerating summary tables...")
df_nii_summary = print_simulation_summary(rates_matrix, nii_matrix, cet1_matrix, lcr_matrix, months=[6, 12, 18, 24])
df_eve_summary = print_eve_summary_table(rates_matrix, bs_dict, irs_notional=IRS_NOTIONAL, months=[6, 12, 18, 24])

# Export results to CSV
export_alm_summary_to_csv(
    rates_matrix, 
    nii_matrix, 
    bs_dict, 
    cet1_matrix=cet1_matrix, 
    lcr_matrix=lcr_matrix, 
    irs_notional=IRS_NOTIONAL,
    output_filename=f"alm_executive_summary_col_{COL_IDX}.csv"
)

print(f"\nDone (column {COL_IDX}).")