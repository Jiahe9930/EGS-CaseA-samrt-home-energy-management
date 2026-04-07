import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from pathlib import Path

# =========================
# Parameters
# =========================
dt = 0.5  # hours
E_max = 5.0          # kWh
P_ch_max = 2.5       # kW
P_dis_max = 2.5      # kW
eta_ch = 0.95
eta_dis = 0.95
E0 = 2.5             # kWh, 50% initial SOC
DEGR_COST = 0.01     # £/kWh equivalent throughput for extension

DATA_FILE = "caseA_smart_home_30min_summer.csv"
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


# =========================
# Model 1: Self-consumption-first baseline
# =========================
def simulate_self_consumption(df, E_max, P_ch_max, P_dis_max, eta_ch, eta_dis, E0, dt):
    E = E0
    results = []

    for _, row in df.iterrows():
        pv = row["pv_kw"]
        load = row["base_load_kw"]
        import_tariff = row["import_tariff_gbp_per_kwh"]
        export_price = row["export_price_gbp_per_kwh"]

        # PV serves load first
        pv_to_load = min(pv, load)
        remaining_pv = pv - pv_to_load
        remaining_load = load - pv_to_load

        # Charge battery using remaining PV
        max_charge_by_headroom = max((E_max - E) / (eta_ch * dt), 0)
        pv_to_bat = min(remaining_pv, P_ch_max, max_charge_by_headroom)
        E = E + eta_ch * pv_to_bat * dt
        remaining_pv -= pv_to_bat

        # Discharge battery to serve remaining load
        max_discharge_by_energy = max(E * eta_dis / dt, 0)
        bat_to_load = min(remaining_load, P_dis_max, max_discharge_by_energy)
        E = E - bat_to_load * dt / eta_dis
        remaining_load -= bat_to_load

        # Grid balances the rest
        grid_to_load = remaining_load
        pv_to_grid = remaining_pv

        P_ch = pv_to_bat
        P_dis = bat_to_load
        P_imp = grid_to_load
        P_exp = pv_to_grid

        balance_residual = pv + P_imp + P_dis - (load + P_ch + P_exp)

        import_cost = P_imp * import_tariff * dt
        export_revenue = P_exp * export_price * dt
        net_cost = import_cost - export_revenue

        results.append({
            "timestamp": row["timestamp"],
            "pv_kw": pv,
            "load_kw": load,
            "pv_to_load_kw": pv_to_load,
            "pv_to_bat_kw": pv_to_bat,
            "pv_to_grid_kw": pv_to_grid,
            "bat_to_load_kw": bat_to_load,
            "grid_to_load_kw": grid_to_load,
            "P_ch_kw": P_ch,
            "P_dis_kw": P_dis,
            "P_imp_kw": P_imp,
            "P_exp_kw": P_exp,
            "E_kwh": E,
            "SOC": E / E_max,
            "import_cost_gbp": import_cost,
            "export_revenue_gbp": export_revenue,
            "degradation_cost_gbp": 0.0,
            "net_cost_gbp": net_cost,
            "balance_residual_kw": balance_residual,
        })

    return pd.DataFrame(results)


# =========================
# Model 2/3: Tariff-aware optimisation
# =========================
def optimise_tariff_aware(df, E_max, P_ch_max, P_dis_max, eta_ch, eta_dis, E0, dt, degr_cost_per_kwh=0.0):
    N = len(df)
    pv = df["pv_kw"].values
    load = df["base_load_kw"].values
    c_imp = df["import_tariff_gbp_per_kwh"].values
    c_exp = df["export_price_gbp_per_kwh"].values

    # Variable order:
    # [P_ch(0...N-1), P_dis(0...N-1), P_imp(0...N-1), P_exp(0...N-1), E(0...N-1)]
    n_vars = 5 * N

    def idx_ch(t): return t
    def idx_dis(t): return N + t
    def idx_imp(t): return 2 * N + t
    def idx_exp(t): return 3 * N + t
    def idx_E(t): return 4 * N + t

    # Objective
    c = np.zeros(n_vars)
    for t in range(N):
        c[idx_imp(t)] = c_imp[t] * dt
        c[idx_exp(t)] = -c_exp[t] * dt
        c[idx_ch(t)] = degr_cost_per_kwh * dt / 2
        c[idx_dis(t)] = degr_cost_per_kwh * dt / 2

    # Equality constraints
    A_eq = []
    b_eq = []

    # Power balance: pv + imp + dis = load + ch + exp
    for t in range(N):
        row = np.zeros(n_vars)
        row[idx_ch(t)] = -1
        row[idx_dis(t)] = 1
        row[idx_imp(t)] = 1
        row[idx_exp(t)] = -1
        A_eq.append(row)
        b_eq.append(load[t] - pv[t])

    # Battery dynamics
    for t in range(N):
        row = np.zeros(n_vars)
        row[idx_E(t)] = 1
        row[idx_ch(t)] = -eta_ch * dt
        row[idx_dis(t)] = dt / eta_dis
        if t == 0:
            b = E0
        else:
            row[idx_E(t - 1)] = -1
            b = 0
        A_eq.append(row)
        b_eq.append(b)

    # Inequality constraints: terminal energy E_end >= E0
    A_ub = []
    b_ub = []
    row = np.zeros(n_vars)
    row[idx_E(N - 1)] = -1
    A_ub.append(row)
    b_ub.append(-E0)

    # Bounds
    bounds = []
    bounds.extend([(0, P_ch_max)] * N)
    bounds.extend([(0, P_dis_max)] * N)
    bounds.extend([(0, None)] * N)
    bounds.extend([(0, None)] * N)
    bounds.extend([(0, E_max)] * N)

    res = linprog(
        c,
        A_ub=np.array(A_ub),
        b_ub=np.array(b_ub),
        A_eq=np.array(A_eq),
        b_eq=np.array(b_eq),
        bounds=bounds,
        method="highs",
    )

    if not res.success:
        raise RuntimeError(f"Optimisation failed: {res.message}")

    x = res.x
    P_ch = x[0:N]
    P_dis = x[N:2 * N]
    P_imp = x[2 * N:3 * N]
    P_exp = x[3 * N:4 * N]
    E = x[4 * N:5 * N]

    results = pd.DataFrame({
        "timestamp": df["timestamp"],
        "pv_kw": df["pv_kw"],
        "load_kw": df["base_load_kw"],
        "P_ch_kw": P_ch,
        "P_dis_kw": P_dis,
        "P_imp_kw": P_imp,
        "P_exp_kw": P_exp,
        "E_kwh": E,
        "SOC": E / E_max,
        "import_cost_gbp": P_imp * c_imp * dt,
        "export_revenue_gbp": P_exp * c_exp * dt,
        "degradation_cost_gbp": degr_cost_per_kwh * (P_ch + P_dis) * dt / 2,
        "net_cost_gbp": P_imp * c_imp * dt - P_exp * c_exp * dt + degr_cost_per_kwh * (P_ch + P_dis) * dt / 2,
        "balance_residual_kw": df["pv_kw"].values + P_imp + P_dis - (df["base_load_kw"].values + P_ch + P_exp),
    })
    return results


# =========================
# Summary helper
# =========================
def summarise_results(results, dt, E0):
    return {
        "total_pv_kwh": (results["pv_kw"] * dt).sum(),
        "total_load_kwh": (results["load_kw"] * dt).sum(),
        "grid_import_kwh": (results["P_imp_kw"] * dt).sum(),
        "grid_export_kwh": (results["P_exp_kw"] * dt).sum(),
        "battery_charge_kwh_ac": (results["P_ch_kw"] * dt).sum(),
        "battery_discharge_kwh_ac": (results["P_dis_kw"] * dt).sum(),
        "import_cost_gbp": results["import_cost_gbp"].sum(),
        "export_revenue_gbp": results["export_revenue_gbp"].sum(),
        "degradation_cost_gbp": results["degradation_cost_gbp"].sum(),
        "net_cost_gbp": results["net_cost_gbp"].sum(),
        "initial_E_kwh": E0,
        "final_E_kwh": results["E_kwh"].iloc[-1],
        "final_SOC": results["SOC"].iloc[-1],
        "max_abs_balance_residual_kw": results["balance_residual_kw"].abs().max(),
    }


# =========================
# Plot helpers
# =========================
def save_plots(results_sc, results_opt, results_deg):
    # SOC comparison
    plt.figure(figsize=(12, 4))
    plt.plot(results_sc["timestamp"], results_sc["SOC"], label="Self-consumption")
    plt.plot(results_opt["timestamp"], results_opt["SOC"], label="Tariff-aware optimisation")
    plt.plot(results_deg["timestamp"], results_deg["SOC"], label="Optimisation + degradation")
    plt.title("Battery SOC comparison")
    plt.xlabel("Time")
    plt.ylabel("SOC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "soc_comparison.png", dpi=200)
    plt.close()

    # Cumulative net cost
    plt.figure(figsize=(12, 4))
    plt.plot(results_sc["timestamp"], results_sc["net_cost_gbp"].cumsum(), label="Self-consumption")
    plt.plot(results_opt["timestamp"], results_opt["net_cost_gbp"].cumsum(), label="Tariff-aware optimisation")
    plt.plot(results_deg["timestamp"], results_deg["net_cost_gbp"].cumsum(), label="Optimisation + degradation")
    plt.title("Cumulative net cost comparison")
    plt.xlabel("Time")
    plt.ylabel("Cumulative net cost (£)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cumulative_net_cost.png", dpi=200)
    plt.close()

    # Cumulative equivalent throughput
    equiv_opt = (results_opt["P_ch_kw"] + results_opt["P_dis_kw"]) * dt / 2
    equiv_deg = (results_deg["P_ch_kw"] + results_deg["P_dis_kw"]) * dt / 2
    plt.figure(figsize=(12, 4))
    plt.plot(results_opt["timestamp"], equiv_opt.cumsum(), label="Without degradation cost")
    plt.plot(results_deg["timestamp"], equiv_deg.cumsum(), label="With degradation cost")
    plt.title("Cumulative equivalent battery throughput")
    plt.xlabel("Time")
    plt.ylabel("Equivalent throughput (kWh)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "cumulative_equivalent_throughput.png", dpi=200)
    plt.close()


# =========================
# Main
# =========================
def main():
    data_path = Path(DATA_FILE)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Could not find '{DATA_FILE}'. Put this .py file in the same folder as the CSV, or edit DATA_FILE."
        )

    df = pd.read_csv(data_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    results_sc = simulate_self_consumption(df, E_max, P_ch_max, P_dis_max, eta_ch, eta_dis, E0, dt)
    results_opt = optimise_tariff_aware(df, E_max, P_ch_max, P_dis_max, eta_ch, eta_dis, E0, dt, degr_cost_per_kwh=0.0)
    results_deg = optimise_tariff_aware(df, E_max, P_ch_max, P_dis_max, eta_ch, eta_dis, E0, dt, degr_cost_per_kwh=DEGR_COST)

    # Save detailed outputs
    results_sc.to_csv(OUT_DIR / "results_self_consumption.csv", index=False)
    results_opt.to_csv(OUT_DIR / "results_tariff_aware.csv", index=False)
    results_deg.to_csv(OUT_DIR / "results_tariff_aware_degradation.csv", index=False)

    summary_sc = summarise_results(results_sc, dt, E0)
    summary_opt = summarise_results(results_opt, dt, E0)
    summary_deg = summarise_results(results_deg, dt, E0)

    summary_df = pd.DataFrame([summary_sc, summary_opt, summary_deg], index=[
        "self_consumption",
        "tariff_aware_optimisation",
        "tariff_aware_with_degradation",
    ])
    summary_df.to_csv(OUT_DIR / "summary_results.csv")

    comparison_df = pd.DataFrame({
        "Metric": [
            "Grid import (kWh)",
            "Grid export (kWh)",
            "Battery charge (kWh AC)",
            "Battery discharge (kWh AC)",
            "Import cost (£)",
            "Export revenue (£)",
            "Degradation cost (£)",
            "Net cost (£)",
            "Final battery energy (kWh)",
            "Final SOC",
        ],
        "Self-consumption": [
            summary_sc["grid_import_kwh"],
            summary_sc["grid_export_kwh"],
            summary_sc["battery_charge_kwh_ac"],
            summary_sc["battery_discharge_kwh_ac"],
            summary_sc["import_cost_gbp"],
            summary_sc["export_revenue_gbp"],
            summary_sc["degradation_cost_gbp"],
            summary_sc["net_cost_gbp"],
            summary_sc["final_E_kwh"],
            summary_sc["final_SOC"],
        ],
        "Tariff-aware optimisation": [
            summary_opt["grid_import_kwh"],
            summary_opt["grid_export_kwh"],
            summary_opt["battery_charge_kwh_ac"],
            summary_opt["battery_discharge_kwh_ac"],
            summary_opt["import_cost_gbp"],
            summary_opt["export_revenue_gbp"],
            summary_opt["degradation_cost_gbp"],
            summary_opt["net_cost_gbp"],
            summary_opt["final_E_kwh"],
            summary_opt["final_SOC"],
        ],
        "Optimisation + degradation": [
            summary_deg["grid_import_kwh"],
            summary_deg["grid_export_kwh"],
            summary_deg["battery_charge_kwh_ac"],
            summary_deg["battery_discharge_kwh_ac"],
            summary_deg["import_cost_gbp"],
            summary_deg["export_revenue_gbp"],
            summary_deg["degradation_cost_gbp"],
            summary_deg["net_cost_gbp"],
            summary_deg["final_E_kwh"],
            summary_deg["final_SOC"],
        ],
    })
    comparison_df.to_csv(OUT_DIR / "comparison_table.csv", index=False)

    save_plots(results_sc, results_opt, results_deg)

    print("Run complete.")
    print(f"Detailed outputs saved in: {OUT_DIR.resolve()}")
    print("\nSummary results:")
    print(summary_df.round(4))


if __name__ == "__main__":
    main()
