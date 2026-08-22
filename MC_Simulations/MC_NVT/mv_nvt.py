"""
Post-processing and analysis script for Monte Carlo NVT simulations.

This script performs the following tasks:
1. Q1: Block averaging of pressure and energy time series with confidence intervals
2. Q2: Replica-averaged Widom insertion analysis
3. Q3: Equation of state analysis and thermodynamic integration,
       including comparison with Widom insertion results

All outputs (tables and figures) are saved in the working directory.
"""

import numpy as np
from scipy.stats import t
from scipy.integrate import cumulative_trapezoid
import glob
import os
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt

# Base directory where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Output directory: simulations/data
OUT_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================
# Q1: Block averaging utilities
# ============================================================

class MCNVTBlockAnalysis:
    """
    Performs autocorrelation analysis, block averaging,
    and confidence interval estimation for MC time series.
    """

    def __init__(self, data, n_particles):
        """
        Parameters
        ----------
        data : ndarray
            Time series data [step, P_inst, U_inst]
        n_particles : int
            Number of particles in the simulation
        """
        self.p = data[:, 1]                 # Instantaneous pressure
        self.u = data[:, 2] / n_particles   # Energy per particle
        self.max_lag = 200                  # Max lag for autocorrelation

    def autocorrelation(self, series):
        """
        Computes normalized autocorrelation function C(t).

        C(t) = <x(0)x(t)> / <x^2>
        """
        x = series - np.mean(series)
        var = np.var(x)
        n = len(x)
        c = np.zeros(self.max_lag)

        for t in range(self.max_lag):
            c[t] = np.sum(x[:n - t] * x[t:]) / ((n - t) * var)
        return c

    def correlation_time(self):
        """
        Estimates correlation time as first lag
        where C(t) < exp(-1).
        """
        c = self.autocorrelation(self.p)
        idx = np.where(c < np.exp(-1))[0]
        return idx[0] if len(idx) > 0 else self.max_lag

    def block_average(self, series, block_size):
        """
        Performs block averaging to reduce correlation effects.
        """
        n_blocks = len(series) // block_size
        return np.array([
            np.mean(series[i * block_size:(i + 1) * block_size])
            for i in range(n_blocks)
        ])

    @staticmethod
    def mean_and_ci(blocks, confidence=0.90):
        """
        Computes mean and confidence interval using Student-t statistics.
        """
        mean = np.mean(blocks)
        stderr = np.std(blocks, ddof=1) / np.sqrt(len(blocks))
        tval = t.ppf((1 + confidence) / 2, len(blocks) - 1)
        return mean, tval * stderr


def parse_timeseries_state(filename):
    """
    Extracts (T*, rho*, replica index) from time series filename.
    """
    parts = os.path.basename(filename).replace(".dat", "").split("_")
    T = float(parts[1][1:])
    rho = float(parts[2][3:])
    rep = int(parts[3][3:])
    return T, rho, rep

# ============================================================
# Q2: Widom replica averaging
# ============================================================

def analyze_widom(filename):
    """
    Computes replica-averaged Widom insertion results
    and saves them as a CSV table.
    """
    data = np.loadtxt(filename)
    df = pd.DataFrame(
        data, columns=["T", "rho", "beta_mu", "n_insert", "seed"]
    )

    rows = []
    for (T, rho), g in df.groupby(["T", "rho"]):
        mu = g["beta_mu"].mean()
        err = g["beta_mu"].std(ddof=1) / np.sqrt(len(g))
        rows.append([T, rho, len(g), mu, err])

    df_final = pd.DataFrame(
        rows,
        columns=["T*", "rho*", "N_rep", "<βμ_ex>", "err(βμ_ex)"]
    )
    df_final.to_csv(os.path.join(OUT_DIR, "Q2_FINAL_TABLE.csv"), index=False)

    print("\nQ2 FINAL RESULTS (Widom)")
    print(tabulate(df_final, headers="keys", tablefmt="github", showindex=False, floatfmt=".6f"))
    return df_final

# ============================================================
# Q3: EOS and thermodynamic integration
# ============================================================

def analyze_eos(filename, widom_df=None):
    """
    Performs EOS analysis:
    - Computes compressibility factor Z
    - Computes excess chemical potential via thermodynamic integration
    - Produces required plots
    """
    data = np.loadtxt(filename)
    T = data[0, 0]
    rho = data[:, 1]
    P = data[:, 2]

    # Compressibility factor
    Z = P / (rho * T)

    # Thermodynamic integration:
    # βμ_ex(ρ) = Z - 1 + ∫_0^ρ [(Z(ρ') - 1)/ρ'] dρ'
    integrand = (Z - 1) / rho
    beta_mu_TI = Z - 1 + cumulative_trapezoid(integrand, rho, initial=0.0)

    # Save numerical results
    df_ti = pd.DataFrame({
        "rho*": rho,
        "Z": Z,
        "beta_mu_TI": beta_mu_TI
    })
    df_ti.to_csv(os.path.join(OUT_DIR, "Q3_TI_results.csv"), index=False)

    # -----------------------------------------
    # Save Widom vs TI comparison table
    # -----------------------------------------
    if widom_df is not None:
        widom_T = widom_df[widom_df["T*"] == T]

        if not widom_T.empty:
            rows = []

            for _, row in widom_T.iterrows():
                rho_w = row["rho*"]

                # Find nearest rho index in EOS data
                idx = np.argmin(np.abs(rho - rho_w))

                rows.append([
                    T,
                    rho_w,
                    beta_mu_TI[idx],
                    row["<βμ_ex>"],
                    row["err(βμ_ex)"]
                ])

            df_compare = pd.DataFrame(
                rows,
                columns=[
                    "T*",
                    "rho*",
                    "beta_mu_TI",
                    "beta_mu_Widom",
                    "err_beta_mu_Widom"
                ]
            )

            df_compare = df_compare.round(6)

            df_compare.to_csv(
                os.path.join(OUT_DIR, "Q3_mu_TI_vs_Widom.csv"),
                index=False
            )

            print("\nQ3: μ_ex comparison (TI vs Widom)")
            print(tabulate(
                df_compare,
                headers="keys",
                tablefmt="github",
                showindex=False,
                floatfmt=".6f"
            ))

    # -----------------------------
    # Plot 1: Isotherm P* vs rho*
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(rho, P, "o-", color='steelblue', label="MC (NVT)")
    plt.xlabel(r"$\rho^*→$", fontsize=13)
    plt.ylabel(r"$P^*→$", fontsize=13)
    plt.title(rf"Isotherm $P^*(\rho^*)$ at $T^* = {T}$", fontsize=14)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.tick_params(axis='both', labelsize=11)
    plt.savefig(os.path.join(OUT_DIR, "Q3_isotherm_P_vs_rho.png"), dpi=600)
    plt.close()

    # -----------------------------
    # Plot 2: Compressibility factor
    # -----------------------------
    plt.figure(figsize=(7, 5))
    plt.plot(rho, Z, "s-", color='steelblue', label=r"$Z = P/(\rho T)$")
    plt.axhline(1.0, color="k", linestyle="--", label="Ideal gas")
    plt.xlabel(r"$\rho^*→$", fontsize=13)
    plt.ylabel(r"$Z→$", fontsize=13)
    plt.title(rf"Compressibility factor at $T^* = {T}$", fontsize=14)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.tick_params(axis='both', labelsize=11)
    plt.savefig(os.path.join(OUT_DIR, "Q3_Z_vs_rho.png"), dpi=600)
    plt.close()

    # -----------------------------
    # Plot 3: μ_ex comparison (TI vs Widom)
    # -----------------------------
    if widom_df is not None:
        widom_T = widom_df[widom_df["T*"] == T]

        if not widom_T.empty:
            plt.figure(figsize=(7, 5))
            plt.plot(rho, beta_mu_TI, "o-", color='steelblue', label="Thermodynamic integration")
            plt.errorbar(
                widom_T["rho*"],
                widom_T["<βμ_ex>"],
                yerr=widom_T["err(βμ_ex)"],
                fmt="s",
                color="orange",
                ecolor="orange",
                capsize=4,
                label="Widom insertion",
                markersize=7,
                elinewidth=1.2,
                capthick=1.2
            )
            plt.xlabel(r"$\rho^*→$", fontsize=13)
            plt.ylabel(r"$\beta \mu^{(ex)}→$", fontsize=13)
            plt.title(rf"Excess chemical potential at $T^* = {T}$", fontsize=14)
            plt.grid(alpha=0.2)
            plt.legend()
            plt.tight_layout()
            plt.tick_params(axis='both', labelsize=11)
            plt.savefig(os.path.join(OUT_DIR, "Q3_mu_comparison.png"), dpi=600)
            plt.close()

# ============================================================
# MAIN DRIVER
# ============================================================

def main():
    """
    Main analysis pipeline for Q1, Q2, and Q3.
    """
    n_particles = 256

    # ---------------- Q1: Block Averaging ----------------
    ts_files = sorted(
        glob.glob(os.path.join(BASE_DIR, "timeseries_T*_rho*_rep*.dat"))
    )
    replica_rows = []

    for f in ts_files:
        data = np.loadtxt(f, comments="#")
        T, rho, rep = parse_timeseries_state(f)

        analysis = MCNVTBlockAnalysis(data, n_particles)
        tau = analysis.correlation_time()
        block_size = max(1, 10 * tau)

        p_blocks = analysis.block_average(analysis.p, block_size)
        u_blocks = analysis.block_average(analysis.u, block_size)

        p_mean, _ = analysis.mean_and_ci(p_blocks)
        u_mean, _ = analysis.mean_and_ci(u_blocks)

        replica_rows.append([T, rho, rep, p_mean, u_mean])
    
    df_rep = pd.DataFrame(
    replica_rows,
    columns=["T*", "rho*", "rep", "<P>", "<U/N>"])

    # Replica-averaged results with confidence intervals
    CONFIDENCE = 0.90
    tval = t.ppf((1 + CONFIDENCE) / 2, 6 - 1)  # 6 replicas → dof = 5
    rows = []
    for (T, rho), g in df_rep.groupby(["T*", "rho*"]):
        n = len(g)
        P_mean = g["<P>"].mean()
        U_mean = g["<U/N>"].mean()

        P_std = g["<P>"].std(ddof=1)
        U_std = g["<U/N>"].std(ddof=1)

        CI_P = tval * P_std / np.sqrt(n)
        CI_U = tval * U_std / np.sqrt(n)
        rows.append([
            T, rho, n,
            P_mean, CI_P,
            U_mean, CI_U
    ])
    df_q1_final = pd.DataFrame(
        rows,
        columns=[
            "T*", "rho*", "N_rep",
            "<P>", "CI_P",
            "<U/N>", "CI_U/N"
        ]
    )
    df_q1_final = df_q1_final.round(6)
    df_q1_final.to_csv(os.path.join(OUT_DIR, "Q1_FINAL_TABLE.csv"), index=False)

    print("\nQ1 FINAL RESULTS (Pressure & Energy)")
    print(tabulate(df_q1_final, headers="keys", tablefmt="github", showindex=False, floatfmt=".6f"))

    # ---------------- Q2: Widom insertion ----------------
    widom_file = os.path.join(BASE_DIR, "widom_results.dat")
    if os.path.exists(widom_file):
        widom_df = analyze_widom(widom_file)

    # ---------------- Q3: Thermodynamic integration ----------------
    for f in glob.glob(os.path.join(BASE_DIR, "eos_T*.dat")):
        analyze_eos(f, widom_df)

    print("\nAnalysis complete.")

if __name__ == "__main__":
    main()