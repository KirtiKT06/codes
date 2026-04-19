# Standard imports
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import glob
import os
import re

# External EOS modules (Johnson LJ EOS + coexistence solver)
from johnson_lj_eos import jhonson_lj_eos
from eos_coexistence import compute_eos_psat

plt.style.use("seaborn-v0_8-darkgrid")
PLOT_DPI = 300
FIG_SIZE = (8, 6)

# Parameters
T = 1.12            # reduced temperature (LJ units)

# Directory of this script (used to build file paths safely)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# VAPOR PHASE (MC DATA + THERMODYNAMIC INTEGRATION)
# ============================================================
def process_vapor():
    """
    Reads Monte Carlo (MC) NVT simulation data for vapor phase.
    Computes:
        - Compressibility factor Z = P / (rho T)
        - Chemical potential via thermodynamic integration

    Returns:
        P_v        → pressure array (vapor)
        beta_mu_v  → dimensionless chemical potential (βμ)
    """

    # Get all MC data file
    files = sorted(glob.glob(os.path.join(BASE_DIR, "data_mc_nvt_modified", "exam_T*.dat")))

    rho_list = []
    Z_list = []
    Z_err_list = []

    # ---- Loop over all simulation files ----
    for file in files:
        filename = os.path.basename(file)

        # Extract density from filename using regex
        rho_match = re.search(r"rho(\d+\.?\d*)", filename)
        rho_val = float(rho_match.group(1))

        # Load data (skip header)
        data = np.loadtxt(file, skiprows=1)

        # Z is in column 3 (0-based indexing → 4th column)
        Z = data[:, 3]

        # Average Z over trajectory
        Z_mean = np.mean(Z)
        Z_sem  = np.std(Z) / np.sqrt(len(Z))
        Z_err = 1.645 * Z_sem
        rho_list.append(rho_val)
        Z_list.append(Z_mean)
        Z_err_list.append(Z_err)

    # Convert to numpy arrays
    rho = np.array(rho_list)
    Z = np.array(Z_list)
    Z_err = np.array(Z_err_list)

    # Add ideal gas reference point (rho → 0, Z → 1)
    rho = np.insert(rho, 0, 1e-6)
    Z = np.insert(Z, 0, 1.0)
    Z_err = np.insert(Z_err, 0, 0.0)

    # Sort data by density (important for integration)
    idx = np.argsort(rho)
    rho = rho[idx]
    Z = Z[idx]
    Z_err = Z_err[idx]

    # ---- Thermodynamic integration ----
    # Formula:
    # βμ = ln(ρ) + ∫[(Z-1)/ρ dρ] + (Z-1)
    integrand = (Z - 1.0) / np.maximum(rho, 1e-8)

    # Cumulative integral using trapezoidal rule
    integral = cumulative_trapezoid(integrand, rho, initial=0.0)

    # Final chemical potential expression
    beta_mu_v = np.log(rho) + integral + (Z - 1.0)

    # Pressure from EOS: P = Z ρ T
    P_v = Z * rho * T

    # Save data for report
    np.savetxt("Z_vs_rho.txt", np.column_stack((rho, Z, Z_err)), 
            header="rho    Z    Z_error")

    # Plot Z vs rho
    plt.figure(figsize=FIG_SIZE)
    plt.plot(rho, Z, 'o-')
    plt.errorbar(rho, Z,
                yerr=Z_err, 
                fmt='o-')
    plt.xlabel(f"Density ($\\rho$) $\\rightarrow$")
    plt.ylabel(f"Compressibility Factor ($Z$) $\\rightarrow$")
    plt.title("Z vs Density plot for Vapor Phase (with 90% CI)")
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig("Z_vs_rho.png", dpi=PLOT_DPI)
    plt.show()

    return P_v, beta_mu_v

# ============================================================
# LIQUID PHASE (LAMMPS DATA PROCESSING)
# ============================================================
def process_liquid(log_dir):
    """
    Reads LAMMPS log files and extracts:
        - Pressure
        - Density

    Uses only equilibrated portion (second half of trajectory).

    Returns:
        P   → pressure array
        rho → density array
    """

    files = sorted(glob.glob(os.path.join(log_dir, "log_*.lammps")))

    P_list, P_err_list = [], []
    rho_list, rho_err_list = [], []

    for file in files:
        pressures = []
        densities = []

        # Read file line-by-line
        with open(file, "r") as f:
            lines = f.readlines()

        reading = False

        for line in lines:
            # Detect start of thermo output block
            if "Step" in line and "Temp" in line:
                reading = True
                continue

            if reading:
                parts = line.split()
                # Skip malformed lines
                if len(parts) < 4:
                    continue

                try:
                    # Extract pressure and density
                    P = float(parts[2])
                    rho = float(parts[3])
                    pressures.append(P)
                    densities.append(rho)

                except:
                    continue

        pressures = np.array(pressures)
        densities = np.array(densities)

        # ---- Use equilibrated region ----
        # Take second half of simulation (better than fixed cutoff)
        half = len(pressures) // 2
        P_data = pressures[half:]
        rho_data = densities[half:]

        P_mean = np.mean(P_data)
        rho_mean = np.mean(rho_data)

        P_sem = np.std(P_data) / np.sqrt(len(P_data))
        rho_sem = np.std(rho_data) / np.sqrt(len(rho_data))
        P_err = 1.645 * P_sem
        rho_err = 1.645 * rho_sem

        P_list.append(P_mean)
        rho_list.append(rho_mean)
        P_err_list.append(P_err)
        rho_err_list.append(rho_err)

    # Convert to arrays
    P = np.array(P_list)
    rho = np.array(rho_list)
    P_er = np.array(P_err_list)
    rho_er = np.array(rho_err_list)

    # Sort by pressure
    idx = np.argsort(P)
    P = P[idx]
    rho = rho[idx]

    return P, rho, P_er[idx], rho_er[idx]

# ============================================================
# LIQUID CHEMICAL POTENTIAL (INTEGRATION IN PRESSURE SPACE)
# ============================================================
def compute_mu_liquid(P, rho, P0, rho0, beta_mu0):
    """
    Computes liquid chemical potential using pressure integration:

    βμ(P) = βμ(P0)
            + β [P0/ρ0 - P/ρ]
            + β ∫(dP / ρ)

    This avoids needing Z explicitly for liquid phase.

    Arguments:
        P, rho → simulation data
        P0, rho0, beta_mu0 → reference state (from EOS)

    Returns:
        beta_mu_l → liquid chemical potential
    """

    beta = 1.0 / T
    
    # Interpolation of rho(P)
    rho_interp = interp1d(P, rho, fill_value="extrapolate")

    # Density at reference pressure
    rho_P0 = rho_interp(P0)
    
    # Compute integral ∫ dP / rho
    integrand = 1.0 / rho
    integral_full = cumulative_trapezoid(integrand, P, initial=0.0)

    # Shift integral so that it is zero at P0
    integral_interp = interp1d(P, integral_full, fill_value="extrapolate")
    integral_shift = integral_full - integral_interp(P0)

    # Final expression
    beta_mu_l = (
        beta_mu0
        + beta * (P0 / rho0 - P / rho)
        + beta * integral_shift
    )

    return beta_mu_l

# ============================================================
# MAIN
# ============================================================
def main():

    # ---------------- Vapor ----------------
    P_v, mu_v = process_vapor()

    # ---------------- Liquid ----------------
    log_dir = "/data/lammps_log"
    P_l, rho_l, P_err, rho_err = process_liquid(log_dir)

    # ---- Reference state from Johnson EOS ----
    gamma = 3.0

    # EOS parameters (pre-fitted constants)
    x = np.array([
            0.8623085097507421, 2.976218765822098, -8.402230115796038,
            0.1054136629203555, -0.8564583828174598, 1.582759470107601,
            0.7639421948305453, 1.753173414312048, 2.798291772190376e+03,
            -4.8394220260857657e-02, 0.9963265197721935,
            -3.698000291272493e+01, 2.084012299434647e+01,
            8.305402124717285e+01, -9.574799715203068e+02,
            -1.477746229234994e+02, 6.398607852471505e+01,
            1.603993673294834e+01, 6.805916615864377e+01,
            -2.791293578795945e+03, -6.245128304568454,
            -8.116836104958410e+03, 1.488735559561229e+01,
            -1.059346754655084e+04, -1.131607632802822e+02,
            -8.867771540418822e+03, -3.986982844450543e+01,
            -4.689270299917261e+03, 2.593535277438717e+02,
            -2.694523589434903e+03, -7.218487631550215e+02,
            1.721802063863269e+02
        ])
    eos = jhonson_lj_eos(gamma, x)

    # Reference density (chosen in liquid region)
    rho0 = 0.7

    # Get EOS values at reference state
    P0, Z0, beta_mu_ex0, beta_mu0 = eos.lj_eos(T, rho0)

    # Compute liquid chemical potential
    mu_l = compute_mu_liquid(P_l, rho_l, P0, rho0, beta_mu0)

    # ============================================================
    # ALIGN LIQUID μ WITH VAPOR μ (REFERENCE CORRECTION)
    # ============================================================

    # Define overlap region
    P_min = max(min(P_v), min(P_l))
    P_max = min(max(P_v), max(P_l))

    if P_min >= P_max:
        raise ValueError("No overlapping pressure region!")

    # Interpolation functions
    f_v = interp1d(P_v, mu_v, fill_value="extrapolate")
    f_l = interp1d(P_l, mu_l, fill_value="extrapolate")

    # Compute average shift between curves (liquid → vapor reference)
    P_overlap = np.linspace(P_min, P_max, 100)
    shift_liquid = np.mean(f_v(P_overlap) - f_l(P_overlap))

    # Apply shift → ensures both μ are on same reference scale
    mu_l = mu_l + shift_liquid

    # Update interpolation
    f_l = interp1d(P_l, mu_l, fill_value="extrapolate")

    # Function whose root gives coexistence
    def diff(P):
        return f_v(P) - f_l(P)
    
    # Solve μ_v = μ_l
    P_sat = brentq(diff, P_min, P_max)
    print(f"\nP_sat = {P_sat:.5f}")

    # # Compare with EOS prediction
    # P_sat_eos, rho_v_eos, rho_l_eos= compute_eos_psat(T)
    # print(f"EOS P_sat ≈ {P_sat_eos:.5f}")

    # -------- Save data --------
    np.savetxt("mu_v_vs_P.txt", np.column_stack((P_v, mu_v)))
    np.savetxt("mu_l_vs_P.txt", np.column_stack((P_l, mu_l)))
    np.savetxt("rho_vs_P_liquid.txt",
           np.column_stack((P_l, rho_l, P_err, rho_err)),
           header="P    rho    P_error    rho_error")

    # ============================================================
    # DIAGNOSTIC PLOTS
    # ============================================================
    # P_test = np.linspace(P_min, P_max, 200)
    # # Check intersection visually
    # plt.figure(figsize=FIG_SIZE)
    # plt.plot(P_test, f_v(P_test), label="mu_v (vapor)")
    # plt.plot(P_test, f_l(P_test), label="mu_l (liquid)")
    # plt.xlabel(f"Pressure (P) $\\rightarrow$")
    # plt.ylabel(f"Chemical Potential ($\\mu$) as function of P $\\rightarrow$")
    # plt.legend()
    # plt.grid(alpha=0.5)
    # plt.tight_layout()
    # plt.title("Check intersection before solving")
    # plt.savefig("P_vs_mu_intersection.png", dpi=PLOT_DPI)
    # plt.show()

    # P_plot = np.linspace(P_min, P_max, 200)
    # # Plot difference (root = coexistence)
    # plt.figure(figsize=FIG_SIZE)
    # plt.plot(P_plot, diff(P_plot))
    # plt.axhline(0, linestyle='--')
    # plt.xlabel("Pressure (P) $\\rightarrow$")
    # plt.ylabel(f"$(\\mu_v - \\mu_l) \\rightarrow$")
    # plt.title("Coexistence Condition")
    # plt.grid(alpha=0.5)
    # plt.tight_layout()
    # plt.savefig("check_coexistence_condition.png", dpi=PLOT_DPI)
    # plt.show()

    plt.figure(figsize=FIG_SIZE)
    plt.errorbar(P_l, rho_l, 
                xerr=P_err, yerr=rho_err, 
                fmt='o-')
    plt.xlabel(f"Pressure (P) $\\rightarrow$")
    plt.ylabel(f"Density ($\\rho$) $\\rightarrow$")
    plt.title(f"$\\rho$ vs Pressure plot for liquid phase (with 90% CI)")
    plt.tight_layout()
    plt.savefig("rho_vs_P_liquid.png", dpi=PLOT_DPI)
    plt.show()

    # Final coexistence plot
    plt.figure(figsize=FIG_SIZE)
    plt.plot(P_v[5:12], mu_v[5:12], 'o-', label="Vapor Phase")
    plt.plot(P_l[:5], mu_l[:5], 'o-', label="Liquid Phase")
    plt.axvline(x=P_sat, linestyle=':', linewidth=1.5, color='black', label="P_sat")
    mu_sat = f_v(P_sat)
    plt.scatter(P_sat, mu_sat, color='black', zorder=5, label="Coexistence") 
    plt.text(P_sat, mu_sat+0.03, f"{P_sat:.3f}", ha='center')
    plt.title(f"$\\beta \\mu$ vs. Pressure (P) plot to find for coexistence point for liquid and vapour phases of a LJ fluid")
    plt.xlabel(f"Pressure (P) $\\rightarrow$")
    plt.ylabel(f"$\\beta \\mu \\rightarrow$")
    plt.legend()
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.savefig("Coexistence_plot.png")
    plt.show()

if __name__ == "__main__":
    main()