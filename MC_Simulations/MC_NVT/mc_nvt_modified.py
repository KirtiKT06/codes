import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import glob
import os
import re
from johnson_lj_eos import jhonson_lj_eos
from eos_coexistence import compute_eos_psat

# ============================================================
# PARAMETERS
# ============================================================
T = 1.12
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# VAPOR PHASE (MC DATA + INTEGRATION)
# ============================================================
def process_vapor():

    files = sorted(glob.glob(os.path.join(BASE_DIR, "data_mc_nvt_modified", "exam_T*.dat")))

    rho_list = []
    Z_list = []

    for file in files:
        filename = os.path.basename(file)

        rho_match = re.search(r"rho(\d+\.?\d*)", filename)
        rho_val = float(rho_match.group(1))

        data = np.loadtxt(file, skiprows=1)

        Z = data[:, 3]
        Z_mean = np.mean(Z)

        rho_list.append(rho_val)
        Z_list.append(Z_mean)

    # Sort by density
    rho = np.array(rho_list)
    Z = np.array(Z_list)

    rho = np.insert(rho, 0, 1e-6)
    Z = np.insert(Z, 0, 1.0)

    idx = np.argsort(rho)
    rho = rho[idx]
    Z = Z[idx]

    # ---- Thermodynamic integration ----
    integrand = (Z - 1.0) / np.maximum(rho, 1e-8)
    integral = cumulative_trapezoid(integrand, rho, initial=0.0)

    beta_mu_v = np.log(rho) + integral + (Z - 1.0)

    # Pressure
    P_v = Z * rho * T

    print("Vapor μ range:", np.min(beta_mu_v), np.max(beta_mu_v))

    return P_v, beta_mu_v


# ============================================================
# LIQUID PHASE (LAMMPS DATA)
# ============================================================
def process_liquid(log_dir):

    files = sorted(glob.glob(os.path.join(log_dir, "log_*.lammps")))

    P_list = []
    rho_list = []

    for file in files:
        pressures = []
        densities = []

        with open(file, "r") as f:
            lines = f.readlines()

        reading = False

        for line in lines:
            if "Step" in line and "Temp" in line:
                reading = True
                continue

            if reading:
                parts = line.split()
                if len(parts) < 4:
                    continue

                try:
                    P = float(parts[2])
                    rho = float(parts[3])

                    pressures.append(P)
                    densities.append(rho)

                except:
                    continue

        pressures = np.array(pressures)
        densities = np.array(densities)

        # Use second half of data (better than fixed 100)
        half = len(pressures) // 2
        P_list.append(np.mean(pressures[half:]))
        rho_list.append(np.mean(densities[half:]))

    P = np.array(P_list)
    rho = np.array(rho_list)

    # Sort by pressure
    idx = np.argsort(P)
    P = P[idx]
    rho = rho[idx]

    return P, rho


# ============================================================
# LIQUID CHEMICAL POTENTIAL (INTEGRATION)
# ============================================================
def compute_mu_liquid(P, rho, P0, rho0, beta_mu0):

    beta = 1.0 / T
    
    rho_interp = interp1d(P, rho, fill_value="extrapolate")

    # Evaluate rho at P0 (density at reference point)
    rho_P0 = rho_interp(P0)
    
    integrand = 1.0 / rho
    integral_full = cumulative_trapezoid(integrand, P, initial=0.0)

    # Interpolate integral at P0
    integral_interp = interp1d(P, integral_full, fill_value="extrapolate")
    integral_shift = integral_full - integral_interp(P0)

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
    P_l, rho_l = process_liquid(log_dir)

    # ---- Reference state from EOS ----
  

    gamma = 3.0
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

    rho0 = 0.7
    P0, Z0, beta_mu_ex0, beta_mu0 = eos.lj_eos(T, rho0)

    mu_l = compute_mu_liquid(P_l, rho_l, P0, rho0, beta_mu0)

    # ============================================================
    # ALIGN LIQUID μ (REFERENCE CORRECTION)
    # ============================================================

    # Define overlap region
    P_min = max(min(P_v), min(P_l))
    P_max = min(max(P_v), max(P_l))

    if P_min >= P_max:
        raise ValueError("No overlapping pressure region!")


    f_v = interp1d(P_v, mu_v, fill_value="extrapolate")
    f_l = interp1d(P_l, mu_l, fill_value="extrapolate")

    # Compute average shift (liquid → vapor reference)
    P_overlap = np.linspace(P_min, P_max, 100)
    shift_liquid = np.mean(f_v(P_overlap) - f_l(P_overlap))

    print("Applying liquid μ shift:", shift_liquid)

    # Apply shift
    mu_l = mu_l + shift_liquid

    f_l = interp1d(P_l, mu_l, fill_value="extrapolate")

    def diff(P):
        return f_v(P) - f_l(P)
    print("P_min, P_max:", P_min, P_max)
    print("diff(P_min):", diff(P_min))
    print("diff(P_max):", diff(P_max))

    P_sat = brentq(diff, P_min, P_max)

    print(f"\nP_sat = {P_sat:.5f}")

    P_sat_eos, rho_v_eos, rho_l_eos= compute_eos_psat(T)
    print(f"EOS P_sat ≈ {P_sat_eos:.5f}")

    P_test = np.linspace(P_min, P_max, 200)

    plt.figure()
    plt.plot(P_test, f_v(P_test), label="mu_v (vapor)")
    plt.plot(P_test, f_l(P_test), label="mu_l (liquid)")
    plt.legend()
    plt.grid()
    plt.title("Check intersection before solving")
    plt.show()

    # ============================================================
    # PLOTS
    # ============================================================
    P_plot = np.linspace(P_min, P_max, 200)

    plt.figure()
    plt.plot(P_plot, diff(P_plot))
    plt.axhline(0, linestyle='--')
    plt.xlabel("P")
    plt.ylabel("μ_v - μ_l")
    plt.title("Coexistence Condition")
    plt.grid()

    plt.figure()
    plt.plot(P_v[5:12], mu_v[5:12], 'o-', label="Vapor")
    plt.plot(P_l[:5], mu_l[:5], 'o-', label="Liquid")
    plt.axvline(x=P_sat, linestyle=':', linewidth=1.5, color='black', label="P_sat")
    mu_sat = f_v(P_sat)
    plt.scatter(P_sat, mu_sat, color='black', zorder=5, label="Coexistence")
    ymin = plt.ylim()[0]    
    plt.text(P_sat, mu_sat+0.03, f"{P_sat:.3f}", ha='center')
    plt.xlabel("Pressure")
    plt.ylabel("βμ")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()