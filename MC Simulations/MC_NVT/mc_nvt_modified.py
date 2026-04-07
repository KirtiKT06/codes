import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.optimize import brentq
import glob
import os
import re

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

    idx = np.argsort(rho)
    rho = rho[idx]
    Z = Z[idx]

    # ---- Thermodynamic integration ----
    integrand = (Z - 1.0) / rho
    integral = cumulative_trapezoid(integrand, rho, initial=0.0)

    beta_mu_v = np.log(rho) + integral + (Z - 1.0)

    # Pressure
    P_v = Z * rho * T

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

    integrand = 1.0 / rho
    integral_full = cumulative_trapezoid(integrand, P, initial=0.0)

    # shift relative to reference state
    idx0 = np.argmin(np.abs(P - P0))
    integral = integral_full - integral_full[idx0]

    beta_mu_l = (
        beta_mu0
        + beta * (P0 / rho0 - P / rho)
        + beta * integral
    )

    return beta_mu_l


# ============================================================
# MAIN
# ============================================================
def main():

    # ---------------- Vapor ----------------
    P_v, mu_v = process_vapor()

    # ---------------- Liquid ----------------
    log_dir = "/home3/kelvin/lammps_logs"
    P_l, rho_l = process_liquid(log_dir)

    # ---- Reference state from EOS ----
    from johnson_lj_eos import jhonson_lj_eos

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
    # OVERLAP REGION (define FIRST)
    # ============================================================
    P_min = max(min(P_v), min(P_l))
    P_max = min(max(P_v), max(P_l))

    if P_min >= P_max:
        raise ValueError("No overlapping pressure region!")

    # ============================================================
    # INTERPOLATION
    # ============================================================
    f_v = interp1d(P_v, mu_v, fill_value="extrapolate")
    f_l = interp1d(P_l, mu_l, fill_value="extrapolate")

    # ============================================================
    # SHIFT (average over overlap)
    # ============================================================
    P_overlap = np.linspace(P_min, P_max, 100)

    shift = np.mean(f_l(P_overlap) - f_v(P_overlap))
    mu_v = mu_v + shift

    # Recreate interpolation after shift
    f_v = interp1d(P_v, mu_v, fill_value="extrapolate")

    def diff(P):
        return f_v(P) - f_l(P)
    print("P_min, P_max:", P_min, P_max)
    print("diff(P_min):", diff(P_min))
    print("diff(P_max):", diff(P_max))

    P_sat = brentq(diff, P_min, P_max)

    print(f"\nP_sat = {P_sat:.5f}")

    # ============================================================
    # 🔵 EOS COEXISTENCE (CORRECT VERSION)
    # ============================================================
    rho_v_range = np.linspace(0.01, 0.15, 300)
    rho_l_range = np.linspace(0.5, 0.9, 300)

    P_v_eos, mu_v_eos = [], []
    P_l_eos, mu_l_eos = [], []

    for rho in rho_v_range:
        P, Z, mu_ex, mu = eos.lj_eos(T, rho)
        P_v_eos.append(P)
        mu_v_eos.append(mu)

    for rho in rho_l_range:
        P, Z, mu_ex, mu = eos.lj_eos(T, rho)
        P_l_eos.append(P)
        mu_l_eos.append(mu)
    
    idx_v = np.argsort(P_v_eos)
    idx_l = np.argsort(P_l_eos)

    P_v_eos = np.array(P_v_eos)[idx_v]
    mu_v_eos = np.array(mu_v_eos)[idx_v]

    P_l_eos = np.array(P_l_eos)[idx_l]
    mu_l_eos = np.array(mu_l_eos)[idx_l]

    f_v = interp1d(P_v_eos, mu_v_eos, fill_value="extrapolate")
    f_l = interp1d(P_l_eos, mu_l_eos, fill_value="extrapolate")

    P_min = max(min(P_v_eos), min(P_l_eos))
    P_max = min(max(P_v_eos), max(P_l_eos))

    def diff(P):
        return f_v(P) - f_l(P)

    P_sat_eos = brentq(diff, P_min, P_max)

    print(f"EOS P_sat ≈ {P_sat_eos:.5f}")
            
    # Plot EOS curve with your data
    
    plt.figure()
    mask = (P_sat_eos >= P_min - 0.02) & (P_sat_eos <= P_max + 0.1)
    plt.plot(P_sat_eos[mask], mu_v_eos[mask], '--', label="EOS (shifted)")
    plt.plot(P_sat_eos[mask], mu_l_eos[mask], '--', label="EOS (shifted)")
    plt.plot(P_v, mu_v, 'o-', label="Vapor (MC)")
    plt.plot(P_l, mu_l, 'o-', label="Liquid (MD)")
    plt.xlabel("Pressure")
    plt.ylabel(r"$\beta \mu$")
    plt.title("EOS Validation vs Simulation")
    plt.legend()
    plt.grid()

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
    plt.plot(P_v, mu_v, 'o-', label="Vapor")
    plt.plot(P_l, mu_l, 'o-', label="Liquid")
    plt.xlabel("Pressure")
    plt.ylabel("βμ")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()