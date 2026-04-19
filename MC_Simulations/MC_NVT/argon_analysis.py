import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

plt.style.use("seaborn-v0_8-darkgrid")
PLOT_DPI = 300
FIG_SIZE = (8, 6)

class argon_analysis:

    def __init__(self, run_dir):
        self.run_dir = run_dir

    # ----------------------------
    # 1. RDF
    # ----------------------------
    def load_rdf(self):
        filepath = f"{self.run_dir}/rdf.dat"
        blocks = []
        current = []

        with open(filepath) as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue

                parts = line.split()

                if len(parts) == 2:
                    if current:
                        blocks.append(current)
                        current = []
                    continue

                if len(parts) >= 4:
                    current.append([float(parts[1]), float(parts[2])])

        if current:
            blocks.append(current)

        # average all blocks
        blocks = [np.array(b) for b in blocks]
        r = blocks[0][:, 0]
        g = np.mean([b[:, 1] for b in blocks], axis=0)

        return r, g

    def plot_rdf(self):
        r, g_r = self.load_rdf()
        plt.figure(figsize=FIG_SIZE)
        plt.plot(r, g_r, label="g(r)")
        # ---- Find peaks ----
        peaks, _ = find_peaks(g_r, height=1.0)

        # ---- Annotate peaks ----
        for p in peaks[:3]:  # first 2–3 peaks only (cleaner)
            plt.scatter(r[p], g_r[p], zorder=5)
            plt.text(r[p], g_r[p] + 0.1,
                    f"({r[p]:.2f}, {g_r[p]:.2f})",
                    ha='center', fontsize=9)
        plt.xlabel(f"r ($\u00C5$) $\\rightarrow$")
        plt.ylabel(f"$g(r) \\rightarrow$")
        plt.axhline(y=1, linestyle='--', linewidth=2, label='g(r) = 1')
        plt.title("Radial Distribution Function g(r) plot for Argon at 94.4 K")
        plt.savefig("argon_rdf.png", dpi=PLOT_DPI)
        plt.tight_layout()
        plt.show()
        plt.close()

    # ----------------------------
    # 2. Temperature & Energy
    # ----------------------------
    def load_log(self):
        data = []
        headers = None

        with open(f"{self.run_dir}/argon.log") as f:
            for line in f:
                if line.strip().startswith("Step"):
                    headers = line.split()
                    data = []   # reset → keeps latest block
                    continue

                if headers is not None:
                    if line.strip() == "" or "Loop" in line:
                        continue

                    parts = line.split()
                    if len(parts) == len(headers):
                        try:
                            data.append([float(x) for x in parts])
                        except:
                            continue

        return headers, np.array(data)

    def plot_temperature(self):
        headers, data = self.load_log()
        step = data[:, headers.index("Step")]
        temp = data[:, headers.index("Temp")]
        temp_smooth = gaussian_filter1d(temp, sigma=5)
        plt.figure(figsize=FIG_SIZE)
        plt.plot(step, temp, alpha=0.3, label="Raw")
        plt.plot(step, temp_smooth, linewidth=2, label="Smoothed")
        plt.xlabel(f"timestep $\\rightarrow$")
        plt.ylabel(f"Temperature/K $\\rightarrow$")
        plt.axhline(y=94.4, linestyle='--', linewidth=2, label='T = 298.15 K')
        plt.title("A plot of temperature vs time for argon ast 94.4 K")
        plt.tight_layout()
        plt.savefig("argon_temp.png", dpi=PLOT_DPI)
        plt.show()
        plt.close()

    def plot_energy(self):
        headers, data = self.load_log()
        step = data[:, headers.index("Step")]
        pe = data[:, headers.index("PotEng")]
        ke = data[:, headers.index("KinEng")]
        etot = data[:, headers.index("TotEng")]
        plt.figure(figsize=FIG_SIZE)
        plt.plot(step, pe, label="Potential Energy")
        plt.plot(step, ke, label="Kinetic Energy")
        plt.plot(step, etot, label="Total Energy")
        plt.xlabel(f"timesteps $\\rightarrow$")
        plt.ylabel(f"Energy/($kcal mol^{-1}$) $\\rightarrow$")
        plt.legend()
        plt.title("Plot of variation of energy vs time of Argon at 94.4 K")
        plt.savefig("energy_argon.png", dpi=PLOT_DPI)
        plt.tight_layout()
        plt.show()
        plt.close()

    # ----------------------------
    # 3. MSD
    # ----------------------------
    def load_msd(self):
        data = np.loadtxt(f"{self.run_dir}/msd.dat")
        timestep = 1.0
        time = data[:, 0] * timestep * 1e-3
        msd = data[:, 1]
        return time, msd

    def plot_msd(self):
        time, msd = self.load_msd()
        n = len(time)
        start = int(0.3 * n)
        end = int(0.8 * n)
        slope, intercept = np.polyfit(time[start:end], msd[start:end], 1)
        plt.figure(figsize=FIG_SIZE)
        plt.plot(time, msd, label="MSD")
        plt.plot(time[start:end],
             slope*time[start:end] + intercept,
             '--', label=f"Fit (D={slope/6:.3f})")
        plt.xlabel(f"timestep $\\rightarrow$")
        plt.ylabel(f"MSD/Å $\\rightarrow$")
        plt.title("Mean Square Displacement plot og Argon at 94.4 K")
        plt.legend()
        plt.tight_layout()
        plt.savefig("msd_argon.png", dpi=PLOT_DPI)
        plt.show()
        plt.close()

    def compute_diffusion_msd(self):
        time, msd = self.load_msd()

        # linear fit (last half of data)
        n = len(time)
        start = int(0.3 * n)
        end = int(0.8 * n)
        slope, _ = np.polyfit(time[start: end], msd[start: end], 1)

        D = slope / 6.0
        print(f"Diffusion coefficient (MSD): {D}")
        return D

    # ----------------------------
    # 4. VACF
    # ----------------------------
    def load_vacf(self):
        data = np.loadtxt(f"{self.run_dir}/vacf.dat", comments="#")
        timestep_fs = 1.0
        time_ps = (data[:, 0] - data[0, 0]) * timestep_fs * 1e-3
        vacf = data[:, 1] * 1e6

        return time_ps, vacf

    def plot_vacf(self):
        time_ps, vacf = self.load_vacf() 
        vacf_smooth = gaussian_filter1d(vacf, sigma=1)
        plt.figure(figsize=FIG_SIZE)
        plt.plot(time_ps, vacf, alpha=0.3, label="Raw")
        plt.plot(time_ps, vacf_smooth, linewidth=2, label="Smoothed")
        plt.axhline(0, linestyle='--')              
        plt.xlabel(f"timestep $\\rightarrow$")
        plt.ylabel(f"VACF/ $(Å/ps)^2$ $\\rightarrow$")
        plt.title("VACF plot og Argon at 94.4 K")
        plt.legend()
        plt.tight_layout()
        plt.savefig("vacf_argon.png", dpi=PLOT_DPI)
        plt.show()
        plt.close()

    def compute_diffusion_vacf(self):
        time_ps, vacf = self.load_vacf()

        # smooth slightly (important)
        from scipy.ndimage import gaussian_filter1d
        vacf_smooth = gaussian_filter1d(vacf, sigma=1)

        # find zero crossing
        zero_cross_idx = np.where(vacf_smooth < 0)[0]

        if len(zero_cross_idx) == 0:
            print("No zero crossing found!")
            return 0.0

        zero_cross = zero_cross_idx[0]

        # integrate properly
        D = np.trapezoid(vacf_smooth[:zero_cross], time_ps[:zero_cross]) / 3.0

        print(f"Diffusion coefficient (VACF): {D}")
        return D

    # ----------------------------
    # 5. Compare Diffusion
    # ----------------------------
    def compare_diffusion(self):
        D_msd = self.compute_diffusion_msd()
        D_vacf = self.compute_diffusion_vacf()

        print("\nComparison:")
        print(f"MSD  : {D_msd}")
        print(f"VACF : {D_vacf}")
        print(f"Relative difference: {abs(D_msd - D_vacf)/D_msd * 100:.2f}%")

def main():
    run_dir = "/data/argon_sim/run1"
    analysis = argon_analysis(run_dir)
    analysis.plot_rdf()
    analysis.plot_temperature()
    analysis.plot_energy()
    analysis.plot_msd()
    analysis.compute_diffusion_msd()
    analysis.plot_vacf()
    analysis.compute_diffusion_vacf()
    analysis.compare_diffusion()

if __name__ == "__main__":
    main()