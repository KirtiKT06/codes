import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

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
        plt.plot(r, g_r)
        plt.xlabel("r")
        plt.ylabel("g(r)")
        plt.title("Radial Distribution Function")
        plt.grid()
        plt.show()

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
        step_idx = headers.index("Step")
        temp_idx = headers.index("Temp")

        plt.plot(data[:, step_idx], data[:, temp_idx])
        plt.xlabel("Step")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature vs Time")
        plt.grid()
        plt.show()

    def plot_energy(self):
        headers, data = self.load_log()
        step = data[:, headers.index("Step")]
        pe = data[:, headers.index("PotEng")]
        ke = data[:, headers.index("KinEng")]
        etot = data[:, headers.index("TotEng")]

        plt.plot(step, pe, label="PE")
        plt.plot(step, ke, label="KE")
        plt.plot(step, etot, label="Total")

        plt.xlabel("Step")
        plt.ylabel("Energy")
        plt.legend()
        plt.title("Energy vs Time")
        plt.grid()
        plt.show()

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
        plt.plot(time, msd)
        plt.xlabel("Time")
        plt.ylabel("MSD")
        plt.title("Mean Square Displacement")
        plt.grid()
        plt.show()

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
        filepath = f"{self.run_dir}/vacf.dat"

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # --- Find last block header ---
        header_idx = None

        for i in range(len(lines) - 1, -1, -1):
            parts = lines[i].split()
            if len(parts) == 2:
                try:
                    int(parts[0])
                    int(parts[1])
                    header_idx = i
                    break
                except:
                    continue

        if header_idx is None:
            raise ValueError("Could not find VACF block header")

        nrows = int(lines[header_idx].split()[1])

        # --- Read block ---
        data = []

        for j in range(header_idx + 1, header_idx + 1 + nrows):
            parts = lines[j].split()

            if len(parts) != 6:
                continue

            try:
                time_lag = float(parts[1])   # fs
                ncount   = int(parts[2])

                if ncount == 0:
                    continue   # skip bad points

                vx = float(parts[3])
                vy = float(parts[4])
                vz = float(parts[5])

                data.append([time_lag, vx, vy, vz])

            except:
                continue

        data = np.array(data)

        time_ps = data[:, 0] / 1000.0
        vacf = data[:, 1] + data[:, 2] + data[:, 3]

        return time_ps, vacf

    def plot_vacf(self):
        time_ps, vacf = self.load_vacf()        
        plt.plot(time_ps, vacf)
        plt.xlabel("Time (ps)")
        plt.ylabel("VACF")
        #plt.xlim(0, 2)
        plt.title("VACF (short time)")
        plt.grid()
        plt.show()

    def compute_diffusion_vacf(self):
        time_ps, vacf = self.load_vacf()

        # ---- smooth VACF (important for noise) ----
        vacf_smooth = gaussian_filter1d(vacf, sigma=1)

        # ---- choose cutoff window ----
        n = len(time_ps)
        cutoff = int(0.2 * n)   # try 0.1–0.3 if needed

        # ---- integrate VACF ----
        integral = cumulative_trapezoid(
            vacf_smooth[:cutoff],
            time_ps[:cutoff],
            initial=0
        )

        # ---- diffusion coefficient ----
        D = integral[-1] / 3.0

        print(f"Diffusion coefficient (VACF): {D} Å²/ps")
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
    run_dir = "/data/argon_sim/run2"
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