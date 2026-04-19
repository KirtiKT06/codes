import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.signal import find_peaks
from multiprocessing import Pool, cpu_count

plt.style.use("seaborn-v0_8-darkgrid")
PLOT_DPI = 300
FIG_SIZE = (8, 6)
GLOBAL_FRAMES = None

def init_worker(frames):
    global GLOBAL_FRAMES
    GLOBAL_FRAMES = frames


def rdf_worker(args):
    idx, pair, r_max, bins = args
    atoms, box = GLOBAL_FRAMES[idx]

    # --- split atoms ---
    O, H = [], []
    for _, element, x, y, z in atoms:
        if element == 'O':
            O.append([x, y, z])
        else:
            H.append([x, y, z])

    O = np.array(O) % box
    H = np.array(H) % box

    if pair == "OO":
        A, B = O, O
    elif pair == "OH":
        A, B = O, H
    else:
        A, B = H, H

    if len(A) == 0 or len(B) == 0:
        return np.zeros(bins)

    tree_A = cKDTree(A, boxsize=box)
    tree_B = cKDTree(B, boxsize=box)

    pairs = tree_A.query_ball_tree(tree_B, r_max)

    dist = []
    for i, neigh in enumerate(pairs):
        for j in neigh:
            if pair in ["OO", "HH"] and j <= i:
                continue
            d = np.linalg.norm(A[i] - B[j])
            dist.append(d)

    if not dist:
        return np.zeros(bins)

    hist, _ = np.histogram(dist, bins=bins, range=(0, r_max))
    return hist


class water_analysis:
    def __init__(self, filename, max_frames=1000, r_max=10.0, bins=200):
        self.filename = filename
        self.max_frames = max_frames
        self.r_max = r_max
        self.bins = bins
        self.frames = []
        self.thermo = {}

    # -----------------------------
    # TRAJECTORY LOADING (optimized)
    # -----------------------------
    def read_trajectory(self):
        with open(self.filename, 'r') as f:
            count = 0

            while True:
                line = f.readline()
                if not line:
                    break

                if "ITEM: TIMESTEP" in line:
                    f.readline()  # timestep value
                    f.readline()  # ITEM: NUMBER OF ATOMS
                    natoms = int(f.readline())

                    f.readline()  # ITEM: BOX BOUNDS
                    box = []
                    for _ in range(3):
                        lo, hi = map(float, f.readline().split())
                        box.append(hi - lo)

                    f.readline()  # ITEM: ATOMS

                    atoms = []
                    for _ in range(natoms):
                        parts = f.readline().split()
                        mol = int(parts[1])
                        element = parts[3]
                        x, y, z = map(float, parts[4:7])
                        atoms.append((mol, element, x, y, z))

                    self.frames.append((atoms, np.array(box)))
                    count += 1

                    if count >= self.max_frames:
                        break

        print(f"[INFO] Loaded {len(self.frames)} frames")

    # -----------------------------
    # RDF (FAST)
    # -----------------------------
    def compute_rdf(self, pair="OO", nproc=6):
        args = [(i, pair, self.r_max, self.bins) for i in range(len(self.frames))]

        with Pool(processes=nproc, initializer=init_worker, initargs=(self.frames,)) as pool:
            results = pool.map(rdf_worker, args)

        rdf = np.sum(results, axis=0)

        # --- normalization ---
        r = np.linspace(0, self.r_max, self.bins)
        dr = self.r_max / self.bins

        atoms, box = self.frames[0]
        O = sum(1 for a in atoms if a[1] == 'O')
        H = len(atoms) - O

        V = np.prod(box)

        if pair == "OO":
            N, rho = O, O / V
        elif pair == "OH":
            N, rho = O, H / V
        else:
            N, rho = H, H / V

        shell = 4 * np.pi * r**2 * dr
        denominator = len(self.frames) * shell * N * rho

        g = np.zeros_like(rdf, dtype=float)

        valid = denominator > 1e-12
        g[valid] = rdf[valid] / denominator[valid]

        # extra safety
        g[np.isnan(g)] = 0
        g[np.isinf(g)] = 0

        return r, g
    
    def analyze_rdf(self, pair, rdf_data=None):
        if rdf_data is None:
            r, g = self.compute_rdf(pair)
        else:
            r, g = rdf_data[pair]

        mask = r > 0.8
        r = r[mask]
        g = g[mask]

        peaks, _ = find_peaks(g, height=0.5)

        print(f"\n[RESULT] RDF Analysis ({pair})")

        for i, p in enumerate(peaks[:3]):
            print(f"Peak {i+1}: r = {r[p]:.3f} Å, g(r) = {g[p]:.3f}")

        # --- first minimum ---
        if len(peaks) > 0:
            first_peak = peaks[0]

            minima = np.where(
                (g[1:-1] < g[:-2]) & (g[1:-1] < g[2:])
            )[0] + 1

            minima = minima[minima > first_peak]

            if len(minima) > 0:
                m = minima[0]
                print(f"First minimum: r = {r[m]:.3f} Å, g(r) = {g[m]:.3f}")

    # -----------------------------
    # RDF PLOT
    # -----------------------------
    def plot_rdf(self):
        plt.figure(figsize=FIG_SIZE)
        rdf_data = {}
        for pair in ["OO", "OH", "HH"]:
            r, g = self.compute_rdf(pair)
            rdf_data[pair] = (r, g)
            mask = r > 0.8
            r_plot = r[mask]
            g_plot = g[mask]
            plt.plot(r_plot, np.clip(g_plot, 0, 10), label=pair)
        plt.axhline(y=1, linestyle='--', linewidth=2, label='g(r) = 1')
        plt.title("A radial distribution function g(r) plot for water at 298.15 K using TIP4P model")
        plt.legend()
        plt.xlabel(f"r ($\u00C5$) $\\rightarrow$")
        plt.ylabel(f"$g(r) \\rightarrow$")
        plt.savefig("water_rdf.png", dpi=PLOT_DPI)
        plt.close()
        print("[INFO] RDF done")
        return rdf_data

    # -----------------------------
    # ORIENTATION (optimized)
    # -----------------------------
    def compute_orientation(self, max_dt=300):
        vecs_all = []
        for atoms, box in self.frames:
            mol_dict = {}
            for mol, el, x, y, z in atoms:
                mol_dict.setdefault(mol, []).append((el, np.array([x, y, z])))
            vecs = {}
            for mol_id, mol in mol_dict.items():
                O = [p for e, p in mol if e == 'O'][0]
                Hs = [p for e, p in mol if e != 'O']
                d1 = Hs[0] - O
                d2 = Hs[1] - O
                vec = d1 + d2
                vec = vec / np.linalg.norm(vec)
                vecs[mol_id] = vec
            vecs_all.append(vecs)

        C = np.zeros(max_dt)

        for dt in range(max_dt):
            vals = []
            for t in range(len(vecs_all) - dt):
                v0 = vecs_all[t]
                vt = vecs_all[t + dt]
                common = set(v0.keys()) & set(vt.keys())
                if not common:
                    continue
                dots = []
                for key in common:
                    dot = np.dot(v0[key], vt[key])
                    P2 = (3 * dot**2 - 1) / 2
                    dots.append(P2)
                if dots:
                    vals.append(np.mean(dots))
            if vals:
                C[dt] = np.mean(vals)
            
            if C[0] != 0:
                C = C / C[0]
        return C

    # -----------------------------
    # HYDROGEN BONDS (optimized)
    # -----------------------------
    def compute_hbonds_frame_fast(self, frame):
        atoms, box = frame

        # --- group molecules ---
        mol_dict = {}
        for mol, el, x, y, z in atoms:
            mol_dict.setdefault(mol, []).append((el, np.array([x, y, z])))

        mol_ids = list(mol_dict.keys())

        O_pos = []
        mol_map = []

        for mol in mol_ids:
            O = next(p for e, p in mol_dict[mol] if e == 'O')
            O_pos.append(O)
            mol_map.append(mol)

        O_pos = np.array(O_pos) % box

        # --- KDTree on oxygen atoms ---
        tree = cKDTree(O_pos, boxsize=box)
        pairs = tree.query_pairs(r=3.5)  # ONLY neighbors within cutoff

        hbonds = set()

        for i, j in pairs:
            mol_i = mol_dict[mol_map[i]]
            mol_j = mol_dict[mol_map[j]]

            Oi = O_pos[i]
            Oj = O_pos[j]

            # --- check hydrogens of i ---
            for e, H in mol_i:
                if e == 'O':
                    continue

                v1 = H - Oi
                v2 = Oj - H

                cos = np.dot(v1, v2) / (np.linalg.norm(v1)*np.linalg.norm(v2))

                if cos > 0.866:
                    hbonds.add((mol_map[i], mol_map[j]))
                    break

        return hbonds

    def compute_hbond_lifetime_fast(self, max_dt=150):
        hb_all = [self.compute_hbonds_frame_fast(f) for f in self.frames]

        C = np.zeros(max_dt)

        for dt in range(1, max_dt):
            num = 0
            den = 0

            for t in range(len(hb_all) - dt):
                h0 = hb_all[t]
                if not h0:
                    continue

                ht = hb_all[t+dt]
                num += len(h0 & ht)
                den += len(h0)

            C[dt] = num / den if den else 0

        return C
    
    def plot_orientation(self):
        C = self.compute_orientation()

        # remove t=0 spike
        C = C[1:]
        t = np.arange(len(C))
        # ---- NEW PART (ADD HERE) ----
        dt_ps = 10.0   # from dump frequency (10000 steps × 1 fs)

        if len(C) > 1 and C[1] > 0:
            tau_ps = -dt_ps / np.log(C[1])
            print(f"[RESULT] Orientation relaxation time τ ≈ {tau_ps:.2f} ps")

        # # --- find tau (1/e decay) ---
        # tau_idx = np.argmin(np.abs(C - 1/np.e))
        # tau = tau_idx
        # print(f"[RESULT] Orientation relaxation time τ ≈ {tau} timesteps")

        plt.figure(figsize=FIG_SIZE)
        plt.plot(C)
        plt.xlabel(f"timestep $\\rightarrow$")
        plt.ylabel(f"C(t) $\\rightarrow$")
        plt.title("Orientation Relaxation plot of water using TIP4P model")
        plt.ylim(-0.1, 1)
        plt.tight_layout()
        plt.savefig("orientation_water.png", dpi=PLOT_DPI)
        plt.close()
        print("[INFO] Orientation plot saved")

    # =========================
    # Thermodynamic Analysis
    # =========================
    def read_log_file(self, logfile="log.lammps"):
        self.thermo = {}

        with open(logfile, 'r') as f:
            lines = f.readlines()

        headers = []
        data_started = False

        for line in lines:
            # detect header
            if "Step" in line and "Temp" in line:
                headers = line.split()
                self.thermo = {h: [] for h in headers}
                self.thermo["Density"] = []
                data_started = True
                continue

            if not data_started:
                continue

            parts = line.split()

            # case 1: thermo line (many columns)
            if len(parts) == len(headers):
                try:
                    values = [float(x) for x in parts]
                except ValueError:
                    continue

                for h, val in zip(headers, values):
                    self.thermo[h].append(val)

            # case 2: density line (single value)
            elif len(parts) == 1:
                try:
                    density = float(parts[0])
                    self.thermo["Density"].append(density)
                except ValueError:
                    continue
        print("[INFO] Thermo data loaded")
    
    def plot_energy(self):
        equil_cut = int(0.1 * len(self.thermo["Step"]))
        step = np.array(self.thermo["Step"][equil_cut:])
        PE = np.array(self.thermo["PotEng"][equil_cut:])
        KE = np.array(self.thermo["KinEng"][equil_cut:])
        TE = np.array(self.thermo["TotEng"][equil_cut:])                
        plt.figure(figsize=FIG_SIZE)        
        plt.plot(step, PE, label="Potential Energy")
        plt.plot(step, KE, label="Kinetic Energy")
        plt.plot(step, TE, label="Total Energy")
        plt.xlabel(f"timesteps $\\rightarrow$")
        plt.ylabel(f"Energy/$(kcal mol^{-1})$ $\\rightarrow$")
        plt.legend()
        plt.title("Plot of variation of energy vs time of water at 298.15 K using TIP4P model")
        plt.savefig("energy_water.png", dpi=PLOT_DPI)
        #plt.show()
        plt.close()
        print("[INFO] Energy plot saved")

    def plot_temp_pressure(self):
        equil_cut = int(0.1 * len(self.thermo["Step"]))
        step = np.array(self.thermo["Step"][equil_cut:])
        temp = np.array(self.thermo["Temp"][equil_cut:])
        press = np.array(self.thermo["Press"][equil_cut:])

        # --- smoothing ---
        window = 50
        temp_smooth = np.convolve(temp, np.ones(window)/window, mode='valid')
        press_smooth = np.convolve(press, np.ones(window)/window, mode='valid')

        offset = window // 2
        step_s = step[offset : offset + len(temp_smooth)]

        # Temperature
        plt.figure(figsize=FIG_SIZE)
        plt.plot(step, temp, alpha=0.3, label="Raw")
        plt.plot(step_s, temp_smooth, linewidth=2, label="Smoothed")
        plt.axhline(y=298.15, linestyle='--', linewidth=2, label='T = 298.15 K')
        plt.xlabel(f"timesteps $\\rightarrow$")
        plt.ylabel(f"Temperature/K $\\rightarrow$")
        plt.title("Plot of variation of temperature with time of water at 298.15 K using TIP4P model")
        plt.legend()
        plt.tight_layout()
        plt.savefig("temperature_water.png", dpi=PLOT_DPI)
        #plt.show()
        plt.close()

        # Pressure
        plt.figure(figsize=FIG_SIZE)
        plt.plot(step, press, alpha=0.3, label="Raw")
        plt.plot(step_s, press_smooth, linewidth=2, label="Smoothed")
        plt.axhline(y=1, linestyle='--', linewidth=2, label='P = 1 atm')
        plt.xlabel(f"timesteps $\\rightarrow$")
        plt.ylabel(f"Pressure/atm $\\rightarrow$")
        plt.title("Plot of variation of pressure with time of water at 298.15 K using TIP4P model")
        plt.legend()
        plt.tight_layout()
        plt.savefig("pressure_water.png", dpi=PLOT_DPI)
        #plt.show()
        plt.close()
        print("[INFO] Temp & Pressure plots saved")

    def plot_density(self):
        equil_cut = int(0.1 * len(self.thermo["Step"]))
        step = np.array(self.thermo["Step"][equil_cut:])
        density = np.array(self.thermo["Density"][equil_cut:])

        window = 200
        density_smooth = np.convolve(density, np.ones(window)/window, mode='valid')

        offset = window // 2
        step_s = step[offset : offset + len(density_smooth)]

        plt.figure(figsize=FIG_SIZE)
        plt.plot(step, density, alpha=0.3, label="Raw")
        plt.plot(step_s, density_smooth, linewidth=2, label="Smoothed")
        plt.axhline(y=0.99704, linestyle='--', linewidth=2, label='$\\rho = 0.99704 gcm^{-3}$')
        plt.xlabel(f"timesteps $\\rightarrow$")
        plt.ylabel(f"$\\rho/ gcm^{-3}$ $\\rightarrow$")
        plt.title("Plot of variation of density with time of water at 298.15 K using TIP4P model")
        plt.legend()
        plt.tight_layout()
        plt.savefig("density_water.png", dpi=PLOT_DPI)
        #plt.show()
        plt.close()

        print("[INFO] Density plot saved")

    def compute_specific_heat(self):
        E = np.array(self.thermo["TotEng"])
        T = np.mean(self.thermo["Temp"])

        if len(E) == 0:
            print("[ERROR] No energy data found!")
            return None

        kB = 0.001987  # kcal/mol-K (LAMMPS real units)

        E_mean = np.mean(E)
        E2_mean = np.mean(E**2)

        N = 1000
        Cv = (E2_mean - E_mean**2) / (kB * T**2 * N)

        print(f"[RESULT] Specific Heat Cv = {Cv:.4f} kcal/mol-K")
        return Cv
    
    def plot_hbond_lifetime_fast(self):
        C = self.compute_hbond_lifetime_fast()

        C = C[1:]  # remove spike
        t = np.arange(len(C))
        # ---- NEW PART (ADD HERE) ----
        dt_ps = 10.0   # from dump frequency (10000 steps × 1 fs)

        if len(C) > 1 and C[1] > 0:
            tau_ps = -dt_ps / np.log(C[1])
            print(f"[RESULT] Estimated H-bond lifetime τ ≈ {tau_ps:.2f} ps")

        # # --- find tau ---
        # tau_idx = np.argmin(np.abs(C - 1/np.e))
        # tau = tau_idx

        # print(f"[RESULT] H-bond lifetime τ ≈ {tau} timesteps")

        plt.figure(figsize=FIG_SIZE)
        plt.plot(C)
        plt.xlabel(f"timesteps $\\rightarrow$")
        plt.ylabel(f"C(t) $\\rightarrow$")
        plt.title("Plot of Hydrogen Bond Lifetime of water at 298.15 K")
        plt.tight_layout()
        plt.savefig("hbond_lifetime.png", dpi=PLOT_DPI)
        #plt.show()
        plt.close()
        print("[INFO] H-bond lifetime plot saved")


# =========================
# MAIN FUNCTION
# =========================
def main():
    fname = "/home/feynman/Downloads/water_lammps/dump.lammpstrj"
    analyzer = water_analysis(
        filename=fname,
        max_frames=5000,
        r_max=10.0,
        bins=200
    )

    # ---- Thermo ----
    analyzer.read_log_file("/home/feynman/Downloads/water_lammps/log.lammps")
    analyzer.plot_energy()
    analyzer.plot_temp_pressure()
    analyzer.plot_density()
    analyzer.compute_specific_heat()

    # ---- Trajectory ----
    analyzer.read_trajectory()
    rdf_data = analyzer.plot_rdf()
    analyzer.plot_orientation()
    analyzer.analyze_rdf("OO", rdf_data)
    analyzer.analyze_rdf("OH", rdf_data)
    analyzer.analyze_rdf("HH", rdf_data)
    analyzer.plot_hbond_lifetime_fast()

    print("[DONE] Analysis complete.")

if __name__ == "__main__":
    main()