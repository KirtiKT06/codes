import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from scipy.signal import find_peaks

plt.style.use("seaborn-v0_8-darkgrid")
PLOT_DPI = 300
FIG_SIZE = (7, 5)

def rdf_single_frame(args):
    frame, pair, r_max, bins = args
    analyzer = water_lammps_analysis("", r_max=r_max, bins=bins)

    O, H, box = analyzer.split_atoms(frame)

    if pair == "OO":
        A, B = O, O
    elif pair == "OH":
        A, B = O, H
    elif pair == "HH":
        A, B = H, H

    A_pos = A
    B_pos = B

    delta = A_pos[:, None, :] - B_pos[None, :, :]
    delta -= box * np.round(delta / box)
    dist = np.linalg.norm(delta, axis=2)

    if pair in ["OO", "HH"]:
        mask = np.triu(np.ones_like(dist, dtype=bool), k=1)
    else:
        mask = np.ones_like(dist, dtype=bool)

    dist = dist[mask]
    dist = dist[dist < r_max]

    hist, _ = np.histogram(dist, bins=bins, range=(0, r_max))

    return hist    


class water_lammps_analysis:
    def __init__(self, filename, max_frames=100, r_max=10.0, bins=200):
        self.filename = filename
        self.max_frames = max_frames
        self.r_max = r_max
        self.bins = bins
        self.frames = []

    # =========================
    # READ TRAJECTORY
    # =========================
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

    # =========================
    # SPLIT ATOMS
    # =========================
    def split_atoms(self, frame):
        atoms, box = frame
        O, H = [], []

        for mol, element, x, y, z in atoms:
            if element == 'O':
                O.append([x, y, z])
            else:
                H.append([x, y, z])

        return np.array(O), np.array(H), box

    # =========================
    # RDF
    # =========================
    def compute_rdf(self, pair="OO"):
        dr = self.r_max / self.bins
        rdf = np.zeros(self.bins)

        # -------- PARALLEL EXECUTION --------
        args = [(frame, pair, self.r_max, self.bins) for frame in self.frames]

        with Pool(10) as pool:
            results = pool.map(rdf_single_frame, args)

        rdf = np.sum(results, axis=0)

        # normalization
        r_vals = np.linspace(0, self.r_max, self.bins)
        norm_rdf = np.zeros_like(r_vals)

        O, H, box = self.split_atoms(self.frames[0])
        volume = np.prod(box)
        if pair == "OO":
            N = len(O)
            rho = len(O) / volume
        elif pair == "OH":
            N = len(O)
            rho = len(H) / volume
        elif pair == "HH":
            N = len(H)
            rho = len(H) / volume

        for i in range(self.bins):
            r = r_vals[i]
            if r < 1e-6:
                norm_rdf[i] = 0
                continue

            shell_vol = 4 * np.pi * r**2 * dr
            if shell_vol < 1e-6 or rho == 0:
                norm_rdf[i] = 0
                continue
            
            norm_rdf[i] = rdf[i] / (len(self.frames) * shell_vol * N * rho)

        return r_vals, norm_rdf

    # =========================
    # ORIENTATION VECTORS
    # =========================
    def get_OH_vectors(self, frame):
        atoms, _ = frame
        O, H, box = self.split_atoms(frame)
        mol_dict = {}

        for mol, element, x, y, z in atoms:
            if mol not in mol_dict:
                mol_dict[mol] = []
            mol_dict[mol].append((element, np.array([x, y, z])))

        vectors = []

        for mol in mol_dict:
            O = None
            Hs = []

            for element, pos in mol_dict[mol]:
                if element == 'O':
                    O = pos
                else:
                    Hs.append(pos)

            for H in Hs:
                delta = H - O
                delta -= box * np.round(delta / box)
                v = delta / np.linalg.norm(delta)
                vectors.append(v)

        return np.array(vectors)

    def compute_orientation(self):
        max_dt = 500
        C = []

        vecs_all = [self.get_OH_vectors(f) for f in self.frames]

        for dt in range(max_dt):
            vals = []

            for t0 in range(len(self.frames) - dt):
                v0 = vecs_all[t0]
                vt = vecs_all[t0 + dt]

                n = min(len(v0), len(vt))
                dot = np.sum(v0[:n] * vt[:n], axis=1)

                P2 = (3 * dot**2 - 1) / 2
                vals.append(np.mean(P2))

            C.append(np.mean(vals))

        return np.array(C)

    def analyze_rdf(self, pair="OO"):
        r, g = self.compute_rdf(pair)

        mask = r > 0.8
        r = r[mask]
        g = g[mask]

        # find peaks
        peaks, _ = find_peaks(g, height=0.5)

        print(f"\n[RESULT] RDF Analysis ({pair})")

        for i, p in enumerate(peaks[:3]):  # first 3 peaks
            print(f"Peak {i+1}: r = {r[p]:.3f} Å, g(r) = {g[p]:.3f}")

        # first minimum (after first peak)
        if len(peaks) > 0:
            first_peak = peaks[0]

            minima = np.where(
                (g[1:-1] < g[:-2]) & (g[1:-1] < g[2:])
            )[0] + 1

            minima = minima[minima > first_peak]

            if len(minima) > 0:
                m = minima[0]
                print(f"First minimum: r = {r[m]:.3f} Å, g(r) = {g[m]:.3f}")

        return r, g

    def plot_rdf(self):
        plt.figure(figsize=FIG_SIZE)

        for pair in ["OO", "OH", "HH"]:
            r, g = self.compute_rdf(pair)

            if pair == "OH":
                # clip spike for visualization
                g = np.clip(g, 0, 10)

            plt.plot(r, g, label=pair)

        plt.xlabel("r (Å)")
        plt.ylabel("g(r)")
        plt.legend()
        plt.title("Radial Distribution Function")
        plt.xlim(0, 6)
        plt.tight_layout()
        plt.savefig("rdf.png", dpi=PLOT_DPI)
        #plt.show()
        plt.close()

    print("[INFO] RDF plot saved")

    def plot_orientation(self):
        C = self.compute_orientation()

        # remove t=0 spike
        C = C[1:]

        plt.figure(figsize=FIG_SIZE)
        plt.plot(C)

        plt.xlabel("Time step")
        plt.ylabel("C(t)")
        plt.title("Orientation Relaxation")

        plt.ylim(-0.1, 1)

        plt.tight_layout()
        plt.savefig("orientation.png", dpi=PLOT_DPI)
        #plt.show()
        plt.close()

        print("[INFO] Orientation plot saved")

    # def coordination_number(self, pair="OO"):
    #     r, g = self.compute_rdf(pair)

    #     dr = r[1] - r[0]

    #     # find first minimum
    #     peaks, _ = find_peaks(g)
    #     first_peak = peaks[0]

    #     minima = np.where((g[1:-1] < g[:-2]) & (g[1:-1] < g[2:]))[0] + 1

    #     minima = minima[minima > first_peak]
    #     r_cut = r[minima[0]]

    #     rho = self.number_density

    #     integral = 0
    #     for i in range(len(r)):
    #         if r[i] > r_cut:
    #             break
    #         integral += 4 * np.pi * r[i]**2 * g[i] * dr

    #     CN = rho * integral

    #     print(f"[RESULT] Coordination Number ({pair}) = {CN:.3f}")

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
        plt.plot(step, PE, label="PE")
        plt.plot(step, KE, label="KE")
        plt.plot(step, TE, label="TE")
        plt.xlabel("Steps")
        plt.ylabel("Energy")
        plt.legend()
        plt.title("Energy vs Time")
        plt.savefig("energy.png", dpi=PLOT_DPI)
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
        plt.xlabel("Steps")
        plt.ylabel("Temperature (K)")
        plt.title("Temperature vs Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig("temperature.png", dpi=PLOT_DPI)
        #plt.show()
        plt.close()

        # Pressure
        plt.figure(figsize=FIG_SIZE)
        plt.plot(step, press, alpha=0.3, label="Raw")
        plt.plot(step_s, press_smooth, linewidth=2, label="Smoothed")
        plt.xlabel("Steps")
        plt.ylabel("Pressure")
        plt.title("Pressure vs Time")
        plt.legend()
        plt.tight_layout()
        plt.savefig("pressure.png", dpi=PLOT_DPI)
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
        plt.xlabel("Steps")
        plt.ylabel("Density (g/cm³)")
        plt.title("Density vs Steps")
        plt.legend()
        plt.tight_layout()
        plt.savefig("density.png", dpi=PLOT_DPI)
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
    
    def compute_hbonds_frame(self, frame):
        atoms, box = frame

        mol_dict = {}
        for mol, element, x, y, z in atoms:
            if mol not in mol_dict:
                mol_dict[mol] = []
            mol_dict[mol].append((element, np.array([x, y, z])))

        hbonds = []

        mols = list(mol_dict.keys())

        for i in range(len(mols)):
            for j in range(i+1, len(mols)):
            
                mol_i = mol_dict[mols[i]]
                mol_j = mol_dict[mols[j]]

                O_i = None
                H_i = []
                O_j = None

                for el, pos in mol_i:
                    if el == 'O':
                        O_i = pos
                    else:
                        H_i.append(pos)

                for el, pos in mol_j:
                    if el == 'O':
                        O_j = pos

                if O_i is None or O_j is None:
                    continue

                delta = O_i - O_j
                delta -= box * np.round(delta / box)
                OO = np.linalg.norm(delta)

                if OO > 3.5:
                    continue

                # Check angle for each H
                for H in H_i:
                    v1 = H - O_i
                    v2 = O_j - H
                    cos_theta = np.dot(v1, v2) / (
                        np.linalg.norm(v1) * np.linalg.norm(v2)
                    )
                    
                    if cos_theta > 0.866:  # ~30 degrees
                        hbonds.append((mols[i], mols[j]))
                        break

        return set(hbonds)
    
    def compute_hbond_lifetime(self):
        print("[INFO] Computing hydrogen bond lifetime...")

        hbonds_all = [self.compute_hbonds_frame(f) for f in self.frames]
        C = []
        n_frames = len(hbonds_all)
        max_dt = 300
        for dt in range(1, max_dt):
            count = 0
            total = 0

            for t0 in range(n_frames - dt):
                h0 = hbonds_all[t0]
                ht = hbonds_all[t0 + dt]

                if len(h0) == 0:
                    continue

                count += len(h0.intersection(ht))
                total += len(h0)

            if total == 0:
                C.append(0)
            else:
                C.append(count / total)

        return np.array(C)
    
    def plot_hbond_lifetime(self):
        C = self.compute_hbond_lifetime()

        C = C[1:]  # remove spike

        plt.figure(figsize=FIG_SIZE)
        plt.plot(C)

        plt.xlabel("Time step")
        plt.ylabel("C(t)")
        plt.title("Hydrogen Bond Lifetime")

        plt.tight_layout()
        plt.savefig("hbond_lifetime.png", dpi=PLOT_DPI)
        plt.show()
        plt.close()

        print("[INFO] H-bond lifetime plot saved")


# =========================
# MAIN FUNCTION
# =========================
def main():
    fname = "/home/feynman/Downloads/water_lammps/dump.lammpstrj"
    analyzer = water_lammps_analysis(
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
    analyzer.plot_rdf()
    analyzer.plot_orientation()
    analyzer.analyze_rdf("OO")
    analyzer.analyze_rdf("OH")
    analyzer.analyze_rdf("HH")
    # analyzer.coordination_number("OO")
    # analyzer.coordination_number("OH")
    # analyzer.coordination_number("HH")
    analyzer.plot_hbond_lifetime()

    print("[DONE] Analysis complete.")

if __name__ == "__main__":
    main()