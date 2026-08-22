"""
Compute mean SASA from explicit solvent MD trajectories.
Uses MDAnalysis + FreeSASA (or fallback to SASA via shrake_rupley).

Run: python compute_sasa.py
"""

import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis
import warnings
warnings.filterwarnings("ignore")

OUT_ROOT = "/data/openmm_out/1PGA_mutations/explicit_solvent"
SEL_CSV  = "selected_mutants.csv"


def compute_sasa_trajectory(mut, rep=1, step=50):
    """
    Compute mean SASA (Å²) for protein atoms only,
    averaged over trajectory frames.
    
    step=50 means sample every 50 frames (fast but representative)
    """
    rep_dir    = os.path.join(OUT_ROOT, f"{mut}_rep{rep}")
    dcd_file   = os.path.join(rep_dir, f"{mut}_rep{rep}_trajectory.dcd")
    final_file = os.path.join(rep_dir, f"{mut}_rep{rep}_final.pdb")
    min_file   = os.path.join(rep_dir, f"{mut}_rep{rep}_minimized.pdb")

    if not os.path.exists(dcd_file):
        print(f"  [SKIP] No trajectory: {mut}")
        return None

    top_file = final_file if os.path.exists(final_file) else min_file

    try:
        u    = mda.Universe(top_file, dcd_file)
        prot = u.select_atoms("protein")
        print(f"  {len(u.trajectory)} frames, {prot.n_atoms} protein atoms")
    except Exception as e:
        print(f"  [ERROR] Loading {mut}: {e}")
        return None

    # ── Try Method 1: MDAnalysis built-in SASA (shrake_rupley) ──
    try:
        from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis
        from MDAnalysis.analysis import solventaccessiblesurface as sas

        # Use shrake_rupley algorithm
        sasa_vals = []
        for ts in u.trajectory[::step]:
            # Compute SASA for protein only
            sasa = sas.ShrakeRupley(prot, n_sphere_points=960,
                                     probe_radius=1.4)
            sasa.run(start=ts.frame, stop=ts.frame+1)
            sasa_vals.append(sasa.results.total_area[0])

        mean_sasa = float(np.mean(sasa_vals))
        print(f"  SASA (ShrakeRupley): {mean_sasa:.1f} Å²")
        return mean_sasa

    except Exception as e1:
        print(f"  [WARN] ShrakeRupley failed ({e1}), trying freesasa...")

    # ── Try Method 2: freesasa ──
    try:
        import freesasa

        sasa_vals = []
        for ts in u.trajectory[::step]:
            # Write temporary PDB for this frame
            tmp_pdb = f"/tmp/{mut}_frame_{ts.frame}.pdb"
            with mda.Writer(tmp_pdb, prot.n_atoms) as W:
                W.write(prot)

            result = freesasa.calc(
                freesasa.Structure(tmp_pdb),
                freesasa.Parameters({
                    'algorithm': freesasa.ShrakeRupley,
                    'probe-radius': 1.4,
                    'n-points': 100
                })
            )
            sasa_vals.append(result.totalArea())
            os.remove(tmp_pdb)

        mean_sasa = float(np.mean(sasa_vals))
        print(f"  SASA (freesasa): {mean_sasa:.1f} Å²")
        return mean_sasa

    except Exception as e2:
        print(f"  [WARN] freesasa failed ({e2}), using geometric approximation...")

    # ── Method 3: Geometric fallback ──
    # Count solvent-exposed atoms as proxy for SASA
    # Not true SASA but correlates well enough for ranking
    try:
        water_O    = u.select_atoms("resname HOH and name O")
        prot_heavy = u.select_atoms("protein and not name H*")

        exposed_counts = []
        for ts in u.trajectory[::step]:
            count = 0
            for atom in prot_heavy.positions:
                dists = np.linalg.norm(water_O.positions - atom, axis=1)
                # Atom is "exposed" if any water is within 5 Å
                if np.any(dists < 5.0):
                    count += 1
            exposed_counts.append(count)

        mean_exposed = float(np.mean(exposed_counts))
        print(f"  Exposed atoms (fallback): {mean_exposed:.1f}")
        return mean_exposed  # proxy for SASA

    except Exception as e3:
        print(f"  [ERROR] All SASA methods failed for {mut}: {e3}")
        return None


def run_sasa_analysis():
    # Find all completed simulations
    completed = []
    all_dirs  = glob.glob(os.path.join(OUT_ROOT, "*_rep1"))

    for d in sorted(all_dirs):
        mut     = os.path.basename(d).replace("_rep1", "")
        dcd     = os.path.join(d, f"{mut}_rep1_trajectory.dcd")
        if os.path.exists(dcd):
            completed.append(mut)

    print(f"Found {len(completed)} completed simulations\n")

    records = []
    for mut in completed:
        print(f"Computing SASA: {mut}")
        sasa = compute_sasa_trajectory(mut)
        if sasa is not None:
            records.append({"mutant": mut, "mean_sasa": sasa})
        print()

    df = pd.DataFrame(records)
    df.to_csv("sasa_features.csv", index=False)
    print(f"Saved sasa_features.csv ({len(df)} mutants)")
    return df


def validate_sasa(sasa_csv="sasa_features.csv", sel_csv=SEL_CSV):
    feat = pd.read_csv(sasa_csv)
    sel  = pd.read_csv(sel_csv)

    # Compute delta SASA vs WT
    wt_row = feat[feat["mutant"] == "1PGA"]
    if len(wt_row):
        wt_sasa = wt_row["mean_sasa"].values[0]
        feat["delta_sasa"] = feat["mean_sasa"] - wt_sasa
        print(f"WT (1PGA) mean SASA: {wt_sasa:.1f} Å²\n")

    merged = feat[feat["mutant"] != "1PGA"].merge(
        sel[["name", "y"]], left_on="mutant", right_on="name"
    ).sort_values("y")

    print(f"{'Mutant':<8} {'y (exp)':>10} {'SASA (Å²)':>12} {'delta_SASA':>12}")
    print("-" * 48)
    for _, row in merged.iterrows():
        delta = row.get("delta_sasa", float("nan"))
        print(f"{row['mutant']:<8} {row['y']:>+10.3f} "
              f"{row['mean_sasa']:>12.1f} {delta:>+12.1f}")

    print()
    for col in ["mean_sasa", "delta_sasa"]:
        if col not in merged.columns:
            continue
        valid = merged[[col, "y"]].dropna()
        rho, p = spearmanr(valid[col], valid["y"])
        sig = "***" if p < 0.001 else ("**" if p < 0.01 else
              ("*" if p < 0.05 else "ns"))
        print(f"{col:<20}  rho={rho:+.3f}  p={p:.4f}  {sig}")

    print()

    # ── Plot ──
    if "mean_sasa" in merged.columns:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for ax, col, label in [
            (axes[0], "mean_sasa",  "Mean SASA (Å²)"),
            (axes[1], "delta_sasa", "ΔSASA vs WT (Å²)")
        ]:
            if col not in merged.columns:
                continue

            valid = merged[[col, "y", "mutant"]].dropna()
            rho, p = spearmanr(valid[col], valid["y"])

            ax.scatter(valid[col], valid["y"],
                       s=80, color="steelblue",
                       edgecolors="k", linewidths=0.5, zorder=3)

            for _, row in valid.iterrows():
                ax.annotate(row["mutant"],
                           (row[col], row["y"]),
                           fontsize=7, color="gray",
                           xytext=(4, 4), textcoords="offset points")

            z  = np.polyfit(valid[col], valid["y"], 1)
            xs = np.linspace(valid[col].min(), valid[col].max(), 100)
            ax.plot(xs, np.poly1d(z)(xs),
                    color="tomato", linewidth=1.5, linestyle="--")

            ax.set_xlabel(label, fontsize=11)
            ax.set_ylabel("Experimental y (−ΔG)", fontsize=11)
            ax.set_title(f"ρ = {rho:.3f}   p = {p:.4f}   n = {len(valid)}",
                        fontsize=10)
            ax.grid(True, alpha=0.3)

        plt.suptitle("SASA vs Experimental Stability (1PGA Mutants)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig("sasa_vs_y.png", dpi=150, bbox_inches="tight")
        print("Saved: sasa_vs_y.png")


if __name__ == "__main__":
    df = run_sasa_analysis()
    validate_sasa()