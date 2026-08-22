# import pandas as pd
# import numpy as np
# from scipy.stats import spearmanr

# feat = pd.read_csv('md_features.csv')
# sel  = pd.read_csv('selected_mutants.csv')

# OUT_ROOT = "/data/openmm_out/1PGA_mutations/run_330K"

# # Read potential energies from log CSVs
# def get_mean_pe(mut, rep=1):
#     log = f"{OUT_ROOT}/{mut}_rep{rep}/{mut}_rep{rep}_log.csv"
#     try:
#         df = pd.read_csv(log)
#         # Column name from your StateDataReporter
#         pe_col = [c for c in df.columns if 'Potential' in c][0]
#         # Skip first 20% (equilibration drift)
#         n = len(df)
#         return df[pe_col].iloc[int(0.2*n):].mean()
#     except Exception as e:
#         print(f"  Failed {mut}: {e}")
#         return np.nan

# print("Computing mean potential energies...")
# pe_values = {}
# for mut in feat['mutant'].tolist():
#     pe = get_mean_pe(mut)
#     pe_values[mut] = pe
#     print(f"  {mut}: {pe:.1f} kJ/mol")

# wt_pe = pe_values.get('1PGA', np.nan)
# print(f"\nWT PE: {wt_pe:.1f}")

# # Compute delta PE
# records = []
# for mut, pe in pe_values.items():
#     if mut == '1PGA':
#         continue
#     records.append({'mutant': mut, 'delta_PE': pe - wt_pe})

# pe_df = pd.DataFrame(records)
# merged = pe_df.merge(sel[['name','y']], left_on='mutant', right_on='name')

# rho, p = spearmanr(merged['delta_PE'], merged['y'])
# print(f"\ndelta_PE vs Nisthal y:  rho={rho:+.3f}  p={p:.4f}")

"""
Compute mean potential energy per mutant using OpenMM
by replaying saved trajectory frames.
Saves results to pe_features.csv
"""

import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── OpenMM imports ──
from openmm.app import *
from openmm import *
from openmm.unit import picoseconds, nanometers, kilojoules_per_mole
import MDAnalysis as mda

# ── Config ──
OUT_ROOT  = "/data/openmm_out/1PGA_mutations/run_330K"
PDB_DIR   = "/home/feynman/projects/codes/Mutation_studies/mutant_pdbs"
SEL_CSV   = "selected_mutants.csv"
OUT_CSV   = "pe_features.csv"
N_FRAMES  = 100   # sample this many frames evenly from trajectory


def build_simulation(pdb_file):
    """Build OpenMM simulation context for energy evaluation."""
    pdb = PDBFile(pdb_file)
    ff  = ForceField("charmm36.xml", "implicit/obc2.xml")
    mod = Modeller(pdb.topology, pdb.positions)
    mod.addHydrogens(ff)

    system = ff.createSystem(
        mod.topology,
        nonbondedMethod=NoCutoff,
        constraints=HBonds
    )
    integrator = VerletIntegrator(0.001 * picoseconds)
    platform   = Platform.getPlatformByName("CUDA")
    properties = {"CudaPrecision": "mixed"}

    sim = Simulation(mod.topology, system, integrator,
                     platform, properties)
    return sim, mod.topology


def compute_mean_pe(mut, rep=1):
    """
    Compute mean potential energy by replaying trajectory frames
    through OpenMM energy evaluator.
    Returns mean PE in kJ/mol.
    """
    rep_dir  = os.path.join(OUT_ROOT, f"{mut}_rep{rep}")
    dcd_file = os.path.join(rep_dir,  f"{mut}_rep{rep}_trajectory.dcd")
    min_file = os.path.join(rep_dir,  f"{mut}_rep{rep}_minimized.pdb")

    # Use mutant PDB for WT, use 1PGA.pdb
    if mut == "1PGA":
        pdb_file = os.path.join(PDB_DIR, "1PGA.pdb")
    else:
        pdb_file = os.path.join(PDB_DIR, f"{mut}.pdb")

    if not os.path.exists(dcd_file) or not os.path.exists(min_file):
        print(f"  [SKIP] Missing files for {mut}")
        return np.nan

    # Load trajectory
    u = mda.Universe(min_file, dcd_file)
    n_frames = len(u.trajectory)
    sample_idx = np.linspace(
        int(0.2 * n_frames),   # skip first 20% equilibration
        n_frames - 1,
        N_FRAMES,
        dtype=int
    )

    # Build simulation for energy evaluation
    try:
        sim, topology = build_simulation(min_file)
    except Exception as e:
        print(f"  [ERROR] Build failed for {mut}: {e}")
        return np.nan

    pe_values = []
    protein = u.select_atoms("protein")

    for idx in sample_idx:
        u.trajectory[idx]
        positions = protein.positions * 0.1  # Å → nm

        # Set positions in OpenMM context
        sim.context.setPositions(
            [Vec3(p[0], p[1], p[2]) for p in positions] * nanometers
        )

        # Get potential energy
        state = sim.context.getState(getEnergy=True)
        pe = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        pe_values.append(pe)

    return float(np.mean(pe_values))


if __name__ == "__main__":
    import pickle

    # Load mutant list
    with open("mutant_dataset.pkl", "rb") as f:
        pkl_df = pickle.load(f)

    mutant_list = ["1PGA"] + pkl_df["name"].tolist()

    print(f"Computing PE for {len(mutant_list)} mutants...")
    print(f"Sampling {N_FRAMES} frames per mutant (skipping first 20%)")
    print()

    records = []
    for i, mut in enumerate(mutant_list):
        print(f"[{i+1}/{len(mutant_list)}] {mut}...", end=" ", flush=True)
        pe = compute_mean_pe(mut)
        print(f"PE = {pe:.1f} kJ/mol" if not np.isnan(pe) else "FAILED")
        records.append({"mutant": mut, "mean_PE": pe})

    pe_df = pd.DataFrame(records)

    # Compute delta PE vs WT
    wt_pe = pe_df[pe_df["mutant"] == "1PGA"]["mean_PE"].values[0]
    print(f"\nWT (1PGA) mean PE: {wt_pe:.1f} kJ/mol")

    pe_df["delta_PE"] = pe_df["mean_PE"] - wt_pe
    pe_df.to_csv(OUT_CSV, index=False)
    print(f"Saved to {OUT_CSV}")

    # Validate against Nisthal y
    sel_df = pd.read_csv(SEL_CSV)
    merged = pe_df[pe_df["mutant"] != "1PGA"].merge(
        sel_df[["name", "y"]],
        left_on="mutant", right_on="name"
    ).dropna()

    if len(merged) > 5:
        rho_abs, p_abs = spearmanr(merged["mean_PE"],  merged["y"])
        rho_del, p_del = spearmanr(merged["delta_PE"], merged["y"])
        print(f"\nmean_PE  vs Nisthal y:  rho={rho_abs:+.3f}  p={p_abs:.4f}")
        print(f"delta_PE vs Nisthal y:  rho={rho_del:+.3f}  p={p_del:.4f}")

        if abs(rho_del) > 0.35:
            print("\n[GOOD] delta_PE is a useful stability proxy → use as y label")
        else:
            print("\n[WEAK] delta_PE correlation is weak — see discussion")
    else:
        print("\nNot enough matched mutants for validation")

    print("\nTop 10 most stable by delta_PE (most negative = most stable):")
    print(pe_df.nsmallest(10, "delta_PE")[["mutant","mean_PE","delta_PE"]].to_string())