"""
Relative-Change Feature Extraction (Hou et al. style)
======================================================
Instead of one scalar per mutant, compute PER-RESIDUE dynamic
properties for WT and mutant, then take relative change per
residue, then aggregate (mean of |relative change|).

This is the same logic as ESMDance's mutation-effect formula:
    Si = mean( |P_mut - P_WT| / P_WT )

Applied here to:
  - per-residue water contacts (proxy for local SASA)
  - per-residue RMSF
  - per-residue native-contact loss

Requires explicit_features.csv to already have per-mutant trajectories
processed (uses the same trajectory loading as analyze_explicit.py).

Usage:
  python compute_relative_features.py
"""

import os
import numpy as np
import pandas as pd
import pickle
from scipy.stats import spearmanr
import MDAnalysis as mda
from MDAnalysis.analysis import rms
import warnings
warnings.filterwarnings("ignore")

OUT_ROOT = "/data/openmm_out/1PGA_mutations/explicit_solvent"
SEL_CSV  = "selected_mutants.csv"
PKL_PATH = "mutant_dataset.pkl"
OUT_CSV  = "relative_features.csv"


def load_trajectory(mut, rep=1):
    rep_dir    = os.path.join(OUT_ROOT, f"{mut}_rep{rep}")
    dcd_file   = os.path.join(rep_dir, f"{mut}_rep{rep}_trajectory.dcd")
    final_file = os.path.join(rep_dir, f"{mut}_rep{rep}_final.pdb")
    min_file   = os.path.join(rep_dir, f"{mut}_rep{rep}_minimized.pdb")

    if not os.path.exists(dcd_file):
        return None
    top_file = final_file if os.path.exists(final_file) else min_file
    try:
        return mda.Universe(top_file, dcd_file)
    except ValueError:
        return None


def per_residue_water_contacts(u, step=100):
    """
    For each residue, count mean number of water oxygens within 4.5 Å
    of any polar sidechain atom. Returns a vector of length n_residues.
    """
    residues = u.select_atoms("protein").residues
    water_O  = u.select_atoms("resname HOH and name O")

    n_res = len(residues)
    counts = np.zeros((len(u.trajectory[::step]), n_res))

    for fi, ts in enumerate(u.trajectory[::step]):
        for ri, res in enumerate(residues):
            polar_atoms = res.atoms.select_atoms("name N* O* S*")
            if len(polar_atoms) == 0:
                continue
            c = 0
            for a in polar_atoms.positions:
                d = np.linalg.norm(water_O.positions - a, axis=1)
                c += np.sum(d < 4.5)
            counts[fi, ri] = c

    return counts.mean(axis=0)   # mean per-residue, shape (n_residues,)


def per_residue_rmsf(u):
    """Per-residue Cα RMSF vector."""
    ca = u.select_atoms("protein and name CA")
    r  = rms.RMSF(ca).run()
    return r.rmsf   # shape (n_residues,)


def relative_change_score(wt_vec, mut_vec):
    """
    Hou et al. style: Si = mean( |P_mut - P_WT| / P_WT )
    Handles zero/near-zero WT values safely.
    """
    wt_vec  = np.asarray(wt_vec, dtype=float)
    mut_vec = np.asarray(mut_vec, dtype=float)

    if len(wt_vec) != len(mut_vec):
        n = min(len(wt_vec), len(mut_vec))
        wt_vec, mut_vec = wt_vec[:n], mut_vec[:n]

    # avoid division by zero — add small epsilon
    # eps = 1e-3
    # rel = np.abs(mut_vec - wt_vec) / (np.abs(wt_vec) + eps)
    mask = np.abs(wt_vec) > 0.05
    rel = np.abs(mut_vec[mask] - wt_vec[mask]) / (np.abs(wt_vec[mask]))
    return float(np.median(rel))


def run():
    with open(PKL_PATH, "rb") as f:
        pkl_df = pickle.load(f)
    mutant_list = pkl_df["name"].tolist()

    print("Loading WT trajectory...")
    u_wt = load_trajectory("1PGA")
    if u_wt is None:
        print("WT trajectory not found.")
        return

    wt_water_vec = per_residue_water_contacts(u_wt)
    wt_rmsf_vec  = per_residue_rmsf(u_wt)
    print(f"WT: {len(wt_water_vec)} residues\n")

    records = []
    for i, mut in enumerate(mutant_list):
        u = load_trajectory(mut)
        if u is None:
            print(f"[{i+1}/{len(mutant_list)}] {mut} — [SKIP] no trajectory")
            continue

        print(f"[{i+1}/{len(mutant_list)}] {mut}")
        mut_water_vec = per_residue_water_contacts(u)
        mut_rmsf_vec  = per_residue_rmsf(u)

        score_water = relative_change_score(wt_water_vec, mut_water_vec)
        score_rmsf  = relative_change_score(wt_rmsf_vec,  mut_rmsf_vec)

        records.append({
            "mutant": mut,
            "rel_water_score": score_water,
            "rel_rmsf_score":  score_rmsf,
        })
        print(f"  rel_water_score={score_water:.4f}  rel_rmsf_score={score_rmsf:.4f}")

    df = pd.DataFrame(records)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")
    return df


def validate(rel_csv=OUT_CSV, sel_csv=SEL_CSV):
    feat = pd.read_csv(rel_csv)
    sel  = pd.read_csv(sel_csv)

    merged = feat.merge(sel[["name", "y"]], left_on="mutant", right_on="name")

    print("\n" + "=" * 50)
    print("VALIDATION — relative-change scores vs Nisthal y")
    print("=" * 50)

    for col in ["rel_water_score", "rel_rmsf_score"]:
        valid = merged[[col, "y"]].dropna()
        rho, p = spearmanr(valid[col], valid["y"])
        # Note: higher relative change = MORE disrupted by mutation
        # = expected destabilizing → should correlate NEGATIVELY with y
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
        print(f"{col:<20}  rho={rho:+.3f}  p={p:.4f}  {sig}")

    # Combined score (geometric mean, like their approach)
    merged["combined"] = np.sqrt(
        merged["rel_water_score"] * merged["rel_rmsf_score"]
    )
    rho, p = spearmanr(merged["combined"], merged["y"])
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    print(f"{'combined (geo mean)':<20}  rho={rho:+.3f}  p={p:.4f}  {sig}")


if __name__ == "__main__":
    df = run()
    if df is not None and len(df) > 5:
        validate()