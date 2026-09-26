"""
Phase II: real 3D structural features, extracted directly from the atomic
coordinates the label pipeline already produced (the .pqr files pdb2pqr
generated for APBS) -- no new downloads, no re-running the PB solve.

Motivation: the C1' local signal survived every Phase I attack
(composition, same-support base identity, wide k-mer context, sequence-
cluster splitting). Per the decision tree, the next question is whether
genuine 3D geometry -- not just sequence -- explains what's left. This
does NOT include explicit electrostatic descriptors (that would trivially
reconstruct the target); it's purely geometric/packing information.

Features (per residue, centered on its C1' atom):
  - n_neighbors_5A / n_neighbors_8A: heavy-atom count within 5/8 A,
    excluding the residue's own atoms. A local packing-density proxy.
  - n_tertiary_neighbors_5A: the subset of n_neighbors_5A belonging to
    residues more than 2 positions away in sequence (or on a different
    chain) -- i.e. genuine through-space (tertiary) contacts, not atoms
    that are just sequence-adjacent. This is the feature most directly
    testing "does this require knowing the fold", since sequence-
    adjacent packing is already implicitly available to any sequence
    model via local context.
  - dist_nearest_P_other: distance from C1' to the nearest phosphorus
    atom belonging to a DIFFERENT residue -- a simple proxy for how
    close this spot sits to another strand/backbone segment.
  - burial_proxy: 1 / (1 + n_neighbors_8A). NOT true solvent-accessible
    surface area -- that needs a proper rolling-probe calculation this
    script does not attempt. An honestly-labelled, cheap stand-in based
    on local atom crowding; treat accordingly, don't cite as "SASA".

Requirements: scipy (already a label-pipeline dependency). No ViennaRNA,
no torch, no GPU.

Runtime note: this loops per-residue with a KD-tree neighbor query plus a
small Python loop per residue for the tertiary-contact count -- expect
something in the same ballpark as the ViennaRNA folding step (order of an
hour for ~2000 structures). Start with --limit-structures for a quick
sanity check before the full run.

Usage:
    python structural_3d_baseline.py \
        --labels-dir /data/rna_mg/cutoff_4_8_12 \
        --label-column potential_at_c1prime_kT_e \
        --outdir /data/rna_mg/priority_results_c1prime/structural_3d
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("structural_3d_baseline")

from rna_electrostatics_pipeline_v2 import parse_pqr_residues  # noqa: E402
from rna_embedding_probe_v2 import load_structure_labels  # noqa: E402
from composition_position_baselines import split_ridge_r2  # noqa: E402

SEQ_ADJACENT_WINDOW = 2  # residues within this many sequence positions are NOT "tertiary"

STRUCT3D_FEATURE_NAMES = ["n_neighbors_5A", "n_neighbors_8A", "n_tertiary_neighbors_5A",
                           "dist_nearest_P_other", "burial_proxy"]


def _is_heavy_atom(atom_name: str) -> bool:
    """True unless `atom_name` names a hydrogen (or deuterium) atom.

    PQR files carry no explicit element column, so element has to be
    inferred from the atom name. pdb2pqr/AMBER hydrogen names are NOT
    always a plain leading "H" -- stereo-labelled hydrogens are commonly
    written with a leading digit, e.g. "1H5'", "2H5'", "1H2''" -- so a
    naive `name.startswith("H")` check silently keeps roughly half the
    hydrogens in the file. No standard RNA heavy-atom name starts with a
    digit followed by H/D, so stripping leading digits before checking
    the first letter is both necessary and safe here.
    """
    stripped = atom_name.lstrip("0123456789")
    return stripped[:1] not in ("H", "D")


def extract_structural_features(pqr_path: Path, with_sasa: bool = False) -> pd.DataFrame:
    """One row per residue: (chain, resnum, resname) + the 3D features
    above, computed from the PQR's atom coordinates via a KD-tree.

    Hydrogens are dropped before any of this runs (see _is_heavy_atom):
    pdb2pqr's PQR output includes explicit hydrogens (needed for correct
    protonation states in the PB solve), and hydrogen COUNT differs
    systematically between purines and pyrimidines. Left in, that
    silently reintroduces a base-identity confound into features that
    are supposed to be purely geometric -- exactly the kind of thing
    Phase I was designed to rule out.

    with_sasa: also compute a real per-residue solvent-accessible
    surface area (sum of per-atom Shrake-Rupley SASA over the residue's
    heavy atoms) as "residue_sasa_A2" -- a principled replacement for
    `burial_proxy`'s neighbor-count heuristic, computed from the same
    coordinates/radii, still no simulation required. Adds real runtime
    cost (see compute_per_atom_sasa), so it's opt-in.
    """
    from scipy.spatial import cKDTree

    atoms = parse_pqr_residues(pqr_path)
    if atoms.empty:
        return pd.DataFrame()

    atoms = atoms[atoms["atom_name"].map(_is_heavy_atom)].reset_index(drop=True)
    if atoms.empty:
        return pd.DataFrame()

    all_xyz = atoms[["x", "y", "z"]].to_numpy()
    tree = cKDTree(all_xyz)

    p_atoms = atoms[atoms["atom_name"] == "P"]
    p_tree = cKDTree(p_atoms[["x", "y", "z"]].to_numpy()) if len(p_atoms) else None

    atom_sasa = compute_per_atom_sasa(atoms) if with_sasa else None

    rows = []
    for (chain, resnum, resname), group in atoms.groupby(["chain", "resnum", "resname"], sort=False):
        c1p = group[group["atom_name"].isin(["C1'", "C1*"])]
        if c1p.empty:
            continue
        center = c1p[["x", "y", "z"]].iloc[0].to_numpy()
        own_atom_idxs = set(group.index)

        idx_5 = tree.query_ball_point(center, r=5.0)
        idx_8 = tree.query_ball_point(center, r=8.0)
        n5 = len([i for i in idx_5 if atoms.index[i] not in own_atom_idxs])
        n8 = len([i for i in idx_8 if atoms.index[i] not in own_atom_idxs])

        n_tertiary = 0
        for i in idx_5:
            if atoms.index[i] in own_atom_idxs:
                continue
            row_i = atoms.iloc[i]
            same_chain = row_i["chain"] == chain
            close_in_seq = same_chain and abs(int(row_i["resnum"]) - int(resnum)) <= SEQ_ADJACENT_WINDOW
            if not close_in_seq:
                n_tertiary += 1

        dist_other_p = np.nan
        if p_tree is not None and len(p_atoms):
            own_p_idx = set(group[group["atom_name"] == "P"].index)
            k = min(5, len(p_atoms))
            dists, idxs = p_tree.query(center, k=k)
            dists = np.atleast_1d(dists)
            idxs = np.atleast_1d(idxs)
            for d, pi in zip(dists, idxs):
                if p_atoms.index[pi] not in own_p_idx:
                    dist_other_p = float(d)
                    break

        row = {
            "chain": chain, "resnum": resnum, "resname": resname,
            "n_neighbors_5A": n5, "n_neighbors_8A": n8,
            "n_tertiary_neighbors_5A": n_tertiary,
            "dist_nearest_P_other": dist_other_p,
            "burial_proxy": 1.0 / (1.0 + n8),
        }
        if atom_sasa is not None:
            own_positions = [atoms.index.get_loc(idx) for idx in own_atom_idxs]
            row["residue_sasa_A2"] = float(atom_sasa[own_positions].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _fibonacci_sphere(n_points: int) -> np.ndarray:
    """n_points roughly-uniformly-spaced points on a unit sphere."""
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    i = np.arange(n_points)
    z = 1 - 2 * (i + 0.5) / n_points
    radius_xy = np.sqrt(np.clip(1 - z * z, 0.0, None))
    theta = golden_angle * i
    return np.stack([radius_xy * np.cos(theta), radius_xy * np.sin(theta), z], axis=1)


def compute_per_atom_sasa(atoms: pd.DataFrame, probe_radius: float = 1.4,
                           n_sphere_points: int = 92) -> np.ndarray:
    """Shrake-Rupley solvent-accessible surface area, per atom (A^2).

    Real geometry -- a rolling probe sphere over the atomic coordinates
    -- rather than a crowding heuristic like `burial_proxy`. Uses the
    PQR's own AMBER radii (now returned by `parse_pqr_residues` as the
    "radius" column) rather than an external radius table, so it stays
    consistent with the exact atoms/radii the PB solve itself used.

    O(n_atoms^2)-ish per structure (each atom checks candidate neighbors
    via a KD-tree, then tests ~90 sphere points against them), noticeably
    slower than the other structural_3d features -- opt in via
    --with-sasa rather than always computing it.
    """
    from scipy.spatial import cKDTree

    coords = atoms[["x", "y", "z"]].to_numpy()
    ext_radii = atoms["radius"].to_numpy() + probe_radius
    sphere = _fibonacci_sphere(n_sphere_points)
    tree = cKDTree(coords)
    max_ext = ext_radii.max() if len(ext_radii) else 0.0

    sasa = np.zeros(len(atoms), dtype=float)
    for i in range(len(atoms)):
        neighbor_idx = [j for j in tree.query_ball_point(coords[i], r=ext_radii[i] + max_ext) if j != i]
        if not neighbor_idx:
            sasa[i] = 4 * np.pi * ext_radii[i] ** 2
            continue
        test_points = coords[i] + sphere * ext_radii[i]                     # (P, 3)
        nbr_coords = coords[neighbor_idx]                                    # (M, 3)
        nbr_radii = ext_radii[neighbor_idx]                                  # (M,)
        d2 = ((test_points[:, None, :] - nbr_coords[None, :, :]) ** 2).sum(-1)  # (P, M)
        buried = (d2 < (nbr_radii[None, :] ** 2)).any(axis=1)
        sasa[i] = (1.0 - buried.mean()) * 4 * np.pi * ext_radii[i] ** 2
    return sasa


def build_residue_level_table(labels_dir: Path, pdb_ids: list[str], label_column: str,
                               include_flagged: bool, with_sasa: bool = False) -> pd.DataFrame:
    feature_names = STRUCT3D_FEATURE_NAMES + (["residue_sasa_A2"] if with_sasa else [])
    frames = []
    for pdb_id in pdb_ids:
        labels = load_structure_labels(labels_dir, pdb_id, include_flagged)
        if labels is None or label_column not in labels.columns:
            continue
        pqr_path = labels_dir / "work" / pdb_id.upper() / f"{pdb_id.lower()}.pqr"
        if not pqr_path.exists():
            continue
        struct_feats = extract_structural_features(pqr_path, with_sasa=with_sasa)
        if struct_feats.empty:
            continue
        merged = labels.merge(struct_feats, on=["chain", "resnum", "resname"], how="inner")
        merged = merged.dropna(subset=[label_column] + feature_names)
        if merged.empty:
            continue
        merged = merged.copy()
        merged["chain_id"] = merged["chain"].apply(lambda c, p=pdb_id: f"{p}::{c}")
        frames.append(merged)
    if not frames:
        raise RuntimeError(
            "No usable rows assembled -- check .pqr files exist under labels_dir/work/<PDB_ID>/"
        )
    df = pd.concat(frames, ignore_index=True)
    df["y"] = df[label_column].astype(float)
    return df


def add_centered_and_rank(df: pd.DataFrame) -> None:
    mu = df.groupby("chain_id")["y"].transform("mean")
    df["y_centered"] = df["y"] - mu
    df["y_rank_pct"] = df.groupby("chain_id")["y"].rank(pct=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--label-column", required=True)
    ap.add_argument("--outdir", type=Path, default=Path("./structural_3d_baseline"))
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--limit-structures", type=int, default=None)
    ap.add_argument("--with-sasa", action="store_true",
                     help="Also compute real per-residue SASA (Shrake-Rupley) from the PQR's own "
                          "atomic radii and report it alongside the crowding-based features -- "
                          "noticeably slower (see compute_per_atom_sasa).")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    all_pot_path = args.labels_dir / "all_residue_potentials.csv"
    pdb_ids = sorted(pd.read_csv(all_pot_path, low_memory=False)["pdb_id"].unique())
    if args.limit_structures:
        pdb_ids = pdb_ids[: args.limit_structures]

    log.info("=== Extracting 3D structural features for %d structures ===", len(pdb_ids))
    df = build_residue_level_table(args.labels_dir, pdb_ids, args.label_column, args.include_flagged,
                                    with_sasa=args.with_sasa)
    add_centered_and_rank(df)
    df.to_csv(args.outdir / "residue_level_structural_3d_table.csv", index=False)
    log.info("Assembled %d residues across %d chains", len(df), df["chain_id"].nunique())

    splits = df["split"].to_numpy() if "split" in df.columns else np.full(len(df), "train")

    feature_sets = {"structural_3d": STRUCT3D_FEATURE_NAMES}
    if args.with_sasa:
        feature_sets["structural_3d_plus_sasa"] = STRUCT3D_FEATURE_NAMES + ["residue_sasa_A2"]
        feature_sets["sasa_only"] = ["residue_sasa_A2"]

    targets = {
        "raw": df["y"].to_numpy(dtype=float),
        "within_chain_centered": df["y_centered"].to_numpy(dtype=float),
        "within_chain_rank_pct": df["y_rank_pct"].to_numpy(dtype=float),
    }

    rows = []
    log.info("=== Regressing each target on 3D structural features ===")
    for target_name, y in targets.items():
        for feat_name, feat_cols in feature_sets.items():
            X = df[feat_cols].to_numpy(dtype=float)
            r2 = split_ridge_r2(X, y, splits)
            rows.append({"target": target_name, "features": feat_name, "test_r2": r2})
            log.info("target=%-24s features=%-24s test_R^2=%.4f", target_name, feat_name, r2)

    pd.DataFrame(rows).to_csv(args.outdir / "structural_3d_summary.csv", index=False)
    log.info("Wrote outputs to %s", args.outdir)


if __name__ == "__main__":
    main()