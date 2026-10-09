"""
Does a sequence-only LM embedding, decoded and then smoothly interpolated
across 3D space, approximate the REAL electrostatic field (the actual
APBS .dx solution) -- and does it do so BETTER than the obvious baseline
of just smoothly interpolating the sparse true per-residue values you
already have, with no embedding at all?

WHY THE BASELINE IS THE WHOLE POINT of this script, not an afterthought:
comparing an LM-derived field to the real APBS field, on its own, risks
being close to circular -- of course a field fit to correlate with samples
of that same field will correlate with it somewhat. The informative
question is comparative: on TEST-split structures the decoder never saw
during fitting, does filling in the gaps between residues using ONLY
sequence information get you anywhere near as close to the true field as
filling in those same gaps using the true values at the residues you
already have? If yes, that's the extraordinary result (a usable
approximate field for a structure you'd otherwise need to run APBS on).
If the LM field is far worse than the naive baseline, that's still a real
result -- it bounds how far this line of work can currently go -- but a
very different one to report.

Ground truth: the actual APBS potential.dx grid for each structure,
already sitting on disk at
    {labels_dir}/work/{PDB_ID}/potential.dx
(never deleted by the label pipeline) -- read via the SAME
`trilinear_interpolate` used to build potential_at_c1prime_kT_e /
potential_at_phosphorus_kT_e in the first place, imported directly rather
than reimplemented, so there is exactly one trilinear-interpolation
implementation in this whole project.

Two evaluation protocols, both computed ONLY on TEST-split structures and
ONLY at points that are NOT the residue positions used to fit anything
(otherwise this just re-derives the R^2 numbers already in hand):

  Backbone-midpoint evaluation (always run, no extra parameters to tune):
    for every pair of sequence-adjacent residues (seq_index differing by
    exactly 1, same chain), evaluate all three sources -- true APBS field,
    LM-field interpolant, baseline-field interpolant -- at the 3D
    midpoint between them. A genuinely held-out point by construction:
    it was never a fitting location for either interpolant.

  Grid-sampling evaluation (--grid-eval, off by default -- slower):
    random .dx grid VOXELS within --max-dist-from-atom of any heavy atom
    (heavy-atom filter reused from structural_3d_baseline.py's
    _is_heavy_atom, for the same reason it mattered there: PQR files
    include explicit hydrogens). This is the fuller "does the field look
    right throughout space" comparison, at real cost (more points, RBF
    evaluation per point) -- run it once the cheap backbone-midpoint
    numbers look promising.

The "LM field" itself is a single-target frozen ridge decoder (same
StandardScaler + RidgeCV convention as run_ridge_probe elsewhere in this
project), NOT the shared PLS subspace from electrostatic_latent_manifold.py
-- this experiment asks what sequence information can do for ONE specific
target at its best, not about the shared-subspace question, which is
already answered elsewhere.

Usage:
    python electrostatic_field_mapping.py \
        --labels-dir /data/rna_mg/res_phi \
        --model ernierna \
        --target potential_at_c1prime_kT_e \
        --outdir /data/rna_mg/field_mapping/ernierna \
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("electrostatic_field_mapping")

from rna_electrostatics_pipeline_v2 import trilinear_interpolate, parse_pqr_residues  # noqa: E402
from rna_embedding_probe_v2 import select_best_layer_and_test  # noqa: E402
from rna_probe_priority_experiments_extended import build_dataset_for_model_with_chains  # noqa: E402
from structural_3d_baseline import _is_heavy_atom  # noqa: E402


def fit_frozen_decoder(H_train: np.ndarray, y_train: np.ndarray):
    """Same StandardScaler + RidgeCV convention as run_ridge_probe, but
    returns the fitted (scaler, model) pair so it can be applied to
    arbitrary new structures later -- run_ridge_probe only returns
    metrics, not the fitted objects."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(H_train)
    model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(scaler.transform(H_train), y_train)
    return scaler, model


def load_dx_grid(dx_path: Path):
    from gridData import Grid

    grid = Grid(str(dx_path))
    origin = np.asarray(grid.origin, dtype=float)
    delta = np.asarray(grid.delta, dtype=float)
    dims = np.array(grid.grid.shape)
    return grid.grid, origin, delta, dims


def dedupe_points(coords: np.ndarray, *value_arrays: np.ndarray, decimals: int = 2):
    """Merge points whose coordinates coincide (after rounding to
    `decimals` places, i.e. 0.01 A) by averaging every value array over
    the group. Coincident points make RBF interpolation matrices
    singular (LinAlgError: Singular matrix), and they do occur in real
    structures: residues sharing a representative-atom position, or
    duplicated chains/models. Returns (unique_coords, *averaged_values,
    n_dropped). Both interpolants (LM and baseline) must be built from
    the SAME deduplicated points so the comparison stays fair."""
    finite = np.isfinite(coords).all(axis=1)
    for v in value_arrays:
        finite &= np.isfinite(v)
    coords = coords[finite]
    value_arrays = [v[finite] for v in value_arrays]
    keys = np.round(coords, decimals)
    _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inverse = inverse.ravel()
    n_unique = len(counts)
    uniq_coords = np.zeros((n_unique, coords.shape[1]))
    np.add.at(uniq_coords, inverse, coords)
    uniq_coords /= counts[:, None]
    averaged = []
    for v in value_arrays:
        acc = np.zeros(n_unique)
        np.add.at(acc, inverse, v)
        averaged.append(acc / counts)
    return (uniq_coords, *averaged, int(len(coords) - n_unique))


def build_field_interpolant(coords: np.ndarray, values: np.ndarray, kernel: str, neighbors: int):
    from scipy.interpolate import RBFInterpolator

    n = coords.shape[0]
    # tiny smoothing regularizes near-singular local systems (nearly
    # coincident points that survive rounding); negligible vs. kT/e values.
    return RBFInterpolator(coords, values, kernel=kernel, neighbors=min(neighbors, n), smoothing=1e-6)


def backbone_midpoint_eval(pdb_id: str, coords: np.ndarray, seq_idx: np.ndarray,
                            lm_interp, baseline_interp, dx_values, dx_origin, dx_delta, dx_dims) -> list[dict]:
    order = np.argsort(seq_idx)
    coords, seq_idx = coords[order], seq_idx[order]
    rows = []
    for i in range(len(seq_idx) - 1):
        if seq_idx[i + 1] - seq_idx[i] != 1:
            continue  # not truly sequence-adjacent (chain break / gap) -- skip
        midpoint = (coords[i] + coords[i + 1]) / 2.0
        true_val = trilinear_interpolate(dx_values, dx_origin, dx_delta, dx_dims, midpoint)
        if not np.isfinite(true_val):
            continue
        lm_val = float(lm_interp(midpoint[None, :])[0])
        baseline_val = float(baseline_interp(midpoint[None, :])[0])
        rows.append({"pdb_id": pdb_id, "eval_type": "backbone_midpoint",
                      "x": midpoint[0], "y": midpoint[1], "z": midpoint[2],
                      "true_apbs": true_val, "lm_field": lm_val, "baseline_field": baseline_val})
    return rows


def grid_sample_eval(pdb_id: str, pqr_path: Path, dx_values, dx_origin, dx_delta, dx_dims,
                      lm_interp, baseline_interp, max_dist_from_atom: float, n_samples: int,
                      rng: np.random.Generator) -> list[dict]:
    from scipy.spatial import cKDTree

    atoms = parse_pqr_residues(pqr_path)
    atoms = atoms[atoms["atom_name"].map(_is_heavy_atom)]
    if atoms.empty:
        return []
    atom_tree = cKDTree(atoms[["x", "y", "z"]].to_numpy())

    lo = np.floor((atoms[["x", "y", "z"]].min().to_numpy() - max_dist_from_atom - dx_origin) / dx_delta).astype(int)
    hi = np.ceil((atoms[["x", "y", "z"]].max().to_numpy() + max_dist_from_atom - dx_origin) / dx_delta).astype(int)
    lo, hi = np.clip(lo, 0, dx_dims), np.clip(hi, 0, dx_dims)
    if np.any(hi <= lo):
        return []

    ii = rng.integers(lo[0], hi[0], size=n_samples * 4)
    jj = rng.integers(lo[1], hi[1], size=n_samples * 4)
    kk = rng.integers(lo[2], hi[2], size=n_samples * 4)
    candidate_pts = dx_origin + np.stack([ii, jj, kk], axis=1) * dx_delta
    dist_to_nearest_atom, _ = atom_tree.query(candidate_pts)
    keep = dist_to_nearest_atom <= max_dist_from_atom
    ii, jj, kk, candidate_pts = ii[keep], jj[keep], kk[keep], candidate_pts[keep]
    if len(candidate_pts) == 0:
        return []
    take = rng.choice(len(candidate_pts), size=min(n_samples, len(candidate_pts)), replace=False)

    rows = []
    for idx in take:
        pt = candidate_pts[idx]
        true_val = float(dx_values[ii[idx], jj[idx], kk[idx]])
        lm_val = float(lm_interp(pt[None, :])[0])
        baseline_val = float(baseline_interp(pt[None, :])[0])
        rows.append({"pdb_id": pdb_id, "eval_type": "grid_sample",
                      "x": pt[0], "y": pt[1], "z": pt[2],
                      "true_apbs": true_val, "lm_field": lm_val, "baseline_field": baseline_val})
    return rows


def summarize(rows: list[dict], label: str):
    from scipy.stats import pearsonr, spearmanr

    if not rows:
        log.warning("%s: no evaluation points collected, skipping summary", label)
        return None
    df = pd.DataFrame(rows)
    out = {}
    for field_col in ("lm_field", "baseline_field"):
        r, _ = pearsonr(df["true_apbs"], df[field_col])
        rho, _ = spearmanr(df["true_apbs"], df[field_col])
        rmse = float(np.sqrt(np.mean((df["true_apbs"] - df[field_col]) ** 2)))
        out[field_col] = {"pearson_r": float(r), "spearman_rho": float(rho), "rmse": rmse}
    log.info("=== %s (n=%d points pooled across test structures) ===", label, len(df))
    log.info("  LM field:       r=%.3f  rho=%.3f  RMSE=%.3f",
              out["lm_field"]["pearson_r"], out["lm_field"]["spearman_rho"], out["lm_field"]["rmse"])
    log.info("  Baseline field: r=%.3f  rho=%.3f  RMSE=%.3f",
              out["baseline_field"]["pearson_r"], out["baseline_field"]["spearman_rho"], out["baseline_field"]["rmse"])
    return df, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default="potential_at_c1prime_kT_e")
    ap.add_argument("--layer", type=int, default=None)
    ap.add_argument("--outdir", type=Path, default=Path("./field_mapping"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--limit-structures", type=int, default=None)
    ap.add_argument("--min-residues-per-structure", type=int, default=10)
    ap.add_argument("--rbf-kernel", default="linear")
    ap.add_argument("--rbf-neighbors", type=int, default=12)
    ap.add_argument("--grid-eval", action="store_true", help="Also run the (slower) grid-sampling evaluation")
    ap.add_argument("--max-dist-from-atom", type=float, default=15.0)
    ap.add_argument("--n-grid-samples", type=int, default=2000, help="Per structure, for --grid-eval")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_pot_path = args.labels_dir / "all_residue_potentials.csv"
    if not all_pot_path.exists():
        raise FileNotFoundError(f"{all_pot_path} not found -- run the label pipeline first.")
    pdb_ids = sorted(pd.read_csv(all_pot_path, low_memory=False)["pdb_id"].unique())
    if args.limit_structures:
        pdb_ids = pdb_ids[: args.limit_structures]

    log.info("=== Building embeddings for %s (%d structures) ===", args.model, len(pdb_ids))
    X, y_by_column, pdb_groups, chain_groups, splits, base_ids, resnums, seq_indices, eff_max_len = \
        build_dataset_for_model_with_chains(args.labels_dir, pdb_ids, args.model, args.device,
                                             args.include_flagged, args.max_len, [args.target])

    y = y_by_column[args.target]
    valid = ~np.isnan(y)

    if args.layer is not None:
        layer = args.layer
    else:
        X_valid = {i: arr[valid] for i, arr in X.items()}
        result = select_best_layer_and_test(X_valid, y[valid], splits[valid], args.model, args.target)
        layer = result["best_layer"]
        log.info("Selected layer=%d via %s (own test_R^2=%.3f)", layer, args.target, result["test_r2"])

    H = X[layer]
    trainval = valid & ((splits == "train") | (splits == "val"))
    test = valid & (splits == "test")
    log.info("Fitting frozen decoder on %d train+val residues", int(trainval.sum()))
    scaler, model = fit_frozen_decoder(H[trainval], y[trainval])

    labels_df = pd.read_csv(all_pot_path, low_memory=False)

    backbone_rows, grid_rows = [], []
    n_evaluated = n_skipped = total_dropped = 0
    test_pdb_ids = sorted(set(pdb_groups[test]))
    log.info("=== Building fields for %d test-split structures ===", len(test_pdb_ids))
    for pdb_id in test_pdb_ids:
        struct_mask = test & (pdb_groups == pdb_id)
        n_res = int(struct_mask.sum())
        if n_res < args.min_residues_per_structure:
            continue

        struct_labels = labels_df[(labels_df["pdb_id"] == pdb_id) & labels_df["x"].notna()]
        struct_labels = struct_labels.set_index(["chain", "resnum"])
        # chain_groups holds "PDBID::chain" composites (see
        # build_dataset_for_model_with_chains), NOT the bare chain letter the
        # label table's `chain` column uses -- strip the prefix before lookup.
        chains = np.array([c.split("::", 1)[1] for c in chain_groups[struct_mask]])
        resns = resnums[struct_mask]
        struct_labels = struct_labels[~struct_labels.index.duplicated(keep="first")]
        try:
            coord_rows = struct_labels.loc[list(zip(chains, resns))]
        except KeyError as e:
            log.warning("%s: could not align coordinates for all residues (%s), skipping", pdb_id, e)
            continue
        coords = coord_rows[["x", "y", "z"]].to_numpy()
        y_true = y[struct_mask]
        seq_idx = seq_indices[struct_mask]
        H_struct = H[struct_mask]

        y_lm_pred = model.predict(scaler.transform(H_struct))

        dx_path = args.labels_dir / "work" / pdb_id.upper() / "potential.dx"
        pqr_path = args.labels_dir / "work" / pdb_id.upper() / f"{pdb_id.lower()}.pqr"
        if not dx_path.exists() or not pqr_path.exists():
            log.warning("%s: missing potential.dx or .pqr, skipping", pdb_id)
            continue
        dx_values, dx_origin, dx_delta, dx_dims = load_dx_grid(dx_path)

        u_coords, u_lm, u_true, n_dropped = dedupe_points(coords, y_lm_pred, y_true)
        total_dropped += n_dropped
        if len(u_coords) < max(4, args.min_residues_per_structure):
            log.warning("%s: only %d distinct points after deduplication, skipping", pdb_id, len(u_coords))
            n_skipped += 1
            continue

        try:
            lm_interp = build_field_interpolant(u_coords, u_lm, args.rbf_kernel, args.rbf_neighbors)
            baseline_interp = build_field_interpolant(u_coords, u_true, args.rbf_kernel, args.rbf_neighbors)
            new_backbone = backbone_midpoint_eval(pdb_id, coords, seq_idx, lm_interp, baseline_interp,
                                                   dx_values, dx_origin, dx_delta, dx_dims)
            new_grid = []
            if args.grid_eval:
                new_grid = grid_sample_eval(pdb_id, pqr_path, dx_values, dx_origin, dx_delta, dx_dims,
                                             lm_interp, baseline_interp, args.max_dist_from_atom,
                                             args.n_grid_samples, rng)
        except np.linalg.LinAlgError:
            log.warning("%s: interpolation matrix still singular after deduplication, skipping", pdb_id)
            n_skipped += 1
            continue
        backbone_rows += new_backbone
        grid_rows += new_grid
        n_evaluated += 1

    log.info("Structures evaluated: %d | skipped: %d | coincident points merged: %d",
              n_evaluated, n_skipped, total_dropped)

    backbone_result = summarize(backbone_rows, "Backbone-midpoint evaluation")
    if backbone_result is not None:
        backbone_result[0].to_csv(args.outdir / "field_mapping_backbone_eval.csv", index=False)

    if args.grid_eval:
        grid_result = summarize(grid_rows, "Grid-sampling evaluation")
        if grid_result is not None:
            grid_result[0].to_csv(args.outdir / "field_mapping_grid_eval.csv", index=False)

    log.info("All field-mapping outputs written to %s", args.outdir)


if __name__ == "__main__":
    main()