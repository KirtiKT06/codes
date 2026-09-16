"""
Composition, length, and position-only baselines for the RNA electrostatic
label set. No GPU, no LM embeddings, no HuggingFace downloads -- runs
directly against per_residue_potential.csv / seq_index_map.csv / .fasta
already on disk from the label pipeline. Meant to run alongside (or before)
any GPU-bound probing, since it's a couple of minutes on CPU for ~2k
structures.

Motivation: three independent diagnostics (mean-pooled-embedding scale
probe, within-chain label shuffling, and now the within-chain
centered/rank probe) point to the LM embeddings mostly encoding a
per-chain electrostatic SCALE rather than local per-residue variation.
This script asks the next obvious question: is that scale just explained
by simple physical/compositional covariates -- chain length, GC content,
base composition -- rather than anything the LM specifically learned?
And separately, does raw sequence position (residue index / chain length)
carry any of the raw target's signal on its own (Set VI-C/D from the
original review)?

Two experiments:

  A) STRUCTURE-LEVEL: length + GC% + per-base fraction -> mu_chain (the
     same chain-level scalar target used in the embedding scale probe).
     KFold CV over chains (no GPU-driven train/val/test split needed,
     since there's nothing to overfit to at this feature count).

  B) RESIDUE-LEVEL: (a) the same composition features, broadcast constant
     across every residue in a chain, and (b) position-in-chain (i / L)
     as the sole feature -- both regressed against the RAW per-residue
     target using the existing PDB-level train/val/test split, for a
     direct comparison against the LM's raw-target R^2 from the main
     probing run.

Usage:
    python composition_position_baselines.py --labels-dir /data/rna_mg/cutoff_4_8_12 \
        --label-column potential_at_phosphorus_kT_e --outdir /data/rna_mg/priority_results/composition
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("composition_baselines")

from rna_embedding_probe_v2 import load_structure_labels, load_fasta_by_chain  # noqa: E402

BASES = ["A", "C", "G", "U"]


def composition_features(seq: str) -> np.ndarray:
    seq = seq.upper()
    length = len(seq)
    if length == 0:
        return np.zeros(6, dtype=float)
    counts = {b: seq.count(b) for b in BASES}
    fracs = np.array([counts[b] / length for b in BASES], dtype=float)
    gc = fracs[BASES.index("C")] + fracs[BASES.index("G")]
    return np.concatenate([[float(length)], [gc], fracs])


FEATURE_NAMES = ["length", "gc_frac", "a_frac", "c_frac", "g_frac", "u_frac"]


def build_per_chain_table(labels_dir: Path, pdb_ids: list[str], label_column: str,
                           include_flagged: bool) -> pd.DataFrame:
    """One row per (pdb_id, chain): composition features + mu_chain (mean
    raw target) + split (inherited from the label pipeline's assign_split,
    same convention used throughout)."""
    rows = []
    for pdb_id in pdb_ids:
        labels = load_structure_labels(labels_dir, pdb_id, include_flagged)
        if labels is None or label_column not in labels.columns:
            continue
        seqs = load_fasta_by_chain(labels_dir, pdb_id)
        for chain, chain_rows in labels.groupby("chain"):
            seq = seqs.get(chain)
            y = chain_rows[label_column].dropna()
            if not seq or y.empty:
                continue
            feats = composition_features(seq)
            row = dict(zip(FEATURE_NAMES, feats))
            row["pdb_id"] = pdb_id
            row["chain"] = chain
            row["mu_chain"] = float(y.mean())
            row["split"] = chain_rows["split"].iloc[0] if "split" in chain_rows.columns else "train"
            rows.append(row)
    return pd.DataFrame(rows)


def build_residue_level_table(labels_dir: Path, pdb_ids: list[str], label_column: str,
                               include_flagged: bool) -> pd.DataFrame:
    """One row per residue: composition features (constant within a
    chain), position-in-chain fraction, the raw target, and split."""
    rows = []
    for pdb_id in pdb_ids:
        labels = load_structure_labels(labels_dir, pdb_id, include_flagged)
        if labels is None or label_column not in labels.columns:
            continue
        seqs = load_fasta_by_chain(labels_dir, pdb_id)
        for chain, chain_rows in labels.groupby("chain"):
            seq = seqs.get(chain)
            if not seq:
                continue
            length = len(seq)
            feats = composition_features(seq)
            valid = chain_rows.dropna(subset=[label_column])
            if valid.empty:
                continue
            for _, r in valid.iterrows():
                pos_frac = float(r["seq_index"]) / max(length - 1, 1)
                d = dict(zip(FEATURE_NAMES, feats))
                d["position_frac"] = pos_frac
                d["y"] = float(r[label_column])
                d["split"] = r["split"] if "split" in r else "train"
                rows.append(d)
    return pd.DataFrame(rows)


def kfold_ridge_r2(X: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 0) -> float:
    from sklearn.model_selection import KFold
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    n_splits = min(n_splits, len(y))
    if n_splits < 2:
        return float("nan")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in kf.split(X):
        scaler = StandardScaler().fit(X[train_idx])
        Xtr, Xte = scaler.transform(X[train_idx]), scaler.transform(X[test_idx])
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, y[train_idx])
        preds[test_idx] = model.predict(Xte)
    return float(r2_score(y, preds))


def split_ridge_r2(X: np.ndarray, y: np.ndarray, splits: np.ndarray) -> float:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    train, val, test = splits == "train", splits == "val", splits == "test"
    trainval = train | val
    if trainval.sum() == 0 or test.sum() == 0:
        return float("nan")
    scaler = StandardScaler().fit(X[trainval])
    Xtr, Xte = scaler.transform(X[trainval]), scaler.transform(X[test])
    model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, y[trainval])
    pred = model.predict(Xte)
    return float(r2_score(y[test], pred))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--label-column", required=True)
    ap.add_argument("--outdir", type=Path, default=Path("./composition_baselines"))
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--limit-structures", type=int, default=None)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    all_pot_path = args.labels_dir / "all_residue_potentials.csv"
    if not all_pot_path.exists():
        raise FileNotFoundError(f"{all_pot_path} not found -- run the label pipeline first.")
    pdb_ids = sorted(pd.read_csv(all_pot_path, low_memory=False)["pdb_id"].unique())
    if args.limit_structures:
        pdb_ids = pdb_ids[: args.limit_structures]

    # ---- A) structure-level: composition -> mu_chain ----
    log.info("=== A) composition/length -> per-chain mean potential ===")
    chain_df = build_per_chain_table(args.labels_dir, pdb_ids, args.label_column, args.include_flagged)
    chain_df.to_csv(args.outdir / "per_chain_composition_table.csv", index=False)
    X_chain = chain_df[FEATURE_NAMES].to_numpy(dtype=float)
    y_chain = chain_df["mu_chain"].to_numpy(dtype=float)
    r2_chain = kfold_ridge_r2(X_chain, y_chain)
    log.info("composition/length -> mu_chain: 5-fold CV R^2=%.3f over %d chains", r2_chain, len(chain_df))

    # ---- B) residue-level: composition (constant per chain) + position -> raw target ----
    log.info("=== B) residue-level composition and position baselines ===")
    res_df = build_residue_level_table(args.labels_dir, pdb_ids, args.label_column, args.include_flagged)
    splits = res_df["split"].to_numpy()
    y = res_df["y"].to_numpy(dtype=float)

    X_comp = res_df[FEATURE_NAMES].to_numpy(dtype=float)
    r2_comp = split_ridge_r2(X_comp, y, splits)
    log.info("composition (broadcast per chain) -> raw target: test R^2=%.3f", r2_comp)

    X_pos = res_df[["position_frac"]].to_numpy(dtype=float)
    r2_pos = split_ridge_r2(X_pos, y, splits)
    log.info("position-in-chain only -> raw target: test R^2=%.3f", r2_pos)

    X_comp_pos = res_df[FEATURE_NAMES + ["position_frac"]].to_numpy(dtype=float)
    r2_comp_pos = split_ridge_r2(X_comp_pos, y, splits)
    log.info("composition + position -> raw target: test R^2=%.3f", r2_comp_pos)

    summary = pd.DataFrame([
        {"experiment": "composition/length -> mu_chain (5-fold CV)", "r2": r2_chain, "n": len(chain_df)},
        {"experiment": "composition (per-chain, broadcast) -> raw target", "r2": r2_comp, "n": len(res_df)},
        {"experiment": "position-in-chain only -> raw target", "r2": r2_pos, "n": len(res_df)},
        {"experiment": "composition + position -> raw target", "r2": r2_comp_pos, "n": len(res_df)},
    ])
    summary.to_csv(args.outdir / "composition_position_summary.csv", index=False)
    log.info("Summary:\n%s", summary.to_string(index=False))
    log.info("Wrote outputs to %s", args.outdir)


if __name__ == "__main__":
    main()
