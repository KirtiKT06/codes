"""
Secondary-structure / base-pairing physics baseline for the RNA
electrostatic label set. Uses ViennaRNA's classical partition-function
folding (McCaskill algorithm -- the same thermodynamics RNAfold/mfold use,
Turner nearest-neighbour parameters, no ML, no GPU, no MSA). Complements
composition_position_baselines.py.

WHY THIS IS THE HIGH-VALUE NEXT TEST, not just another baseline:

Composition (length/GC%) and all four tested LMs explain a lot of RAW
per-residue variance but essentially NONE of the WITHIN-CHAIN
CENTERED/RANK variance (Priority 1b: R^2 ~ 0.00 across rnabert, rnafm,
splicebert, ernierna). Base-pairing status (paired vs. unpaired, and
pairing probability) is genuinely LOCAL, position-specific information
that is cheaply computable from sequence alone via classical folding, but
which none of the composition or LM features explicitly encode.

  - If base-pairing features recover ANY of the within-chain signal that
    composition and the LMs both missed, that's a direct, falsifiable
    result: local electrostatic variation is tied to base-pairing
    context, and current sequence LMs are not capturing it even though
    it's derivable from the same sequence via classical thermodynamics.
    That's concrete motivation for a physics-informed architecture (e.g.
    one that bakes base-pairing/stacking energies into attention) to
    succeed where pure sequence LMs failed -- rather than a speculative
    claim.
  - If it does NOT recover the signal either, that's also a real,
    reportable result, and it tempers how strongly a base-pairing-aware
    architecture specifically should be expected to help (the missing
    local information might be more three-dimensional / tertiary-contact
    driven than secondary-structure driven).

Requires: pip install ViennaRNA

Usage:
    python secondary_structure_baseline.py \
        --labels-dir /data/rna_mg/cutoff_4_8_12 \
        --label-column potential_at_phosphorus_kT_e \
        --outdir /data/rna_mg/priority_results/secondary_structure
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("secondary_structure_baseline")

from rna_embedding_probe_v2 import load_structure_labels, load_fasta_by_chain  # noqa: E402
from composition_position_baselines import composition_features, FEATURE_NAMES  # noqa: E402

STRUCT_FEATURE_NAMES = ["is_paired_mfe", "prob_paired"]


def pairing_features(seq: str) -> np.ndarray:
    """Per-residue (is_paired_mfe, prob_paired) from ViennaRNA's MFE
    structure and partition-function base-pair probabilities. Returns an
    (len(seq), 2) array. Sequences ViennaRNA can't fold (empty, or a
    folding error) return NaN rows -- callers should drop those."""
    import RNA

    seq = seq.upper().replace("T", "U")
    n = len(seq)
    if n == 0:
        return np.zeros((0, 2), dtype=float)

    try:
        fc = RNA.fold_compound(seq)
        mfe_struct, mfe = fc.mfe()
        fc.exp_params_rescale(mfe)
        _, _ = fc.pf()
        bpp = fc.bpp()  # (n+1) x (n+1) upper-triangular pairing probabilities, 1-indexed
    except Exception:
        log.warning("ViennaRNA folding failed for a sequence of length %d, returning NaNs", n)
        return np.full((n, 2), np.nan)

    is_paired = np.array([1.0 if c in "()" else 0.0 for c in mfe_struct], dtype=float)
    prob_paired = np.zeros(n, dtype=float)
    for i in range(1, n + 1):
        row_sum = sum(bpp[i][j] for j in range(i + 1, n + 1))
        col_sum = sum(bpp[j][i] for j in range(1, i))
        prob_paired[i - 1] = min(row_sum + col_sum, 1.0)

    return np.stack([is_paired, prob_paired], axis=1)


def add_centered_and_rank(df: pd.DataFrame, value_col: str, group_col: str = "chain_id") -> None:
    """In-place: within-chain-centered and within-chain-percentile-rank
    twins of value_col, same convention as Priority 1b in
    rna_probe_priority_experiments.py."""
    mu = df.groupby(group_col)[value_col].transform("mean")
    df[f"{value_col}_centered"] = df[value_col] - mu
    df[f"{value_col}_rank_pct"] = df.groupby(group_col)[value_col].rank(pct=True)


def build_residue_level_table(labels_dir: Path, pdb_ids: list[str], label_column: str,
                               include_flagged: bool) -> pd.DataFrame:
    """One row per residue: composition features (constant per chain),
    ViennaRNA base-pairing features (position-specific), the raw target,
    its within-chain-centered/rank twins, and split."""
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
            comp = composition_features(seq)
            pair_feats = pairing_features(seq)
            valid = chain_rows.dropna(subset=[label_column])
            if valid.empty:
                continue
            for _, r in valid.iterrows():
                idx = int(r["seq_index"])
                if idx >= pair_feats.shape[0] or np.isnan(pair_feats[idx]).any():
                    continue
                d = dict(zip(FEATURE_NAMES, comp))
                d["is_paired_mfe"] = pair_feats[idx, 0]
                d["prob_paired"] = pair_feats[idx, 1]
                d["y"] = float(r[label_column])
                d["chain_id"] = f"{pdb_id}::{chain}"
                d["split"] = r["split"] if "split" in r else "train"
                rows.append(d)
    df = pd.DataFrame(rows)
    if not df.empty:
        add_centered_and_rank(df, "y")
    return df


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
    return float(r2_score(y[test], model.predict(Xte)))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--label-column", required=True)
    ap.add_argument("--outdir", type=Path, default=Path("./secondary_structure_baseline"))
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

    log.info("=== Folding %d structures with ViennaRNA and assembling features ===", len(pdb_ids))
    df = build_residue_level_table(args.labels_dir, pdb_ids, args.label_column, args.include_flagged)
    if df.empty:
        raise RuntimeError("No usable rows assembled -- check ViennaRNA install and label column.")
    df.to_csv(args.outdir / "residue_level_structure_features.csv", index=False)
    log.info("Assembled %d residues across %d chains", len(df), df["chain_id"].nunique())

    splits = df["split"].to_numpy()
    X_struct = df[STRUCT_FEATURE_NAMES].to_numpy(dtype=float)
    X_comp = df[FEATURE_NAMES].to_numpy(dtype=float)
    X_both = df[FEATURE_NAMES + STRUCT_FEATURE_NAMES].to_numpy(dtype=float)

    targets = {
        "raw": df["y"].to_numpy(dtype=float),
        "within_chain_centered": df["y_centered"].to_numpy(dtype=float),
        "within_chain_rank_pct": df["y_rank_pct"].to_numpy(dtype=float),
    }
    feature_sets = {
        "composition_only": X_comp,
        "structure_only": X_struct,
        "composition_plus_structure": X_both,
    }

    rows = []
    log.info("=== Regressing each target on each feature set ===")
    for target_name, y in targets.items():
        for feat_name, X in feature_sets.items():
            r2 = split_ridge_r2(X, y, splits)
            rows.append({"target": target_name, "features": feat_name, "test_r2": r2})
            log.info("target=%-24s features=%-26s test_R^2=%.4f", target_name, feat_name, r2)

    summary = pd.DataFrame(rows)
    summary.to_csv(args.outdir / "secondary_structure_summary.csv", index=False)
    log.info("Summary:\n%s", summary.pivot(index="target", columns="features", values="test_r2").to_string())
    log.info("Wrote outputs to %s", args.outdir)


if __name__ == "__main__":
    main()
