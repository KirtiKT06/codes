"""
Final Phase I/II combination test: base identity + real 3D structural
features TOGETHER, regressed against the within-chain-centered target,
compared to the LM's own within-chain-centered R^2.

This directly operationalizes the advisor's Delta R^2 test:

    Delta R^2 = R^2(LM) - R^2(identity + structure)

structural_3d_baseline.py showed generic 3D geometry alone actually
underperforms trivial base identity (0.12-0.13 vs 0.42-0.56 on the same
kind of target) -- so geometry-alone couldn't have explained the LM's
edge over identity in the first place. The real question is whether
identity+geometry COMBINED closes the gap to the LM, or whether a
genuine residual survives even the strongest simple explanation available
so far.

Reuses extract_structural_features() from structural_3d_baseline.py and
base_identity_features() from secondary_structure_baseline_modified.py -- keep all
three scripts in the same directory.

Usage:
    python combined_identity_structure_baseline.py \
        --labels-dir /data/rna_mg/cutoff_4_8_12 \
        --label-column potential_at_c1prime_kT_e \
        --outdir /data/rna_mg/priority_results_c1prime/combined
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("combined_identity_structure_baseline")

from rna_embedding_probe_v2 import load_structure_labels, load_fasta_by_chain  # noqa: E402
from composition_position_baselines import split_ridge_r2  # noqa: E402
from structural_3d_baseline import extract_structural_features, STRUCT3D_FEATURE_NAMES  # noqa: E402
from secondary_structure_baseline_modified import base_identity_features, BASE_FEATURE_NAMES  # noqa: E402


def build_combined_table(labels_dir: Path, pdb_ids: list[str], label_column: str,
                          include_flagged: bool) -> pd.DataFrame:
    frames = []
    for pdb_id in pdb_ids:
        labels = load_structure_labels(labels_dir, pdb_id, include_flagged)
        if labels is None or label_column not in labels.columns:
            continue
        pqr_path = labels_dir / "work" / pdb_id.upper() / f"{pdb_id.lower()}.pqr"
        if not pqr_path.exists():
            continue
        struct_feats = extract_structural_features(pqr_path)
        if struct_feats.empty:
            continue
        merged = labels.merge(struct_feats, on=["chain", "resnum", "resname"], how="inner")
        merged = merged.dropna(subset=[label_column] + STRUCT3D_FEATURE_NAMES)
        if merged.empty:
            continue

        seqs = load_fasta_by_chain(labels_dir, pdb_id)
        rows_out = []
        for chain, group in merged.groupby("chain"):
            seq = seqs.get(chain)
            if not seq:
                continue
            base_feats = base_identity_features(seq)
            for _, r in group.iterrows():
                idx = int(r["seq_index"])
                if idx >= base_feats.shape[0]:
                    continue
                d = r.to_dict()
                for bi, bname in enumerate(BASE_FEATURE_NAMES):
                    d[bname] = base_feats[idx, bi]
                d["chain_id"] = f"{pdb_id}::{chain}"
                rows_out.append(d)
        if rows_out:
            frames.append(pd.DataFrame(rows_out))

    if not frames:
        raise RuntimeError("No usable rows assembled -- check .pqr files exist under labels_dir/work/<PDB_ID>/")
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
    ap.add_argument("--outdir", type=Path, default=Path("./combined_baseline"))
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--limit-structures", type=int, default=None)
    # Fill these in from your own runs so the summary can show the LM gap directly.
    ap.add_argument("--lm-centered-r2", type=float, default=None,
                     help="LM's within-chain-centered test R^2 (from Priority 1b), for reference in the summary.")
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    all_pot_path = args.labels_dir / "all_residue_potentials.csv"
    pdb_ids = sorted(pd.read_csv(all_pot_path, low_memory=False)["pdb_id"].unique())
    if args.limit_structures:
        pdb_ids = pdb_ids[: args.limit_structures]

    log.info("=== Assembling combined identity + structure table for %d structures ===", len(pdb_ids))
    df = build_combined_table(args.labels_dir, pdb_ids, args.label_column, args.include_flagged)
    add_centered_and_rank(df)
    df.to_csv(args.outdir / "residue_level_combined_table.csv", index=False)
    log.info("Assembled %d residues across %d chains", len(df), df["chain_id"].nunique())

    splits = df["split"].to_numpy() if "split" in df.columns else np.full(len(df), "train")
    X_base = df[BASE_FEATURE_NAMES].to_numpy(dtype=float)
    X_struct = df[STRUCT3D_FEATURE_NAMES].to_numpy(dtype=float)
    X_combined = df[BASE_FEATURE_NAMES + STRUCT3D_FEATURE_NAMES].to_numpy(dtype=float)

    targets = {
        "raw": df["y"].to_numpy(dtype=float),
        "within_chain_centered": df["y_centered"].to_numpy(dtype=float),
        "within_chain_rank_pct": df["y_rank_pct"].to_numpy(dtype=float),
    }
    feature_sets = {
        "base_identity_only": X_base,
        "structural_3d_only": X_struct,
        "base_identity_plus_structural_3d": X_combined,
    }

    rows = []
    log.info("=== Regressing each target on each feature set ===")
    for target_name, y in targets.items():
        for feat_name, X in feature_sets.items():
            r2 = split_ridge_r2(X, y, splits)
            rows.append({"target": target_name, "features": feat_name, "test_r2": r2})
            log.info("target=%-24s features=%-32s test_R^2=%.4f", target_name, feat_name, r2)

    summary = pd.DataFrame(rows)
    if args.lm_centered_r2 is not None:
        combined_centered = summary[(summary.target == "within_chain_centered") &
                                     (summary.features == "base_identity_plus_structural_3d")]["test_r2"].iloc[0]
        gap = args.lm_centered_r2 - combined_centered
        log.info("Delta R^2 = LM(%.3f) - identity+structure(%.3f) = %.3f",
                  args.lm_centered_r2, combined_centered, gap)
        summary = pd.concat([summary, pd.DataFrame([{
            "target": "within_chain_centered", "features": "LM_minus_combined_gap",
            "test_r2": gap,
        }])], ignore_index=True)

    summary.to_csv(args.outdir / "combined_summary.csv", index=False)
    log.info("Wrote outputs to %s", args.outdir)


if __name__ == "__main__":
    main()
