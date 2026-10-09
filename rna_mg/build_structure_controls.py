"""
Build the per-residue STRUCTURE-CONTROL table used by centered_latent_manifold.py
(--structure-controls) and cross_model_latent_agreement.py (--controls).

What it computes for every labelled residue (one row each, keyed by pdb_id, chain, seq_index):
    is_paired_mfe, prob_paired      predicted base pairing from sequence (ViennaRNA)
    n_neighbors_5A, n_neighbors_8A  heavy-atom packing density around the residue's C1'
    n_tertiary_neighbors_5A         ... counting only atoms from residues > 2 positions away
    dist_nearest_P_other            distance from C1' to the nearest other residue's phosphorus
    burial_proxy                    1 / (1 + n_neighbors_8A)  (crowding, NOT true SASA)

It re-uses the exact feature code from your structural_3d_baseline.py and
secondary_structure_baseline.py, so the numbers match what you already ran. The only new
thing is that the output carries `seq_index`, which the earlier tables did not, so the
features can be lined up with the LM embeddings residue by residue.

Why this exists: the question "do LMs know electrostatics, or only secondary structure /
burial?" needs those features as controls. Pairing is PREDICTED from sequence (imperfect);
the 3D features come from the real coordinates (stronger control).

Runtime: dominated by ViennaRNA folding and the per-residue KD-tree loop. Structures are
processed in parallel (--workers). Try --limit-structures 20 first.

Usage:
    python build_structure_controls.py \
        --labels-dir /data/rna_electrostats/phi_res \
        --out /data/rna_electrostats/structure_controls.csv.gz \
        --workers 8
"""

from __future__ import annotations

import argparse
import logging
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("build_structure_controls")

from rna_embedding_probe_v2 import load_structure_labels, load_fasta_by_chain  # noqa: E402
from structural_3d_baseline import extract_structural_features, STRUCT3D_FEATURE_NAMES  # noqa: E402
from secondary_structure_baseline import pairing_features  # noqa: E402

OUT_COLS = (["pdb_id", "chain", "seq_index", "is_paired_mfe", "prob_paired"]
            + STRUCT3D_FEATURE_NAMES)


def process_structure(job):
    """All control features for ONE structure. Returns a DataFrame (possibly empty)."""
    labels_dir, pdb_id, include_flagged, max_fold_len = job
    labels_dir = Path(labels_dir)
    try:
        labels = load_structure_labels(labels_dir, pdb_id, include_flagged)
        if labels is None or labels.empty:
            return pd.DataFrame(columns=OUT_COLS)
        pqr_path = labels_dir / "work" / pdb_id.upper() / f"{pdb_id.lower()}.pqr"
        if not pqr_path.exists():
            return pd.DataFrame(columns=OUT_COLS)

        # --- 3D geometry from the PQR coordinates (same function as structural_3d_baseline) ---
        geom = extract_structural_features(pqr_path)
        if geom.empty:
            return pd.DataFrame(columns=OUT_COLS)
        key_cols = ["chain", "resnum", "resname", "seq_index"]
        merged = labels[key_cols].drop_duplicates().merge(
            geom, on=["chain", "resnum", "resname"], how="inner")
        if merged.empty:
            return pd.DataFrame(columns=OUT_COLS)

        # --- predicted pairing from sequence, one fold per chain ---
        seqs = load_fasta_by_chain(labels_dir, pdb_id)
        paired = np.full(len(merged), np.nan)
        prob = np.full(len(merged), np.nan)
        for chain, idx in merged.groupby("chain").groups.items():
            seq = seqs.get(chain)
            if not seq or len(seq) > max_fold_len:
                continue  # left as NaN -> dropped downstream
            pf = pairing_features(seq)
            pos = np.flatnonzero(merged.index.isin(idx))
            si = merged["seq_index"].to_numpy()[pos].astype(int)
            good = si < pf.shape[0]
            paired[pos[good]] = pf[si[good], 0]
            prob[pos[good]] = pf[si[good], 1]
        merged["is_paired_mfe"], merged["prob_paired"] = paired, prob
        merged["pdb_id"] = pdb_id
        merged["seq_index"] = merged["seq_index"].astype(int)
        return merged[OUT_COLS]
    except Exception as exc:  # one bad structure must not kill a multi-hour run
        logging.getLogger("build_structure_controls").warning("%s failed: %s", pdb_id, exc)
        return pd.DataFrame(columns=OUT_COLS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="Output CSV (.csv.gz is fine)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-fold-len", type=int, default=3000,
                    help="Chains longer than this are not folded (pairing left NaN)")
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--limit-structures", type=int, default=None)
    args = ap.parse_args()

    pdb_ids = sorted(pd.read_csv(args.labels_dir / "all_residue_potentials.csv",
                                 low_memory=False)["pdb_id"].unique())
    if args.limit_structures:
        pdb_ids = pdb_ids[: args.limit_structures]
    log.info("Building controls for %d structures with %d workers", len(pdb_ids), args.workers)

    jobs = [(str(args.labels_dir), p, args.include_flagged, args.max_fold_len) for p in pdb_ids]
    frames = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for n, df in enumerate(pool.map(process_structure, jobs, chunksize=4), 1):
            if not df.empty:
                frames.append(df)
            if n % 100 == 0:
                log.info("  %d / %d structures done", n, len(jobs))
    if not frames:
        raise RuntimeError("No control rows built -- check .pqr files under labels_dir/work/<PDB_ID>/")
    out = pd.concat(frames, ignore_index=True)
    n_before = len(out)
    out = out.dropna(subset=["is_paired_mfe", "prob_paired"] + STRUCT3D_FEATURE_NAMES)
    log.info("%d residues with complete controls (%d dropped for missing values)",
             len(out), n_before - len(out))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False, compression="gzip" if str(args.out).endswith(".gz") else None)
    log.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
