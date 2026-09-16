"""
Priority follow-up experiments for the RNA LM electrostatic probing pipeline.

Builds on:
  - rna_electrostatics_pipeline_v2.py  (label generation; provides assign_split)
  - rna_embedding_probe_v2.py          (LM embedding extraction; provides
                                          build_dataset_for_model, run_ridge_probe,
                                          run_mlp_probe, load_structure_labels,
                                          load_fasta_by_chain)

Rationale for what's implemented here (see accompanying discussion):

  Priority 1 -- STRUCTURE-LEVEL SCALE PROBE (new, not in the original review).
    Directly tests the mechanism behind the raw-vs-z-scored collapse: does the
    LM mostly encode "which structure is this / what's its overall electrostatic
    character", rather than local per-residue information? Mean-pool embeddings
    per structure, regress against mu_RNA (or std_RNA) of the raw potential.

  Priority 2 -- k-MER SEQUENCE BASELINES (doc-3 Set II).
    One-hot local sequence context of width k, same Ridge probe. Answers
    "how much of the LM's advantage over 1-mer is just local sequence stats".

  Priority 3 -- SEQUENCE-CLUSTER SPLIT (doc-3 Set III-A).
    Fast k-mer-Jaccard greedy clustering (O(n^2) pairwise comparisons -- at
    ~2k structures that's ~2M comparisons, which is fine but not instant;
    switch to MMseqs2/CD-HIT if you scale much past this) used to keep
    near-duplicate/close-homolog structures on the same side of the
    train/val/test split, then re-run the SAME probe.

  Group-array note: this script tracks BOTH a structure-level group
  ("pdb_id") and a chain-level group ("pdb_id::chain"). At ~2k structures
  you almost certainly have multi-chain entries, so "within one RNA" for
  centering/ranking (Priority 1) and within-structure shuffling (Priority
  5) uses the chain-level group; the train/val/test split, clustering
  (Priority 3), and the bootstrap CI (Priority 6) stay at the structure
  level, matching assign_split()'s existing convention in the label
  pipeline.

  Priority 4 -- FORCE MLP ON THE BEST RIDGE LAYER (doc-3 Set V-A).
    Your current pipeline only tries an MLP when Ridge's best val R^2 < 0.3.
    That's fine as a compute-saving heuristic but means you don't actually
    know whether nonlinear readout adds anything when Ridge already succeeds
    (which is exactly your phosphorus case). This runs it regardless.

  Priority 5 -- NEGATIVE CONTROLS (doc-3 Set VI-A/B).
    Within-structure and global label shuffling. Should both give R^2 ~ 0;
    if they don't, something is leaking (e.g. through the split itself).

  Priority 6 -- STRUCTURE-LEVEL BOOTSTRAP CI (doc-3 Set I-C).
    Resample structures (not residues) with replacement to get a CI on the
    final test R^2 -- residues within one RNA are not independent samples.

Deferred (not implemented here, see accompanying discussion for why):
  multi-seed repeats (trivial loop, add once the above settles), MLP on every
  layer (expensive, priority-4 already answers the key question), true
  RNA-family holdout (needs external family annotations you don't have yet),
  composition/position-only baselines (cheap add-ons, stubbed at the bottom
  if you want them).

Usage:
    python rna_probe_priority_experiments.py \
        --labels-dir ./rna_labels --model splicebert \
        --label-column potential_at_phosphorus_kT_e \
        --outdir ./priority_results
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("rna_probe_priority")

# These two imports assume both scripts sit on PYTHONPATH / in the same dir.
from rna_electrostatics_pipeline_v2 import assign_split  # noqa: E402
from rna_embedding_probe_v2 import (  # noqa: E402
    build_dataset_for_model, run_ridge_probe, run_mlp_probe, load_model,
    extract_layer_embeddings, load_structure_labels, load_fasta_by_chain,
    select_best_layer_and_test,
)


# ----------------------------------------------------------------------
# Chain-aware dataset builder
# ----------------------------------------------------------------------
#
# build_dataset_for_model() from the base pipeline only tracks pdb_id per
# residue, not chain -- fine when structures are single-chain-dominated
# (tRNAs), but at ~2k structures you almost certainly have multi-chain
# complexes (ribosomal fragments, multi-strand assemblies, etc.), and two
# chains of the same PDB entry are NOT the same "one RNA" for the
# within-structure centering/ranking/shuffling experiments below. This
# wrapper duplicates the base builder's loop but also emits a combined
# "pdb_id::chain" group array (`chain_groups`) alongside the original
# pdb_id-only array (`pdb_groups`, kept for anything that should stay at
# the structure level -- the train/val/test split itself, clustering, and
# the bootstrap CI, all of which already operate per assign_split()'s
# pdb_id-level convention).

def build_dataset_for_model_with_chains(labels_dir: Path, pdb_ids: list[str], model_key: str,
                                         device: str, include_flagged: bool, max_len: int,
                                         label_columns: list[str]):
    """Same as build_dataset_for_model, but additionally returns
    chain_groups (pdb_id::chain per residue) alongside pdb_groups
    (pdb_id per residue, identical to the base builder's `groups`)."""
    tokenizer, model, n_layers, model_max_len = load_model(model_key, device)
    effective_max_len = max(1, min(max_len, model_max_len - 2))
    log.info("%s: model_max_len=%d, effective_max_len=%d", model_key, model_max_len, effective_max_len)

    per_layer_X = {i: [] for i in range(n_layers)}
    y_by_column_lists = {col: [] for col in label_columns}
    pdb_groups_all, chain_groups_all, splits_all, base_all = [], [], [], []

    for pdb_id in pdb_ids:
        labels = load_structure_labels(labels_dir, pdb_id, include_flagged)
        if labels is None:
            continue
        seqs = load_fasta_by_chain(labels_dir, pdb_id)
        for chain, chain_rows in labels.groupby("chain"):
            seq = seqs.get(chain)
            if not seq:
                continue
            emb, _ = extract_layer_embeddings(tokenizer, model, seq, device, max_len=effective_max_len)
            if emb is None:
                continue
            valid = chain_rows[chain_rows["seq_index"] < emb.shape[1]]
            if valid.empty:
                continue
            idxs = valid["seq_index"].to_numpy()
            for layer_i in range(n_layers):
                per_layer_X[layer_i].append(emb[layer_i, idxs, :])
            for col in label_columns:
                y_by_column_lists[col].append(
                    valid[col].to_numpy(dtype=float) if col in valid.columns
                    else np.full(len(valid), np.nan)
                )
            pdb_groups_all.append(np.full(len(valid), pdb_id))
            chain_groups_all.append(np.full(len(valid), f"{pdb_id}::{chain}"))
            splits_all.append(valid["split"].to_numpy() if "split" in valid.columns
                               else np.full(len(valid), "train"))
            base_all.append(valid["base"].to_numpy())

    if not pdb_groups_all:
        raise RuntimeError(f"No usable (embedding, label) pairs assembled for model {model_key}")

    X = {i: np.concatenate(v, axis=0) for i, v in per_layer_X.items()}
    pdb_groups = np.concatenate(pdb_groups_all)
    chain_groups = np.concatenate(chain_groups_all)
    splits = np.concatenate(splits_all)
    base_ids = np.concatenate(base_all)
    y_by_column = {col: np.concatenate(v) for col, v in y_by_column_lists.items()}

    log.info("%s: assembled %d residues, %d structures, %d distinct chains",
              model_key, len(pdb_groups), len(set(pdb_groups)), len(set(chain_groups)))
    return X, y_by_column, pdb_groups, chain_groups, splits, base_ids, effective_max_len


# ----------------------------------------------------------------------
# Priority 1: structure-level scale probe
# ----------------------------------------------------------------------

def compute_structure_scalars(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """Given raw per-residue targets and a group array, return
    (centered, rank_pct, mu_by_group). Pass CHAIN-level groups
    ("pdb_id::chain", from build_dataset_for_model_with_chains) here --
    centering/ranking "within one RNA" should be per chain, not per
    structure, whenever a structure has more than one RNA chain."""
    df = pd.DataFrame({"y": y, "group": groups})
    mu = df.groupby("group")["y"].transform("mean")
    centered = (df["y"] - mu).to_numpy()
    rank_pct = df.groupby("group")["y"].rank(pct=True).to_numpy()
    mu_by_group = df.groupby("group")["y"].mean().to_dict()
    return centered, rank_pct, mu_by_group


def structure_level_scale_probe(X_layer: np.ndarray, groups: np.ndarray,
                                 mu_by_group: dict, n_splits: int = 5) -> dict:
    """Mean-pool embeddings per structure, regress the pooled vector
    against that structure's mean raw potential (mu_RNA). If this R^2 is
    close to the residue-level raw R^2, the LM is mostly encoding a
    structure-level scalar rather than local per-residue variation --
    this is the direct mechanistic check behind the raw-vs-z collapse."""
    from sklearn.model_selection import KFold
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score

    unique_pdbs = np.array(sorted(set(groups)))
    pooled = np.stack([X_layer[groups == p].mean(axis=0) for p in unique_pdbs])
    y = np.array([mu_by_group[p] for p in unique_pdbs])

    n_splits = min(n_splits, len(unique_pdbs))
    if n_splits < 2:
        return {"r2": float("nan"), "n_structures": len(unique_pdbs),
                "note": "too few structures for CV"}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=0)
    preds = np.full(len(y), np.nan)
    for train_idx, test_idx in kf.split(pooled):
        scaler = StandardScaler().fit(pooled[train_idx])
        Xtr, Xte = scaler.transform(pooled[train_idx]), scaler.transform(pooled[test_idx])
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, y[train_idx])
        preds[test_idx] = model.predict(Xte)

    r2 = r2_score(y, preds)
    return {"r2": float(r2), "n_structures": len(unique_pdbs)}


# ----------------------------------------------------------------------
# Priority 2: k-mer sequence baselines
# ----------------------------------------------------------------------

_ALPHABET = "ACGUN"  # N = padding/unknown


def kmer_context_features(seq: str, k: int) -> np.ndarray:
    """One-hot flattened local sequence window of width k centered on
    each position (k should be odd). Edge positions are padded with N."""
    assert k % 2 == 1, "k should be odd so the window is centered"
    half = k // 2
    padded = "N" * half + seq.upper() + "N" * half
    code = {c: i for i, c in enumerate(_ALPHABET)}
    n_sym = len(_ALPHABET)
    feats = np.zeros((len(seq), k * n_sym), dtype=float)
    for i in range(len(seq)):
        window = padded[i:i + k]
        for j, c in enumerate(window):
            feats[i, j * n_sym + code.get(c, code["N"])] = 1.0
    return feats


def build_kmer_dataset(labels_dir: Path, pdb_ids: list[str], k: int,
                        include_flagged: bool, label_column: str):
    """Sequence-only analogue of build_dataset_for_model: same label
    loading, but features are local one-hot k-mer context instead of LM
    embeddings. Returns X (n_residues, k*5), y, groups, splits."""
    X_list, y_list, groups_list, splits_list = [], [], [], []
    for pdb_id in pdb_ids:
        labels = load_structure_labels(labels_dir, pdb_id, include_flagged)
        if labels is None or label_column not in labels.columns:
            continue
        seqs = load_fasta_by_chain(labels_dir, pdb_id)
        for chain, chain_rows in labels.groupby("chain"):
            seq = seqs.get(chain)
            if not seq:
                continue
            feats = kmer_context_features(seq, k)
            valid = chain_rows[chain_rows["seq_index"] < feats.shape[0]]
            if valid.empty:
                continue
            idxs = valid["seq_index"].to_numpy()
            X_list.append(feats[idxs])
            y_list.append(valid[label_column].to_numpy(dtype=float))
            groups_list.append(np.full(len(valid), pdb_id))
            splits_list.append(valid["split"].to_numpy() if "split" in valid.columns
                                else np.full(len(valid), "train"))
    if not X_list:
        raise RuntimeError(f"No usable rows assembled for k-mer baseline (k={k})")
    return (np.concatenate(X_list), np.concatenate(y_list),
            np.concatenate(groups_list), np.concatenate(splits_list))


def run_kmer_baseline_sweep(labels_dir: Path, pdb_ids: list[str], label_column: str,
                             include_flagged: bool, ks: tuple[int, ...] = (1, 3, 5, 7)) -> pd.DataFrame:
    rows = []
    for k in ks:
        X, y, groups, splits = build_kmer_dataset(labels_dir, pdb_ids, k, include_flagged, label_column)
        valid = ~np.isnan(y)
        X, y, splits = X[valid], y[valid], splits[valid]
        train, val, test = splits == "train", splits == "val", splits == "test"
        if train.sum() == 0 or val.sum() == 0 or test.sum() == 0:
            log.warning("k=%d: missing a split, skipping", k)
            continue
        val_result = run_ridge_probe(X[train], y[train], X[val], y[val])
        test_result = run_ridge_probe(X[train | val], y[train | val], X[test], y[test])
        rows.append({"k": k, "val_r2": val_result["r2"], "test_r2": test_result["r2"],
                      "n_residues": int(valid.sum())})
        log.info("k-mer baseline k=%d: val_R2=%.3f test_R2=%.3f", k, val_result["r2"], test_result["r2"])
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Priority 3: sequence-cluster split (fast k-mer-Jaccard approximation)
# ----------------------------------------------------------------------

def _kmer_set(seq: str, k: int = 6) -> set:
    return {seq[i:i + k] for i in range(max(0, len(seq) - k + 1))}


def cluster_structures_by_similarity(pdb_seqs: dict[str, str], k: int = 6,
                                      identity_threshold: float = 0.5) -> dict[str, list[str]]:
    """Greedy single-linkage clustering by k-mer Jaccard similarity, used
    as a fast proxy for sequence identity. NOT an exact alignment identity
    -- adequate for flagging near-duplicate/close-homolog structures
    before assigning cluster-level splits, but for a few hundred to low
    thousands of structures only (O(n^2) pairwise comparisons). For
    larger datasets, swap in MMseqs2/CD-HIT clustering and just feed the
    resulting cluster assignments into cluster_split() below.
    pdb_seqs: pdb_id -> concatenated sequence across all its chains.
    Returns cluster_root -> list of pdb_ids.
    """
    ids = list(pdb_seqs)
    kmer_sets = {i: _kmer_set(pdb_seqs[i], k) for i in ids}
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for i in range(len(ids)):
        si = kmer_sets[ids[i]]
        if not si:
            continue
        for j in range(i + 1, len(ids)):
            sj = kmer_sets[ids[j]]
            union_size = len(si | sj)
            if union_size == 0:
                continue
            jaccard = len(si & sj) / union_size
            if jaccard >= identity_threshold:
                union(ids[i], ids[j])

    clusters: dict[str, list[str]] = {}
    for i in ids:
        clusters.setdefault(find(i), []).append(i)
    return clusters


def cluster_split(clusters: dict[str, list[str]]) -> dict[str, str]:
    """Deterministic 80/10/10 split at the CLUSTER level (all members of
    a cluster go to the same split), same hashing convention as
    assign_split() in the label pipeline."""
    pdb_to_split = {}
    for members in clusters.values():
        cluster_key = ",".join(sorted(members))
        h = int(hashlib.md5(cluster_key.encode()).hexdigest(), 16)
        bucket = h % 100
        split = "train" if bucket < 80 else ("val" if bucket < 90 else "test")
        for m in members:
            pdb_to_split[m] = split
    return pdb_to_split


def rerun_probe_with_cluster_split(X: dict, y: np.ndarray, groups: np.ndarray,
                                    pdb_seqs: dict[str, str], model_key: str,
                                    label_column: str, identity_threshold: float = 0.5) -> dict:
    """Re-derives splits by sequence-cluster instead of by individual PDB
    ID, then re-runs the same best-layer-selection + held-out-test
    protocol as the main pipeline. Compare this test_r2 against the
    PDB-level-split test_r2 you already have."""
    clusters = cluster_structures_by_similarity(pdb_seqs, identity_threshold=identity_threshold)
    n_singleton = sum(1 for m in clusters.values() if len(m) == 1)
    log.info("cluster split: %d clusters from %d structures (%d singletons)",
              len(clusters), len(pdb_seqs), n_singleton)
    pdb_to_split = cluster_split(clusters)
    new_splits = np.array([pdb_to_split.get(p, "train") for p in groups])

    valid = ~np.isnan(y)
    X_f = {i: arr[valid] for i, arr in X.items()}
    y_f, splits_f = y[valid], new_splits[valid]
    result = select_best_layer_and_test(X_f, y_f, splits_f, model_key, label_column + "_clustersplit")
    result["n_clusters"] = len(clusters)
    result["n_singleton_clusters"] = n_singleton
    return result


# ----------------------------------------------------------------------
# Priority 4: force MLP on the best Ridge layer
# ----------------------------------------------------------------------

def force_mlp_on_best_layer(X: dict, y: np.ndarray, splits: np.ndarray,
                             best_layer: int) -> dict:
    """Runs the MLP probe on whichever layer Ridge selected as best,
    regardless of whether Ridge's val R^2 cleared the success threshold.
    Directly answers: does nonlinear readout add anything once linear
    already works well (your phosphorus case, where the MLP branch never
    fires under the original threshold logic)."""
    train, val = splits == "train", splits == "val"
    X_layer = X[best_layer]
    ridge = run_ridge_probe(X_layer[train], y[train], X_layer[val], y[val])
    mlp = run_mlp_probe(X_layer[train], y[train], X_layer[val], y[val])
    return {"layer": best_layer, "ridge_val_r2": ridge["r2"], "mlp_val_r2": mlp["r2"],
            "delta": mlp["r2"] - ridge["r2"]}


# ----------------------------------------------------------------------
# Priority 5: negative controls
# ----------------------------------------------------------------------

def shuffle_within_structure(y: np.ndarray, groups: np.ndarray, seed: int = 0) -> np.ndarray:
    """Pass CHAIN-level groups here so "within structure" really means
    within one RNA chain, not across all chains of a multi-chain entry."""
    rng = np.random.default_rng(seed)
    y2 = y.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        y2[idx] = y[rng.permutation(idx)]
    return y2


def shuffle_globally(y: np.ndarray, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.permutation(y)


def run_negative_controls(X: dict, y: np.ndarray, chain_groups: np.ndarray, splits: np.ndarray,
                           model_key: str, label_column: str) -> pd.DataFrame:
    """chain_groups should be the CHAIN-level ("pdb_id::chain") array --
    within-structure shuffling should stay within one RNA molecule."""
    valid = ~np.isnan(y)
    X_f = {i: arr[valid] for i, arr in X.items()}
    y_f, groups_f, splits_f = y[valid], chain_groups[valid], splits[valid]

    rows = []
    real = select_best_layer_and_test(X_f, y_f, splits_f, model_key, label_column)
    rows.append({"control": "real", **real})

    y_within = shuffle_within_structure(y_f, groups_f)
    within = select_best_layer_and_test(X_f, y_within, splits_f, model_key, label_column + "_shuf_within")
    rows.append({"control": "shuffled_within_structure", **within})

    y_global = shuffle_globally(y_f)
    glob = select_best_layer_and_test(X_f, y_global, splits_f, model_key, label_column + "_shuf_global")
    rows.append({"control": "shuffled_globally", **glob})

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Priority 6: structure-level bootstrap CI
# ----------------------------------------------------------------------

def select_best_layer_and_test_with_preds(X: dict, y: np.ndarray, splits: np.ndarray) -> dict:
    """Same protocol as select_best_layer_and_test but also returns the
    held-out test predictions + groups needed for structure-level
    bootstrapping."""
    train, val, test = splits == "train", splits == "val", splits == "test"
    best_layer, best_val_r2 = None, -np.inf
    for layer, X_layer in X.items():
        result = run_ridge_probe(X_layer[train], y[train], X_layer[val], y[val])
        if result["r2"] > best_val_r2:
            best_val_r2, best_layer = result["r2"], layer

    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    trainval = train | val
    X_best = X[best_layer]
    scaler = StandardScaler().fit(X_best[trainval])
    Xtr, Xte = scaler.transform(X_best[trainval]), scaler.transform(X_best[test])
    model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xtr, y[trainval])
    pred = model.predict(Xte)

    return {"best_layer": int(best_layer), "val_r2": float(best_val_r2),
            "test_pred": pred, "test_true": y[test]}


def bootstrap_r2_by_structure(y_true: np.ndarray, y_pred: np.ndarray, groups_test: np.ndarray,
                               n_boot: int = 2000, seed: int = 0) -> dict:
    """Resamples whole structures (not individual residues) with
    replacement -- residues within one RNA aren't independent, so a
    naive residue-level bootstrap understates the true uncertainty."""
    from sklearn.metrics import r2_score
    rng = np.random.default_rng(seed)
    structs = np.unique(groups_test)
    scores = []
    for _ in range(n_boot):
        sampled = rng.choice(structs, size=len(structs), replace=True)
        idx = np.concatenate([np.where(groups_test == s)[0] for s in sampled])
        if len(np.unique(y_true[idx])) < 2:
            continue
        scores.append(r2_score(y_true[idx], y_pred[idx]))
    scores = np.array(scores)
    return {"mean": float(scores.mean()), "ci_lo": float(np.percentile(scores, 2.5)),
            "ci_hi": float(np.percentile(scores, 97.5)), "n_boot_used": len(scores)}


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--model", required=True, help="Single model key, e.g. splicebert")
    ap.add_argument("--label-column", required=True,
                     help="e.g. potential_at_phosphorus_kT_e")
    ap.add_argument("--outdir", type=Path, default=Path("./priority_results"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--kmer-ks", nargs="+", type=int, default=[1, 3, 5, 7])
    ap.add_argument("--cluster-identity-threshold", type=float, default=0.5)
    ap.add_argument("--limit-structures", type=int, default=None)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    import torch
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    all_pot_path = args.labels_dir / "all_residue_potentials.csv"
    if not all_pot_path.exists():
        raise FileNotFoundError(f"{all_pot_path} not found -- run the label pipeline first.")
    pdb_ids = sorted(pd.read_csv(all_pot_path)["pdb_id"].unique())
    if args.limit_structures:
        pdb_ids = pdb_ids[: args.limit_structures]

    log.info("=== Building embeddings for %s (%d structures) ===", args.model, len(pdb_ids))
    X, y_by_column, pdb_groups, chain_groups, splits, base_ids, eff_max_len = build_dataset_for_model_with_chains(
        args.labels_dir, pdb_ids, args.model, device, args.include_flagged,
        args.max_len, [args.label_column],
    )
    y = y_by_column[args.label_column]
    valid = ~np.isnan(y)
    groups = pdb_groups  # structure-level: used for split/clustering/bootstrap below

    # ---- Priority 1: structure-level scale probe ----
    # NOTE: "structure-level" here means per CHAIN (one RNA molecule),
    # since that's the correct unit for "within one RNA" comparisons.
    log.info("=== Priority 1: structure-level (per-chain) scale probe ===")
    centered, rank_pct, mu_by_group = compute_structure_scalars(y[valid], chain_groups[valid])
    # use the best layer from a quick residue-level pass as the layer to pool
    quick = select_best_layer_and_test(
        {i: arr[valid] for i, arr in X.items()}, y[valid], splits[valid], args.model, args.label_column,
    )
    scale_result = structure_level_scale_probe(X[quick["best_layer"]][valid], chain_groups[valid], mu_by_group)
    pd.DataFrame([scale_result]).to_csv(args.outdir / "p1_structure_scale_probe.csv", index=False)
    log.info("Per-chain scale probe (layer %d): R^2=%.3f over %d chains",
              quick["best_layer"], scale_result["r2"], scale_result["n_structures"])

    # ---- Priority 1b: direct centered / within-chain-rank probes ----
    # This is the quantitative counterpart to the Priority-5 within-chain
    # shuffle control: instead of asking "does shuffling within a chain
    # hurt performance" (it shouldn't, if only chain-level info is used),
    # directly probe the residue-level embeddings against the
    # within-chain-centered value (phi_i - mu_chain) and the within-chain
    # percentile rank. If the raw target's R^2 is mostly chain-level scale,
    # both of these should come back near zero -- reusing the SAME
    # already-extracted embeddings, no new extraction needed.
    log.info("=== Priority 1b: within-chain centered / rank probes ===")
    X_valid = {i: arr[valid] for i, arr in X.items()}
    splits_valid = splits[valid]
    centered_result = select_best_layer_and_test(
        X_valid, centered, splits_valid, args.model, args.label_column + "_centered",
    )
    rank_result = select_best_layer_and_test(
        X_valid, rank_pct, splits_valid, args.model, args.label_column + "_rank_pct",
    )
    pd.DataFrame([
        {"target": "raw", **quick},
        {"target": "within_chain_centered", **centered_result},
        {"target": "within_chain_rank_pct", **rank_result},
    ]).to_csv(args.outdir / "p1b_centered_rank_probe.csv", index=False)
    log.info("Raw target test_R^2=%.3f | within-chain centered test_R^2=%.3f | "
              "within-chain rank test_R^2=%.3f",
              quick["test_r2"], centered_result["test_r2"], rank_result["test_r2"])

    # ---- Priority 2: k-mer baselines ----
    log.info("=== Priority 2: k-mer baselines ===")
    kmer_df = run_kmer_baseline_sweep(args.labels_dir, pdb_ids, args.label_column,
                                       args.include_flagged, ks=tuple(args.kmer_ks))
    kmer_df.to_csv(args.outdir / "p2_kmer_baselines.csv", index=False)

    # ---- Priority 3: sequence-cluster split ----
    log.info("=== Priority 3: sequence-cluster split ===")
    pdb_seqs = {}
    for pdb_id in pdb_ids:
        chains = load_fasta_by_chain(args.labels_dir, pdb_id)
        if chains:
            pdb_seqs[pdb_id] = "".join(chains.values())
    cluster_result = rerun_probe_with_cluster_split(
        X, y, pdb_groups, pdb_seqs, args.model, args.label_column,
        identity_threshold=args.cluster_identity_threshold,
    )
    pd.DataFrame([cluster_result]).to_csv(args.outdir / "p3_cluster_split.csv", index=False)
    log.info("Cluster-split test R^2=%.3f (vs pdb-level-split %.3f) -- %d clusters, %d singletons",
              cluster_result["test_r2"], quick["test_r2"],
              cluster_result["n_clusters"], cluster_result["n_singleton_clusters"])

    # ---- Priority 4: force MLP on best layer ----
    log.info("=== Priority 4: MLP on best Ridge layer ===")
    mlp_result = force_mlp_on_best_layer(
        {i: arr[valid] for i, arr in X.items()}, y[valid], splits[valid], quick["best_layer"],
    )
    pd.DataFrame([mlp_result]).to_csv(args.outdir / "p4_mlp_on_best_layer.csv", index=False)
    log.info("Layer %d: ridge val R^2=%.3f, mlp val R^2=%.3f (delta=%.3f)",
              mlp_result["layer"], mlp_result["ridge_val_r2"], mlp_result["mlp_val_r2"], mlp_result["delta"])

    # ---- Priority 5: negative controls ----
    log.info("=== Priority 5: negative controls ===")
    neg_df = run_negative_controls(X, y, chain_groups, splits, args.model, args.label_column)
    neg_df.to_csv(args.outdir / "p5_negative_controls.csv", index=False)

    # ---- Priority 6: structure-level bootstrap CI ----
    # Bootstrapping resamples whole PDB entries (pdb_groups), not chains --
    # chains of the same crystal/cryo-EM entry still share correlated
    # experimental conditions, so the conservative independence unit for
    # the CI is the structure, even though centering/shuffling above used
    # the chain as the "one RNA" unit.
    log.info("=== Priority 6: structure-level bootstrap CI ===")
    with_preds = select_best_layer_and_test_with_preds(
        {i: arr[valid] for i, arr in X.items()}, y[valid], splits[valid],
    )
    test_mask = splits[valid] == "test"
    boot = bootstrap_r2_by_structure(with_preds["test_true"], with_preds["test_pred"],
                                      pdb_groups[valid][test_mask])
    pd.DataFrame([boot]).to_csv(args.outdir / "p6_bootstrap_ci.csv", index=False)
    log.info("Bootstrap test R^2 = %.3f [95%% CI %.3f, %.3f] (n_boot=%d)",
              boot["mean"], boot["ci_lo"], boot["ci_hi"], boot["n_boot_used"])

    log.info("All priority experiment outputs written to %s", args.outdir)


if __name__ == "__main__":
    main()