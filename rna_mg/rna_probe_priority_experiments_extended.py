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
    select_best_layer_and_test, _onehot_base_features,
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
    pdb_groups_all, chain_groups_all, splits_all, base_all, resnum_all, seq_index_all = [], [], [], [], [], []

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
            resnum_all.append(valid["resnum"].to_numpy())
            seq_index_all.append(valid["seq_index"].to_numpy())

    if not pdb_groups_all:
        raise RuntimeError(f"No usable (embedding, label) pairs assembled for model {model_key}")

    X = {i: np.concatenate(v, axis=0) for i, v in per_layer_X.items()}
    pdb_groups = np.concatenate(pdb_groups_all)
    chain_groups = np.concatenate(chain_groups_all)
    splits = np.concatenate(splits_all)
    base_ids = np.concatenate(base_all)
    resnums = np.concatenate(resnum_all)
    seq_indices = np.concatenate(seq_index_all)
    y_by_column = {col: np.concatenate(v) for col, v in y_by_column_lists.items()}

    log.info("%s: assembled %d residues, %d structures, %d distinct chains",
              model_key, len(pdb_groups), len(set(pdb_groups)), len(set(chain_groups)))
    return X, y_by_column, pdb_groups, chain_groups, splits, base_ids, resnums, seq_indices, effective_max_len


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
                        include_flagged: bool, label_column: str,
                        max_len: int | None = None):
    """Sequence-only analogue of build_dataset_for_model: same label
    loading, but features are local one-hot k-mer context instead of LM
    embeddings. Returns X (n_residues, k*5), y, groups, splits.

    max_len: restrict to seq_index < max_len, matching exactly the
    truncation condition build_dataset_for_model_with_chains applies
    (seq_index < emb.shape[1], i.e. seq_index < effective_max_len).
    Without this, the k-mer baseline runs over EVERY residue in
    labels_dir regardless of which model this baseline is meant to sit
    next to -- so a model with heavy truncation (e.g. rnabert's 438-token
    limit, which keeps only ~41% of residues) gets compared against a
    baseline fit on roughly 2.4x more data than its own LM ever saw. Pass
    the model's own `effective_max_len` here for a fair, same-support
    comparison (this is exactly what Priority 1c already does for the
    base-identity baseline; Priority 2 was the odd one out)."""
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
            cap = feats.shape[0] if max_len is None else min(feats.shape[0], max_len)
            valid = chain_rows[chain_rows["seq_index"] < cap]
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
                             include_flagged: bool, ks: tuple[int, ...] = (1, 3, 5, 7),
                             max_len: int | None = None) -> pd.DataFrame:
    rows = []
    for k in ks:
        X, y, groups, splits = build_kmer_dataset(labels_dir, pdb_ids, k, include_flagged, label_column,
                                                   max_len=max_len)
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


def load_pdb_clusters_from_csv(csv_path: Path) -> dict[str, list[str]]:
    """Loads a chain_id -> cluster_rep CSV from mmseqs_cluster.py and
    collapses it to the PDB level (matching cluster_split()'s existing
    pdb_id-granularity). A structure's pdb_id is assigned to whichever
    cluster its FIRST chain belongs to; if a structure's chains disagree
    (rare -- usually only chimeric/multi-family complexes), a warning is
    logged and the first is used anyway."""
    df = pd.read_csv(csv_path)
    pdb_to_clusters: dict[str, set] = {}
    for _, row in df.iterrows():
        pdb_id = str(row["chain_id"]).split("::")[0]
        pdb_to_clusters.setdefault(pdb_id, set()).add(row["cluster_rep"])

    clusters: dict[str, list[str]] = {}
    for pdb_id, cluster_set in pdb_to_clusters.items():
        if len(cluster_set) > 1:
            log.warning("%s: chains fall in %d different sequence clusters, using the first",
                        pdb_id, len(cluster_set))
        cluster_rep = sorted(str(c) for c in cluster_set)[0]
        clusters.setdefault(cluster_rep, []).append(pdb_id)
    return clusters


def build_cluster_splits(pdb_seqs: dict[str, str], groups: np.ndarray,
                          identity_threshold: float = 0.5,
                          precomputed_clusters: Optional[dict] = None) -> tuple[np.ndarray, dict]:
    """Returns (new_splits aligned to `groups`, clusters dict). Uses
    precomputed_clusters (e.g. real MMseqs2/CD-HIT identity clusters from
    mmseqs_cluster.py) when given; otherwise falls back to the built-in
    k-mer-Jaccard approximation."""
    if precomputed_clusters is not None:
        clusters = precomputed_clusters
    else:
        clusters = cluster_structures_by_similarity(pdb_seqs, identity_threshold=identity_threshold)
    pdb_to_split = cluster_split(clusters)
    new_splits = np.array([pdb_to_split.get(p, "train") for p in groups])
    return new_splits, clusters


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

    log.info("Negative controls: real R^2=%.3f, shuffled-within R^2=%.3f, shuffled-global R^2=%.3f",
              real["test_r2"], within["test_r2"], glob["test_r2"])
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

def build_composition_lookup(labels_dir: Path, pdb_ids: list[str]) -> dict[str, np.ndarray]:
    """chain_id ('pdb_id::chain') -> composition feature vector (length,
    gc_frac, a/c/g/u fracs), for the composition-residualization probe
    below. Reuses composition_features() from composition_position_baselines.py
    -- keep that file in the same directory."""
    from composition_position_baselines import composition_features
    lookup = {}
    for pdb_id in pdb_ids:
        seqs = load_fasta_by_chain(labels_dir, pdb_id)
        for chain, seq in seqs.items():
            lookup[f"{pdb_id}::{chain}"] = composition_features(seq)
    return lookup


def composition_residualize(y: np.ndarray, chain_ids: np.ndarray, comp_lookup: dict,
                             splits: np.ndarray) -> np.ndarray:
    """Fits a Ridge composition model on TRAIN rows only, then returns the
    residual y - y_hat for every row (train/val/test). Unlike within-chain
    centering (which zeroes out anything chain-constant by construction),
    this only removes what a simple length/GC/base-fraction model can
    actually predict -- a genuine, non-tautological per-residue target
    remains for the LM to be probed against."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    X_comp = np.stack([comp_lookup[c] for c in chain_ids])
    train = splits == "train"
    scaler = StandardScaler().fit(X_comp[train])
    model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(scaler.transform(X_comp[train]), y[train])
    y_hat = model.predict(scaler.transform(X_comp))
    return y - y_hat


def build_kmer_dataset_with_centered_target(labels_dir: Path, pdb_ids: list[str], k: int,
                                             include_flagged: bool, label_column: str,
                                             max_len: Optional[int] = None):
    """Same idea as build_kmer_dataset, but also returns the within-chain
    centered/rank twins of the target (same convention as Priority 1b),
    so k-mer context can be compared against the LM on the SAME
    local-signal target, not just the raw one.

    max_len: if given, restricts to seq_index < max_len per chain, the
    same simple front-truncation rule extract_layer_embeddings() uses
    (sequence[:max_len]) -- pass the LM's own effective_max_len so this
    sweep runs over the EXACT SAME residues the LM actually saw. Without
    this, long RNAs the LM never got to look at (it truncates first, drops
    everything past its length limit) would be included here but not in
    the LM's own numbers -- an unfair, easier-looking comparison."""
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
            if max_len is not None:
                seq = seq[:max_len]
            feats = kmer_context_features(seq, k)
            valid_rows = chain_rows[chain_rows["seq_index"] < feats.shape[0]]
            valid_rows = valid_rows.dropna(subset=[label_column])
            if valid_rows.empty:
                continue
            idxs = valid_rows["seq_index"].to_numpy()
            X_list.append(feats[idxs])
            y_list.append(valid_rows[label_column].to_numpy(dtype=float))
            groups_list.append(np.full(len(valid_rows), f"{pdb_id}::{chain}"))
            splits_list.append(valid_rows["split"].to_numpy() if "split" in valid_rows.columns
                                else np.full(len(valid_rows), "train"))
    if not X_list:
        raise RuntimeError(f"No usable rows assembled for k-mer sweep (k={k})")
    X = np.concatenate(X_list)
    y = np.concatenate(y_list)
    groups = np.concatenate(groups_list)
    splits = np.concatenate(splits_list)
    centered, rank_pct, _ = compute_structure_scalars(y, groups)
    return X, y, centered, rank_pct, splits


def run_kmer_centered_sweep(labels_dir: Path, pdb_ids: list[str], label_column: str,
                             include_flagged: bool,
                             ks: tuple[int, ...] = (1, 3, 5, 7, 9, 11, 15, 21),
                             max_len: Optional[int] = None) -> pd.DataFrame:
    """Sweeps k-mer context width against the within-chain-centered target,
    restricted (via max_len) to the exact same residues the LM saw. If
    R^2(k) keeps climbing toward the LM's own within-chain-centered R^2,
    the LM's apparent local signal is explainable by longer sequence
    context (a 'fancier k-mer model'). If it saturates well below the LM,
    that's real, unexplained representational advantage."""
    rows = []
    for k in ks:
        X, y, centered, rank_pct, splits = build_kmer_dataset_with_centered_target(
            labels_dir, pdb_ids, k, include_flagged, label_column, max_len=max_len)
        train, val, test = splits == "train", splits == "val", splits == "test"
        if train.sum() == 0 or test.sum() == 0:
            log.warning("k=%d: missing a split, skipping", k)
            continue
        trainval = train | val
        raw_r2 = run_ridge_probe(X[trainval], y[trainval], X[test], y[test])["r2"]
        centered_r2 = run_ridge_probe(X[trainval], centered[trainval], X[test], centered[test])["r2"]
        rank_r2 = run_ridge_probe(X[trainval], rank_pct[trainval], X[test], rank_pct[test])["r2"]
        rows.append({"k": k, "raw_test_r2": raw_r2, "within_chain_centered_test_r2": centered_r2,
                      "within_chain_rank_test_r2": rank_r2, "n_residues": int(len(y))})
        log.info("k=%2d: raw_R2=%.3f centered_R2=%.3f rank_R2=%.3f", k, raw_r2, centered_r2, rank_r2)
    return pd.DataFrame(rows)


def build_structural_feature_lookup(labels_dir: Path, pdb_ids: list[str],
                                     label_column: str, include_flagged: bool) -> dict:
    """(pdb_id, chain, seq_index) -> 3D structural feature vector,
    extracted from the label pipeline's own .pqr coordinate files via
    structural_3d_baseline.py's extractor -- keep that file in the same
    directory. Keyed by seq_index (not resnum): extract_structural_features
    only has resnum from the PQR atoms, so this looks each residue's
    seq_index up from the labels table (which carries both) before
    storing it, so the final lookup matches align_structural_features'
    seq_index-keyed convention. Lazy-imported so scipy is only required
    when this priority actually runs."""
    from structural_3d_baseline import extract_structural_features, STRUCT3D_FEATURE_NAMES
    lookup = {}
    for pdb_id in pdb_ids:
        pqr_path = labels_dir / "work" / pdb_id.upper() / f"{pdb_id.lower()}.pqr"
        if not pqr_path.exists():
            continue
        feats = extract_structural_features(pqr_path)
        if feats.empty:
            continue
        labels = load_structure_labels(labels_dir, pdb_id, include_flagged)
        if labels is None or label_column not in labels.columns:
            continue
        resnum_to_seqidx = labels.set_index(["chain", "resnum"])["seq_index"].to_dict()
        for _, r in feats.iterrows():
            seq_idx = resnum_to_seqidx.get((r["chain"], r["resnum"]))
            if seq_idx is None:
                continue
            lookup[(pdb_id, r["chain"], int(seq_idx))] = r[STRUCT3D_FEATURE_NAMES].to_numpy(dtype=float)
    return lookup


def align_structural_features(lookup: dict, pdb_groups: np.ndarray, chain_groups: np.ndarray,
                               seq_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X_struct, ok_mask) aligned row-for-row with pdb_groups/
    chain_groups/seq_indices. Keyed by seq_index rather than resnum --
    seq_index is unique-by-construction (it's the FASTA position), so
    this sidesteps any risk of resnum collisions from insertion-code
    edge cases even after the label pipeline's own renumbering. ok_mask
    is also False where a found vector isn't fully finite (e.g.
    dist_nearest_P_other with no other-residue phosphate nearby)."""
    n = len(pdb_groups)
    X_struct = None
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        chain = chain_groups[i].split("::")[1]
        key = (pdb_groups[i], chain, seq_indices[i])
        vec = lookup.get(key)
        if vec is None:
            continue
        vec = np.asarray(vec, dtype=float)
        if X_struct is None:
            X_struct = np.full((n, vec.shape[0]), np.nan)
        if np.isfinite(vec).all():
            X_struct[i] = vec
            ok[i] = True
    if X_struct is None:
        from structural_3d_baseline import STRUCT3D_FEATURE_NAMES
        X_struct = np.full((n, len(STRUCT3D_FEATURE_NAMES)), np.nan)
    return X_struct, ok


def align_feature_lookup(lookup: dict, pdb_groups: np.ndarray, chain_groups: np.ndarray,
                          seq_indices: np.ndarray, n_feat: int) -> tuple[np.ndarray, np.ndarray]:
    """Generic version of align_structural_features for any (pdb_id, chain,
    seq_index) -> feature-vector lookup -- used by Priority 10's 21-mer
    residual lookup so the same safe, seq_index-keyed alignment pattern
    applies everywhere a per-residue feature gets joined to the LM's rows."""
    n = len(pdb_groups)
    X = np.full((n, n_feat), np.nan)
    ok = np.zeros(n, dtype=bool)
    for i in range(n):
        chain = chain_groups[i].split("::")[1]
        key = (pdb_groups[i], chain, seq_indices[i])
        vec = lookup.get(key)
        if vec is None:
            continue
        vec = np.asarray(vec, dtype=float)
        if np.isfinite(vec).all():
            X[i] = vec
            ok[i] = True
    return X, ok


def build_kmer21_lookup(labels_dir: Path, pdb_ids: list[str], label_column: str,
                         include_flagged: bool, max_len: Optional[int]) -> dict:
    """(pdb_id, chain, seq_index) -> 21-mer one-hot context feature
    vector, restricted to max_len (the LM's own effective_max_len) so it
    stays support-matched -- for the residual-after-21mer probe
    (Priority 10)."""
    lookup = {}
    for pdb_id in pdb_ids:
        labels = load_structure_labels(labels_dir, pdb_id, include_flagged)
        if labels is None or label_column not in labels.columns:
            continue
        seqs = load_fasta_by_chain(labels_dir, pdb_id)
        for chain, chain_rows in labels.groupby("chain"):
            seq = seqs.get(chain)
            if not seq:
                continue
            if max_len is not None:
                seq = seq[:max_len]
            feats = kmer_context_features(seq, 21)
            valid_rows = chain_rows[chain_rows["seq_index"] < feats.shape[0]]
            for _, r in valid_rows.iterrows():
                idx = int(r["seq_index"])
                lookup[(pdb_id, chain, idx)] = feats[idx]
    return lookup


def fit_kmer_baseline_and_compute_residual(X_kmer: np.ndarray, y: np.ndarray,
                                            splits: np.ndarray) -> np.ndarray:
    """Out-of-fold residualization, NOT a single train+val fit predicting
    everything. A single trainval fit would make val's residual in-sample
    (the kmer model saw val's own targets), which biases layer selection
    in select_best_layer_and_test (layer choice is driven by val R^2).
    Instead: a model fit on TRAIN ONLY generates train's residual
    (in-sample for train -- standard and unavoidable in two-stage
    residual learning) AND val's residual (genuinely out-of-sample,
    since this model never saw val); a separate model fit on TRAIN+VAL
    generates test's residual (genuinely out-of-sample), mirroring
    exactly the protocol select_best_layer_and_test itself uses."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    train = splits == "train"
    val = splits == "val"
    test = splits == "test"
    trainval = train | val

    residual = np.full_like(y, np.nan, dtype=float)

    scaler_train = StandardScaler().fit(X_kmer[train])
    model_train = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(scaler_train.transform(X_kmer[train]), y[train])
    residual[train] = y[train] - model_train.predict(scaler_train.transform(X_kmer[train]))
    residual[val] = y[val] - model_train.predict(scaler_train.transform(X_kmer[val]))

    scaler_trainval = StandardScaler().fit(X_kmer[trainval])
    model_trainval = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(
        scaler_trainval.transform(X_kmer[trainval]), y[trainval])
    residual[test] = y[test] - model_trainval.predict(scaler_trainval.transform(X_kmer[test]))

    return residual


def local_context_strings(seq: str, half_window: int = 2) -> list[str]:
    """Local sequence context string per position (e.g. length 5 for
    half_window=2), padded with N at chain boundaries -- used purely as a
    grouping key for the electrostatic-twins experiment (Priority 11),
    not as a regression feature."""
    seq = seq.upper()
    padded = "N" * half_window + seq + "N" * half_window
    return [padded[i:i + 2 * half_window + 1] for i in range(len(seq))]


def build_twin_key_lookup(labels_dir: Path, pdb_ids: list[str], half_window: int = 2,
                           max_len: Optional[int] = None) -> dict:
    """(pdb_id, chain, seq_index) -> (base, is_paired_mfe, local_context)
    grouping key. Residues sharing this key look identical to any simple
    sequence-based model (same base, same secondary-structure status,
    same immediate neighbourhood) -- so any embedding-distance vs.
    electrostatic-difference correlation found within a group can't be
    explained by those superficial features, since they're held fixed by
    construction.

    max_len: truncates each sequence to the LM's own effective_max_len
    BEFORE folding. Without this, ViennaRNA's partition-function step
    (roughly O(n^3)) gets paid in full on every chain, including the
    longest RNAs in the dataset (several run past 4000 nt) -- for
    residues well past what the LM ever saw and that this lookup would
    never even be queried for downstream. That mismatch is what was
    burning hours: truncating first cuts the cost on the longest chains
    by roughly (full_len / max_len)^3, not just proportionally."""
    from secondary_structure_baseline import pairing_features
    lookup = {}
    for pdb_id in pdb_ids:
        seqs = load_fasta_by_chain(labels_dir, pdb_id)
        for chain, seq in seqs.items():
            if not seq:
                continue
            if max_len is not None:
                seq = seq[:max_len]
            pair_feats = pairing_features(seq)
            contexts = local_context_strings(seq, half_window=half_window)
            seq_upper = seq.upper()
            for i, base in enumerate(seq_upper):
                if i >= pair_feats.shape[0] or np.isnan(pair_feats[i]).any():
                    continue
                is_paired = int(round(pair_feats[i, 0]))
                lookup[(pdb_id, chain, i)] = (base, is_paired, contexts[i])
    return lookup


def fit_probe_and_get_predictions(X_layer: np.ndarray, y: np.ndarray, splits: np.ndarray) -> np.ndarray:
    """Refits the Ridge probe on train+val (same final-fit protocol
    select_best_layer_and_test itself uses for the chosen best layer),
    then returns w^T h + b -- the probe's own scalar prediction -- for
    EVERY row. This is what Priority 11 should compare twins on: not raw
    Euclidean distance in the full embedding space (which mixes in every
    direction the probe doesn't even use), but the actual signed
    difference in what the trained probe itself would predict."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    trainval = (splits == "train") | (splits == "val")
    scaler = StandardScaler().fit(X_layer[trainval])
    model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(scaler.transform(X_layer[trainval]), y[trainval])
    return model.predict(scaler.transform(X_layer))


def sample_pairs_from_group(idxs: list[int], max_pairs: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    """All pairs if the group is small enough, else a random sample of
    unique pairs (bounded attempts to avoid pathological loops on
    degenerate/huge groups)."""
    import itertools
    s = len(idxs)
    total_possible = s * (s - 1) // 2
    if total_possible <= max_pairs:
        return list(itertools.combinations(idxs, 2))
    idxs_arr = np.array(idxs)
    pairs = set()
    attempts = 0
    while len(pairs) < max_pairs and attempts < max_pairs * 20:
        a, b = rng.choice(idxs_arr, size=2, replace=False)
        pairs.add((int(min(a, b)), int(max(a, b))))
        attempts += 1
    return list(pairs)


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
    ap.add_argument("--cluster-assignments-csv", type=Path, default=None,
                     help="chain_id,cluster_rep CSV from mmseqs_cluster.py -- if given, "
                          "Priority 3 uses these real sequence-identity clusters instead of "
                          "the built-in k-mer-Jaccard approximation.")
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
    X, y_by_column, pdb_groups, chain_groups, splits, base_ids, resnums, seq_indices, eff_max_len = build_dataset_for_model_with_chains(
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

    # ---- Priority 1c: same-support base-identity baseline (fair comparison) ----
    # The LM only ever sees residues that survived truncation to its max
    # sequence length -- `valid` (well, the tokenizer-truncation filter
    # baked into build_dataset_for_model_with_chains) already restricts
    # every X/y/base_ids row to that exact support. So simply one-hot
    # encoding base_ids on these SAME rows, through the SAME
    # select_best_layer_and_test protocol, gives an apples-to-apples
    # number to put next to the LM's raw/centered/rank R^2 above --
    # unlike an external baseline run over the full untruncated set.
    log.info("=== Priority 1c: same-support base-identity baseline ===")
    base_onehot = _onehot_base_features(base_ids[valid])
    base_X = {0: base_onehot}  # single "layer" -- select_best_layer_and_test always picks it
    base_raw = select_best_layer_and_test(base_X, y[valid], splits_valid, args.model,
                                           args.label_column + "_baseident_raw")
    base_centered = select_best_layer_and_test(base_X, centered, splits_valid, args.model,
                                                args.label_column + "_baseident_centered")
    base_rank = select_best_layer_and_test(base_X, rank_pct, splits_valid, args.model,
                                            args.label_column + "_baseident_rank")
    pd.DataFrame([
        {"target": "raw", "lm_test_r2": quick["test_r2"], "base_identity_test_r2": base_raw["test_r2"]},
        {"target": "within_chain_centered", "lm_test_r2": centered_result["test_r2"],
         "base_identity_test_r2": base_centered["test_r2"]},
        {"target": "within_chain_rank_pct", "lm_test_r2": rank_result["test_r2"],
         "base_identity_test_r2": base_rank["test_r2"]},
    ]).to_csv(args.outdir / "p1c_same_support_base_identity.csv", index=False)
    log.info("Same-support comparison -- raw: LM=%.3f base=%.3f | centered: LM=%.3f base=%.3f | "
              "rank: LM=%.3f base=%.3f", quick["test_r2"], base_raw["test_r2"],
              centered_result["test_r2"], base_centered["test_r2"],
              rank_result["test_r2"], base_rank["test_r2"])
    log.info("=== Priority 2: k-mer baselines (same support as %s, seq_index < %d) ===",
              args.model, eff_max_len)
    kmer_df = run_kmer_baseline_sweep(args.labels_dir, pdb_ids, args.label_column,
                                       args.include_flagged, ks=tuple(args.kmer_ks),
                                       max_len=eff_max_len)
    kmer_df.to_csv(args.outdir / "p2_kmer_baselines.csv", index=False)

    # ---- Priority 3: sequence-cluster split (raw / centered / rank) ----
    log.info("=== Priority 3: sequence-cluster split ===")
    pdb_seqs = {}
    for pdb_id in pdb_ids:
        chains = load_fasta_by_chain(args.labels_dir, pdb_id)
        if chains:
            pdb_seqs[pdb_id] = "".join(chains.values())

    precomputed_clusters = None
    if args.cluster_assignments_csv:
        precomputed_clusters = load_pdb_clusters_from_csv(args.cluster_assignments_csv)
        log.info("Using external sequence-identity clusters from %s", args.cluster_assignments_csv)

    new_splits, clusters = build_cluster_splits(pdb_seqs, pdb_groups, args.cluster_identity_threshold,
                                                 precomputed_clusters)
    new_splits_valid = new_splits[valid]
    n_singleton = sum(1 for m in clusters.values() if len(m) == 1)

    cluster_raw = select_best_layer_and_test(X_valid, y[valid], new_splits_valid, args.model,
                                              args.label_column + "_clustersplit_raw")
    cluster_centered = select_best_layer_and_test(X_valid, centered, new_splits_valid, args.model,
                                                   args.label_column + "_clustersplit_centered")
    cluster_rank = select_best_layer_and_test(X_valid, rank_pct, new_splits_valid, args.model,
                                               args.label_column + "_clustersplit_rank")
    pd.DataFrame([
        {"target": "raw", "pdb_split_r2": quick["test_r2"], "cluster_split_r2": cluster_raw["test_r2"]},
        {"target": "within_chain_centered", "pdb_split_r2": centered_result["test_r2"],
         "cluster_split_r2": cluster_centered["test_r2"]},
        {"target": "within_chain_rank_pct", "pdb_split_r2": rank_result["test_r2"],
         "cluster_split_r2": cluster_rank["test_r2"]},
    ]).to_csv(args.outdir / "p3_cluster_split.csv", index=False)
    log.info("Cluster split (n_clusters=%d, singletons=%d) -- raw: pdb=%.3f cluster=%.3f | "
              "centered: pdb=%.3f cluster=%.3f | rank: pdb=%.3f cluster=%.3f",
              len(clusters), n_singleton, quick["test_r2"], cluster_raw["test_r2"],
              centered_result["test_r2"], cluster_centered["test_r2"],
              rank_result["test_r2"], cluster_rank["test_r2"])

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

    # ---- Priority 7: composition-residualized LM probe ----
    # Fits a simple length/GC/base-fraction model to the RAW target on
    # TRAIN rows only, then probes the LM against the *residual*
    # (y - composition_prediction) for every row -- a cleaner, single-number
    # version of "does the LM explain variance a fitted composition model
    # cannot", on the same rows/protocol as everything else here.
    log.info("=== Priority 7: composition-residualized LM probe ===")
    comp_lookup = build_composition_lookup(args.labels_dir, pdb_ids)
    residual = composition_residualize(y[valid], chain_groups[valid], comp_lookup, splits_valid)
    residual_result = select_best_layer_and_test(X_valid, residual, splits_valid, args.model,
                                                  args.label_column + "_comp_residual")
    pd.DataFrame([residual_result]).to_csv(args.outdir / "p7_composition_residualized.csv", index=False)
    log.info("Composition-residualized target: LM test_R^2=%.3f (vs raw target R^2=%.3f)",
              residual_result["test_r2"], quick["test_r2"])

    # ---- Priority 8: k-mer context sweep (k=1..21) vs within-chain-centered target ----
    log.info("=== Priority 8: k-mer context sweep vs within-chain-centered target ===")
    kmer_sweep_df = run_kmer_centered_sweep(args.labels_dir, pdb_ids, args.label_column, args.include_flagged,
                                             max_len=eff_max_len)
    kmer_sweep_df["lm_within_chain_centered_r2"] = centered_result["test_r2"]
    kmer_sweep_df.to_csv(args.outdir / "p8_kmer_sweep_vs_centered.csv", index=False)

    # ---- Priority 9: the actual Delta R^2 test -- LM+structure vs structure alone ----
    # This is the test as specified: Delta R^2 = R^2(LM+structure) - R^2(structure),
    # using the LM's own embeddings (its best layer for this target) concatenated
    # with real 3D structural features, compared against structure alone. This is
    # distinct from -- and more decisive than -- comparing the LM alone against
    # identity+structure, which answers a related but different question.
    log.info("=== Priority 9: Delta R^2 = R^2(LM+structure) - R^2(structure) ===")
    struct_lookup = build_structural_feature_lookup(args.labels_dir, pdb_ids, args.label_column,
                                                      args.include_flagged)
    X_struct_full, struct_ok_full = align_structural_features(struct_lookup, pdb_groups, chain_groups,
                                                                seq_indices)
    struct_ok_valid = struct_ok_full[valid]
    log.info("Structural features matched for %d/%d LM-valid residues",
              int(struct_ok_valid.sum()), len(struct_ok_valid))

    if struct_ok_valid.sum() < 100:
        log.warning("Too few residues with matched structural features -- skipping Priority 9")
    else:
        X_struct_valid = X_struct_full[valid][struct_ok_valid]
        y_centered_9 = centered[struct_ok_valid]
        splits_9 = splits_valid[struct_ok_valid]
        best_layer = quick["best_layer"]  # reuse the layer already chosen for the LM alone on this target
        X_lm_best_9 = X_valid[best_layer][struct_ok_valid]
        X_lm_plus_struct_9 = np.concatenate([X_lm_best_9, X_struct_valid], axis=1)

        struct_alone_result = select_best_layer_and_test(
            {0: X_struct_valid}, y_centered_9, splits_9, args.model, args.label_column + "_structure_alone")
        lm_alone_result_9 = select_best_layer_and_test(
            {0: X_lm_best_9}, y_centered_9, splits_9, args.model, args.label_column + "_lm_alone_samesupport")
        lm_plus_struct_result = select_best_layer_and_test(
            {0: X_lm_plus_struct_9}, y_centered_9, splits_9, args.model, args.label_column + "_lm_plus_structure")

        delta_r2 = lm_plus_struct_result["test_r2"] - struct_alone_result["test_r2"]
        pd.DataFrame([
            {"quantity": "structure_alone", "test_r2": struct_alone_result["test_r2"]},
            {"quantity": "lm_alone_same_support", "test_r2": lm_alone_result_9["test_r2"]},
            {"quantity": "lm_plus_structure", "test_r2": lm_plus_struct_result["test_r2"]},
            {"quantity": "delta_r2_lm_plus_structure_minus_structure", "test_r2": delta_r2},
        ]).to_csv(args.outdir / "p9_delta_r2_lm_plus_structure.csv", index=False)
        log.info("structure_alone=%.3f | LM_alone(same support)=%.3f | LM+structure=%.3f | "
                  "Delta R^2 (LM+structure - structure) = %.3f",
                  struct_alone_result["test_r2"], lm_alone_result_9["test_r2"],
                  lm_plus_struct_result["test_r2"], delta_r2)

    # ---- Priority 10: residual-after-21mer probe, plus the proper decomposition ----
    # Fits a 21-nucleotide-context model to the raw target with PROPER
    # out-of-fold residualization (see fit_kmer_baseline_and_compute_residual),
    # then probes the LM against what's left -- directly answering "is the
    # LM just a fancy k-mer model". Also reports the mathematically correct
    # decomposition Delta R^2 = R^2(kmer+LM) - R^2(kmer), since R^2 on a
    # residual target isn't simply additive with the original R^2 -- the
    # residual number below is a genuine, honest diagnostic, but this
    # second number is the one to actually cite. Finally, 10c sanity-checks
    # the residual itself: base identity, k=7, and k=21 should ALL score
    # ~0 against it (they're exactly what it was built to remove); if they
    # don't, the residual construction has a problem worth chasing down
    # before trusting the LM number above it.
    log.info("=== Priority 10: residual-after-21mer probe ===")
    kmer21_lookup = build_kmer21_lookup(args.labels_dir, pdb_ids, args.label_column,
                                         args.include_flagged, eff_max_len)
    n_kmer_feat = 21 * len(_ALPHABET)
    X_kmer21_full, kmer21_ok_full = align_feature_lookup(kmer21_lookup, pdb_groups, chain_groups,
                                                           seq_indices, n_kmer_feat)
    kmer21_ok_valid = kmer21_ok_full[valid]
    log.info("21-mer features matched for %d/%d LM-valid residues",
              int(kmer21_ok_valid.sum()), len(kmer21_ok_valid))

    if kmer21_ok_valid.sum() < 100:
        log.warning("Too few residues with matched 21-mer features -- skipping Priority 10")
    else:
        X_kmer21_valid = X_kmer21_full[valid][kmer21_ok_valid]
        y_10 = y[valid][kmer21_ok_valid]
        splits_10 = splits_valid[kmer21_ok_valid]

        # 10a: out-of-fold residual, LM probed against it (diagnostic, not the headline number)
        residual_21mer = fit_kmer_baseline_and_compute_residual(X_kmer21_valid, y_10, splits_10)
        X_lm_10 = {i: arr[kmer21_ok_valid] for i, arr in X_valid.items()}
        residual_result_10 = select_best_layer_and_test(
            X_lm_10, residual_21mer, splits_10, args.model, args.label_column + "_residual_after_21mer")
        log.info("10a (diagnostic) Residual-after-21mer: LM test_R^2=%.3f (n=%d)",
                  residual_result_10["test_r2"], int(kmer21_ok_valid.sum()))

        # 10b: the correct decomposition -- Delta R^2 = R^2(kmer+LM) - R^2(kmer)
        best_layer_10 = residual_result_10["best_layer"] if residual_result_10["best_layer"] in X_lm_10 \
            else quick["best_layer"]
        X_lm_best_10 = X_lm_10[best_layer_10]
        X_kmer_plus_lm_10 = np.concatenate([X_kmer21_valid, X_lm_best_10], axis=1)
        kmer_alone_10 = select_best_layer_and_test(
            {0: X_kmer21_valid}, y_10, splits_10, args.model, args.label_column + "_kmer21_alone")
        kmer_plus_lm_10 = select_best_layer_and_test(
            {0: X_kmer_plus_lm_10}, y_10, splits_10, args.model, args.label_column + "_kmer21_plus_lm")
        delta_r2_10b = kmer_plus_lm_10["test_r2"] - kmer_alone_10["test_r2"]
        log.info("10b (headline) kmer21_alone=%.3f | kmer21+LM=%.3f | Delta R^2 = %.3f",
                  kmer_alone_10["test_r2"], kmer_plus_lm_10["test_r2"], delta_r2_10b)

        # 10c: sanity check -- base identity / k=7 / k=21 should all score ~0 against the residual
        base_onehot_10 = _onehot_base_features(base_ids[valid][kmer21_ok_valid])
        base_vs_resid = select_best_layer_and_test(
            {0: base_onehot_10}, residual_21mer, splits_10, args.model, args.label_column + "_resid_vs_base")
        k7_lookup_10 = {}
        for pdb_id in pdb_ids:
            seqs = load_fasta_by_chain(args.labels_dir, pdb_id)
            for chain, seq in seqs.items():
                seq7 = seq[:eff_max_len] if eff_max_len is not None else seq
                feats7 = kmer_context_features(seq7, 7)
                for idx in range(feats7.shape[0]):
                    k7_lookup_10[(pdb_id, chain, idx)] = feats7[idx]
        X_k7_full, k7_ok_full = align_feature_lookup(k7_lookup_10, pdb_groups, chain_groups,
                                                       seq_indices, 7 * len(_ALPHABET))
        k7_ok_10 = k7_ok_full[valid][kmer21_ok_valid]
        if k7_ok_10.sum() > 100:
            k7_vs_resid = select_best_layer_and_test(
                {0: X_k7_full[valid][kmer21_ok_valid][k7_ok_10]}, residual_21mer[k7_ok_10],
                splits_10[k7_ok_10], args.model, args.label_column + "_resid_vs_k7")
        else:
            k7_vs_resid = {"test_r2": float("nan")}
        k21_vs_resid = select_best_layer_and_test(
            {0: X_kmer21_valid}, residual_21mer, splits_10, args.model, args.label_column + "_resid_vs_k21")
        log.info("10c sanity check -- base_identity vs residual R^2=%.3f | k=7 vs residual R^2=%.3f | "
                  "k=21 vs residual R^2=%.3f (all should be ~0)",
                  base_vs_resid["test_r2"], k7_vs_resid["test_r2"], k21_vs_resid["test_r2"])

        pd.DataFrame([
            {"quantity": "10a_residual_after_21mer_LM", "test_r2": residual_result_10["test_r2"]},
            {"quantity": "10b_kmer21_alone", "test_r2": kmer_alone_10["test_r2"]},
            {"quantity": "10b_kmer21_plus_LM", "test_r2": kmer_plus_lm_10["test_r2"]},
            {"quantity": "10b_delta_r2_headline", "test_r2": delta_r2_10b},
            {"quantity": "10c_sanity_base_identity_vs_residual", "test_r2": base_vs_resid["test_r2"]},
            {"quantity": "10c_sanity_k7_vs_residual", "test_r2": k7_vs_resid["test_r2"]},
            {"quantity": "10c_sanity_k21_vs_residual", "test_r2": k21_vs_resid["test_r2"]},
        ]).to_csv(args.outdir / "p10_residual_after_21mer.csv", index=False)

    # ---- Priority 11: electrostatic twin experiment ----
    # Groups residues by (base identity, ViennaRNA paired/unpaired status,
    # local 5-nt sequence context) -- everything a simple sequence model
    # could plausibly use -- pooled across ALL structures. Within each
    # group, residues are indistinguishable on every superficial feature.
    # If the LM's embedding distance between two "twins" still tracks
    # their ACTUAL electrostatic difference, that's the embedding space
    # organized by real environment, not by the features defining the
    # group. No new structures or APBS runs needed -- this only re-uses
    # data you already have.
    log.info("=== Priority 11: electrostatic twin experiment ===")
    twin_key_lookup = build_twin_key_lookup(args.labels_dir, pdb_ids, max_len=eff_max_len)
    pg_valid = pdb_groups[valid]
    cg_valid = chain_groups[valid]
    si_valid = seq_indices[valid]
    n_valid_rows = len(pg_valid)

    from collections import defaultdict
    groups_by_key = defaultdict(list)
    n_matched_11 = 0
    for i in range(n_valid_rows):
        chain = cg_valid[i].split("::")[1]
        key = (pg_valid[i], chain, int(si_valid[i]))
        gk = twin_key_lookup.get(key)
        if gk is not None:
            groups_by_key[gk].append(i)
            n_matched_11 += 1
    log.info("Twin-key matched for %d/%d LM-valid residues, forming %d distinct groups",
              n_matched_11, n_valid_rows, len(groups_by_key))

    rng11 = np.random.default_rng(0)
    pair_i, pair_j = [], []
    n_groups_used = 0
    for gk, idxs in groups_by_key.items():
        if len(idxs) < 2:
            continue
        n_groups_used += 1
        for a, b in sample_pairs_from_group(idxs, max_pairs=100, rng=rng11):
            pair_i.append(a)
            pair_j.append(b)

    if len(pair_i) < 50:
        log.warning("Too few twin pairs formed -- skipping Priority 11")
    else:
        pair_i = np.array(pair_i)
        pair_j = np.array(pair_j)
        y_valid_arr = y[valid]
        emb_best_11 = X_valid[quick["best_layer"]]

        # Signed probe-prediction difference (w^T h_i - w^T h_j), not raw
        # Euclidean embedding distance. This isolates exactly the direction
        # in embedding space the trained probe actually uses, and keeping
        # it SIGNED (rather than a magnitude-only norm) tests whether the
        # probe gets the DIRECTION of the electrostatic difference right,
        # not just how large it thinks the gap is -- a strictly sharper
        # claim than "distance tracks distance".
        probe_preds_11 = fit_probe_and_get_predictions(emb_best_11, y_valid_arr, splits_valid)
        delta_phi_signed = y_valid_arr[pair_i] - y_valid_arr[pair_j]
        delta_pred_signed = probe_preds_11[pair_i] - probe_preds_11[pair_j]

        from scipy.stats import pearsonr, spearmanr
        pear_r, pear_p = pearsonr(delta_pred_signed, delta_phi_signed)
        spear_r, spear_p = spearmanr(delta_pred_signed, delta_phi_signed)

        # negative control: shuffle delta_phi across pairs -- correlation should collapse to ~0
        shuffled_delta_phi = rng11.permutation(delta_phi_signed)
        pear_r_shuf, pear_p_shuf = pearsonr(delta_pred_signed, shuffled_delta_phi)

        log.info("Electrostatic twins: n_pairs=%d, n_groups=%d | real: Pearson r=%.3f (p=%.2e), "
                  "Spearman rho=%.3f (p=%.2e) | shuffled-control: Pearson r=%.3f (p=%.2e)",
                  len(pair_i), n_groups_used, pear_r, pear_p, spear_r, spear_p, pear_r_shuf, pear_p_shuf)
        log.info("Base rate within twin groups: median|delta_phi|=%.4f, 90th pct=%.4f",
                  float(np.median(np.abs(delta_phi_signed))), float(np.percentile(np.abs(delta_phi_signed), 90)))

        pd.DataFrame({"delta_phi_signed": delta_phi_signed, "delta_probe_pred_signed": delta_pred_signed}).to_csv(
            args.outdir / "p11_electrostatic_twin_pairs.csv", index=False)
        pd.DataFrame([{
            "n_pairs": len(pair_i), "n_groups_used": n_groups_used,
            "pearson_r": pear_r, "pearson_p": pear_p,
            "spearman_rho": spear_r, "spearman_p": spear_p,
            "pearson_r_shuffled_control": pear_r_shuf, "pearson_p_shuffled_control": pear_p_shuf,
            "median_abs_delta_phi": float(np.median(np.abs(delta_phi_signed))),
            "p90_abs_delta_phi": float(np.percentile(np.abs(delta_phi_signed), 90)),
        }]).to_csv(args.outdir / "p11_electrostatic_twin_summary.csv", index=False)

    log.info("All priority experiment outputs written to %s", args.outdir)


if __name__ == "__main__":
    main()