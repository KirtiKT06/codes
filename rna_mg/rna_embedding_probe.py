"""
Layer-wise probing of RNA language model embeddings for electrostatic
potential information.

Question being tested: do per-residue hidden states of published RNA LMs
linearly (or with a small MLP) encode local electrostatic potential, as
computed by rna_electrostatics_pipeline_v2.py?

Design decisions and why:

  * MODELS. Uses the `multimolecule` package (pip install multimolecule),
    which wraps several published RNA LMs -- RNA-FM, RNA-MSM, RNABERT,
    SpliceBERT, RiNALMo, UTR-LM among them -- behind a single
    HuggingFace-compatible AutoModel/AutoTokenizer interface, hosted under
    the `multimolecule/<name>` org on the HF Hub. This is a real,
    maintained integration -- not a hypothetical -- and it means one
    embedding-extraction code path covers every model instead of vendoring
    each paper's own repo. Swap/add entries in MODEL_REGISTRY freely; if a
    model you want isn't on `multimolecule`, you'll need a per-model
    loader function, but the rest of the probing pipeline (alignment,
    ridge/MLP loop, reporting) is model-agnostic and doesn't change.

  * ALIGNMENT. Joins on the `seq_index_map.csv` that the label pipeline
    now emits per structure -- NOT on resnum arithmetic. This is the
    single most important correctness point in this script: if the label
    and the embedding aren't from the exact same tokenization of the exact
    same sequence, everything downstream is silently wrong. The FASTA
    written by the label pipeline is re-tokenized here; the tokenizer's
    special tokens (CLS/EOS/etc.) are stripped by position before
    embeddings are matched back to seq_index.

  * SPLITTING. Uses the `split` column already assigned in the label
    pipeline (hashed by pdb_id) so probe train/val/test never mixes
    residues from the same RNA across splits.

  * BASELINE. Before trusting any probe result, this also fits a
    sequence-identity-only baseline (one-hot nucleotide + local GC/AU
    dinucleotide context) per split. Electrostatic potential correlates
    with base identity and backbone density on its own; if the LM
    embedding doesn't clearly beat this baseline, the LM isn't
    contributing electrostatic information beyond "knows what base this
    is."

  * LINEAR FIRST, MLP ON FAILURE. Per your plan: Ridge regression per
    layer first (fast, interpretable, standard for probing). If the best
    layer's held-out R^2 doesn't clear --linear-success-r2 (default 0.3,
    an arbitrary but explicit threshold -- adjust it), a small MLP probe
    is run on every layer as a fallback, since a real but nonlinearly
    encoded signal is a different (and still interesting) answer than "no
    signal at all."

  * FILTERING. Excludes charges_possibly_incomplete and
    had_modified_nucleotides rows by default (--include-flagged to keep
    them) since both are lower-confidence electrostatic labels and you
    don't want the probe result contaminated by label noise you already
    know about.

NOT verified end-to-end in this environment: no network access here to
actually pip install multimolecule, download model weights, or run a real
forward pass. The embedding-extraction and alignment logic mirrors
multimolecule's documented tokenizer/model API (RnaTokenizer +
AutoModel.from_pretrained(..., output_hidden_states=True)), but you should
smoke-test extract_layer_embeddings() on one small structure (e.g. 1EHZ,
76 nt) before launching the full 1.8k-structure run, and diff the printed
sequence against the FASTA to confirm tokenizer alignment before trusting
anything past that.

Requirements:
    pip install multimolecule torch transformers scikit-learn pandas numpy scipy

Usage:
    python rna_embedding_probe.py \
        --labels-dir ./rna_labels \
        --models rnafm rinalmo rnabert \
        --outdir ./probe_results
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("rna_embedding_probe")

# ----------------------------------------------------------------------
# Model registry -- multimolecule HF hub IDs.
# Add/remove freely. Each is loaded via the same generic code path below.
# ----------------------------------------------------------------------

MODEL_REGISTRY = {
    "rnafm":    "multimolecule/rnafm",
    "rnabert":  "multimolecule/rnabert",
    "rnamsm":   "multimolecule/rnamsm",
    "rinalmo":  "multimolecule/rinalmo",
    "splicebert": "multimolecule/splicebert",
    "utrlm":    "multimolecule/utrlm",
    "ernierna": "multimolecule/ernierna",
}

BASES = ["A", "C", "G", "U"]


# ----------------------------------------------------------------------
# Step 1: gather label data (potential + sequence index map) per structure
# ----------------------------------------------------------------------

def load_structure_labels(labels_dir: Path, pdb_id: str,
                           include_flagged: bool) -> Optional[pd.DataFrame]:
    """Join per_residue_potential.csv with seq_index_map.csv for one
    structure -> one row per residue with both the electrostatic-label
    columns (potential_mean_kT_e_{4,8,12}A, potential_at_c1prime_kT_e,
    potential_at_phosphorus_kT_e) and `seq_index` (the position the
    tokenizer will assign it), keyed by chain so multi-chain structures
    don't cross-contaminate seq_index."""
    work_dir = labels_dir / "work" / pdb_id.upper()
    pot_path = work_dir / "per_residue_potential.csv"
    idx_path = work_dir / "seq_index_map.csv"
    if not pot_path.exists() or not idx_path.exists():
        return None

    pot = pd.read_csv(pot_path)
    idx = pd.read_csv(idx_path)

    if not include_flagged:
        if "charges_possibly_incomplete" in pot.columns:
            pot = pot[~pot["charges_possibly_incomplete"]]
        if "had_modified_nucleotides" in pot.columns:
            pot = pot[~pot["had_modified_nucleotides"]]

    merged = pot.merge(idx, on=["chain", "resnum", "resname"], how="inner",
                        suffixes=("", "_idxmap"))
    if merged.empty:
        return None
    merged["pdb_id"] = pdb_id.upper()

    for col in (
        "potential_mean_kT_e_4A",
        "potential_mean_kT_e_8A",
        "potential_mean_kT_e_12A",
    ):
        if col in merged.columns:

            mean = merged[col].mean()
            std = merged[col].std()

            if std > 0:
                merged[f"{col}_z"] = (merged[col] - mean) / std
            else:
                merged[f"{col}_z"] = 0.0

    return merged


def load_fasta_by_chain(labels_dir: Path, pdb_id: str) -> dict[str, str]:
    fasta_path = labels_dir / "work" / pdb_id.upper() / f"{pdb_id.lower()}.fasta"
    seqs = {}
    if not fasta_path.exists():
        return seqs
    chain = None
    for line in fasta_path.read_text().splitlines():
        if line.startswith(">"):
            chain = line.split("_")[-1].strip()
        elif chain is not None:
            seqs[chain] = line.strip()
    return seqs


# ----------------------------------------------------------------------
# Step 2: embedding extraction
# ----------------------------------------------------------------------

def load_model(model_key: str, device: str):
    """Generic loader for any multimolecule-hosted RNA LM. Returns
    (tokenizer, model, n_layers). All these models expose
    output_hidden_states, giving one embedding tensor per layer
    (including the input embedding layer at index 0) -- that's the full
    per-layer sweep the probing loop iterates over."""
    import torch
    from multimolecule import RnaTokenizer, AutoModel

    hf_id = MODEL_REGISTRY[model_key]
    tokenizer = RnaTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id, output_hidden_states=True)
    log.info(
    "%s: tokenizer.model_max_length=%s, config.max_position_embeddings=%s",
    model_key,
    getattr(tokenizer, "model_max_length", "NA"),
    getattr(model.config, "max_position_embeddings", "NA"),)
    model.to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1  # +1 for the embedding layer
    tokenizer_max_len = getattr(tokenizer, "model_max_length", None)
    config_max_len = getattr(model.config, "max_position_embeddings", None)

    if tokenizer_max_len is None or tokenizer_max_len > 100000:
        tokenizer_max_len = 1000000

    if config_max_len is None or config_max_len > 100000:
        config_max_len = 1000000

    max_model_len = min(tokenizer_max_len, config_max_len)
    return tokenizer, model, n_layers, max_model_len


def extract_layer_embeddings(tokenizer, model, sequence: str, device: str,
                              max_len: int = 1024) -> Optional[np.ndarray]:
    """Returns an array of shape (n_layers, seq_len, hidden_dim) with
    special tokens already stripped, so array position i corresponds
    directly to seq_index i in the label table.

    Alignment assumption (SPOT-CHECK THIS): the tokenizer emits exactly
    one token per input character with a fixed number of special tokens
    at the start/end (typically CLS ... EOS for these BERT-style RNA
    LMs). This holds for multimolecule's models as documented, but if you
    add a model with a different tokenization scheme (e.g. k-mer
    tokenization), this stripping logic needs to change accordingly --
    a silent off-by-one here would corrupt every probe result.
    """
    import torch

    if len(sequence) == 0:
        return None
    original_length = len(sequence)
    if original_length > max_len:
        dropped = original_length - max_len
        log.warning(
            "Sequence length %d exceeds max_len=%d "
            "(dropping %d residues)",
            original_length,
            max_len,
            dropped,
        )
        sequence = sequence[:max_len]

    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
        log.warning(
            "Tokenized sequence length %d exceeds model limit %d",
            inputs["input_ids"].shape[1],
            model.config.max_position_embeddings,
        )
    n_special_start = int((inputs["input_ids"][0] == tokenizer.cls_token_id).sum()) if \
        tokenizer.cls_token_id is not None else 0

    with torch.no_grad():
        out = model(**inputs)

    hidden_states = out.hidden_states  # tuple of (1, seq_len_with_specials, dim)
    n_tokens_with_specials = hidden_states[0].shape[1]
    n_special_end = n_tokens_with_specials - n_special_start - len(sequence)
    if n_special_end < 0:
        log.error("Token count mismatch for sequence of length %d (got %d tokens); "
                   "tokenizer alignment assumption failed for this model -- "
                   "inspect tokenizer output before trusting any probe result.",
                   len(sequence), n_tokens_with_specials)
        return None

    layers = []
    for h in hidden_states:
        h = h[0]  # drop batch dim -> (seq_len_with_specials, dim)
        stripped = h[n_special_start: n_special_start + len(sequence)]
        layers.append(stripped.cpu().numpy())
    return np.stack(layers, axis=0)  # (n_layers, seq_len, dim)


# ----------------------------------------------------------------------
# Step 3: build the (embedding, label) dataset for one model
# ----------------------------------------------------------------------

def build_dataset_for_model(labels_dir: Path, pdb_ids: list[str], model_key: str,
                             device: str, include_flagged: bool, max_len: int,
                             label_columns: list[str]):
    """Returns:
        X: dict layer_idx -> (n_residues, hidden_dim) array
        y_by_column: dict label_column -> (n_residues,) array (may contain NaN
            where that particular averaging radius/point value wasn't available
            for a residue, e.g. a chain terminus with no resolved phosphate)
        groups: (n_residues,) pdb_id per row, for GroupKFold-style splitting
        splits: (n_residues,) split label ('train'/'val'/'test') from the label pipeline
        base_ids: (n_residues,) one-hot-able nucleotide identity, for the baseline probe

    Embeddings are extracted ONCE per model here; every requested
    label_column (e.g. the 4/8/12 A averages and the point values at C1'/P)
    is carried alongside the same embeddings, so comparing which
    electrostatic-label definition correlates best with a given layer -- the
    open experimental-design question raised in review -- costs nothing
    beyond re-fitting cheap linear/MLP probes, not re-running the LM.
    """
    tokenizer, model, n_layers, model_max_len = load_model(model_key, device)
    effective_max_len = min(max_len, model_max_len-2)
    log.info(
        "%s: model_max_len=%d, effective_max_len=%d",
        model_key,
        model_max_len,
        effective_max_len,
    )

    per_layer_X = {i: [] for i in range(n_layers)}
    y_by_column_lists = {col: [] for col in label_columns}
    groups_all, splits_all, base_all = [], [], []
    total_before = 0
    total_after = 0

    for pdb_id in pdb_ids:
        labels = load_structure_labels(labels_dir, pdb_id, include_flagged)
        if labels is None:
            continue
        missing_cols = [c for c in label_columns if c not in labels.columns]
        if missing_cols and pdb_id == pdb_ids[0]:
            log.warning("%s: label column(s) %s not found (older label pipeline run?); "
                        "those columns will be all-NaN", pdb_id, missing_cols)
        seqs = load_fasta_by_chain(labels_dir, pdb_id)

        for chain, chain_rows in labels.groupby("chain"):
            seq = seqs.get(chain)
            if not seq:
                continue
            emb = extract_layer_embeddings(tokenizer, model, seq, device, max_len=effective_max_len)
            if emb is None:
                continue

            valid = chain_rows[chain_rows["seq_index"] < emb.shape[1]]
            total_before += len(chain_rows)
            total_after += len(valid)

            if len(valid) < len(chain_rows):
                log.info(
                    "%s chain %s: retained %d/%d residues after truncation",
                    pdb_id,
                    chain,
                    len(valid),
                    len(chain_rows),
                )
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
            groups_all.append(np.full(len(valid), pdb_id))
            splits_all.append(valid["split"].to_numpy() if "split" in valid.columns
                               else np.full(len(valid), "train"))
            base_all.append(valid["base"].to_numpy())

    if not groups_all:
        raise RuntimeError(f"No usable (embedding, label) pairs assembled for model {model_key}")

    X = {i: np.concatenate(v, axis=0) for i, v in per_layer_X.items()}
    y_by_column = {col: np.concatenate(v) for col, v in y_by_column_lists.items()}
    groups = np.concatenate(groups_all)
    splits = np.concatenate(splits_all)
    base_ids = np.concatenate(base_all)
    log.info("%s: assembled %d residues across %d structures, %d layers, dim=%d",
              model_key, len(groups), len(set(groups)), n_layers, next(iter(X.values())).shape[1])
    if total_before:
        retained_fraction = total_after / total_before
        log.info(
            "%s: retained %.2f%% of residues after tokenizer filtering (%d/%d)",
            model_key,
            retained_fraction * 100.0,
            total_after,
            total_before,
        )

        log.info(
            "%s: dropped %.2f%% of residues due to model length limits",
            model_key,
            100.0 * (1.0 - retained_fraction),
        )

    log.info(
    "%s split distribution: train=%d val=%d test=%d",
    model_key,
    (splits == "train").sum(),
    (splits == "val").sum(),
    (splits == "test").sum(),)
    return X, y_by_column, groups, splits, base_ids, effective_max_len



# ----------------------------------------------------------------------
# Step 4: probes
# ----------------------------------------------------------------------

def _onehot_base_features(base_ids: np.ndarray) -> np.ndarray:
    return np.stack([(base_ids == b).astype(float) for b in BASES], axis=1)


def run_ridge_probe(X_train, y_train, X_val, y_val) -> dict:
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xva = scaler.transform(X_val)

    model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model.fit(Xtr, y_train)
    pred = model.predict(Xva)

    r2 = model.score(Xva, y_val)
    r, _ = pearsonr(y_val, pred) if len(set(y_val)) > 1 else (np.nan, np.nan)
    return {"r2": float(r2), "pearson_r": float(r), "alpha": float(model.alpha_)}


def run_mlp_probe(X_train, y_train, X_val, y_val, hidden=(128, 32)) -> dict:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xva = scaler.transform(X_val)

    model = MLPRegressor(hidden_layer_sizes=hidden, early_stopping=True,
                          max_iter=500, random_state=0)
    model.fit(Xtr, y_train)
    pred = model.predict(Xva)

    r2 = model.score(Xva, y_val)
    r, _ = pearsonr(y_val, pred) if len(set(y_val)) > 1 else (np.nan, np.nan)
    return {"r2": float(r2), "pearson_r": float(r)}

def select_best_layer_and_test(
    X: dict,
    y: np.ndarray,
    splits: np.ndarray,
    model_key: str,
    label_column: str,
):
    """
    Select best layer on VAL.
    Retrain on TRAIN+VAL.
    Report final TEST performance.
    """

    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr

    train = splits == "train"
    val = splits == "val"
    test = splits == "test"

    if test.sum() == 0:
        raise RuntimeError(
            f"{model_key}/{label_column}: no test samples available"
        )

    best_layer = None
    best_val_r2 = -np.inf

    # layer selection on validation set
    for layer, X_layer in X.items():

        result = run_ridge_probe(
            X_layer[train],
            y[train],
            X_layer[val],
            y[val],
        )

        if result["r2"] > best_val_r2:
            best_val_r2 = result["r2"]
            best_layer = layer

    # retrain using train+val
    trainval = train | val

    X_best = X[best_layer]

    scaler = StandardScaler().fit(X_best[trainval])

    Xtr = scaler.transform(X_best[trainval])
    Xte = scaler.transform(X_best[test])

    model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model.fit(Xtr, y[trainval])

    pred = model.predict(Xte)

    r2 = model.score(Xte, y[test])

    r, _ = pearsonr(y[test], pred)

    return {
        "best_layer": int(best_layer),
        "val_r2": float(best_val_r2),
        "test_r2": float(r2),
        "test_pearson_r": float(r),
    }

def probe_model(model_key: str, X: dict, y: np.ndarray, splits: np.ndarray,
                 base_ids: np.ndarray, linear_success_r2: float,
                 label_column: str, model_max_len) -> pd.DataFrame:
    """Probes one (model, label_column) pair across all layers. Rows
    with a NaN label for this particular column (e.g. a chain-terminus
    residue with no resolved phosphorus atom for potential_at_phosphorus_kT_e)
    are dropped up front -- which residues are droppped can differ across
    label_columns, so this filtering is done per-column, not once globally.
    """
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 20:
        log.warning("%s / %s: only %d non-NaN labels, skipping", model_key, label_column,
                    valid_mask.sum())
        return pd.DataFrame()

    y = y[valid_mask]
    splits_f = splits[valid_mask]
    base_f = base_ids[valid_mask]
    X_f = {i: arr[valid_mask] for i, arr in X.items()}

    train_mask = splits_f == "train"
    val_mask = splits_f == "val"
    test_mask = splits_f == "test"

    if train_mask.sum() == 0:
        log.warning(
            "%s / %s: no training samples available",
            model_key,
            label_column,
        )
        return pd.DataFrame()

    if val_mask.sum() == 0:
        log.warning(
            "%s / %s: no validation samples available",
            model_key,
            label_column,
        )
        return pd.DataFrame()

    if test_mask.sum() == 0:
        log.warning(
            "%s / %s: no test samples available",
            model_key,
            label_column,
        )
        return pd.DataFrame()
    
    log.info(
        "%s / %s: train=%d val=%d test=%d",
        model_key,
        label_column,
        train_mask.sum(),
        val_mask.sum(),
        (splits_f == "test").sum(),
    )
    base_feats = _onehot_base_features(base_f)
    baseline = run_ridge_probe(base_feats[train_mask], y[train_mask],
                                base_feats[val_mask], y[val_mask])
    log.info("%s / %s: sequence-identity-only baseline R^2=%.3f (r=%.3f)",
              model_key, label_column, baseline["r2"], baseline["pearson_r"])

    rows = []
    best_linear_r2 = -np.inf
    for layer_i, X_layer in X_f.items():
        lin = run_ridge_probe(X_layer[train_mask], y[train_mask],
                               X_layer[val_mask], y[val_mask])
        rows.append({"model": model_key, "model_max_len": model_max_len, "label_column": label_column, "layer": layer_i,
                      "probe": "linear", **lin, "baseline_r2": baseline["r2"],
                      "n_residues": int(valid_mask.sum())})
        best_linear_r2 = max(best_linear_r2, lin["r2"])
        log.info("%s / %s layer %d [linear] R^2=%.3f r=%.3f", model_key, label_column,
                  layer_i, lin["r2"], lin["pearson_r"])

    if best_linear_r2 < linear_success_r2:
        log.info("%s / %s: best linear R^2=%.3f below threshold %.2f -- running MLP fallback",
                  model_key, label_column, best_linear_r2, linear_success_r2)
        for layer_i, X_layer in X_f.items():
            mlp = run_mlp_probe(X_layer[train_mask], y[train_mask],
                                 X_layer[val_mask], y[val_mask])
            rows.append({"model": model_key, "model_max_len": model_max_len, "label_column": label_column, "layer": layer_i,
                         "probe": "mlp", **mlp, "baseline_r2": baseline["r2"],
                         "n_residues": int(valid_mask.sum())})
            log.info("%s / %s layer %d [mlp] R^2=%.3f r=%.3f", model_key, label_column,
                      layer_i, mlp["r2"], mlp["pearson_r"])
    final_test = select_best_layer_and_test(
        X_f,
        y,
        splits_f,
        model_key,
        label_column,
    )

    rows.append({
    "model": model_key,
    "model_max_len": model_max_len,
    "label_column": label_column,
    "layer": final_test["best_layer"],
    "probe": "final_test",
    "r2": final_test["test_r2"],
    "pearson_r": final_test["test_pearson_r"],
    "baseline_r2": baseline["r2"],
    "n_residues": int(valid_mask.sum()),
    })

    log.info(
        "%s / %s FINAL TEST: best layer=%d  test_R2=%.3f  test_r=%.3f",
        model_key,
        label_column,
        final_test["best_layer"],
        final_test["test_r2"],
        final_test["test_pearson_r"],
    )

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True,
                     help="Output dir of rna_electrostatics_pipeline_v2.py "
                          "(expects labels-dir/work/<PDBID>/...).")
    ap.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                     choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--outdir", type=Path, default=Path("./probe_results"))
    ap.add_argument("--device", default="cuda", help="'cuda' or 'cpu'.")
    ap.add_argument("--max-len", type=int, default=1024,
                     help="Truncate sequences longer than this (memory guard for e.g. rRNA).")
    ap.add_argument("--include-flagged", action="store_true",
                     help="Include charges_possibly_incomplete / had_modified_nucleotides rows.")
    ap.add_argument("--linear-success-r2", type=float, default=0.3,
                     help="Held-out R^2 threshold below which the MLP fallback probe runs.")
    ap.add_argument("--label-columns", nargs="+",
                     default=[
                        "potential_mean_kT_e_4A",
                        "potential_mean_kT_e_8A",
                        "potential_mean_kT_e_12A",

                        "potential_mean_kT_e_4A_z",
                        "potential_mean_kT_e_8A_z",
                        "potential_mean_kT_e_12A_z",

                        "potential_at_c1prime_kT_e",
                        "potential_at_phosphorus_kT_e",
                    ],
                     help="Which electrostatic-label column(s) to probe against, from "
                          "per_residue_potential.csv. Averaging radius (and sphere-average "
                          "vs. point value) is an open experimental-design question, not a "
                          "fixed choice -- probing all of them (the default) directly answers "
                          "which definition correlates best with a given model/layer, at the "
                          "cost of extra probe fits only (embeddings are extracted once and "
                          "reused across every label column).")
    ap.add_argument("--limit-structures", type=int, default=None)
    args = ap.parse_args()

    import torch
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    if device != args.device:
        log.warning("CUDA not available, falling back to CPU")

    all_pot_path = args.labels_dir / "all_residue_potentials.csv"
    if not all_pot_path.exists():
        raise FileNotFoundError(f"{all_pot_path} not found -- run the label pipeline first.")
    pdb_ids = sorted(pd.read_csv(all_pot_path)["pdb_id"].unique())
    if args.limit_structures:
        pdb_ids = pdb_ids[: args.limit_structures]
    log.info("Probing against %d structures, label columns: %s", len(pdb_ids), args.label_columns)

    args.outdir.mkdir(parents=True, exist_ok=True)
    all_results = []
    for model_key in args.models:
        log.info("=== Model: %s (%s) ===", model_key, MODEL_REGISTRY[model_key])
        try:
            X, y_by_column, groups, splits, base_ids, effective_max_len = build_dataset_for_model(
                args.labels_dir, pdb_ids, model_key, device,
                args.include_flagged, args.max_len, args.label_columns,
            )
            for label_column in args.label_columns:
                results = probe_model(model_key, X, y_by_column[label_column], splits,
                                       base_ids, args.linear_success_r2, label_column, effective_max_len)
                if results.empty:
                    continue
                safe_col = label_column.replace("/", "_")
                results.to_csv(args.outdir / f"{model_key}_{safe_col}_probe_results.csv",
                                index=False)
                all_results.append(results)
        except Exception as e:
            log.error("%s: probing failed: %s", model_key, e, exc_info=True)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.outdir / "all_probe_results.csv", index=False)
        log.info("Wrote combined probe results to %s", args.outdir / "all_probe_results.csv")

        summary = (
            combined[combined["probe"] == "final_test"]
            .sort_values("r2", ascending=False)
            [
                [
                    "model",
                    "label_column",
                    "layer",
                    "r2",
                    "pearson_r",
                    "baseline_r2",
                ]
            ]
        )
        log.info(
            "Final held-out test results:\n%s",
            summary.to_string(index=False),
        )

if __name__ == "__main__":
    main()