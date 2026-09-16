"""
Layer-wise probing of RNA language model embeddings for electrostatic
potential information.

Question being tested: do per-residue hidden states of published RNA LMs
linearly (or with a small MLP) encode local electrostatic potential, as
computed by rna_electrostatics_pipeline_v2.py?

Design decisions and why:

  * MODELS. Uses the `multimolecule` package (pip install multimolecule),
    which wraps several published RNA LMs behind a single HuggingFace-
    compatible AutoModel/AutoTokenizer interface. RNA-MSM is deliberately
    NOT in the registry (post-review removal): it requires an actual MSA
    per sequence (confirmed by the "works best with MSA inputs" warning
    and hard tokenizer failure on single-sequence input in the first full
    run), and current published benchmarks (BEACON-style RNA-LM
    comparisons, e.g. the ERNIE-RNA paper's F1 tables) show general-
    purpose single-sequence models -- ERNIE-RNA and RiNALMo specifically
    -- already match or beat RNA-MSM without needing alignment. Building
    an Infernal/Rfam-based MSA pipeline for one model the field has
    already moved past is not worth the engineering cost here. RiNALMo
    and UTR-LM are gated repos on the HF Hub (401 in the first run, not a
    code bug) -- request access on their model pages and run
    `huggingface-cli login` / set HF_TOKEN before using them.

  * ALIGNMENT. Joins on the `seq_index_map.csv` that the label pipeline
    emits per structure -- NOT on resnum arithmetic. The FASTA written by
    the label pipeline is re-tokenized here; special tokens are stripped
    by position before embeddings are matched back to seq_index.

  * SPLITTING. Uses the `split` column already assigned in the label
    pipeline (hashed by pdb_id) so probe train/val/test never mixes
    residues from the same RNA across splits.

  * TRAIN/VAL/TEST DISCIPLINE. Every layer is fit on train and scored on
    val; that val score picks the best layer only. The winning layer is
    refit on train+val and scored ONCE on test -- report that number, not
    the val score that picked it. Post-review fix: the baseline
    (sequence-identity-only) reported alongside the final test row now
    goes through the IDENTICAL train+val -> test protocol as the model,
    via baseline_train_val_test(). Previously the baseline attached to
    the final_test row was a train->val fit (correct for the per-layer
    diagnostic rows, where the model's own r2 is also val-based, but
    silently mismatched against the model's train+val->test r2 on the
    final_test row -- comparing a test number to a val number under the
    same column name). The per-layer diagnostic rows are unaffected: their
    baseline_r2 is still the train->val baseline, which matches what those
    rows' own r2 measures.

  * Z-SCORED TARGETS. Per-structure z-scored twin of every potential_*
    column (suffix _z), added by add_zscored_targets() inside
    load_structure_labels(). Only ever computed for the sphere-average
    columns in this run -- NOT yet for potential_at_c1prime_kT_e /
    potential_at_phosphorus_kT_e. Add those too before concluding whether
    the strong point-value results reflect real local electrostatic
    encoding versus a structure-level scale confound (the sphere-average
    z-scored results dropped substantially versus raw, suggesting at
    least part of the raw signal there was between-structure scale, not
    fine-grained per-residue reasoning -- worth checking whether the same
    holds for the point-value targets).

  * BASELINE. Sequence-identity-only (one-hot nucleotide). If the LM
    embedding doesn't clearly beat this, it isn't contributing
    electrostatic information beyond "knows what base this is."

  * LINEAR FIRST, MLP ON FAILURE. Ridge regression per layer first. If
    the best layer's held-out val R^2 doesn't clear --linear-success-r2
    (default 0.3), a small MLP probe runs on every layer as a fallback.

  * FILTERING. Excludes charges_possibly_incomplete and
    had_modified_nucleotides rows by default (--include-flagged to keep
    them).

Requirements:
    pip install multimolecule torch transformers scikit-learn pandas numpy scipy

Usage:
    python rna_embedding_probe_v2.py \
        --labels-dir ./rna_labels \
        --models rnafm rinalmo ernierna splicebert \
        --outdir ./probe_results
"""

from __future__ import annotations

import argparse
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
#
# rnamsm removed (post-review): needs real per-sequence MSA input, hard-
# fails on single-sequence input (confirmed in the first full run), and
# published benchmarks show ERNIE-RNA / RiNALMo already beat it without
# alignment -- not worth building an Infernal/Rfam MSA pipeline for.
#
# rinalmo and utrlm are GATED on the HF Hub: request access on their
# model pages, then `huggingface-cli login` (or set HF_TOKEN) before
# running -- the 401s in the first run were an auth issue, not a bug
# here or in multimolecule.
# ----------------------------------------------------------------------

MODEL_REGISTRY = {
    "rnafm":      "multimolecule/rnafm",
    "rnabert":    "multimolecule/rnabert",
    # "rinalmo":    "multimolecule/rinalmo",     # gated -- needs HF auth
    "splicebert": "multimolecule/splicebert",
    # "utrlm":      "multimolecule/utrlm",       # gated -- needs HF auth
    "ernierna":   "multimolecule/ernierna",
    # "rnamsm":     "multimolecule/rnamsm",
}

BASES = ["A", "C", "G", "U"]


# ----------------------------------------------------------------------
# Step 1: gather label data (potential + sequence index map) per structure
# ----------------------------------------------------------------------

def add_zscored_targets(df: pd.DataFrame) -> None:
    """In-place z-scoring of electrostatic targets within one structure."""

    target_cols = [
        "potential_mean_kT_e_4A",
        "potential_mean_kT_e_8A",
        "potential_mean_kT_e_12A",
        "potential_at_c1prime_kT_e",
        "potential_at_phosphorus_kT_e",
    ]

    for col in target_cols:
        if col not in df.columns:
            continue

        mean = df[col].mean()
        std = df[col].std()

        if pd.notna(std) and std > 0:
            df[f"{col}_z"] = (df[col] - mean) / std
        else:
            df[f"{col}_z"] = np.nan


def load_structure_labels(labels_dir: Path, pdb_id: str,
                           include_flagged: bool) -> Optional[pd.DataFrame]:
    """Join per_residue_potential.csv with seq_index_map.csv for one
    structure -> one row per residue with the electrostatic-label columns
    (potential_mean_kT_e_{4,8,12}A [+ _z twins], potential_at_c1prime_kT_e,
    potential_at_phosphorus_kT_e) and `seq_index`, keyed by chain so
    multi-chain structures don't cross-contaminate seq_index."""
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

    merge_fraction = len(merged) / max(len(pot), 1)
    # if merge_fraction < 0.95:
    #     log.warning(
    #         "%s: merge retained only %.1f%% of rows "
    #         "(%d/%d)",
    #         pdb_id,
    #         merge_fraction * 100.0,
    #         len(merged),
    #         len(pot)
    #     )

    if merged.empty:
        return None
    merged["pdb_id"] = pdb_id.upper()
    add_zscored_targets(merged)
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
    (tokenizer, model, n_layers, max_model_len). Derives max_model_len
    from tokenizer.model_max_length and config.max_position_embeddings
    (some tokenizers report an absurd sentinel like 1e30 when unset --
    clamped away here) so callers never hardcode a single model's limit."""
    import torch
    from multimolecule import RnaTokenizer, AutoModel

    hf_id = MODEL_REGISTRY[model_key]
    tokenizer = RnaTokenizer.from_pretrained(hf_id)
    model = AutoModel.from_pretrained(hf_id, output_hidden_states=True)
    model.to(device)
    model.eval()
    n_layers = model.config.num_hidden_layers + 1  # +1 for the embedding layer

    tokenizer_max_len = getattr(tokenizer, "model_max_length", None)
    config_max_len = getattr(model.config, "max_position_embeddings", None)
    if tokenizer_max_len is None or tokenizer_max_len > 100_000:
        tokenizer_max_len = 1_000_000
    if config_max_len is None or config_max_len > 100_000:
        config_max_len = 1_000_000
    max_model_len = min(tokenizer_max_len, config_max_len)

    log.info("%s: tokenizer.model_max_length=%s, config.max_position_embeddings=%s",
              model_key, getattr(tokenizer, "model_max_length", "NA"),
              getattr(model.config, "max_position_embeddings", "NA"))
    return tokenizer, model, n_layers, max_model_len


def extract_layer_embeddings(tokenizer, model, sequence: str, device: str,
                              max_len: int = 1024) -> tuple[Optional[np.ndarray], int]:
    """Returns (embeddings, n_truncated). embeddings has shape
    (n_layers, seq_len, hidden_dim) with special tokens stripped, so
    array position i corresponds directly to seq_index i."""
    import torch

    if len(sequence) == 0:
        return None, 0
    orig_len = len(sequence)
    n_truncated = 0
    if orig_len > max_len:
        n_truncated = orig_len - max_len
        # log.warning("Sequence length %d exceeds max_len=%d (dropping %d residues)",
        #              orig_len, max_len, n_truncated)
        sequence = sequence[:max_len]

    inputs = tokenizer(sequence, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"][0]
    # if len(sequence) <= 50:
    #     tokens = tokenizer.convert_ids_to_tokens(input_ids.detach().cpu().tolist())

    #     log.info(
    #         "TOKENIZATION CHECK\n"
    #         "Sequence: %s\n"
    #         "Tokens: %s",
    #         sequence,
    #         tokens
    #     )

    # ------------------------------------------------------------------
    # SANITY CHECK: tokenizer creates one token per residue
    # ------------------------------------------------------------------
    n_tokens_total = len(input_ids)

    n_cls = 0
    if tokenizer.cls_token_id is not None:
        n_cls = int((input_ids == tokenizer.cls_token_id).sum())

    n_eos = 0
    if getattr(tokenizer, "eos_token_id", None) is not None:
        n_eos = int((input_ids == tokenizer.eos_token_id).sum())

    n_residue_tokens = n_tokens_total - n_cls - n_eos

    if n_residue_tokens != len(sequence):
        raise RuntimeError(
            f"Tokenizer alignment failure: "
            f"sequence length={len(sequence)} "
            f"but residue token count={n_residue_tokens}"
        )
    
    if inputs["input_ids"].shape[1] > model.config.max_position_embeddings:
        log.warning("Tokenized sequence length %d exceeds model limit %d",
                     inputs["input_ids"].shape[1], model.config.max_position_embeddings)

    n_special_start = int((inputs["input_ids"][0] == tokenizer.cls_token_id).sum()) if \
        tokenizer.cls_token_id is not None else 0

    with torch.no_grad():
        out = model(**inputs)

    hidden_states = out.hidden_states

    expected_layers = model.config.num_hidden_layers + 1
    if len(hidden_states) != expected_layers:
        raise RuntimeError(
            f"Hidden-state mismatch: got "
            f"{len(hidden_states)} hidden states, "
            f"expected {expected_layers}"
        )
    
    n_tokens_with_specials = hidden_states[0].shape[1]
    n_special_end = n_tokens_with_specials - n_special_start - len(sequence)
    if n_special_end < 0:
        log.error("Token count mismatch for sequence of length %d (got %d tokens); "
                   "tokenizer alignment assumption failed for this model -- "
                   "inspect tokenizer output before trusting any probe result.",
                   len(sequence), n_tokens_with_specials)
        return None, n_truncated

    layers = []
    for h in hidden_states:
        h = h[0]
        stripped = h[n_special_start: n_special_start + len(sequence)]
        if stripped.shape[0] != len(sequence):
            raise RuntimeError(
                f"Post-strip alignment failure: "
                f"got {stripped.shape[0]} residues, "
                f"expected {len(sequence)}"
            )
        layers.append(stripped.cpu().numpy())
    return np.stack(layers, axis=0), n_truncated


# ----------------------------------------------------------------------
# Step 3: build the (embedding, label) dataset for one model
# ----------------------------------------------------------------------

def build_dataset_for_model(labels_dir: Path, pdb_ids: list[str], model_key: str,
                             device: str, include_flagged: bool, max_len: int,
                             label_columns: list[str]):
    """Returns X, y_by_column, groups, splits, base_ids, effective_max_len.
    Embeddings are extracted ONCE per model; every requested label_column
    is carried alongside the same embeddings."""
    tokenizer, model, n_layers, model_max_len = load_model(model_key, device)
    effective_max_len = max(1, min(max_len, model_max_len - 2))  # -2: room for CLS/EOS
    log.info("%s: model_max_len=%d, effective_max_len=%d",
              model_key, model_max_len, effective_max_len)

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
            # ------------------------------------------------------------------
            # SANITY CHECK: seq_index_map agrees with FASTA sequence
            # ------------------------------------------------------------------

            if "seq_index" in chain_rows.columns:
                for _, row in chain_rows.iterrows():
                    idx = int(row["seq_index"])

                    if idx < 0 or idx >= len(seq):
                        raise RuntimeError(
                            f"{pdb_id} chain {chain}: seq_index={idx} "
                            f"out of range for sequence length {len(seq)}"
                        )

                    fasta_base = seq[idx].upper()

                    if "base" in row:
                        label_base = str(row["base"]).upper()
                    else:
                        label_base = str(row["resname"]).upper()

                    if fasta_base != label_base:
                        raise RuntimeError(
                            f"{pdb_id} chain {chain}: "
                            f"FASTA[{idx}]={fasta_base} "
                            f"but label says {label_base}"
                        )

            emb, n_truncated = extract_layer_embeddings(tokenizer, model, seq, device,
                                                          max_len=effective_max_len)
            if emb is None:
                continue

            valid = chain_rows[chain_rows["seq_index"] < emb.shape[1]]
            total_before += len(chain_rows)
            total_after += len(valid)
            if len(valid) < len(chain_rows):
                log.info("%s chain %s: retained %d/%d residues after truncation",
                          pdb_id, chain, len(valid), len(chain_rows))
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
    groups = np.concatenate(groups_all)
    splits = np.concatenate(splits_all)
    base_ids = np.concatenate(base_all)

    for layer_idx, arr in X.items():
        if np.isnan(arr).any():
            raise RuntimeError(
                f"Layer {layer_idx}: NaNs detected in embeddings"
            )

    n_residues = len(groups)
    for layer_idx, arr in X.items():
        if arr.shape[0] != n_residues:
            raise RuntimeError(
                f"Layer {layer_idx}: "
                f"{arr.shape[0]} embeddings "
                f"but {n_residues} labels"
            )
        
    y_by_column = {col: np.concatenate(v) for col, v in y_by_column_lists.items()}

    log.info("%s: assembled %d residues across %d structures, %d layers, dim=%d",
              model_key, len(groups), len(set(groups)), n_layers, next(iter(X.values())).shape[1])
    if total_before:
        retained = total_after / total_before
        log.info("%s: retained %.2f%% of residues after tokenizer filtering (%d/%d)",
                  model_key, retained * 100.0, total_after, total_before)
        log.info("%s: dropped %.2f%% of residues due to model length limits",
                  model_key, 100.0 * (1.0 - retained))
    log.info("%s split distribution: train=%d val=%d test=%d", model_key,
              (splits == "train").sum(), (splits == "val").sum(), (splits == "test").sum())

    train_pdbs = set(groups[splits == "train"])
    val_pdbs = set(groups[splits == "val"])
    test_pdbs = set(groups[splits == "test"])

    assert len(train_pdbs & val_pdbs) == 0
    assert len(train_pdbs & test_pdbs) == 0
    assert len(val_pdbs & test_pdbs) == 0

    log.info(
        "Split sanity check passed: "
        "%d train structures, "
        "%d val structures, "
        "%d test structures",
        len(train_pdbs),
        len(val_pdbs),
        len(test_pdbs)
    )
    return X, y_by_column, groups, splits, base_ids, effective_max_len


# ----------------------------------------------------------------------
# Step 4: probes
# ----------------------------------------------------------------------

def _onehot_base_features(base_ids: np.ndarray) -> np.ndarray:
    return np.stack([(base_ids == b).astype(float) for b in BASES], axis=1)


def run_ridge_probe(X_train, y_train, X_eval, y_eval) -> dict:
    """Generic fit-on-first-arg, eval-on-second-arg ridge probe. Used both
    for the val-selection sweep (fit train, eval val) and for final
    reporting (fit train+val, eval test) -- the caller decides which
    split goes where."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xev = scaler.transform(X_eval)

    model = RidgeCV(alphas=np.logspace(-3, 3, 13))
    model.fit(Xtr, y_train)
    pred = model.predict(Xev)

    r2 = model.score(Xev, y_eval)
    r, _ = pearsonr(y_eval, pred) if len(set(y_eval)) > 1 else (np.nan, np.nan)
    return {"r2": float(r2), "pearson_r": float(r), "alpha": float(model.alpha_)}


def run_mlp_probe(X_train, y_train, X_eval, y_eval, hidden=(128, 32)) -> dict:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import pearsonr

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xev = scaler.transform(X_eval)

    model = MLPRegressor(hidden_layer_sizes=hidden, early_stopping=True,
                          max_iter=500, random_state=0)
    model.fit(Xtr, y_train)
    pred = model.predict(Xev)

    r2 = model.score(Xev, y_eval)
    r, _ = pearsonr(y_eval, pred) if len(set(y_eval)) > 1 else (np.nan, np.nan)
    return {"r2": float(r2), "pearson_r": float(r)}


def baseline_train_val_test(base_feats: np.ndarray, y: np.ndarray, splits: np.ndarray) -> dict:
    """Post-review fix: the baseline reported alongside the model's
    final_test row must go through the SAME train+val -> test protocol as
    the model, or the two numbers in that row aren't comparable (previously
    this used a train->val fit, which matches the per-layer diagnostic
    rows but not the final_test row it was actually attached to)."""
    trainval = (splits == "train") | (splits == "val")
    test = splits == "test"
    return run_ridge_probe(base_feats[trainval], y[trainval], base_feats[test], y[test])


def select_best_layer_and_test(X: dict, y: np.ndarray, splits: np.ndarray,
                                model_key: str, label_column: str) -> dict:
    """Select best layer on VAL (fit train, score val). Retrain that one
    layer on TRAIN+VAL. Report ONE held-out TEST score. This is the number
    that belongs in a paper -- not the val score that picked the layer."""
    train = splits == "train"
    val = splits == "val"
    test = splits == "test"
    if test.sum() == 0:
        raise RuntimeError(f"{model_key}/{label_column}: no test samples available")

    best_layer = None
    best_val_r2 = -np.inf
    for layer, X_layer in X.items():
        result = run_ridge_probe(X_layer[train], y[train], X_layer[val], y[val])
        if result["r2"] > best_val_r2:
            best_val_r2 = result["r2"]
            best_layer = layer

    trainval = train | val
    X_best = X[best_layer]
    final = run_ridge_probe(X_best[trainval], y[trainval], X_best[test], y[test])

    return {"best_layer": int(best_layer), "val_r2": float(best_val_r2),
            "test_r2": final["r2"], "test_pearson_r": final["pearson_r"]}


def probe_model(model_key: str, X: dict, y: np.ndarray, splits: np.ndarray,
                 base_ids: np.ndarray, linear_success_r2: float,
                 label_column: str, model_max_len) -> pd.DataFrame:
    """Probes one (model, label_column) pair across all layers."""
    valid_mask = ~np.isnan(y)
    if valid_mask.sum() < 20:
        log.warning("%s / %s: only %d non-NaN labels, skipping", model_key, label_column,
                    valid_mask.sum())
        return pd.DataFrame()

    y = y[valid_mask]
    if np.std(y) < 1e-8:
        log.warning(
            "%s / %s: target nearly constant",
            model_key,
            label_column
        )
        return pd.DataFrame()
    splits_f = splits[valid_mask]
    base_f = base_ids[valid_mask]
    X_f = {i: arr[valid_mask] for i, arr in X.items()}

    train_mask = splits_f == "train"
    val_mask = splits_f == "val"
    test_mask = splits_f == "test"
    if train_mask.sum() == 0 or val_mask.sum() == 0 or test_mask.sum() == 0:
        log.warning("%s / %s: missing train/val/test samples, skipping", model_key, label_column)
        return pd.DataFrame()

    log.info("%s / %s: train=%d val=%d test=%d", model_key, label_column,
              train_mask.sum(), val_mask.sum(), test_mask.sum())

    # Diagnostic (val-based) baseline -- matches the per-layer diagnostic
    # rows below, which are ALSO val-based (their r2 is the layer-selection
    # metric, not a final test number).
    base_feats = _onehot_base_features(base_f)
    baseline_val = run_ridge_probe(base_feats[train_mask], y[train_mask],
                                    base_feats[val_mask], y[val_mask])
    log.info("%s / %s: sequence-identity-only baseline (val) R^2=%.3f (r=%.3f)",
              model_key, label_column, baseline_val["r2"], baseline_val["pearson_r"])

    rows = []
    best_linear_r2 = -np.inf
    for layer_i, X_layer in X_f.items():
        lin = run_ridge_probe(X_layer[train_mask], y[train_mask], X_layer[val_mask], y[val_mask])
        rows.append({"model": model_key, "model_max_len": model_max_len,
                      "label_column": label_column, "layer": layer_i, "probe": "linear",
                      **lin, "baseline_r2": baseline_val["r2"], "n_residues": int(valid_mask.sum())})
        best_linear_r2 = max(best_linear_r2, lin["r2"])
        log.info("%s / %s layer %d [linear] R^2=%.3f r=%.3f", model_key, label_column,
                  layer_i, lin["r2"], lin["pearson_r"])

    if best_linear_r2 < linear_success_r2:
        log.info("%s / %s: best linear R^2=%.3f below threshold %.2f -- running MLP fallback",
                  model_key, label_column, best_linear_r2, linear_success_r2)
        for layer_i, X_layer in X_f.items():
            mlp = run_mlp_probe(X_layer[train_mask], y[train_mask], X_layer[val_mask], y[val_mask])
            rows.append({"model": model_key, "model_max_len": model_max_len,
                          "label_column": label_column, "layer": layer_i, "probe": "mlp",
                          **mlp, "baseline_r2": baseline_val["r2"], "n_residues": int(valid_mask.sum())})
            log.info("%s / %s layer %d [mlp] R^2=%.3f r=%.3f", model_key, label_column,
                      layer_i, mlp["r2"], mlp["pearson_r"])

    # --- Final held-out numbers: model AND baseline both go through the
    # identical train+val -> test protocol, so this row's two numbers are
    # directly comparable (post-review fix). ---
    final_test = select_best_layer_and_test(X_f, y, splits_f, model_key, label_column)
    baseline_test = baseline_train_val_test(base_feats, y, splits_f)

    rows.append({
        "model": model_key, "model_max_len": model_max_len, "label_column": label_column,
        "layer": final_test["best_layer"], "probe": "final_test",
        "r2": final_test["test_r2"], "pearson_r": final_test["test_pearson_r"],
        "baseline_r2": baseline_test["r2"], "n_residues": int(valid_mask.sum()),
    })
    log.info("%s / %s FINAL TEST: best layer=%d  test_R2=%.3f  test_r=%.3f  "
              "baseline_test_R2=%.3f", model_key, label_column, final_test["best_layer"],
              final_test["test_r2"], final_test["test_pearson_r"], baseline_test["r2"])

    return pd.DataFrame(rows)


# ----------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--models", nargs="+", default=list(MODEL_REGISTRY.keys()),
                     choices=list(MODEL_REGISTRY.keys()))
    ap.add_argument("--outdir", type=Path, default=Path("./probe_results"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--linear-success-r2", type=float, default=0.3)
    ap.add_argument("--label-columns", nargs="+",
                     default=[
                        "potential_mean_kT_e_4A",
                        "potential_mean_kT_e_8A",
                        "potential_mean_kT_e_12A",

                        "potential_mean_kT_e_4A_z",
                        "potential_mean_kT_e_8A_z",
                        "potential_mean_kT_e_12A_z",

                        "potential_at_c1prime_kT_e",
                        "potential_at_c1prime_kT_e_z",

                        "potential_at_phosphorus_kT_e",
                        "potential_at_phosphorus_kT_e_z",
                    ])
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
                results.to_csv(args.outdir / f"{model_key}_{safe_col}_probe_results.csv", index=False)
                all_results.append(results)
        except Exception as e:
            log.error("%s: probing failed: %s", model_key, e, exc_info=True)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.to_csv(args.outdir / "all_probe_results.csv", index=False)
        log.info("Wrote combined probe results to %s", args.outdir / "all_probe_results.csv")

        summary = (combined[combined["probe"] == "final_test"]
                   .sort_values("r2", ascending=False)
                   [["model", "label_column", "layer", "r2", "pearson_r", "baseline_r2"]])
        log.info("Final held-out test results:\n%s", summary.to_string(index=False))


if __name__ == "__main__":
    main()