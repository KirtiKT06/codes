"""
Centered / residualized re-run of the latent-manifold analysis
(PLS + CCA between RNA-LM embeddings and the electrostatic targets).   [v2]

WHY THIS SCRIPT EXISTS (plain-language version)
-----------------------------------------------
Each per-residue electrostatic label is really several things added together:

    label = (A) how electrostatic THIS WHOLE RNA is      <- chain-level "scale"
          + (B) what LETTER this residue is (A/C/G/U)    <- force-field charge bookkeeping
          + (C) what the residue's 3D surroundings do    <- the part we actually want

We remove (A) and (B) from the TARGET and then ask the same questions again:

    raw                      : the label as before (sanity check)
    centered                 : label minus its chain mean                    (removes A)
    centered_base_resid      : centered, then letter effect removed          (removes A, B)
    centered_nuisance_resid  : also removes chain length / distance to ends  (removes A, B, ends)

OPTIONAL (needs --structure-controls, built by build_structure_controls.py):
    centered_ss_gbm          : also removes what predicted base-pairing explains
    centered_struct_gbm      : also removes what real 3D packing/burial explains
These two use a small nonlinear model (gradient boosting) to do the removing, because
burial and pairing act on potential nonlinearly; a linear remover would understate them.

WHAT CHANGED IN v2 (important)
------------------------------
v1 removed the letter effect with ONE global regression fitted AFTER centering. Chains
differ in base composition, so that regression gave every chain a small leftover offset
that depended on its composition -- a chain-level signal an embedding can predict (the
embedding "knows" composition). The tell-tale in the v1 run: the within-chain shuffle
control was ~0 for `centered` but NOT for the residualized variants (C1' R^2 0.02-0.04,
CCA c1 0.2-0.3). v2 fixes this: the regressors are centered within each chain too (a
"fixed-effects" fit) and the result is re-centered, so EVERY non-raw variant has exactly
zero mean inside every chain. Expected after the fix: the within-chain shuffle control
is ~0 for all variants. If it is not, something chain-level is still leaking.

WHAT CHANGED IN v3
------------------
  * --dssr-controls  : CSV from build_dssr_controls.py (real base pairs, stems, loops, multiplets,
                       stacking, sugar pucker). Enables centered_dssr_gbm / centered_dssr_ctx_gbm.
  * --coulomb-table  : CSV from coulomb_residual_test.py (point-charge screened-Coulomb baseline).
                       Enables centered_coulomb_resid (linear) and centered_dssr_ctx_coulomb_gbm.
  * centered_ctx_gbm : removes what the +-2 neighbouring LETTERS predict (local sequence context).
  * --audit          : macro (chain-weighted) R^2, structure-level bootstrap CIs, an apples-to-apples
                       variance partition (same learner for controls / LM / both), and a nonlinear
                       LM probe. Writes audit_probe_r2.csv and audit_variance_partition.csv.
  * --swap-control   : fits PLS axes to NON-electrostatic structural targets and saves them as
                       variant `swap_geom`, so cross_model_latent_agreement.py can show whether
                       agreement between models is specific to electrostatics or generic structure.

WHAT CHANGED IN v4
------------------
  * --random-init [--random-seed N] : same architecture and tokenizer, RANDOM weights (never pretrained).
                       Use --layer to match the pretrained run. See latent_controls.py.
  * --family-table CSV --family-folds K --family-fold i : leave-family-out split. Whole RNA families go
                       to train or to test; structures that straddle are dropped. Needs a CSV with
                       pdb_id, chain, family (see build_family_table.py). Writes family_split_*.
                       Everything downstream (layer choice, PLS, audit) then uses this split.

IMPORTANT LIMITS
  * Linear removal of letter / length / end-distance; longer k-mer context is NOT removed.
  * Chain means are computed over the residues the LM actually saw
    (after truncation to the model's max length), complete-case across targets
    (and across the structure controls, when those are used).
  * Predicted pairing (ViennaRNA MFE) is an imperfect proxy for real pairing; the 3D
    features are the stronger control.
  * All variants share one layer, picked on --selection-variant.

OUTPUT (per variant, in <outdir>/<variant>/):
    pls_dimensionality_sweep.csv, cca_canonical_correlations.csv,
    latent_axis_interpretation.csv, negative_controls_{pls,cca}.csv
    nuisance_baseline_r2.csv   how much the simple features alone explain (linear variants)
    scores.csv.gz              per-residue PLS latent scores + nuisance (+ control) features,
                               consumed by cross_model_latent_agreement.py
and, when structure controls are used, <outdir>/control_explained_r2.csv: how much of the
centered target the controls alone explain on held-out structures (the bar for the LM).

Usage:
    python centered_latent_manifold.py \
        --labels-dir /data/rna_electrostats/phi_res \
        --model ernierna \
        --outdir /data/rna_electrostats/centered_latent_v2/ernierna \
        --device cuda
    # with structure controls:
        --structure-controls /data/rna_electrostats/structure_controls.csv.gz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("centered_latent_manifold")

# Re-use the pieces we already trust, so numbers stay comparable to earlier runs.
from rna_embedding_probe_v2 import run_ridge_probe, load_fasta_by_chain  # noqa: E402
from rna_probe_priority_experiments_extended import build_dataset_for_model_with_chains  # noqa: E402
from electrostatic_latent_manifold import (  # noqa: E402
    pls_dimensionality_sweep, cca_canonical_correlations, axis_interpretation, negative_controls,
)

BASE_TARGETS = [
    "potential_at_c1prime_kT_e",
    "potential_at_phosphorus_kT_e",
    "potential_mean_kT_e_4A",
    "potential_mean_kT_e_8A",
    "potential_mean_kT_e_12A",
]
# One extra target: "local minus global". NOTE it is a linear combination of two existing
# targets (4A - 12A), so it adds no new independent direction to Y; it is here because it
# reads naturally in the axis table.
DELTA_NAME = "local_minus_global_4A_12A"
ALL_TARGETS = BASE_TARGETS + [DELTA_NAME]

LINEAR_VARIANTS = ["raw", "centered", "centered_base_resid", "centered_nuisance_resid"]
# variants that need extra inputs: name -> (feature blocks used for removal, tables it needs)
#   blocks: F=letter/length/ends, ss=predicted pairing, struct=pairing+3D packing, dssr, ctx=+-k letters, coul
GBM_SPEC = {
    "centered_ss_gbm": (["F", "ss"], {"struct"}),
    "centered_struct_gbm": (["F", "struct"], {"struct"}),
    "centered_ctx_gbm": (["F", "ctx"], set()),
    "centered_dssr_gbm": (["F", "struct", "dssr"], {"struct", "dssr"}),
    "centered_dssr_ctx_gbm": (["F", "struct", "dssr", "ctx"], {"struct", "dssr"}),
    "centered_dssr_ctx_coulomb_gbm": (["F", "struct", "dssr", "ctx", "coul"], {"struct", "dssr", "coul"}),
}
GBM_VARIANTS = list(GBM_SPEC)
COULOMB_LINEAR = "centered_coulomb_resid"      # linear removal of the point-charge baseline (+letters)
ALL_VARIANTS = LINEAR_VARIANTS + GBM_VARIANTS + [COULOMB_LINEAR]
DEFAULT_VARIANTS = ["centered", "centered_base_resid", "centered_nuisance_resid"]
BASES = ["A", "C", "G", "U"]
SWAP_TARGETS = ["n_neighbors_8A", "burial_proxy", "dist_nearest_P_other", "prob_paired",
                "n_tertiary_neighbors_5A"]

KEYS = ["pdb_id", "chain", "seq_index"]
SS_COLS = ["is_paired_mfe", "prob_paired"]
GEOM_COLS = ["n_neighbors_5A", "n_neighbors_8A", "n_tertiary_neighbors_5A",
             "dist_nearest_P_other", "burial_proxy"]
CONTROL_COLS = SS_COLS + GEOM_COLS


# ----------------------------------------------------------------------
# Building blocks
# ----------------------------------------------------------------------

def within_chain_center(Y: np.ndarray, chain_ids: np.ndarray) -> np.ndarray:
    """Subtract each chain's own mean from every column of Y.
    Intuition: 'how much higher/lower than the average residue of THIS RNA'."""
    df = pd.DataFrame(Y)
    means = df.groupby(np.asarray(chain_ids)).transform("mean")
    return (df - means).to_numpy()


def build_nuisance_features(pdb_groups, chain_groups, seq_indices, base_ids, labels_dir):
    """Per-residue features that are NOT electrostatics-of-the-environment:
    letter identity (4 one-hot), chain length, position, distance to chain ends.
    Returns (F, names). The first 4 columns are the letter one-hot."""
    # chain length from the FASTA (true length, not the truncated length the LM saw)
    fasta_cache: dict[str, dict] = {}
    for pdb in np.unique(pdb_groups):
        fasta_cache[pdb] = load_fasta_by_chain(labels_dir, pdb)
    chain_len: dict[str, int] = {}
    for cg in np.unique(chain_groups):
        pdb, chain = cg.split("::", 1)
        chain_len[cg] = len(fasta_cache[pdb].get(chain, ""))
    L = np.array([chain_len[c] for c in chain_groups], dtype=float)
    i = seq_indices.astype(float)
    L = np.maximum(L, i + 1)  # safety: length can never be shorter than the index we hold

    dist5 = i
    dist3 = L - 1 - i
    dist_end = np.minimum(dist5, dist3)
    onehot = np.stack([(base_ids == b).astype(float) for b in BASES], axis=1)
    F = np.column_stack([
        onehot,
        np.log1p(L),
        i / np.maximum(L - 1, 1),
        np.log1p(dist5),
        np.log1p(dist3),
        np.log1p(dist_end),
    ])
    names = [f"nuis_base_{b}" for b in BASES] + [
        "nuis_log_len", "nuis_rel_pos", "nuis_log_dist5", "nuis_log_dist3", "nuis_log_dist_end"]
    return F, names


def load_controls(path: Path, pdb_groups, chain_groups, seq_indices, required=None, prefix=None):
    """Look up a control table (structure / DSSR / Coulomb) for every residue we hold.
    required: exact column list the table must contain (structure controls);
    prefix:   keep only columns that start with this (Coulomb table also holds label columns);
    neither:  keep every non-key column (DSSR table).
    Returns (C, ok, cols): C is (n, len(cols)); ok marks rows where ALL columns exist."""
    ctrl = pd.read_csv(path, dtype={"pdb_id": str, "chain": str})
    if required is not None:
        missing = [c for c in KEYS + required if c not in ctrl.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        cols = list(required)
    elif prefix is not None:
        cols = [c for c in ctrl.columns if c.startswith(prefix)]
        if not cols:
            raise ValueError(f"{path} has no columns starting with '{prefix}'")
    else:
        cols = [c for c in ctrl.columns if c not in KEYS]
    ctrl["seq_index"] = ctrl["seq_index"].astype(int)
    ctrl = ctrl.drop_duplicates(KEYS)
    keys = pd.DataFrame({
        "pdb_id": np.asarray(pdb_groups).astype(str),
        "chain": [c.split("::", 1)[1] for c in chain_groups],
        "seq_index": np.asarray(seq_indices).astype(int),
    })
    merged = keys.merge(ctrl[KEYS + cols], on=KEYS, how="left")  # left merge keeps row order
    C = merged[cols].to_numpy(dtype=float)
    ok = ~np.isnan(C).any(axis=1)
    return C, ok, cols


def build_context_features(chain_groups, seq_indices, labels_dir, pdb_groups, window=2):
    """One-hot letters at offsets -window..-1,+1..+window around each residue (zeros past a chain end).
    Local SEQUENCE context only -- the thing a language model is built to read. Removing what it
    predicts tests whether the LM adds anything beyond 'which letters are next door'."""
    codes = {"A": 0, "C": 1, "G": 2, "U": 3}
    fasta: dict[str, dict] = {p: load_fasta_by_chain(labels_dir, p) for p in np.unique(pdb_groups)}
    n = len(seq_indices)
    offsets = [o for o in range(-window, window + 1) if o != 0]
    ctx = np.zeros((n, 4 * len(offsets)))
    names = [f"ctx_{'m' if o < 0 else 'p'}{abs(o)}_{b}" for o in offsets for b in BASES]
    groups = pd.Series(np.arange(n)).groupby(np.asarray(chain_groups)).indices
    for cg, idx in groups.items():
        pdb, chain = cg.split("::", 1)
        seq = fasta[pdb].get(chain, "").upper().replace("T", "U")
        arr = np.array([codes.get(ch, 4) for ch in seq], dtype=int)
        L = len(arr)
        si = np.asarray(seq_indices)[idx].astype(int)
        for oi, o in enumerate(offsets):
            pos = si + o
            inside = (pos >= 0) & (pos < L)
            c = np.where(inside, arr[np.clip(pos, 0, max(L - 1, 0))] if L else 4, 4)
            for b in range(4):
                ctx[idx, oi * 4 + b] = (c == b)
    return ctx, names


def residualize(Y: np.ndarray, F: np.ndarray, splits: np.ndarray, chain_ids: np.ndarray) -> np.ndarray:
    """Remove what a LINEAR model on F can predict from Y, WITHIN chains.

    F is centered within each chain before fitting, so the model only learns
    'residue differs from its own chain's average in feature F -> differs in Y'.
    It therefore cannot hand any chain a composition-dependent offset (the v1 leak).
    The fit uses TRAIN rows only and is then applied to every row, so val/test are honest.
    The result is re-centered within chain, so every chain has exactly zero mean."""
    from sklearn.linear_model import LinearRegression
    F_c = within_chain_center(F, chain_ids)
    train = splits == "train"
    model = LinearRegression().fit(F_c[train], Y[train])
    return within_chain_center(Y - model.predict(F_c), chain_ids)


def residualize_nonlinear(Y: np.ndarray, F: np.ndarray, splits: np.ndarray, chain_ids: np.ndarray,
                          group_ids: np.ndarray, seed: int = 0, n_folds: int = 5):
    """Same idea as `residualize`, but the remover is gradient boosting, so it can strip out
    NONLINEAR effects of the controls (e.g. burial saturating at high packing).

    Train rows get OUT-OF-FOLD predictions (folds split by structure), val/test rows get
    predictions from a model fit on all train rows. Without this, train residuals would be
    artificially small (the remover memorized them) and train/test would not match.
    Returns (Y_residual, predictions) -- the predictions let us report how much the controls
    alone explain."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    X = np.column_stack([F, within_chain_center(F, chain_ids)])  # raw + within-chain view
    train_idx = np.flatnonzero(splits == "train")
    other_idx = np.flatnonzero(splits != "train")
    pred = np.zeros_like(Y, dtype=float)

    def make():
        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                             min_samples_leaf=200, early_stopping=False,
                                             random_state=seed)

    for j in range(Y.shape[1]):
        gkf = GroupKFold(n_splits=n_folds)
        for tr, te in gkf.split(train_idx, groups=group_ids[train_idx]):
            m = make().fit(X[train_idx[tr]], Y[train_idx[tr], j])
            pred[train_idx[te], j] = m.predict(X[train_idx[te]])
        m = make().fit(X[train_idx], Y[train_idx, j])
        pred[other_idx, j] = m.predict(X[other_idx])
        log.info("    nonlinear remover fitted for target %d/%d", j + 1, Y.shape[1])
    return within_chain_center(Y - pred, chain_ids), pred


def pick_layer(X: dict, Y: np.ndarray, splits: np.ndarray) -> int:
    """Choose the layer with the best TRAIN->VAL ridge R^2, averaged over all
    target columns (so no single target, e.g. C1', decides the layer)."""
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    train, val = splits == "train", splits == "val"
    best_layer, best = None, -np.inf
    for layer, H in X.items():
        scaler = StandardScaler().fit(H[train])
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(scaler.transform(H[train]), Y[train])
        r2 = r2_score(Y[val], model.predict(scaler.transform(H[val])), multioutput="uniform_average")
        log.info("  layer %2d: mean val R^2 over targets = %.3f", layer, r2)
        if r2 > best:
            best, best_layer = r2, layer
    log.info("Selected layer %d (mean val R^2 = %.3f)", best_layer, best)
    return best_layer


def nuisance_baseline(F: np.ndarray, Y: np.ndarray, splits: np.ndarray, targets: list[str],
                      outdir: Path, label: str) -> None:
    """How much of each target can the simple features ALONE explain on held-out data?
    This is the bar the LM has to clear before we credit it with anything."""
    trainval, test = splits != "test", splits == "test"
    rows = []
    for j, t in enumerate(targets):
        r2 = run_ridge_probe(F[trainval], Y[trainval, j], F[test], Y[test, j])["r2"]
        rows.append({"variant": label, "target": t, "nuisance_only_test_r2": r2})
        log.info("  [%s] nuisance-only baseline %-30s test R^2 = %.3f", label, t, r2)
    pd.DataFrame(rows).to_csv(outdir / "nuisance_baseline_r2.csv", index=False)


def save_scores(path: Path, H, pls, meta: dict, F, F_names, Y, targets):
    """Per-residue latent scores (what the PLS axes read off each residue) plus the keys
    needed to line up the SAME residue across different models."""
    Z = pls.transform(H)  # PLS was fit on train+val, so test rows are out-of-sample
    out = pd.DataFrame(meta)
    for a in range(Z.shape[1]):
        out[f"Z{a + 1}"] = Z[:, a]
    for c, name in enumerate(F_names):
        out[name] = F[:, c]
    for j, t in enumerate(targets):
        out[f"y__{t}"] = Y[:, j]
    out.to_csv(path, index=False, compression="gzip")
    log.info("  wrote %s (%d rows, %d latent axes)", path, len(out), Z.shape[1])


# ----------------------------------------------------------------------
# Audit: macro R^2, bootstrap, variance partition, nonlinear LM probe
# ----------------------------------------------------------------------

def structure_stats(y, pred, chain_ids, pdb_ids) -> pd.DataFrame:
    """Per-structure sufficient statistics so R^2 can be bootstrapped cheaply.
    sse/sst  : micro (every residue counts equally)
    sse_m/sst_m : each CHAIN contributes its mean squared error/variance (every chain counts equally),
                  so one 3000-nt rRNA cannot outvote 500 tRNAs."""
    df = pd.DataFrame({"pdb": pdb_ids, "chain": chain_ids, "se": (y - pred) ** 2, "s2": y ** 2})
    g = df.groupby(["pdb", "chain"]).agg(sse=("se", "sum"), sst=("s2", "sum"), n=("se", "size")).reset_index()
    g["sse_m"], g["sst_m"] = g["sse"] / g["n"], g["sst"] / g["n"]
    return g.groupby("pdb")[["sse", "sst", "sse_m", "sst_m"]].sum().sort_index()


def chain_stats(y, pred, chain_ids, pdb_ids) -> pd.DataFrame:
    """Per-CHAIN sums (pdb, chain, n, sse, sst). Saved by the audit so later analyses (chain-length bins,
    paired bootstraps between runs) can be done offline without refitting anything."""
    df = pd.DataFrame({"pdb": pdb_ids, "chain": chain_ids, "se": (y - pred) ** 2, "s2": y ** 2})
    return df.groupby(["pdb", "chain"]).agg(sse=("se", "sum"), sst=("s2", "sum"), n=("se", "size")).reset_index()


def r2_micro_macro(stats: pd.DataFrame, w: np.ndarray):
    return (1.0 - float(w @ stats["sse"].to_numpy()) / float(w @ stats["sst"].to_numpy()),
            1.0 - float(w @ stats["sse_m"].to_numpy()) / float(w @ stats["sst_m"].to_numpy()))


def run_audit(H, Y_cen, blocks: dict, splits, chain_groups, pdb_groups, targets_all, audit_targets,
              outdir: Path, seed=0, n_boot=500, pca_dim=64):
    """Same learner (gradient boosting) for controls, LM and both, so the partition is not
    biased by 'linear probe vs nonlinear controls'. The linear full-embedding ridge is reported
    alongside as a separate row."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler

    train, test = splits == "train", splits == "test"
    rng = np.random.default_rng(seed)

    scaler = StandardScaler().fit(H[train])
    pca = PCA(n_components=min(pca_dim, H.shape[1]), random_state=seed).fit(scaler.transform(H[train]))
    Hp = pca.transform(scaler.transform(H))
    lm_feats = np.column_stack([Hp, within_chain_center(Hp, chain_groups)])
    log.info("AUDIT: LM reduced to %d PCs (%.1f%% of embedding variance on train)",
             Hp.shape[1], 100 * pca.explained_variance_ratio_.sum())

    def view(names):
        M = np.column_stack([blocks[n] for n in names])
        return np.column_stack([M, within_chain_center(M, chain_groups)])

    ctrl_sets = {"controls_geom": [n for n in ("F", "struct", "dssr", "ctx") if n in blocks]}
    if "coul" in blocks:
        ctrl_sets["controls_geom_coulomb"] = ctrl_sets["controls_geom"] + ["coul"]

    def gbm_pred(X, y):
        m = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                          min_samples_leaf=200, early_stopping=False, random_state=seed)
        return m.fit(X[train], y[train]).predict(X[test])

    # linear full-embedding ridge (all targets at once)
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(scaler.transform(H[train]), Y_cen[train])
    ridge_pred = ridge.predict(scaler.transform(H[test]))

    pdb_t, chain_t = np.asarray(pdb_groups)[test], np.asarray(chain_groups)[test]
    rows, part_rows, chain_rows = [], [], []
    for t in audit_targets:
        j = targets_all.index(t)
        y = Y_cen[:, j]
        preds = {"lm_ridge_full": ridge_pred[:, j], "lm_gbm_pca": gbm_pred(lm_feats, y)}
        for cname, names in ctrl_sets.items():
            Xc = view(names)
            preds[cname] = gbm_pred(Xc, y)
            preds[f"{cname}+lm"] = gbm_pred(np.column_stack([Xc, lm_feats]), y)
            log.info("  AUDIT %-28s %-24s fitted", t, cname)
        stats = {k: structure_stats(y[test], p, chain_t, pdb_t) for k, p in preds.items()}
        for k, p in preds.items():
            chain_rows.append(chain_stats(y[test], p, chain_t, pdb_t).assign(target=t, predictors=k))
        S = len(next(iter(stats.values())))
        boots = [np.bincount(rng.integers(0, S, S), minlength=S).astype(float) for _ in range(n_boot)]
        ones = np.ones(S)

        def ci(fn):
            vals = np.array([fn(w) for w in boots])
            return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))

        for k, st in stats.items():
            mi, ma = r2_micro_macro(st, ones)
            lo_i, hi_i = ci(lambda w: r2_micro_macro(st, w)[0])
            lo_a, hi_a = ci(lambda w: r2_micro_macro(st, w)[1])
            rows.append({"target": t, "predictors": k, "micro_r2": mi, "micro_lo": lo_i, "micro_hi": hi_i,
                         "macro_r2": ma, "macro_lo": lo_a, "macro_hi": hi_a, "n_test_structures": S})
            log.info("  AUDIT %-26s %-28s micro R2 %.3f [%.3f, %.3f] | macro R2 %.3f [%.3f, %.3f]",
                     t, k, mi, lo_i, hi_i, ma, lo_a, hi_a)

        # variance partition (paired bootstrap: same resampled structures for every model)
        for cname in ctrl_sets:
            for kind, idx in (("micro", 0), ("macro", 1)):
                def comp(w):
                    rc = r2_micro_macro(stats[cname], w)[idx]
                    rl = r2_micro_macro(stats["lm_gbm_pca"], w)[idx]
                    rb = r2_micro_macro(stats[f"{cname}+lm"], w)[idx]
                    return np.array([rb - rc, rb - rl, rc + rl - rb, 1.0 - rb])  # unique LM, unique ctrl, shared, unexplained
                point = comp(ones)
                bs = np.array([comp(w) for w in boots])
                for name, v, lo, hi in zip(["unique_lm", "unique_controls", "shared", "unexplained"], point,
                                           np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)):
                    part_rows.append({"target": t, "controls": cname, "weighting": kind, "component": name,
                                      "value": float(v), "lo": float(lo), "hi": float(hi)})
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "audit_probe_r2.csv", index=False)
    pd.concat(chain_rows, ignore_index=True).to_csv(outdir / "audit_chain_stats.csv.gz", index=False,
                                                    compression="gzip")
    pd.DataFrame(part_rows).to_csv(outdir / "audit_variance_partition.csv", index=False)
    log.info("AUDIT written to %s", outdir)


def run_swap_control(H, F, C_struct, chain_groups, splits, meta, F_save, F_save_names, k, outdir: Path):
    """Label-swap control for the cross-model agreement test. Same embeddings, same PLS machinery,
    but the targets are NON-electrostatic structural quantities (packing, burial, distance to the
    nearest other phosphate, predicted pairing, tertiary contacts), centered within chain and
    stripped of letter identity. If models agree on THESE axes as strongly as on the electrostatic
    axes, the agreement is generic 'structure' agreement, not something specific to electrostatics."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import r2_score
    j = [CONTROL_COLS.index(c) for c in SWAP_TARGETS]
    Y = within_chain_center(C_struct[:, j], chain_groups)
    Y = residualize(Y, F[:, :4], splits, chain_groups)
    fit, test = splits != "test", splits == "test"
    pls = PLSRegression(n_components=k, scale=True).fit(H[fit], Y[fit])
    r2 = r2_score(Y[test], pls.predict(H[test]), multioutput="raw_values")
    for c, v in zip(SWAP_TARGETS, r2):
        log.info("  swap target %-26s PLS(k=%d) test R^2 = %.3f", c, k, v)
    vdir = outdir / "swap_geom"
    vdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"target": SWAP_TARGETS, "pls_test_r2": r2}).to_csv(vdir / "swap_pls_test_r2.csv", index=False)
    save_scores(vdir / "scores.csv.gz", H, pls, meta, F_save, F_save_names, Y, [f"swap_{c}" for c in SWAP_TARGETS])


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def run_spike_in(H, Y_cen, blocks, splits, chain_groups, pdb_groups, targets_all, args):
    """Power analysis. Adds to the label a synthetic signal that (i) is a function of the LM embedding only and
    (ii) has been made unpredictable from the controls, with a known share f of the final label variance, then
    reruns the audit and reports how much `unique_lm` it recovers."""
    from sklearn.decomposition import PCA
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.preprocessing import StandardScaler

    train, test = splits == "train", splits == "test"
    scaler = StandardScaler().fit(H[train])
    pca = PCA(n_components=min(args.pca_dim, H.shape[1]), random_state=args.seed).fit(scaler.transform(H[train]))
    Hp = pca.transform(scaler.transform(H))
    Hp = Hp / Hp[train].std(0)
    names = [n for n in ("F", "struct", "dssr", "ctx", "coul") if n in blocks]
    M = np.column_stack([blocks[n] for n in names])
    Xc = np.column_stack([M, within_chain_center(M, chain_groups)])
    k = max(1, min(16, Hp.shape[1] // 2))
    all_rows = []
    for stype in args.spike_type:
        rng = np.random.default_rng(args.seed + 17)
        w1, w2 = rng.normal(size=k), rng.normal(size=k)
        if stype == "linear":
            s = Hp[:, :k] @ w1                                   # a probe that is linear in the PCs can find it
        else:
            s = np.tanh(Hp[:, :k] @ w1) * (Hp[:, k:2 * k] @ w2)  # an interaction: needs a nonlinear probe
        s = within_chain_center(s[:, None], chain_groups)[:, 0]
        gb = HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                           min_samples_leaf=200, early_stopping=False, random_state=args.seed)
        s_perp = s - gb.fit(Xc[train], s[train]).predict(Xc)    # remove whatever the controls can predict
        r2_ctrl = 1.0 - float(np.var(s_perp[test]) / max(np.var(s[test]), 1e-12))
        s_perp = s_perp / s_perp[train].std()
        log.info("SPIKE-IN (%s): controls explained %.1f%% of the raw synthetic signal before it was removed",
                 stype, 100 * r2_ctrl)
        for t in args.spike_targets:
            j = targets_all.index(t)
            sd = float(Y_cen[train, j].std())
            for f in args.spike_fractions:
                Y_s = Y_cen.copy()
                Y_s[:, j] = Y_cen[:, j] + sd * np.sqrt(f / (1.0 - f)) * s_perp   # signal = share f of total variance
                sub = args.outdir / f"spike_{stype}_{t}_f{f:g}"
                run_audit(H, Y_s, blocks, splits, chain_groups, pdb_groups, targets_all, [t], sub,
                          seed=args.seed, n_boot=args.spike_n_boot, pca_dim=args.pca_dim)
                P = pd.read_csv(sub / "audit_variance_partition.csv")
                P = P[P.component == "unique_lm"].assign(injected=f, spike_type=stype,
                                                         control_explained_raw_signal=r2_ctrl)
                all_rows.append(P)
                for _, r in P.iterrows():
                    log.info("  spike %-11s %-26s f=%-6g %-24s %-5s recovered unique_lm = %+.4f [%+.4f, %+.4f]",
                             stype, t, f, r.controls, r.weighting, r.value, r.lo, r.hi)
    out = pd.concat(all_rows, ignore_index=True)
    out.to_csv(args.outdir / "spike_in_summary.csv", index=False)
    log.info("SPIKE-IN written to %s", args.outdir / "spike_in_summary.csv")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--structure-controls", type=Path, default=None,
                    help="CSV from build_structure_controls.py (pairing + packing/burial).")
    ap.add_argument("--dssr-controls", type=Path, default=None, help="CSV from build_dssr_controls.py")
    ap.add_argument("--coulomb-table", type=Path, default=None, help="CSV from coulomb_residual_test.py")
    ap.add_argument("--context-window", type=int, default=2, help="+-k neighbouring letters for *_ctx_* variants")
    ap.add_argument("--variants", nargs="+", default=None, choices=ALL_VARIANTS,
                    help="Default: centered, centered_base_resid, centered_nuisance_resid, plus every "
                         "extra variant whose tables you supplied.")
    ap.add_argument("--layer", type=int, default=None, help="Skip layer selection and use this layer")
    ap.add_argument("--selection-variant", default="centered_base_resid", choices=ALL_VARIANTS,
                    help="Which target variant decides the layer when --layer is not given")
    ap.add_argument("--pls-components", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10, 15])
    ap.add_argument("--interpret-k", type=int, default=5)
    ap.add_argument("--cca-components", type=int, default=5)
    ap.add_argument("--audit", action="store_true",
                    help="Macro R^2 + bootstrap CIs + shared/unique variance partition + nonlinear LM probe")
    ap.add_argument("--audit-targets", nargs="+",
                    default=["potential_at_c1prime_kT_e", "potential_mean_kT_e_4A", "potential_mean_kT_e_12A"])
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--spike-fractions", type=float, nargs="*", default=[],
                    help="Power analysis: inject a synthetic LM-only signal with these shares of the label variance "
                         "and report how much unique_lm the audit recovers (requires --audit)")
    ap.add_argument("--spike-type", nargs="+", default=["linear"], choices=["linear", "interaction"])
    ap.add_argument("--spike-targets", nargs="+", default=["potential_mean_kT_e_12A"])
    ap.add_argument("--spike-n-boot", type=int, default=100)
    ap.add_argument("--swap-control", action="store_true",
                    help="Also save non-electrostatic PLS axes as variant `swap_geom` (needs --structure-controls)")
    ap.add_argument("--random-init", action="store_true",
                    help="Control: random (not pretrained) weights, same architecture")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--family-table", type=Path, default=None,
                    help="CSV pdb_id,chain,family -> leave-family-out split (see build_family_table.py)")
    ap.add_argument("--family-folds", type=int, default=5)
    ap.add_argument("--family-fold", type=int, default=0, help="which fold is the held-out test set")
    ap.add_argument("--family-straddle", choices=["drop", "longest"], default="drop")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--limit-structures", type=int, default=None)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    have = {"struct": args.structure_controls is not None, "dssr": args.dssr_controls is not None,
            "coul": args.coulomb_table is not None}
    have_set = {k for k, v in have.items() if v}

    def available(v):
        if v == COULOMB_LINEAR:
            return "coul" in have_set
        return v not in GBM_SPEC or GBM_SPEC[v][1] <= have_set

    if args.variants is None:
        args.variants = DEFAULT_VARIANTS + ([v for v in GBM_VARIANTS + [COULOMB_LINEAR] if available(v)]
                                             if have_set else [])
    unavailable = [v for v in set(args.variants + [args.selection_variant]) if not available(v)]
    if unavailable:
        need = {v: sorted(GBM_SPEC[v][1]) if v in GBM_SPEC else ["coul"] for v in unavailable}
        ap.error(f"these variants need extra tables: {need}  (flags: struct=--structure-controls, "
                 f"dssr=--dssr-controls, coul=--coulomb-table)")
    if args.swap_control and not have["struct"]:
        ap.error("--swap-control needs --structure-controls")

    import torch
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    pdb_ids = sorted(pd.read_csv(args.labels_dir / "all_residue_potentials.csv",
                                 low_memory=False)["pdb_id"].unique())
    if args.limit_structures:
        pdb_ids = pdb_ids[: args.limit_structures]

    if args.random_init:
        from latent_controls import install_random_init_loader
        install_random_init_loader(args.random_seed)
        (args.outdir / "RANDOM_INIT.txt").write_text(f"random-initialised weights, seed {args.random_seed}\n")

    # ---- 1. embeddings + raw labels (same builder as every earlier run) ----
    X, y_by_col, pdb_groups, chain_groups, splits, base_ids, resnums, seq_indices, eff_max_len = \
        build_dataset_for_model_with_chains(args.labels_dir, pdb_ids, args.model, device,
                                            args.include_flagged, args.max_len, BASE_TARGETS)

    # ---- 2. complete-case rows: a residue is kept only if ALL base targets exist ----
    valid = np.logical_and.reduce([~np.isnan(y_by_col[t]) for t in BASE_TARGETS])
    log.info("Complete-case residues: %d of %d", int(valid.sum()), len(valid))
    X = {i: arr[valid] for i, arr in X.items()}
    pdb_groups, chain_groups = pdb_groups[valid], chain_groups[valid]
    splits, base_ids, seq_indices = splits[valid], base_ids[valid], seq_indices[valid]
    Y_raw = np.column_stack([y_by_col[t][valid] for t in BASE_TARGETS])
    delta = y_by_col["potential_mean_kT_e_4A"][valid] - y_by_col["potential_mean_kT_e_12A"][valid]
    Y_raw = np.column_stack([Y_raw, delta])

    # ---- 2b. optional tables (restrict EVERYTHING to rows that have every table you supplied) ----
    tables: dict[str, tuple] = {}
    if have["struct"]:
        tables["struct"] = load_controls(args.structure_controls, pdb_groups, chain_groups, seq_indices,
                                         required=CONTROL_COLS)
    if have["dssr"]:
        tables["dssr"] = load_controls(args.dssr_controls, pdb_groups, chain_groups, seq_indices)
    if have["coul"]:
        tables["coul"] = load_controls(args.coulomb_table, pdb_groups, chain_groups, seq_indices, prefix="dh_")
    blocks_raw: dict[str, np.ndarray] = {}
    block_cols: dict[str, list] = {}
    if tables:
        ok = np.logical_and.reduce([t[1] for t in tables.values()])
        for name, (_, okn, cols) in tables.items():
            log.info("Table '%s' covers %d of %d residues", name, int(okn.sum()), len(okn))
        log.info("Rows with EVERY supplied table: %d of %d (dropping %d); all variants use these rows",
                 int(ok.sum()), len(ok), int((~ok).sum()))
        if ok.mean() < 0.8:
            log.warning("Fewer than 80%% of residues have all tables -- check that the key columns "
                        "(pdb_id, chain, seq_index) match.")
        X = {i: arr[ok] for i, arr in X.items()}
        pdb_groups, chain_groups = pdb_groups[ok], chain_groups[ok]
        splits, base_ids, seq_indices = splits[ok], base_ids[ok], seq_indices[ok]
        Y_raw = Y_raw[ok]
        for name, (M, _, cols) in tables.items():
            blocks_raw[name] = M[ok]
            block_cols[name] = cols
        if "struct" in blocks_raw:
            blocks_raw["ss"] = blocks_raw["struct"][:, :len(SS_COLS)]

    # ---- 2c. optional leave-family-out split (replaces the structure-level split) ----
    if args.family_table is not None:
        from latent_controls import build_family_split, load_family_table, save_family_split_report
        new_splits, keep, fsum, finfo = build_family_split(
            pdb_groups, chain_groups, splits, load_family_table(args.family_table),
            args.family_folds, args.family_fold, args.family_straddle)
        save_family_split_report(args.outdir, fsum, finfo)
        X = {i: arr[keep] for i, arr in X.items()}
        pdb_groups, chain_groups = pdb_groups[keep], chain_groups[keep]
        splits, base_ids, seq_indices = new_splits[keep], base_ids[keep], seq_indices[keep]
        Y_raw = Y_raw[keep]
        blocks_raw = {k: v[keep] for k, v in blocks_raw.items()}
        for part in ("train", "val", "test"):
            if not np.any(splits == part):
                raise RuntimeError(f"family split left no '{part}' residues; change --family-folds/--family-fold")

    # ---- 3. nuisance features + target variants ----
    F, F_names = build_nuisance_features(pdb_groups, chain_groups, seq_indices, base_ids, args.labels_dir)
    blocks_raw["F"] = F
    if any(("ctx" in GBM_SPEC.get(v, ([], set()))[0]) for v in args.variants + [args.selection_variant]) \
            or args.audit:
        blocks_raw["ctx"], ctx_names = build_context_features(chain_groups, seq_indices, args.labels_dir,
                                                              pdb_groups, args.context_window)
        block_cols["ctx"] = ctx_names
    Y_cen = within_chain_center(Y_raw, chain_groups)
    variants = {
        "raw": Y_raw,
        "centered": Y_cen,
        "centered_base_resid": residualize(Y_cen, F[:, :4], splits, chain_groups),
        "centered_nuisance_resid": residualize(Y_cen, F, splits, chain_groups),
    }
    if have_set:
        from sklearn.metrics import r2_score
        wanted = set(args.variants) | {args.selection_variant}
        test = splits == "test"
        ctrl_rows = []

        def log_explained(vname, pred):
            for j, t in enumerate(ALL_TARGETS):
                r2 = float(r2_score(Y_cen[test, j], pred[test, j]))
                ctrl_rows.append({"variant": vname, "target": t, "controls_only_test_r2": r2})
                log.info("  [%s] controls alone explain %-30s test R^2 = %.3f", vname, t, r2)

        if COULOMB_LINEAR in wanted:
            log.info("=== Linear removal of the point-charge baseline: '%s' ===", COULOMB_LINEAR)
            Fv = np.column_stack([F[:, :4], blocks_raw["coul"]])
            variants[COULOMB_LINEAR] = residualize(Y_cen, Fv, splits, chain_groups)
            log_explained(COULOMB_LINEAR, Y_cen - variants[COULOMB_LINEAR])
        for vname, (names, _) in GBM_SPEC.items():
            if vname not in wanted:
                continue
            log.info("=== Nonlinear removal for variant '%s' (blocks: %s) ===", vname, "+".join(names))
            Fv = np.column_stack([blocks_raw[n] for n in names])
            Y_res, pred = residualize_nonlinear(Y_cen, Fv, splits, chain_groups, pdb_groups, args.seed)
            variants[vname] = Y_res
            log_explained(vname, pred)
        pd.DataFrame(ctrl_rows).to_csv(args.outdir / "control_explained_r2.csv", index=False)

    # ---- 4. one shared layer for every variant (so variants are comparable) ----
    if args.layer is not None:
        layer = args.layer
    else:
        log.info("=== Layer selection on variant '%s' ===", args.selection_variant)
        layer = pick_layer(X, variants[args.selection_variant], splits)
    H = X[layer]

    meta = {"pdb_id": pdb_groups, "chain": [c.split("::", 1)[1] for c in chain_groups],
            "seq_index": seq_indices, "split": splits}
    save_blocks, save_names = [F], list(F_names)
    prefix = {"struct": "ctrl_", "dssr": "dssr_", "ctx": "", "coul": "coul_"}
    for bname in ("struct", "dssr", "ctx", "coul"):
        if bname in blocks_raw:
            save_blocks.append(blocks_raw[bname])
            save_names += [(c if bname == "ctx" else f"{prefix[bname]}{c}") for c in block_cols[bname]]
    F_save = np.column_stack(save_blocks)

    # ---- 5. run the analysis for each variant ----
    for name in args.variants:
        Y = variants[name]
        vdir = args.outdir / name
        vdir.mkdir(parents=True, exist_ok=True)
        log.info("################ VARIANT: %s (layer %d) ################", name, layer)

        # What would the simple features alone get on THIS target? (the bar to clear)
        # Centered variants use within-chain-centered features, matching how they were removed.
        # (The nonlinear variants report control_explained_r2.csv instead.)
        if name in LINEAR_VARIANTS:
            Fb = F if name == "raw" else within_chain_center(F, chain_groups)
            nuisance_baseline(Fb, Y, splits, ALL_TARGETS, vdir, name)

        pls_fitted = pls_dimensionality_sweep(H, Y, splits, ALL_TARGETS, args.pls_components, vdir)
        cca_canonical_correlations(H, Y, splits, args.cca_components, vdir)
        axis_interpretation(pls_fitted, args.interpret_k, H, Y, splits, ALL_TARGETS, vdir)
        negative_controls(H, Y, splits, chain_groups, ALL_TARGETS,
                          args.interpret_k, args.cca_components, vdir, seed=args.seed)
        save_scores(vdir / "scores.csv.gz", H, pls_fitted[args.interpret_k], meta,
                    F_save, save_names, Y, ALL_TARGETS)

    if args.swap_control:
        log.info("################ SWAP CONTROL (non-electrostatic PLS axes) ################")
        run_swap_control(H, F, blocks_raw["struct"], chain_groups, splits, meta, F_save, save_names,
                         args.interpret_k, args.outdir)

    if args.audit:
        log.info("################ AUDIT ################")
        bl = {k: v for k, v in blocks_raw.items() if k in ("F", "struct", "dssr", "ctx", "coul")}
        run_audit(H, Y_cen, bl, splits, chain_groups, pdb_groups, ALL_TARGETS, args.audit_targets,
                  args.outdir, seed=args.seed, n_boot=args.n_boot, pca_dim=args.pca_dim)
        if args.spike_fractions:
            run_spike_in(H, Y_cen, bl, splits, chain_groups, pdb_groups, ALL_TARGETS, args)

    log.info("Done. Outputs in %s", args.outdir)


if __name__ == "__main__":
    main()
