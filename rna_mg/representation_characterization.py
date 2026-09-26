"""
Representation-characterization experiments for the RNA LM electrostatic
probe. Where rna_probe_priority_experiments.py asks "is the signal real"
(and by now, repeatedly, yes -- composition, identity, homology, k-mer
context, real structure, and the electrostatic-twins test all agree),
this script asks the next question: what FORM does that signal take
inside the representation?

Implements, in priority order (per the "run this first" recommendation):

  Experiment A -- CROSS-TARGET TRANSFER (the single most decisive test).
    Fit a probe on ONE target (C1' by default) on its best layer, freeze
    it, and see how well that SAME frozen w^T h predicts every other
    electrostatic target (phosphorus, and the 4/8/12 A sphere averages)
    WITHOUT retraining. If it transfers well, w^T h isn't a label-specific
    fit -- it behaves like a general electrostatic latent coordinate. Each
    target's frozen-transfer R^2 is reported alongside a freshly-trained
    probe on that SAME layer (an upper bound for what that layer alone
    can do), so you can see how much of the gap, if any, is "wrong
    coordinate" vs. "this layer just isn't as good for this target".

  Experiment B -- LATENT DIMENSIONALITY (PCA sweep).
    How many principal components of the best layer are actually needed
    to recover most of the electrostatic signal? A compact answer (e.g.
    ~5-20 PCs) means electrostatics occupies a low-dimensional subspace
    of a much higher-dimensional embedding. PCA is fit on TRAIN+VAL only.

  Experiment C -- WHERE ELECTROSTATICS EMERGES (layer-wise curves).
    R^2 per layer, for raw / within-chain-centered / residual-after-21mer
    targets, plotted together. Tests whether local electrostatic content
    concentrates in particular layers rather than spreading uniformly.
    Uses VAL R^2 throughout (not test) since this sweeps every layer for
    a plot, not a single headline number -- test stays reserved for
    Experiment A's numbers and everything already in the main pipeline.

  Experiment D -- SPARSITY (how many DIMENSIONS carry the signal, as
    opposed to Experiment B's basis-rotated PCA answer). LassoCV on the
    best layer; the number of nonzero coefficients at the CV-selected
    alpha is a direct sparsity readout.

Deferred: spatial-field organization (correlation of w^T h vs. 3D
distance between residues) -- needs new per-residue coordinate
extraction and pairwise distance binning, a bigger, separate build.

Usage:
    python representation_characterization.py \
        --labels-dir /data/rna_mg/cutoff_4_8_12 \
        --model ernierna \
        --primary-target potential_at_c1prime_kT_e \
        --outdir /data/rna_mg/characterization/ernierna \
        --device cuda
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
log = logging.getLogger("representation_characterization")

from rna_embedding_probe_v2 import (  # noqa: E402
    run_ridge_probe, select_best_layer_and_test, load_structure_labels, load_fasta_by_chain,
)
from rna_probe_priority_experiments_extended import (  # noqa: E402
    build_dataset_for_model_with_chains, compute_structure_scalars, build_kmer21_lookup,
    align_feature_lookup, fit_kmer_baseline_and_compute_residual, _ALPHABET,
)

DEFAULT_OTHER_TARGETS = [
    "potential_at_phosphorus_kT_e",
    "potential_mean_kT_e_4A",
    "potential_mean_kT_e_8A",
    "potential_mean_kT_e_12A",
]


def experiment_a_cross_target_transfer(X: dict, y_by_column: dict, splits: np.ndarray,
                                        model_key: str, primary_target: str,
                                        target_columns: list[str], outdir: Path) -> int:
    """Returns the primary target's best layer (needed by B/C/D).

    Reports four numbers per target, not just frozen R^2 -- R^2 alone
    conflates "wrong direction" with "right direction, wrong scale/offset",
    and the two have very different implications for whether w^T h behaves
    like a shared electrostatic latent coordinate:

      - frozen_transfer_test_r2 / frozen_transfer_test_pearson_r: the
        frozen primary-target decoder applied AS-IS (same scaler, same
        RidgeCV weights, no refitting at all). R^2 can look catastrophic
        here purely from a scale/offset mismatch between targets (e.g. a
        point-value target vs. a much-lower-variance spatial average),
        even when the underlying direction is informative. Pearson r
        isolates whether the ranking/relationship is any good before
        concluding the direction itself carries nothing.
      - affine_recalibrated_test_r2: fits ONLY a 1D scale+offset
        (a, b in y_hat = a*frozen_score + b) on the OTHER target's own
        train+val split. The decoder DIRECTION (w, and the standardization
        it was fit under) is never touched past this point -- this tests
        calibration only. If this recovers most of
        fresh_probe_test_r2_same_layer, the direction generalizes and only
        the calibration differed; if it stays near the raw frozen R^2, the
        frozen direction genuinely doesn't carry that target's signal.
      - frozen_direction_recovery_frac: affine_recalibrated_test_r2 divided
        by fresh_probe_test_r2_same_layer -- the fraction of "what this
        layer could do if fully refit for this target" that the frozen,
        merely-recalibrated primary-target direction alone recovers. NaN
        when the fresh fit itself has R^2<=0 (nothing to recover).
      - fresh_probe_test_r2_same_layer: a full refit on that layer, as
        before -- the ceiling for what this layer can do on that target.
    """
    from sklearn.linear_model import RidgeCV, LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr

    log.info("=== Experiment A: cross-target transfer (train on %s, freeze, test elsewhere) ===",
              primary_target)
    y_primary = y_by_column[primary_target]
    valid_primary = ~np.isnan(y_primary)
    X_valid_primary = {i: arr[valid_primary] for i, arr in X.items()}
    splits_primary = splits[valid_primary]

    primary_result = select_best_layer_and_test(
        X_valid_primary, y_primary[valid_primary], splits_primary, model_key, primary_target)
    best_layer = primary_result["best_layer"]
    log.info("Primary target %s: best layer=%d, own test_R^2=%.3f",
              primary_target, best_layer, primary_result["test_r2"])

    trainval_primary = (splits_primary == "train") | (splits_primary == "val")
    scaler = StandardScaler().fit(X_valid_primary[best_layer][trainval_primary])
    frozen_model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(
        scaler.transform(X_valid_primary[best_layer][trainval_primary]),
        y_primary[valid_primary][trainval_primary])

    rows = []
    for target in target_columns:
        y_t = y_by_column[target]
        valid_t = ~np.isnan(y_t)
        if valid_t.sum() < 50:
            log.warning("%s: too few valid rows, skipping", target)
            continue
        X_t_layer = X[best_layer][valid_t]
        y_t_valid = y_t[valid_t]
        splits_t = splits[valid_t]
        trainval_t = (splits_t == "train") | (splits_t == "val")
        test_t = splits_t == "test"
        if test_t.sum() == 0 or trainval_t.sum() < 10:
            log.warning("%s: no test rows or too few train/val rows, skipping", target)
            continue

        # Frozen decoder's raw score -- one number per residue, still in
        # the PRIMARY target's units. Nothing below this line touches
        # `frozen_model` or `scaler` again.
        frozen_score_trainval = frozen_model.predict(scaler.transform(X_t_layer[trainval_t]))
        frozen_score_test = frozen_model.predict(scaler.transform(X_t_layer[test_t]))

        frozen_r2 = r2_score(y_t_valid[test_t], frozen_score_test)
        frozen_r, _ = pearsonr(y_t_valid[test_t], frozen_score_test)

        # Affine recalibration: only a, b in y_hat = a*frozen_score + b,
        # fit on this target's own train+val. Direction stays frozen.
        affine = LinearRegression().fit(frozen_score_trainval.reshape(-1, 1), y_t_valid[trainval_t])
        affine_pred_test = affine.predict(frozen_score_test.reshape(-1, 1))
        affine_r2 = r2_score(y_t_valid[test_t], affine_pred_test)

        fresh_result = select_best_layer_and_test(
            {best_layer: X_t_layer}, y_t_valid, splits_t, model_key, target + "_fresh_samelayer")
        fresh_r2 = fresh_result["test_r2"]
        recovery_frac = (affine_r2 / fresh_r2) if fresh_r2 > 0 else float("nan")

        rows.append({
            "target": target, "is_primary": target == primary_target,
            "frozen_transfer_test_r2": frozen_r2,
            "frozen_transfer_test_pearson_r": float(frozen_r),
            "affine_recalibrated_test_r2": affine_r2,
            "frozen_direction_recovery_frac": recovery_frac,
            "fresh_probe_test_r2_same_layer": fresh_r2,
        })
        log.info(
            "Transfer to %-32s: frozen_R2=%7.3f frozen_r=%.3f | affine_recal_R2=%.3f "
            "(%.0f%% of fresh ceiling) | fresh(same layer)=%.3f",
            target, frozen_r2, frozen_r, affine_r2,
            100 * recovery_frac if recovery_frac == recovery_frac else float("nan"), fresh_r2)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "expA_cross_target_transfer.csv", index=False)
    return best_layer

def experiment_a2_decoder_alignment(X: dict, y_by_column: dict, splits: np.ndarray,
                                    model_key: str, target_columns: list[str], best_layer: int, outdir: Path):
    """
    Are different electrostatic observables decoded from the same latent
    subspace?
    For every target:

        phi_t = w_t^T h

    fit a Ridge probe, extract its decoder vector w_t,
    normalize it, and compare all pairs by cosine similarity.

    High cosine similarity:
        same latent electrostatic subspace

    Low similarity:
        different electrostatic factors.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    log.info("=== Experiment A2: decoder alignment across electrostatic targets ===")
    decoder_vectors = {}
    for target in target_columns:
        y = y_by_column[target]
        valid = ~np.isnan(y)
        # Use ONE fixed layer for all targets.
        # We want decoder similarities inside the same representation space.
        X_layer = X[best_layer][valid]
        splits_t = splits[valid]
        trainval = ((splits_t == "train") | (splits_t == "val"))
        scaler = StandardScaler().fit(X_layer[trainval])
        Xs = scaler.transform(X_layer)
        model = RidgeCV(alphas=np.logspace(-3, 3, 13)).fit(Xs[trainval], y[valid][trainval])
        w = model.coef_
        norm = np.linalg.norm(w)
        if norm > 0:
            w = w / norm
        decoder_vectors[target] = w
        log.info("%s: using shared layer=%d", target, best_layer,)
    rows = []
    targets = list(decoder_vectors.keys())
    for t1 in targets:
        for t2 in targets:
            cosine = float(np.dot(decoder_vectors[t1], decoder_vectors[t2]))
            angle_deg = float(
                    np.degrees(
                        np.arccos(
                            np.clip(cosine, -1.0, 1.0)
                        )
                    )
                )
            rows.append({
                "target1": t1,
                "target2": t2,
                "cosine_similarity": cosine,
                "angle_deg": angle_deg,
            })
            log.info("%s vs %s: cosine=%.3f", t1, t2, cosine,)

    pd.DataFrame(rows).to_csv(outdir / "expA2_decoder_alignment.csv", index=False,)

def experiment_b_pca_dimensionality(X: dict, y_by_column: dict, chain_groups: np.ndarray,
                                     splits: np.ndarray, primary_target: str, best_layer: int,
                                     outdir: Path) -> None:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    log.info("=== Experiment B: PCA dimensionality sweep (layer %d) ===", best_layer)
    y_primary = y_by_column[primary_target]
    valid_primary = ~np.isnan(y_primary)
    X_best = X[best_layer][valid_primary]
    splits_p = splits[valid_primary]
    y_p = y_primary[valid_primary]
    centered_p, _, _ = compute_structure_scalars(y_p, chain_groups[valid_primary])

    train_p, val_p, test_p = splits_p == "train", splits_p == "val", splits_p == "test"
    trainval_p = train_p | val_p

    scaler = StandardScaler().fit(X_best[trainval_p])
    Xs_all = scaler.transform(X_best)
    n_components = min(200, Xs_all.shape[1], trainval_p.sum() - 1)
    pca = PCA(n_components=n_components, random_state=0).fit(Xs_all[trainval_p])
    Xp_all = pca.transform(Xs_all)
    full_raw_r2 = run_ridge_probe(
        Xp_all[trainval_p],
        y_p[trainval_p],
        Xp_all[test_p],
        y_p[test_p],
    )["r2"]

    full_centered_r2 = run_ridge_probe(
        Xp_all[trainval_p],
        centered_p[trainval_p],
        Xp_all[test_p],
        centered_p[test_p],
    )["r2"]


    rows = []
    for k in [1, 2, 3, 5, 10, 20, 50, 100, n_components]:
        if k > n_components:
            continue
        Xk = Xp_all[:, :k]
        r2_raw = run_ridge_probe(Xk[trainval_p], y_p[trainval_p], Xk[test_p], y_p[test_p])["r2"]
        r2_centered = run_ridge_probe(Xk[trainval_p], centered_p[trainval_p], Xk[test_p], centered_p[test_p])["r2"]
        rows.append({
        "n_pcs": k,
        "raw_test_r2": r2_raw,
        "centered_test_r2": r2_centered,
        "raw_fraction_of_full":
            r2_raw / full_raw_r2
            if full_raw_r2 > 0 else np.nan,
        "centered_fraction_of_full":
            r2_centered / full_centered_r2
            if full_centered_r2 > 0 else np.nan,
        "cum_explained_variance":
            float(np.sum(pca.explained_variance_ratio_[:k])),
    })
        log.info("PCA k=%3d: raw_R^2=%.3f centered_R^2=%.3f (cum. explained var=%.3f)",
                  k, r2_raw, r2_centered, rows[-1]["cum_explained_variance"])

    pd.DataFrame(rows).to_csv(outdir / "expB_pca_dimensionality.csv", index=False)


def experiment_c_layerwise_curves(X: dict, y_by_column: dict, pdb_groups: np.ndarray,
                                   chain_groups: np.ndarray, splits: np.ndarray, seq_indices: np.ndarray,
                                   labels_dir: Path, pdb_ids: list[str], primary_target: str,
                                   include_flagged: bool, eff_max_len: Optional[int], outdir: Path) -> None:
    log.info("=== Experiment C: layer-wise R^2 (raw / centered / residual-after-21mer) ===")
    y_primary = y_by_column[primary_target]
    valid_primary = ~np.isnan(y_primary)
    X_valid = {i: arr[valid_primary] for i, arr in X.items()}
    splits_p = splits[valid_primary]
    y_p = y_primary[valid_primary]
    centered_p, _, _ = compute_structure_scalars(y_p, chain_groups[valid_primary])

    kmer21_lookup = build_kmer21_lookup(labels_dir, pdb_ids, primary_target, include_flagged, eff_max_len)
    n_kmer_feat = 21 * len(_ALPHABET)
    X_kmer21_full, kmer21_ok_full = align_feature_lookup(kmer21_lookup, pdb_groups, chain_groups,
                                                           seq_indices, n_kmer_feat)
    kmer21_ok_p = kmer21_ok_full[valid_primary]
    if kmer21_ok_p.sum() < 100:
        log.warning("Too few 21-mer-matched rows for the residual curve -- skipping that column")
        residual_p, splits_resid = None, None
    else:
        residual_p = fit_kmer_baseline_and_compute_residual(
            X_kmer21_full[valid_primary][kmer21_ok_p], y_p[kmer21_ok_p], splits_p[kmer21_ok_p])
        splits_resid = splits_p[kmer21_ok_p]

    rows = []
    for layer_i, X_layer in X_valid.items():
        train_l, val_l = splits_p == "train", splits_p == "val"
        r2_raw_l = run_ridge_probe(X_layer[train_l], y_p[train_l], X_layer[val_l], y_p[val_l])["r2"]
        r2_centered_l = run_ridge_probe(X_layer[train_l], centered_p[train_l],
                                         X_layer[val_l], centered_p[val_l])["r2"]
        if residual_p is not None:
            X_layer_resid = X_layer[kmer21_ok_p]
            train_r = splits_resid == "train"
            val_r = splits_resid == "val"
            r2_resid_l = run_ridge_probe(X_layer_resid[train_r], residual_p[train_r],
                                          X_layer_resid[val_r], residual_p[val_r])["r2"]
        else:
            r2_resid_l = float("nan")
        rows.append({"layer": layer_i, "raw_val_r2": r2_raw_l, "centered_val_r2": r2_centered_l,
                      "residual21mer_val_r2": r2_resid_l})
        log.info("Layer %2d: raw_R^2=%.3f centered_R^2=%.3f residual21mer_R^2=%.3f",
                  layer_i, r2_raw_l, r2_centered_l, r2_resid_l)

    pd.DataFrame(rows).to_csv(outdir / "expC_layerwise_curves.csv", index=False)


# def experiment_d_elasticnet_sparsity(X: dict, y_by_column: dict, splits: np.ndarray, 
#                                     primary_target: str, best_layer: int, outdir: Path):
#     """
#     Sparse characterization using ElasticNet.
#     More reliable than pure LASSO since the probe results suggest
#     electrostatic information is highly distributed.
#     Reports:
#         - number of nonzero dimensions
#         - fraction of embedding used
#         - chosen alpha
#         - chosen l1_ratio
#         - test R²
#     """
#     from sklearn.linear_model import ElasticNetCV
#     from sklearn.preprocessing import StandardScaler
#     from sklearn.metrics import r2_score

#     log.info("=== Experiment D: ElasticNet sparsity (layer %d) ===", best_layer)
#     y = y_by_column[primary_target]
#     valid = ~np.isnan(y)
#     X_best = X[best_layer][valid]
#     y = y[valid]
#     splits_p = splits[valid]
#     trainval = ((splits_p == "train") | (splits_p == "val"))
#     test = splits_p == "test"
#     scaler = StandardScaler().fit(X_best[trainval])
#     Xs = scaler.transform(X_best)
#     model = ElasticNetCV(
#     l1_ratio=[
#         0.1,
#         0.25,
#         0.5,
#         0.75,
#         0.9,
#         0.95,
#         1.0,
#     ],
#     alphas=np.logspace(-6, 2, 40),
#     max_iter=200000,
#     tol=1e-3,
#     n_jobs=-1,
#     random_state=0,
# )
#     model.fit(Xs[trainval], y[trainval])
#     pred = model.predict(Xs[test])
#     test_r2 = r2_score(y[test], pred,)
#     n_nonzero = int(np.sum(np.abs(model.coef_) > 1e-6))
#     frac_nonzero = (n_nonzero / Xs.shape[1])
#     log.info("ElasticNet: nonzero=%d/%d (%.1f%%), alpha=%.4g, l1_ratio=%.2f, test_R²=%.3f", n_nonzero, Xs.shape[1], frac_nonzero * 100, model.alpha_, model.l1_ratio_, test_r2,)
#     pd.DataFrame([{
#         "n_nonzero_dims": n_nonzero,
#         "n_total_dims": Xs.shape[1],
#         "frac_nonzero": frac_nonzero,
#         "alpha": model.alpha_,
#         "l1_ratio": model.l1_ratio_,
#         "test_r2": test_r2,}]).to_csv(outdir / "expD_elasticnet_sparsity.csv", index=False,)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--model", required=True, help="Single model key, e.g. ernierna")
    ap.add_argument("--primary-target", default="potential_at_c1prime_kT_e",
                     help="Target the probe is trained/frozen on for Experiments A/B/C/D")
    ap.add_argument("--other-targets", nargs="+", default=DEFAULT_OTHER_TARGETS,
                     help="Additional targets Experiment A tests frozen transfer against")
    ap.add_argument("--outdir", type=Path, default=Path("./characterization"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--include-flagged", action="store_true")
    ap.add_argument("--limit-structures", type=int, default=None)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    import torch
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    all_pot_path = args.labels_dir / "all_residue_potentials.csv"
    if not all_pot_path.exists():
        raise FileNotFoundError(f"{all_pot_path} not found -- run the label pipeline first.")
    pdb_ids = sorted(pd.read_csv(all_pot_path, low_memory=False)["pdb_id"].unique())
    if args.limit_structures:
        pdb_ids = pdb_ids[: args.limit_structures]

    target_columns = [args.primary_target] + [t for t in args.other_targets if t != args.primary_target]

    log.info("=== Building embeddings for %s (%d structures, %d targets) ===",
              args.model, len(pdb_ids), len(target_columns))
    X, y_by_column, pdb_groups, chain_groups, splits, base_ids, resnums, seq_indices, eff_max_len = \
        build_dataset_for_model_with_chains(args.labels_dir, pdb_ids, args.model, device,
                                             args.include_flagged, args.max_len, target_columns)

    best_layer = experiment_a_cross_target_transfer(
        X, y_by_column, splits, args.model, args.primary_target, target_columns, args.outdir)
    
    experiment_a2_decoder_alignment(
        X, y_by_column, splits, args.model, target_columns, best_layer, args.outdir,)

    experiment_b_pca_dimensionality(
        X, y_by_column, chain_groups, splits, args.primary_target, best_layer, args.outdir)

    experiment_c_layerwise_curves(
        X, y_by_column, pdb_groups, chain_groups, splits, seq_indices,
        args.labels_dir, pdb_ids, args.primary_target, args.include_flagged, eff_max_len, args.outdir)

    # experiment_d_elasticnet_sparsity(
    #     X, y_by_column, splits, args.primary_target, best_layer, args.outdir,)

    log.info("All characterization experiment outputs written to %s", args.outdir)

if __name__ == "__main__":
    main()