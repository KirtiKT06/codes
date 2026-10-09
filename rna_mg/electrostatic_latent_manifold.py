"""
Does a single generalized electrostatic latent SUBSPACE exist inside an
RNA LM's embeddings -- one that jointly explains C1', phosphorus, and the
4/8/12 A cutoff-averaged potentials -- rather than five loosely related,
separately-decoded scalars?

Motivation, from what representation_characterization.py already showed:
  - A ridge decoder trained on ONLY potential_at_c1prime_kT_e, merely
    affine-recalibrated (never refit), recovers 38-64% of what a fresh
    decoder achieves on the other four targets, consistently across four
    architecturally distinct LMs. Real shared structure -- but recovery
    never exceeds ~65%, so a SINGLE frozen direction is clearly not the
    whole story.
  - Cosine similarity between the five separately-fit per-target decoder
    weight vectors (Experiment A2) doesn't track that recovery well and
    shouldn't be over-read: rnafm's c1prime<->phosphorus cosine is the
    LOWEST of any model (0.30) yet its affine recovery (48%) is HIGHER
    than rnabert's (38%, cosine 0.58). Comparing five independently
    regularized weight vectors pairwise is not the same question as "is
    there a shared subspace" -- it's noisier and answers a related but
    different question.

This script asks the better-posed version directly, with two standard
multivariate tools that are BUILT to find a shared subspace (rather than
compare pairwise fits after the fact):

  PLS2 (H -> Y jointly, Y = all 5 targets at once): components are
    ordered by covariance shared between H and Y by construction -- this
    is "a shared subspace" operationalized correctly, unlike plain PCA of
    H alone (which only sees embedding variance and is blind to whether
    that variance has anything to do with electrostatics).

  CCA (H vs Y): the standard way to report "how compact is the shared
    subspace" -- canonical correlations, reported OUT OF SAMPLE (on the
    test split) since in-sample canonical correlations are inflated the
    same way in-sample R^2 is.

Experiments implemented here (numbered per the discussion that motivated
this script):

  1A / 2 -- PLS dimensionality sweep. Fit PLSRegression(n_components=k)
    for each k in --pls-components, H->Y jointly (Y stacks all targets in
    --targets, complete-case rows only). For each k, report PER-TARGET
    test R^2 (not just an average -- the whole point is to see whether
    some targets saturate at low k while others need more, which is
    itself evidence about how many electrostatic "kinds of information"
    there are). Each target's row also carries its own single-target
    ridge ceiling (same layer, fit independently, ignoring the other 4
    targets) as a reference column -- this doubles as the bottleneck test
    (originally "Experiment 3"): if PLS at moderate k gets close to that
    ceiling, going through the shared low-rank bottleneck costs little.

  1B -- CCA, --cca-components canonical variate pairs, canonical
    correlation reported on held-out test data (fit on train+val,
    transform test, correlate the resulting scores) -- NOT the training-
    set canonical correlations CCA reports internally, which would be
    optimistic in the same way training R^2 is.

  7 -- Axis interpretation. Using the PLS fit at --interpret-k
    components, correlate each latent axis (out-of-sample test scores)
    against each of the 5 raw targets. This is what would explain the
    transfer asymmetries already seen: e.g. if one axis correlates
    strongly with all three cutoff-averaged targets and weakly with
    c1prime/phosphorus, and another axis does the reverse, that's a
    direct, checkable account of "coarse vs. local electrostatic
    environment" as two separate encoded factors.

Deferred (not implemented here -- see the conversation this script came
out of for the reasoning):
  - UMAP/t-SNE visualization: illustrative only, never evidentiary on its
    own -- add once the numbers here are in hand, not before.
  - Mapping the latent field onto 3D coordinates and comparing to the
    APBS grid: needs a non-circular baseline (compare the sequence-only
    latent field on TEST structures against a naive RBF interpolation of
    the same sparse per-residue LABELS, not against the raw APBS solution
    directly) and needs this script's dimensionality answer first, to
    know how many components to carry into the field.
  - DSSR-residualized structure-independence test: the single strongest
    test on the original list, but needs a DSSR-annotation loader built
    first (per-residue base-pair/triple/junction features joined to what
    build_dataset_for_model_with_chains already returns) -- a separate,
    bigger piece of plumbing, not a flag on this script.

Usage:
    python electrostatic_latent_manifold.py \
        --labels-dir /data/rna_mg/res_phi \
        --model ernierna \
        --outdir /data/rna_mg/latent_manifold/ernierna \
        --device cuda
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                     datefmt="%H:%M:%S")
log = logging.getLogger("electrostatic_latent_manifold")

from rna_embedding_probe_v2 import run_ridge_probe, select_best_layer_and_test  # noqa: E402
from rna_probe_priority_experiments_extended import build_dataset_for_model_with_chains  # noqa: E402

DEFAULT_TARGETS = [
    "potential_at_c1prime_kT_e",
    "potential_at_phosphorus_kT_e",
    "potential_mean_kT_e_4A",
    "potential_mean_kT_e_8A",
    "potential_mean_kT_e_12A",
]


def build_joint_dataset(X: dict, y_by_column: dict, splits: np.ndarray, chain_groups: np.ndarray,
                         targets: list[str], layer: int):
    """Complete-case rows only: a residue enters Y/H only if EVERY target
    in `targets` is non-NaN for it (PLS/CCA need one consistent Y matrix,
    unlike the single-target probes elsewhere in this project that each
    take their own independent valid mask). This is a real, reportable
    restriction -- log how much it costs relative to each target's own
    individual valid count, since e.g. phosphorus alone already drops
    ~1-2% of residues (5' termini with no phosphate) and requiring ALL
    five simultaneously compounds whatever each target drops on its own.

    Also returns chain_groups restricted to the same valid_all mask, so
    callers (the negative controls in particular) can build a
    within-chain-preserving permutation without re-deriving this mask.
    """
    valid_each = {t: ~np.isnan(y_by_column[t]) for t in targets}
    valid_all = np.logical_and.reduce(list(valid_each.values()))
    for t in targets:
        log.info("  %s: %d valid alone -> %d survive requiring all %d targets",
                  t, int(valid_each[t].sum()), int(valid_all.sum()), len(targets))

    H = X[layer][valid_all]
    Y = np.stack([y_by_column[t][valid_all] for t in targets], axis=1)
    splits_valid = splits[valid_all]
    chain_ids_valid = chain_groups[valid_all]
    return H, Y, splits_valid, chain_ids_valid


def pls_dimensionality_sweep(H: np.ndarray, Y: np.ndarray, splits: np.ndarray, targets: list[str],
                              n_components_list: list[int], outdir: Path) -> dict:
    """1A/2/3 combined: per-target test R^2 at each PLS component count,
    against each target's own single-target ridge ceiling (same H, same
    split, that target alone -- ignoring the other 4). Returns the fitted
    PLS models keyed by n_components (so experiment_axis_interpretation
    can reuse one without refitting)."""
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.metrics import r2_score

    train, val, test = splits == "train", splits == "val", splits == "test"
    trainval = train | val

    log.info("=== Single-target ridge ceilings (same layer, no shared bottleneck) ===")
    ceilings = {}
    for j, t in enumerate(targets):
        result = run_ridge_probe(H[trainval], Y[trainval, j], H[test], Y[test, j])
        ceilings[t] = result["r2"]
        log.info("  %-32s ceiling test_R^2=%.3f", t, result["r2"])

    max_k = min(n_components_list[-1], H.shape[1] - 1, trainval.sum() - 1)
    valid_ks = [k for k in n_components_list if k <= max_k]
    if len(valid_ks) < len(n_components_list):
        log.warning("Capping PLS sweep at n_components<=%d (embedding dim / sample size limit)", max_k)

    log.info("=== PLS dimensionality sweep (H -> all %d targets jointly) ===", len(targets))
    rows = []
    fitted = {}
    for k in valid_ks:
        pls = PLSRegression(n_components=k, scale=True)
        pls.fit(H[trainval], Y[trainval])
        fitted[k] = pls
        Y_pred_test = pls.predict(H[test])
        for j, t in enumerate(targets):
            r2 = r2_score(Y[test, j], Y_pred_test[:, j])
            rows.append({"n_components": k, "target": t, "test_r2": r2,
                          "single_target_ceiling_r2": ceilings[t]})
            log.info("  k=%2d  %-32s test_R^2=%.3f  (single-target ceiling=%.3f)",
                      k, t, r2, ceilings[t])

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "pls_dimensionality_sweep.csv", index=False)
    return fitted


def cca_canonical_correlations(H: np.ndarray, Y: np.ndarray, splits: np.ndarray,
                                n_components: int, outdir: Path):
    """1B: canonical correlations, reported OUT OF SAMPLE. CCA's own
    .fit() will happily report near-perfect in-sample correlation for
    high n_components relative to n_samples -- that number belongs
    nowhere near a paper. Transform the held-out test split through the
    frozen canonical directions and correlate THOSE scores instead."""
    from sklearn.cross_decomposition import CCA
    from scipy.stats import pearsonr

    train, val, test = splits == "train", splits == "val", splits == "test"
    trainval = train | val
    max_k = min(n_components, H.shape[1] - 1, Y.shape[1], trainval.sum() - 1)
    if max_k < n_components:
        log.warning("Capping CCA at n_components=%d", max_k)

    cca = CCA(n_components=max_k, scale=True)
    cca.fit(H[trainval], Y[trainval])
    Hc_test, Yc_test = cca.transform(H[test], Y[test])

    log.info("=== CCA canonical correlations (test-set, out-of-sample) ===")
    rows = []
    for i in range(max_k):
        r, _ = pearsonr(Hc_test[:, i], Yc_test[:, i])
        rows.append({"component": i + 1, "test_canonical_correlation": float(r)})
        log.info("  canonical component #%d: test r=%.3f", i + 1, r)

    pd.DataFrame(rows).to_csv(outdir / "cca_canonical_correlations.csv", index=False)


def axis_interpretation(pls_fitted: dict, interpret_k: int, H: np.ndarray, Y: np.ndarray,
                         splits: np.ndarray, targets: list[str], outdir: Path):
    """7: correlate each PLS latent axis (out-of-sample test scores)
    against each raw target. Reuses an already-fitted PLS model from the
    sweep rather than refitting."""
    from scipy.stats import pearsonr

    if interpret_k not in pls_fitted:
        raise ValueError(f"--interpret-k={interpret_k} was not in --pls-components "
                          f"(available: {sorted(pls_fitted)}); add it to the sweep list.")
    pls = pls_fitted[interpret_k]
    test = splits == "test"
    Z_test = pls.transform(H[test])  # (n_test, interpret_k) out-of-sample latent scores

    log.info("=== Axis interpretation (PLS k=%d, out-of-sample test correlations) ===", interpret_k)
    rows = []
    for axis in range(interpret_k):
        axis_rows = []
        for t in targets:
            j = targets.index(t)
            r, _ = pearsonr(Z_test[:, axis], Y[test, j])
            axis_rows.append({"axis": axis + 1, "target": t, "pearson_r": float(r)})
        rows.extend(axis_rows)
        axis_line = " | ".join(f"{row['target']}={row['pearson_r']:.2f}" for row in axis_rows)
        log.info("  axis %d: %s", axis + 1, axis_line)

    pd.DataFrame(rows).to_csv(outdir / "latent_axis_interpretation.csv", index=False)


def _within_chain_permutation(chain_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """A permutation index array such that Y[perm] scrambles which
    residue-within-a-chain each row's Y is paired with, while every row
    still ends up paired with a Y from the SAME chain it started in.
    Chain-constant information (e.g. a per-chain scale factor) survives
    this shuffle by construction; genuinely residue-specific information
    does not."""
    perm = np.arange(len(chain_ids))
    for chain in np.unique(chain_ids):
        idx = np.flatnonzero(chain_ids == chain)
        perm[idx] = rng.permutation(idx)
    return perm


def negative_controls(H: np.ndarray, Y: np.ndarray, splits: np.ndarray, chain_ids: np.ndarray,
                       targets: list[str], pls_k: int, cca_k: int, outdir: Path, seed: int = 0):
    """Mirrors Priority 5's existing negative-control design (real vs.
    shuffled-within-chain vs. shuffled-global label correspondence),
    applied to BOTH PLS and CCA -- both are exactly the kind of high-
    dimensional multivariate fit that can show spurious out-of-sample
    structure even when H and Y are genuinely unrelated, and a reviewer
    who knows CCA's reputation for this will expect to see it before
    trusting a 0.9 canonical correlation.

    shuffled_within_chain preserves each residue's chain membership but
    randomizes which residue-in-that-chain its Y row is drawn from -- if
    an axis is really mostly per-chain SCALE (as axis 1 already looked
    like it might, per axis_interpretation), it should survive this
    shuffle reasonably well, since chain-constant information is
    invariant to a within-chain permutation by construction. This makes
    the control a direct, falsifiable test of that interpretation, not
    just a sanity check.

    shuffled_global destroys chain membership too and should collapse
    both PLS R^2 and CCA canonical correlations toward the values Priority
    5 already reports for the equivalent single-target shuffle (~0). If
    it does NOT collapse, that points at a pipeline bug (e.g. a split or
    row-alignment leak) rather than evidence of real signal -- treat a
    non-collapsing shuffled_global result as a bug report on this
    pipeline, not as a finding.
    """
    from sklearn.cross_decomposition import PLSRegression, CCA
    from sklearn.metrics import r2_score
    from scipy.stats import pearsonr

    rng = np.random.default_rng(seed)
    train, val, test = splits == "train", splits == "val", splits == "test"
    trainval = train | val

    variants = {
        "real": np.arange(len(Y)),
        "shuffled_within_chain": _within_chain_permutation(chain_ids, rng),
        "shuffled_global": rng.permutation(len(Y)),
    }

    log.info("=== Negative controls (PLS k=%d, CCA k=%d) ===", pls_k, cca_k)
    pls_rows, cca_rows = [], []
    for variant_name, perm in variants.items():
        Y_variant = Y[perm]

        pls = PLSRegression(n_components=pls_k, scale=True)
        pls.fit(H[trainval], Y_variant[trainval])
        Y_pred_test = pls.predict(H[test])
        variant_pls_rows = []
        for j, t in enumerate(targets):
            r2 = r2_score(Y_variant[test, j], Y_pred_test[:, j])
            variant_pls_rows.append({"variant": variant_name, "n_components": pls_k,
                                      "target": t, "test_r2": r2})
        pls_rows += variant_pls_rows
        log.info("  PLS [%-22s] %s", variant_name,
                  " | ".join(f"{r['target']}={r['test_r2']:.3f}" for r in variant_pls_rows))

        cca = CCA(n_components=cca_k, scale=True)
        cca.fit(H[trainval], Y_variant[trainval])
        Hc_test, Yc_test = cca.transform(H[test], Y_variant[test])
        variant_cca_rows = []
        for i in range(cca_k):
            r, _ = pearsonr(Hc_test[:, i], Yc_test[:, i])
            variant_cca_rows.append({"variant": variant_name, "component": i + 1,
                                      "test_canonical_correlation": float(r)})
        cca_rows += variant_cca_rows
        log.info("  CCA [%-22s] %s", variant_name,
                  " | ".join(f"c{r['component']}={r['test_canonical_correlation']:.3f}"
                             for r in variant_cca_rows))

    pd.DataFrame(pls_rows).to_csv(outdir / "negative_controls_pls.csv", index=False)
    pd.DataFrame(cca_rows).to_csv(outdir / "negative_controls_cca.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-dir", type=Path, required=True)
    ap.add_argument("--model", required=True, help="Single model key, e.g. ernierna")
    ap.add_argument("--targets", nargs="+", default=DEFAULT_TARGETS,
                     help="Electrostatic targets to jointly explain (complete-case rows only)")
    ap.add_argument("--primary-target", default="potential_at_c1prime_kT_e",
                     help="Used only to pick the shared layer (via best-val-R^2), unless --layer is set")
    ap.add_argument("--layer", type=int, default=None,
                     help="Skip layer selection and use this layer directly")
    ap.add_argument("--pls-components", type=int, nargs="+", default=[1, 2, 3, 5, 8, 10, 15, 20])
    ap.add_argument("--interpret-k", type=int, default=5,
                     help="Which PLS component count to use for axis interpretation "
                          "(must appear in --pls-components)")
    ap.add_argument("--cca-components", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=Path("./latent_manifold"))
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

    target_columns = [args.primary_target] + [t for t in args.targets if t != args.primary_target]

    log.info("=== Building embeddings for %s (%d structures, %d targets) ===",
              args.model, len(pdb_ids), len(target_columns))
    X, y_by_column, pdb_groups, chain_groups, splits, base_ids, resnums, seq_indices, eff_max_len = \
        build_dataset_for_model_with_chains(args.labels_dir, pdb_ids, args.model, device,
                                             args.include_flagged, args.max_len, target_columns)

    if args.layer is not None:
        layer = args.layer
        log.info("Using user-specified layer=%d (no selection run)", layer)
    else:
        y_primary = y_by_column[args.primary_target]
        valid_primary = ~np.isnan(y_primary)
        X_valid_primary = {i: arr[valid_primary] for i, arr in X.items()}
        primary_result = select_best_layer_and_test(
            X_valid_primary, y_primary[valid_primary], splits[valid_primary],
            args.model, args.primary_target)
        layer = primary_result["best_layer"]
        log.info("Selected shared layer=%d via %s (own test_R^2=%.3f)",
                  layer, args.primary_target, primary_result["test_r2"])

    log.info("=== Assembling complete-case joint dataset across %d targets ===", len(args.targets))
    H, Y, splits_valid, chain_ids_valid = build_joint_dataset(X, y_by_column, splits, chain_groups,
                                                               args.targets, layer)
    log.info("Joint dataset: %d residues, %d-dim embedding, %d targets",
              H.shape[0], H.shape[1], Y.shape[1])

    pls_fitted = pls_dimensionality_sweep(H, Y, splits_valid, args.targets,
                                           args.pls_components, args.outdir)
    cca_canonical_correlations(H, Y, splits_valid, args.cca_components, args.outdir)
    axis_interpretation(pls_fitted, args.interpret_k, H, Y, splits_valid, args.targets, args.outdir)
    negative_controls(H, Y, splits_valid, chain_ids_valid, args.targets,
                       args.interpret_k, args.cca_components, args.outdir, seed=args.seed)

    log.info("All latent-manifold experiment outputs written to %s", args.outdir)


if __name__ == "__main__":
    main()