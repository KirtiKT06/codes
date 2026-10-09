"""
Do different RNA language models share the SAME electrostatic latent axes?   [v2]

Idea in one paragraph
---------------------
centered_latent_manifold.py gives every residue a few "latent scores" Z1..Zk per
model (the PLS axes that best predict electrostatics). The axes of two models are
not directly comparable: sign and order are arbitrary, and the models have
different embedding sizes. So we ask a question that does not care about order or sign:

    "Is there ANY linear recombination of model A's axes that tracks ANY
     linear recombination of model B's axes, residue by residue?"

That is exactly what CCA (canonical correlation analysis) measures. A canonical
correlation near 1 means the two models describe that direction the same way;
near 0 means unrelated.

WHAT CHANGED IN v2
------------------
In the v1 run, `shuffle_within_chain` was far from 0 (up to 0.69). Reason: the latent
scores are read off raw embeddings, which carry CHAIN-LEVEL information (composition,
overall character of the RNA), and two models agree about that. A within-chain shuffle
keeps chain membership, so it measures exactly this chain-level agreement. v2 therefore
centers every score (and every nuisance feature) WITHIN each chain before comparing, so
only residue-to-residue differences inside a chain can agree. The honest quantity is now
`within_chain_clean`, and its shuffle controls should be ~0. If they are not, chain-level
information is still leaking.

Conditions reported (mean and per-component canonical correlation on TEST residues):
  raw_scores                    : scores as saved (includes chain-level agreement)
  nuisance_removed              : v1 condition (global removal of letter/length/end-distance)
  shuffle_within_chain_v1       : v1-style control on nuisance_removed (shows the leak)
  within_chain_clean            : PRIMARY. centered within chain, nuisance (+controls) removed
  shuffle_within_chain          : control on within_chain_clean (should be ~0)
  shuffle_global                : control on within_chain_clean (should be ~0)

With --controls (scores produced with --structure-controls) the ctrl_* columns (pairing,
packing, burial) are removed as well. That asks: do the models still agree after
secondary structure and burial are taken out? --residualizer gbm removes them nonlinearly
(stricter; slower).

Everything is fit on train+val residues and reported on TEST residues only. Pairs are
formed on the residues BOTH models saw (rnabert truncates at 438 nt).

Usage (after running centered_latent_manifold.py once per model):
    python cross_model_latent_agreement.py \
        --root /data/rna_electrostats/centered_latent_v2 \
        --models rnabert rnafm splicebert ernierna \
        --variant centered_base_resid \
        --outdir /data/rna_electrostats/centered_latent_v2/cross_model
    # structure-controlled version:
    python cross_model_latent_agreement.py ... --variant centered_struct_gbm --controls --residualizer gbm
"""

from __future__ import annotations

import argparse
import itertools
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cross_model_latent_agreement")

KEYS = ["pdb_id", "chain", "seq_index"]


def load_scores(root: Path, model: str, variant: str):
    """Load one model's saved scores. Latent columns are renamed '<model>__Z1' ... so that
    two models can be merged without name clashes."""
    path = root / model / variant / "scores.csv.gz"
    df = pd.read_csv(path, dtype={"pdb_id": str, "chain": str})
    z_cols = [c for c in df.columns if c.startswith("Z") and c[1:].isdigit()]
    nuis_cols = [c for c in df.columns if c.startswith("nuis_")]
    ctrl_cols = [c for c in df.columns if c.startswith("ctrl_")]
    df = df.rename(columns={c: f"{model}__{c}" for c in z_cols})
    return df, [f"{model}__{c}" for c in z_cols], nuis_cols, ctrl_cols


def within_chain_center(M: np.ndarray, chain_ids: np.ndarray) -> np.ndarray:
    """Subtract each chain's own mean from every column."""
    df = pd.DataFrame(M)
    return (df - df.groupby(np.asarray(chain_ids)).transform("mean")).to_numpy()


def residualize(Z: np.ndarray, F: np.ndarray, splits: np.ndarray) -> np.ndarray:
    """Remove what a LINEAR model on F predicts from each score. Fit on train rows only."""
    from sklearn.linear_model import LinearRegression
    train = splits == "train"
    return Z - LinearRegression().fit(F[train], Z[train]).predict(F)


def residualize_gbm(Z: np.ndarray, F: np.ndarray, splits: np.ndarray, groups: np.ndarray,
                    seed: int = 0, n_folds: int = 5) -> np.ndarray:
    """Nonlinear removal (gradient boosting). Train rows get out-of-fold predictions
    (folds split by structure) so train and test residuals are comparable."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.model_selection import GroupKFold

    train_idx, other_idx = np.flatnonzero(splits == "train"), np.flatnonzero(splits != "train")
    pred = np.zeros_like(Z, dtype=float)

    def make():
        return HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, max_leaf_nodes=31,
                                             min_samples_leaf=200, early_stopping=False,
                                             random_state=seed)

    for j in range(Z.shape[1]):
        for tr, te in GroupKFold(n_splits=n_folds).split(train_idx, groups=groups[train_idx]):
            pred[train_idx[te], j] = make().fit(F[train_idx[tr]], Z[train_idx[tr], j]).predict(F[train_idx[te]])
        pred[other_idx, j] = make().fit(F[train_idx], Z[train_idx, j]).predict(F[other_idx])
    return Z - pred


def within_chain_permutation(chain_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Row order that shuffles residues INSIDE each chain but never moves a row to another chain."""
    perm = np.arange(len(chain_ids))
    for c in np.unique(chain_ids):
        idx = np.flatnonzero(chain_ids == c)
        perm[idx] = rng.permutation(idx)
    return perm


def cca_test_correlations(ZA: np.ndarray, ZB: np.ndarray, splits: np.ndarray, k: int) -> list[float]:
    """Fit CCA on train+val, then correlate the canonical scores on TEST rows."""
    from sklearn.cross_decomposition import CCA
    fit, test = splits != "test", splits == "test"
    cca = CCA(n_components=k, scale=True, max_iter=1000).fit(ZA[fit], ZB[fit])
    A_c, B_c = cca.transform(ZA[test], ZB[test])
    return [float(np.corrcoef(A_c[:, i], B_c[:, i])[0, 1]) for i in range(k)]


def nuisance_r2(Z: np.ndarray, F: np.ndarray, splits: np.ndarray) -> float:
    """How much of a model's latent scores do the simple features explain on test rows?
    High value = the 'latent variable' is mostly those simple features."""
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    fit, test = splits != "test", splits == "test"
    m = LinearRegression().fit(F[fit], Z[fit])
    return float(r2_score(Z[test], m.predict(F[test]), multioutput="uniform_average"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="Folder containing <model>/<variant>/scores.csv.gz")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--variant", default="centered_base_resid")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--controls", action="store_true",
                    help="Also remove the ctrl_* (pairing / packing / burial) columns from the scores")
    ap.add_argument("--residualizer", choices=["linear", "gbm"], default="linear")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    data = {m: load_scores(args.root, m, args.variant) for m in args.models}

    rows = []
    for a, b in itertools.combinations(args.models, 2):
        dfA, zA, nuis_cols, ctrl_cols = data[a]
        dfB, zB, _, _ = data[b]
        merged = dfA.merge(dfB[KEYS + zB], on=KEYS, how="inner")
        if merged.empty:
            log.warning("%s vs %s: no shared residues, skipping", a, b)
            continue
        if args.controls and not ctrl_cols:
            raise SystemExit("--controls given but the scores have no ctrl_* columns; "
                             "rerun centered_latent_manifold.py with --structure-controls")
        splits = merged["split"].to_numpy()
        chain_ids = (merged["pdb_id"] + "::" + merged["chain"]).to_numpy()
        pdb_ids = merged["pdb_id"].to_numpy()
        k = min(len(zA), len(zB))
        ZA, ZB = merged[zA].to_numpy(), merged[zB].to_numpy()
        F_nuis = merged[nuis_cols].to_numpy()
        F_all = merged[nuis_cols + (ctrl_cols if args.controls else [])].to_numpy()
        n_test = int((splits == "test").sum())
        log.info("=== %s vs %s: %d shared residues (%d test), k=%d, controls=%s ===",
                 a, b, len(merged), n_test, k, args.controls)
        if n_test < 100:
            log.warning("too few test residues, skipping pair")
            continue

        # ---- v1-style conditions (global removal, no centering) ----
        A_nr, B_nr = residualize(ZA, F_nuis, splits), residualize(ZB, F_nuis, splits)

        # ---- v2 PRIMARY: center within chain, then remove nuisance (+ controls) ----
        ZA_c, ZB_c = within_chain_center(ZA, chain_ids), within_chain_center(ZB, chain_ids)
        F_all_c = within_chain_center(F_all, chain_ids)
        if args.residualizer == "gbm":
            F_g = np.column_stack([F_all, F_all_c])
            A_cl = residualize_gbm(ZA_c, F_g, splits, pdb_ids, args.seed)
            B_cl = residualize_gbm(ZB_c, F_g, splits, pdb_ids, args.seed)
        else:
            A_cl, B_cl = residualize(ZA_c, F_all_c, splits), residualize(ZB_c, F_all_c, splits)
        A_cl, B_cl = within_chain_center(A_cl, chain_ids), within_chain_center(B_cl, chain_ids)

        within_perm = within_chain_permutation(chain_ids, rng)
        conditions = {
            "raw_scores": (ZA, ZB),
            "nuisance_removed": (A_nr, B_nr),
            "shuffle_within_chain_v1": (A_nr, B_nr[within_perm]),
            "within_chain_clean": (A_cl, B_cl),
            "shuffle_within_chain": (A_cl, B_cl[within_perm]),
            "shuffle_global": (A_cl, B_cl[rng.permutation(len(B_cl))]),
        }

        info = {"nuisance_r2_model_a": nuisance_r2(ZA, F_nuis, splits),
                "nuisance_r2_model_b": nuisance_r2(ZB, F_nuis, splits),
                "within_chain_ctrl_r2_model_a": nuisance_r2(ZA_c, F_all_c, splits),
                "within_chain_ctrl_r2_model_b": nuisance_r2(ZB_c, F_all_c, splits)}

        for cond, (A_mat, B_mat) in conditions.items():
            corrs = cca_test_correlations(A_mat, B_mat, splits, k)
            rows.append({"model_a": a, "model_b": b, "condition": cond, "controls": args.controls,
                         "residualizer": args.residualizer, "n_shared": len(merged), "n_test": n_test,
                         "mean_canonical_corr": float(np.mean(corrs)), **info,
                         **{f"cc{i + 1}": c for i, c in enumerate(corrs)}})
            log.info("  %-24s canonical corrs: %s", cond, " ".join(f"{c:.2f}" for c in corrs))

    summary = pd.DataFrame(rows)
    suffix = args.variant + ("_ctrl_" + args.residualizer if args.controls else "")
    summary.to_csv(args.outdir / f"cross_model_agreement_{suffix}.csv", index=False)
    if not summary.empty:
        pivot = summary.pivot_table(index=["model_a", "model_b"], columns="condition",
                                    values="mean_canonical_corr")
        log.info("Mean canonical correlation (test):\n%s", pivot.round(3).to_string())
    log.info("Wrote results to %s", args.outdir)


if __name__ == "__main__":
    main()