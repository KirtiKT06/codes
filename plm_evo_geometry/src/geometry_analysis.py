"""
geometry_analysis.py

Implements the tightened Q1/Q2 tests:

Q1 (revised): does embedding distance carry information beyond what a
substitution matrix already predicts? Raw embedding-distance-vs-Hamming-
distance is a weak test (a well-trained pLM trivially recovers BLOSUM-like
structure at Hamming-1, which isn't evidence of anything beyond first-order
substitution chemistry). Instead: regress embedding distance on
BLOSUM-predicted distance, keep the RESIDUAL, and test whether that residual
(a) is non-trivial in magnitude and (b) correlates with |DMS score delta|
better than raw Hamming or raw BLOSUM distance do. That residual correlation
is the actual evidence for "the PLM learned something beyond sequence
chemistry."

Q2: are embedding-close variants phenotypically similar? Tested directly via
rank correlation between embedding distance (to WT, or pairwise) and |DMS
score delta|, reported per-protein AND per-position, since pooling over all
positions can mask structure that differs between buried/core, surface, and
interface residues (cf. the AAV/Spike pooling-limitation finding).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from .sequence_utils import hamming_distance, blosum_distance


def regress_out_blosum(blosum_dist: np.ndarray, embed_dist: np.ndarray):
    """
    OLS: embed_dist ~ 1 + blosum_dist. Returns (residuals, (intercept, slope), r2).
    """
    X = np.column_stack([np.ones_like(blosum_dist), blosum_dist])
    beta, *_ = np.linalg.lstsq(X, embed_dist, rcond=None)
    pred = X @ beta
    resid = embed_dist - pred
    ss_res = np.sum((embed_dist - pred) ** 2)
    ss_tot = np.sum((embed_dist - embed_dist.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return resid, tuple(beta), r2


def q1_analysis(df: pd.DataFrame, wt_seq: str,
                 embed_dist_col: str = "embed_dist_to_wt",
                 dms_col: str = "dms_score", wt_dms: float = 0.0) -> dict:
    """
    df must have a 'seq' column (variant sequences) and embed_dist_col already
    computed (embedding distance of each variant to WT). Adds hamming_dist and
    blosum_dist, regresses embedding distance on blosum distance, and reports
    three Spearman correlations against |DMS score - wt_dms| so you can see
    directly whether the BLOSUM-controlled residual beats raw Hamming/BLOSUM.
    """
    df = df.copy()
    df["hamming_dist"] = df["seq"].apply(lambda s: hamming_distance(wt_seq, s))
    df["blosum_dist"] = df["seq"].apply(lambda s: blosum_distance(wt_seq, s))
    df["abs_dms_delta"] = (df[dms_col] - wt_dms).abs()

    resid, coef, r2 = regress_out_blosum(df["blosum_dist"].values, df[embed_dist_col].values)
    df["embed_dist_resid"] = resid

    results = {}
    for col in ["hamming_dist", "blosum_dist", embed_dist_col, "embed_dist_resid"]:
        rho, p = spearmanr(df[col], df["abs_dms_delta"])
        results[col] = {"spearman_rho": rho, "p_value": p}
    results["blosum_regression"] = {"intercept": coef[0], "slope": coef[1], "r2": r2}
    results["interpretation"] = (
        "If 'embed_dist_resid' correlates with |DMS delta| about as strongly as "
        "raw embedding distance (and clearly more than blosum_dist alone), the "
        "PLM is carrying signal beyond substitution chemistry. If the residual "
        "correlation collapses toward zero, most of what looked like 'semantic "
        "structure' in Q1 was actually first-order substitution effects."
    )
    return {"table": df, "stats": results}


def q2_per_position(df: pd.DataFrame, position_col: str = "position",
                     embed_dist_col: str = "embed_dist_to_wt",
                     dms_col: str = "dms_score", wt_dms: float = 0.0,
                     min_variants_per_position: int = 5) -> pd.DataFrame:
    """
    Per-position Spearman correlation between embedding distance and |DMS
    delta|. Positions with too few observed variants are skipped. Compare
    the distribution of per-position rho against buried/surface/interface
    annotations if you have structural data -- pooled, whole-protein
    correlations can average away real heterogeneity.
    """
    rows = []
    for pos, sub in df.groupby(position_col):
        if len(sub) < min_variants_per_position:
            continue
        abs_delta = (sub[dms_col] - wt_dms).abs()
        rho, p = spearmanr(sub[embed_dist_col], abs_delta)
        rows.append({"position": pos, "n_variants": len(sub), "spearman_rho": rho, "p_value": p})
    return pd.DataFrame(rows).sort_values("position").reset_index(drop=True)
