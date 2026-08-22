"""
robustness_evolvability.py

Ground-truth robustness/evolvability are defined ONLY from experimental DMS
data, following Wagner (2008, 2011) and Draghi et al. (2010) -- no embedding
involved. Latent proxies are then computed from PLM geometry and TESTED
against ground truth via correlation. This is the fix for the circularity
risk in the original R(sigma) formula: robustness is not defined as a
property of the embedding, it is defined from phenotype data and the
embedding is asked to predict it.
"""

from __future__ import annotations
import numpy as np
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Ground truth (DMS-only)
# ---------------------------------------------------------------------------

def experimental_robustness(dms_scores: np.ndarray, adj: dict[int, list[int]],
                             mode: str = "variance") -> np.ndarray:
    """
    R_exp(sigma) for every genotype index in adj.

    mode='variance': 1 / (1 + Var_{tau in N(sigma)}[DMS(tau)])  -- bounded in
        (0, 1], higher = neighbors have more uniform phenotype = more robust.
    mode='fraction_neutral': requires a neutral band; see
        experimental_robustness_neutral_band().

    Genotypes with zero neighbors return NaN (undefined, not zero -- do not
    silently treat missing neighborhood data as "not robust").
    """
    n = len(dms_scores)
    out = np.full(n, np.nan)
    for i in range(n):
        neigh = adj.get(i, [])
        if len(neigh) == 0:
            continue
        vals = dms_scores[neigh]
        if mode == "variance":
            out[i] = 1.0 / (1.0 + np.var(vals))
        else:
            raise ValueError(f"Unknown mode: {mode}")
    return out


def experimental_robustness_neutral_band(dms_scores: np.ndarray, adj: dict[int, list[int]],
                                          neutral_halfwidth: float) -> np.ndarray:
    """
    R_exp(sigma) = fraction of Hamming-1 neighbors whose DMS score is within
    `neutral_halfwidth` of sigma's own score. This is closer to the classical
    "fraction of neutral mutations" definition (van Nimwegen et al.) than the
    variance-based version above. Choose neutral_halfwidth from the DMS
    assay's own measurement noise / replicate variance if available -- don't
    pick it arbitrarily.
    """
    n = len(dms_scores)
    out = np.full(n, np.nan)
    for i in range(n):
        neigh = adj.get(i, [])
        if len(neigh) == 0:
            continue
        diffs = np.abs(dms_scores[neigh] - dms_scores[i])
        out[i] = np.mean(diffs <= neutral_halfwidth)
    return out


def experimental_evolvability(dms_scores: np.ndarray, adj: dict[int, list[int]],
                               n_bins: int = 10, bin_edges: np.ndarray | None = None) -> np.ndarray:
    """
    Evo_exp(sigma) = number of DISTINCT phenotype bins reachable among
    sigma's Hamming-1 neighbors, excluding sigma's own bin (classical
    neutral-network "accessible novel phenotypes" definition). Bins are
    computed globally (same edges for every genotype) so counts are
    comparable across genotypes.
    """
    if bin_edges is None:
        bin_edges = np.histogram_bin_edges(dms_scores, bins=n_bins)
    bin_id = np.digitize(dms_scores, bin_edges)

    n = len(dms_scores)
    out = np.full(n, np.nan)
    for i in range(n):
        neigh = adj.get(i, [])
        if len(neigh) == 0:
            continue
        neighbor_bins = set(bin_id[neigh]) - {bin_id[i]}
        out[i] = len(neighbor_bins)
    return out


# ---------------------------------------------------------------------------
# Latent (embedding-derived) candidate proxies -- hypotheses, not ground truth
# ---------------------------------------------------------------------------

def latent_robustness_proxy(embeddings: np.ndarray, adj: dict[int, list[int]]) -> np.ndarray:
    """
    The original candidate: R(sigma) = mean_{tau in N(sigma)} exp(-d_L(sigma,tau)).
    Kept exactly as proposed, but it must be validated against
    experimental_robustness()/experimental_robustness_neutral_band(), not
    assumed to equal biophysical robustness -- it is a measure of local
    embedding contraction, which may or may not track phenotype uniformity.
    """
    n = embeddings.shape[0]
    out = np.full(n, np.nan)
    for i in range(n):
        neigh = adj.get(i, [])
        if len(neigh) == 0:
            continue
        d = np.linalg.norm(embeddings[neigh] - embeddings[i], axis=1)
        out[i] = np.mean(np.exp(-d))
    return out


def latent_neighbor_score_spread(embeddings: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Alternative evolvability proxy: variance of... no, distinct-region count
    of a genotype's k NEAREST NEIGHBORS IN EMBEDDING SPACE (not Hamming
    space). More candidate directions of phenotypic change accessible in
    latent geometry -> proxy for evolvability. Requires no DMS data at all,
    which is the point -- it's a zero-shot geometric evolvability proxy to
    test against experimental_evolvability().
    """
    from scipy.spatial.distance import cdist
    n = embeddings.shape[0]
    D = cdist(embeddings, embeddings)
    np.fill_diagonal(D, np.inf)
    knn_idx = np.argsort(D, axis=1)[:, :k]
    # local spread = mean pairwise distance among the k nearest neighbors
    out = np.zeros(n)
    for i in range(n):
        sub = embeddings[knn_idx[i]]
        out[i] = np.mean(cdist(sub, sub))
    return out


# ---------------------------------------------------------------------------
# Cross-modal validation
# ---------------------------------------------------------------------------

def correlate_ground_truth_vs_latent(exp_values: np.ndarray, latent_values: np.ndarray) -> dict:
    """
    Spearman correlation between an experimental quantity and its candidate
    latent proxy, over indices where both are defined (non-NaN). This is the
    actual test of Objective 3/4's central hypothesis -- report this number,
    not the latent proxy alone.
    """
    mask = ~(np.isnan(exp_values) | np.isnan(latent_values))
    if mask.sum() < 10:
        return {"spearman_rho": np.nan, "p_value": np.nan, "n": int(mask.sum())}
    rho, p = spearmanr(exp_values[mask], latent_values[mask])
    return {"spearman_rho": float(rho), "p_value": float(p), "n": int(mask.sum())}


def robustness_evolvability_tradeoff(robustness: np.ndarray, evolvability: np.ndarray,
                                      n_bins: int = 10) -> dict:
    """
    Bins genotypes by robustness and reports mean evolvability per bin, to
    check for the Draghi et al. (2010) non-monotonic (peaked at intermediate
    robustness) relationship -- a concrete, falsifiable target rather than
    an open-ended 'discover something' search.
    """
    mask = ~(np.isnan(robustness) | np.isnan(evolvability))
    r, e = robustness[mask], evolvability[mask]
    if r.size < n_bins:
        raise ValueError(
            f"Only {r.size} genotypes have both robustness and evolvability "
            f"defined (non-NaN) -- too few for {n_bins} bins. This usually "
            f"means the neighbor graph has near-zero edges; check that "
            f"Variant.substitutions was populated correctly (see "
            f"pipeline._variants_from_df) before build_neighbor_graph()."
        )
    edges = np.quantile(r, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-9
    bin_id = np.digitize(r, edges) - 1
    bin_id = np.clip(bin_id, 0, n_bins - 1)
    means, centers = [], []
    for b in range(n_bins):
        sel = bin_id == b
        if sel.sum() == 0:
            continue
        means.append(e[sel].mean())
        centers.append(r[sel].mean())
    return {"robustness_bin_centers": np.array(centers), "mean_evolvability": np.array(means)}
