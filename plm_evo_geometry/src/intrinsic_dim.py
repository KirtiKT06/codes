"""
intrinsic_dim.py

TwoNN intrinsic dimension estimator (Facco, d'Errico, Rodriguez & Laio,
Scientific Reports 2017), the estimator used by Valeriani et al. (2022) to
characterize pLM hidden-layer geometry. Deliberately NOT PCA-explained-
variance or UMAP-by-eye -- both of those show apparent low dimensionality
on almost any point cloud regardless of whether it reflects real structure,
which is exactly the confound Objective 1 / Q5 need to rule out.

Also provides a local (per-point) MLE estimator as a cross-check, and a
subsample-based stability routine so you report an ID with a spread, not a
single suspiciously-precise number.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform


def two_nn_dimension(X: np.ndarray, discard_fraction: float = 0.1,
                      return_details: bool = False):
    """
    Global intrinsic dimension via the TwoNN estimator.

    X : (n_samples, n_features)
    discard_fraction : fraction of points with the largest mu = r2/r1 ratio
        to discard before the regression (Facco et al. recommend up to ~10%
        to reduce noise from the tail of the mu distribution).

    Returns d_hat (float), or (d_hat, details_dict) if return_details=True.
    """
    n = X.shape[0]
    if n < 20:
        raise ValueError("TwoNN is unreliable below ~20 points; got %d" % n)

    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    sorted_D = np.sort(D, axis=1)
    r1, r2 = sorted_D[:, 0], sorted_D[:, 1]

    valid = r1 > 1e-12
    mu = r2[valid] / r1[valid]
    mu = mu[mu > 1.0]  # mu must be > 1 by construction; guards float noise
    n_valid = len(mu)
    if n_valid < 20:
        raise ValueError("Too few valid points after filtering duplicates.")

    mu_sorted = np.sort(mu)
    F_emp = np.arange(1, n_valid + 1) / n_valid  # empirical CDF (order statistics)

    keep = int(np.floor(n_valid * (1 - discard_fraction)))
    keep = max(keep, 10)
    x = np.log(mu_sorted[:keep])
    y = -np.log(1 - F_emp[:keep])

    # Linear regression through the origin: y = d * x
    d_hat = float(np.sum(x * y) / np.sum(x * x))

    if return_details:
        return d_hat, {"x": x, "y": y, "mu_sorted": mu_sorted, "n_used": keep, "n_total": n}
    return d_hat


def two_nn_dimension_stability(X: np.ndarray, n_subsamples: int = 20,
                                subsample_frac: float = 0.8,
                                discard_fraction: float = 0.1,
                                random_state: int = 0):
    """
    Bootstrap-style stability check: re-estimate TwoNN ID on random
    subsamples and report mean +/- std, following the decimation approach
    used in Valeriani et al. (2022). A single point estimate without this
    is not trustworthy -- report this, not just two_nn_dimension() alone.
    """
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    sub_n = max(int(n * subsample_frac), 21)
    estimates = []
    for _ in range(n_subsamples):
        idx = rng.choice(n, size=sub_n, replace=False)
        try:
            d = two_nn_dimension(X[idx], discard_fraction=discard_fraction)
            estimates.append(d)
        except ValueError:
            continue
    estimates = np.array(estimates)
    return float(estimates.mean()), float(estimates.std()), estimates


def local_mle_dimension(X: np.ndarray, k: int = 10) -> np.ndarray:
    """
    Levina-Bickel local MLE dimension estimator, per point, using k nearest
    neighbors. Useful as a cross-check against TwoNN, and for looking at
    HOW ID varies across the landscape (e.g. near WT vs. far from WT) rather
    than only a single global number -- local intrinsic dimension is often
    the more informative quantity per Kim/Chang-style contextual-embedding
    geometry work.
    """
    n = X.shape[0]
    D = squareform(pdist(X))
    np.fill_diagonal(D, np.inf)
    sorted_D = np.sort(D, axis=1)[:, :k]
    # Levina-Bickel MLE: d_hat(x) = [ (1/(k-1)) sum_{j=1}^{k-1} log(r_k / r_j) ]^-1
    r_k = sorted_D[:, k - 1:k]
    with np.errstate(divide="ignore"):
        log_ratios = np.log(r_k / sorted_D[:, :k - 1])
    log_ratios[~np.isfinite(log_ratios)] = np.nan
    mean_log_ratio = np.nanmean(log_ratios, axis=1)
    d_local = 1.0 / mean_log_ratio
    return d_local
