"""
cv_discovery.py

Objective 4 ("infer candidate macroscopic variables") should not default to
linear PCA -- PCA finds high-variance directions, not necessarily the
directions that organize phenotype. This module treats each mutant
embedding as a "microstate" (the same way you'd treat an MD snapshot) and
uses a diffusion map (Coifman & Lafon 2006) to find low-dimensional,
non-linear coordinates that respect the local connectivity of the mutant
graph -- structurally the same problem as slow-CV discovery from an MD
trajectory, just applied to a mutational graph instead of a time series.

If you want a literal TICA-style analysis, you need a time-like ordering.
One principled way to get one on a mutational landscape: simulate random
walks on the Hamming-1 neighbor graph (built in sequence_utils.py) starting
from WT, generating pseudo-trajectories of embeddings, and hand those to a
standard TICA implementation (e.g. deeptime, PyEMMA) exactly as you would
MD frames. That is a natural extension once diffusion maps establish that
non-linear structure exists worth chasing with a "slow mode" formalism --
see random_walk_trajectory() below.
"""

from __future__ import annotations
import numpy as np
from scipy.spatial.distance import pdist, squareform


def diffusion_map(X: np.ndarray, n_components: int = 5, epsilon: float | None = None,
                   alpha: float = 1.0):
    """
    Basic diffusion map embedding.

    X : (n_samples, n_features)
    epsilon : kernel bandwidth. If None, uses the median of squared pairwise
        distances (a standard heuristic; sensitivity-check this).
    alpha : density-normalization exponent. alpha=1 approximates the
        Laplace-Beltrami operator (recommended default, removes sampling-
        density artifacts); alpha=0 gives the unnormalized graph Laplacian.

    Returns (embedding, eigenvalues) where embedding is
    (n_samples, n_components), using eigenvectors 2..n_components+1 (the
    trivial constant eigenvector/eigenvalue=1 is dropped).
    """
    D2 = squareform(pdist(X, metric="sqeuclidean"))
    if epsilon is None:
        nonzero = D2[D2 > 0]
        epsilon = np.median(nonzero) if nonzero.size else 1.0

    K = np.exp(-D2 / epsilon)

    if alpha != 0:
        d = K.sum(axis=1)
        d_safe = np.where(d > 0, d, 1.0)
        d_alpha = np.power(d_safe, alpha)
        K = K / np.outer(d_alpha, d_alpha)

    d2 = K.sum(axis=1)
    d2[d2 == 0] = 1e-12
    Dinv_sqrt = 1.0 / np.sqrt(d2)
    # Symmetric conjugate of the transition matrix P = D^-1 K, so we can use
    # a stable symmetric eigensolver instead of a general (possibly complex)
    # eigendecomposition of the row-stochastic P directly.
    Ms = (Dinv_sqrt[:, None] * K) * Dinv_sqrt[None, :]
    eigvals, eigvecs = np.linalg.eigh(Ms)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    psi = Dinv_sqrt[:, None] * eigvecs  # right eigenvectors of P = D^-1 K

    k = min(n_components, psi.shape[1] - 1)
    embedding = psi[:, 1:k + 1] * eigvals[1:k + 1]
    return embedding, eigvals[1:k + 1]


def random_walk_trajectory(adj: dict[int, list[int]], embeddings: np.ndarray,
                            start: int = 0, n_steps: int = 5000,
                            random_state: int = 0) -> np.ndarray:
    """
    Generates a pseudo-trajectory of embeddings by random-walking the
    Hamming-1 neighbor graph, for handing to a TICA implementation the same
    way you'd hand it MD frames. Walks that hit a dead end (no neighbors)
    restart from `start`. This is a genuine modeling choice (uniform random
    walk = neutral drift assumption) -- consider biasing steps by DMS score
    if you want to model selection rather than pure neutral exploration.
    """
    rng = np.random.default_rng(random_state)
    traj_idx = [start]
    current = start
    for _ in range(n_steps - 1):
        neigh = adj.get(current, [])
        if not neigh:
            current = start
        else:
            current = int(rng.choice(neigh))
        traj_idx.append(current)
    return embeddings[np.array(traj_idx)]
