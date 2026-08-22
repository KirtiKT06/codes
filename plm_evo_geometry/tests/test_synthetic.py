"""
test_synthetic.py

Validates the pure-numpy modules (no ESM-2 / torch / network required)
against synthetic data with KNOWN ground truth, before you trust them on
real DMS + embedding data:

  1. TwoNN recovers the correct intrinsic dimension on point clouds with a
     known true dimension embedded in high-dimensional space.
  2. Hamming / BLOSUM distance functions behave correctly on hand-checkable
     sequences.
  3. The neighbor-graph builder produces the correct Hamming-1 adjacency for
     a small synthetic single+double mutant library, checked against a brute
     -force O(n^2) Hamming scan.
  4. Ground-truth robustness/evolvability recover sensible values on a
     synthetic landscape with an injected neutral network, and the
     robustness-evolvability tradeoff helper reproduces a designed
     non-monotonic relationship.
  5. Diffusion map recovers the correct 1D latent ordering on points sampled
     along a nonlinear (S-curve) manifold embedded in high dimensions.

Run: python -m pytest tests/test_synthetic.py -v
  (or just: python tests/test_synthetic.py)
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from scipy.stats import spearmanr

from src.intrinsic_dim import two_nn_dimension, two_nn_dimension_stability, local_mle_dimension
from src.sequence_utils import (hamming_distance, blosum_distance, generate_single_mutants,
                                 generate_double_mutants, build_neighbor_graph, AMINO_ACIDS)
from src.robustness_evolvability import (experimental_robustness, experimental_evolvability,
                                          latent_robustness_proxy, correlate_ground_truth_vs_latent,
                                          robustness_evolvability_tradeoff)
from src.cv_discovery import diffusion_map


def test_twonn_recovers_known_dimension():
    rng = np.random.default_rng(0)
    n = 2000

    # Ground truth: 2D uniform square, isometrically embedded in 50D
    # (random orthonormal embedding preserves distances, so true ID = 2).
    low = rng.uniform(0, 1, size=(n, 2))
    Q, _ = np.linalg.qr(rng.normal(size=(50, 50)))
    embed_matrix = Q[:, :2]
    X = low @ embed_matrix.T
    X += rng.normal(scale=1e-4, size=X.shape)  # tiny noise to avoid exact duplicates

    d_hat = two_nn_dimension(X)
    print(f"[TwoNN] true ID=2 (uniform square), estimated={d_hat:.2f}")
    assert 1.5 < d_hat < 2.7, f"expected ~2, got {d_hat}"

    # Ground truth: 5D Gaussian ball, embedded in 100D
    low5 = rng.normal(size=(n, 5))
    Q2, _ = np.linalg.qr(rng.normal(size=(100, 100)))
    X5 = low5 @ Q2[:, :5].T
    d_hat5, d_std5, _ = two_nn_dimension_stability(X5, n_subsamples=15)
    print(f"[TwoNN] true ID=5 (Gaussian ball), estimated={d_hat5:.2f} +/- {d_std5:.2f}")
    assert 3.8 < d_hat5 < 6.5, f"expected ~5, got {d_hat5}"


def test_local_mle_matches_global_scale():
    rng = np.random.default_rng(1)
    n = 1000
    low = rng.uniform(0, 1, size=(n, 3))
    Q, _ = np.linalg.qr(rng.normal(size=(30, 30)))
    X = low @ Q[:, :3].T
    d_local = local_mle_dimension(X, k=15)
    med = np.median(d_local)
    print(f"[local MLE] true ID=3, median local estimate={med:.2f}")
    assert 2.0 < med < 4.5


def test_hamming_and_blosum_distance():
    assert hamming_distance("ACDE", "ACDE") == 0
    assert hamming_distance("ACDE", "ACDF") == 1
    assert hamming_distance("ACDE", "AAAA") == 3

    # Conservative substitution (I<->L, both hydrophobic) should score
    # "cheaper" (smaller distance) than a radical one (D<->W, charged<->big
    # hydrophobic) at the same Hamming distance of 1.
    conservative = blosum_distance("AILE", "AILE".replace("I", "L", 1))
    radical = blosum_distance("ADLE", "ADLE".replace("D", "W", 1))
    print(f"[BLOSUM] conservative(I->L) dist={conservative:.2f}, radical(D->W) dist={radical:.2f}")
    assert conservative < radical


def test_neighbor_graph_matches_bruteforce():
    rng = np.random.default_rng(2)
    wt = "".join(rng.choice(AMINO_ACIDS, size=8))
    positions = list(range(8))
    singles = generate_single_mutants(wt, positions=positions)
    # subsample singles to keep brute force cheap
    singles = list(rng.choice(singles, size=40, replace=False))
    doubles = generate_double_mutants(wt, positions=positions[:4], max_pairs=6)
    doubles = list(rng.choice(doubles, size=min(20, len(doubles)), replace=False))
    variants = singles + doubles

    adj = build_neighbor_graph(wt, variants)

    all_seqs = [wt] + [v.seq for v in variants]
    n = len(all_seqs)
    brute_adj = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if hamming_distance(all_seqs[i], all_seqs[j]) == 1:
                brute_adj[i].append(j)
                brute_adj[j].append(i)
    for k in brute_adj:
        brute_adj[k] = sorted(brute_adj[k])

    mismatches = sum(1 for i in range(n) if adj[i] != brute_adj[i])
    print(f"[neighbor graph] {n} genotypes, {mismatches} mismatched adjacency rows vs brute force")
    assert mismatches == 0


def test_robustness_evolvability_on_synthetic_landscape():
    """
    Build a small synthetic single-mutant landscape around a fake WT where
    a KNOWN subset of positions are "neutral" (mutations barely change
    phenotype) and others are "sensitive" (mutations strongly change
    phenotype). Ground-truth robustness should be higher at neutral
    positions than sensitive ones.
    """
    rng = np.random.default_rng(3)
    wt = "".join(rng.choice(AMINO_ACIDS, size=12))
    neutral_positions = [0, 1, 2, 3]
    sensitive_positions = [4, 5, 6, 7]
    all_positions = neutral_positions + sensitive_positions

    variants = generate_single_mutants(wt, positions=all_positions)
    adj = build_neighbor_graph(wt, variants)

    n = len(variants) + 1
    dms = np.zeros(n)
    dms[0] = 1.0  # WT fitness
    for i, v in enumerate(variants, start=1):
        pos = v.positions[0]
        if pos in neutral_positions:
            dms[i] = 1.0 + rng.normal(scale=0.02)   # tiny phenotype change
        else:
            dms[i] = 1.0 + rng.normal(scale=0.02) - rng.uniform(0.3, 0.9)  # big drop

    R = experimental_robustness(dms, adj)
    neutral_idx = [i + 1 for i, v in enumerate(variants) if v.positions[0] in neutral_positions]
    sensitive_idx = [i + 1 for i, v in enumerate(variants) if v.positions[0] in sensitive_positions]

    mean_R_neutral = np.nanmean(R[neutral_idx])
    mean_R_sensitive = np.nanmean(R[sensitive_idx])
    print(f"[robustness] mean R at neutral positions={mean_R_neutral:.3f}, "
          f"sensitive positions={mean_R_sensitive:.3f}")
    assert mean_R_neutral > mean_R_sensitive


def test_correlate_ground_truth_vs_latent_proxy_recovers_signal():
    """
    Construct a latent embedding that is DELIBERATELY informative about
    ground-truth robustness (small perturbations for genotypes we tag
    'robust'), and check correlate_ground_truth_vs_latent picks up a
    strong positive correlation -- i.e. the pipeline can detect a real
    signal when one is deliberately planted, which is the sanity check you
    want before trusting a null result on real data.
    """
    rng = np.random.default_rng(4)
    wt = "".join(rng.choice(AMINO_ACIDS, size=10))
    positions = list(range(10))
    variants = generate_single_mutants(wt, positions=positions)
    adj = build_neighbor_graph(wt, variants)
    n = len(variants) + 1

    dms = rng.normal(1.0, 0.05, size=n)
    robust_positions = set(positions[:5])
    for i, v in enumerate(variants, start=1):
        if v.positions[0] not in robust_positions:
            dms[i] -= rng.uniform(0.2, 0.6)

    R_exp = experimental_robustness(dms, adj)

    # Build a toy embedding where "robust" genotypes are tightly clustered
    # and "sensitive" genotypes are spread out -- by construction this
    # should predict R_exp well.
    dim = 8
    base = rng.normal(size=dim)
    emb = np.zeros((n, dim))
    emb[0] = base
    for i, v in enumerate(variants, start=1):
        if v.positions[0] in robust_positions:
            emb[i] = base + rng.normal(scale=0.05, size=dim)
        else:
            emb[i] = base + rng.normal(scale=2.0, size=dim)

    R_latent = latent_robustness_proxy(emb, adj)
    result = correlate_ground_truth_vs_latent(R_exp, R_latent)
    print(f"[cross-modal] planted-signal correlation: rho={result['spearman_rho']:.3f}, "
          f"p={result['p_value']:.2e}, n={result['n']}")
    assert result["spearman_rho"] > 0.4


def test_robustness_evolvability_tradeoff_shape():
    rng = np.random.default_rng(5)
    n = 500
    # Design robustness/evolvability with a known peaked (non-monotonic)
    # relationship: evolvability = -(robustness - 0.5)^2 + noise
    robustness = rng.uniform(0, 1, size=n)
    evolvability = -(robustness - 0.5) ** 2 * 4 + 1.0 + rng.normal(scale=0.05, size=n)

    out = robustness_evolvability_tradeoff(robustness, evolvability, n_bins=8)
    centers, means = out["robustness_bin_centers"], out["mean_evolvability"]
    peak_bin = np.argmax(means)
    print(f"[tradeoff] peak evolvability at robustness~{centers[peak_bin]:.2f} (designed peak=0.5)")
    assert 0.25 < centers[peak_bin] < 0.75


def test_diffusion_map_recovers_1d_manifold_ordering():
    """
    Points sampled along a 1D nonlinear curve (helix) embedded in high-D
    should be recoverable, IN ORDER, by the leading diffusion-map
    coordinate -- unlike PCA, which would not linearly separate a curved
    manifold this way if it curves back on itself.
    """
    rng = np.random.default_rng(6)
    n = 400
    t = np.sort(rng.uniform(0, 4 * np.pi, size=n))
    curve = np.column_stack([np.cos(t), np.sin(t), 0.3 * t])  # helix in 3D
    Q, _ = np.linalg.qr(rng.normal(size=(40, 40)))
    X = curve @ Q[:3, :40]
    X += rng.normal(scale=0.01, size=X.shape)

    emb, eigvals = diffusion_map(X, n_components=3)
    rho, p = spearmanr(emb[:, 0], t)
    print(f"[diffusion map] |Spearman rho| between DC1 and true helix parameter t: {abs(rho):.3f}")
    assert abs(rho) > 0.9


if __name__ == "__main__":
    tests = [v for k, v in list(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}\n")
        except AssertionError as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {e}\n")
    print(f"{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(1 if failed else 0)
