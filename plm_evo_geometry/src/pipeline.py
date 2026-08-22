"""
pipeline.py

Orchestrates the full analysis for ONE protein, given (a) a WT sequence,
(b) a DMS dataframe with columns ['seq', 'position', 'dms_score'] for single
mutants (position = 0-indexed mutated position; add 'position2' for doubles
if you have them), and (c) precomputed ESM-2 embeddings (from embeddings.py,
run separately on your GPU machine).

This module does NOT call ESM-2 itself -- it consumes the .npz produced by
embeddings.save_embeddings(). Keeps the network/torch-dependent step fully
separate from the analysis step, which is what actually needs to be
correct and is what's been unit-tested in tests/test_synthetic.py.

Typical usage (see notebooks/01_pilot_GB1.py for a worked example):

    from src.pipeline import run_full_analysis
    results = run_full_analysis(wt_seq, dms_df, embeddings_by_layer,
                                 protein_name="GB1", out_dir="results/GB1")
"""

from __future__ import annotations
import os
import json
import numpy as np
import pandas as pd

from .sequence_utils import Variant, build_neighbor_graph, hamming_distance
from .intrinsic_dim import two_nn_dimension_stability
from .geometry_analysis import q1_analysis, q2_per_position
from .robustness_evolvability import (
    experimental_robustness, experimental_evolvability,
    latent_robustness_proxy, latent_neighbor_score_spread,
    correlate_ground_truth_vs_latent, robustness_evolvability_tradeoff,
)
from .cv_discovery import diffusion_map


def _variants_from_df(dms_df: pd.DataFrame, wt_seq: str) -> list[Variant]:
    """
    Derives .substitutions directly from the sequence diff against WT,
    rather than trusting a 'position' column, so the neighbor graph is
    always self-consistent with the actual sequences even if a position
    column is missing, stale, or off-by-one. build_neighbor_graph() needs
    real (wt_res, pos, mut_res) tuples -- leaving substitutions empty
    silently produces an edgeless graph and NaN everywhere downstream
    (found via the pipeline smoke test).
    """
    variants = []
    for _, row in dms_df.iterrows():
        seq = row["seq"]
        if len(seq) != len(wt_seq):
            raise ValueError(f"Variant length {len(seq)} != WT length {len(wt_seq)}")
        diffs = tuple((wt_seq[i], i, seq[i]) for i in range(len(wt_seq)) if seq[i] != wt_seq[i])
        positions = tuple(sorted(p for _, p, _ in diffs))
        variants.append(Variant(seq=seq, positions=positions,
                                 substitutions=diffs, dms_score=row["dms_score"]))
    return variants


def run_geometry_objective1(embeddings_by_layer: dict[int, np.ndarray],
                             n_subsamples: int = 20) -> pd.DataFrame:
    """
    Objective 1: TwoNN intrinsic dimension per layer, with stability check.
    Run this across ALL extracted layers to locate the plateau layer
    empirically (Valeriani et al. 2022), rather than assuming last-layer.
    """
    rows = []
    for layer, emb in sorted(embeddings_by_layer.items()):
        d_mean, d_std, _ = two_nn_dimension_stability(emb, n_subsamples=n_subsamples)
        rows.append({"layer": layer, "id_mean": d_mean, "id_std": d_std,
                     "embedding_dim": emb.shape[1]})
    df = pd.DataFrame(rows)
    return df


def run_full_analysis(wt_seq: str, dms_df: pd.DataFrame,
                       embeddings_by_layer: dict[int, np.ndarray],
                       protein_name: str, out_dir: str,
                       analysis_layer: int | None = None,
                       wt_dms_score: float = 0.0) -> dict:
    """
    embeddings_by_layer[layer] must be (n_variants + 1, hidden_dim), with
    ROW 0 = WT and rows 1..N matching dms_df row order exactly -- get this
    alignment right or everything downstream is silently wrong. Consider
    asserting len(dms_df) + 1 == embeddings_by_layer[any_layer].shape[0]
    before calling this.

    analysis_layer: which layer's embeddings to use for Objectives 2-4.
    If None, uses the layer with the lowest mean TwoNN ID (the empirical
    plateau) from run_geometry_objective1 -- i.e., let the geometry
    characterization from Objective 1 choose the layer for Objectives 2-4,
    rather than picking one in advance.
    """
    os.makedirs(out_dir, exist_ok=True)
    n_variants = len(dms_df)
    for layer, emb in embeddings_by_layer.items():
        assert emb.shape[0] == n_variants + 1, (
            f"Layer {layer}: embeddings has {emb.shape[0]} rows, expected "
            f"{n_variants + 1} (WT + {n_variants} variants). Check alignment."
        )

    # ---- Objective 1: geometry / intrinsic dimension per layer ----
    geom_df = run_geometry_objective1(embeddings_by_layer)
    geom_df.to_csv(os.path.join(out_dir, "objective1_intrinsic_dimension.csv"), index=False)

    if analysis_layer is None:
        analysis_layer = int(geom_df.loc[geom_df["id_mean"].idxmin(), "layer"])
    emb = embeddings_by_layer[analysis_layer]
    wt_emb, var_emb = emb[0], emb[1:]

    # ---- Objective 2: embedding distance vs genotype/phenotype (Q1, Q2) ----
    df = dms_df.copy().reset_index(drop=True)
    df["embed_dist_to_wt"] = np.linalg.norm(var_emb - wt_emb, axis=1)
    q1_out = q1_analysis(df, wt_seq, embed_dist_col="embed_dist_to_wt",
                          dms_col="dms_score", wt_dms=wt_dms_score)
    q1_out["table"].to_csv(os.path.join(out_dir, "objective2_q1_table.csv"), index=False)
    with open(os.path.join(out_dir, "objective2_q1_stats.json"), "w") as f:
        json.dump(q1_out["stats"], f, indent=2, default=float)

    q2_df = q2_per_position(q1_out["table"], position_col="position",
                             embed_dist_col="embed_dist_to_wt",
                             dms_col="dms_score", wt_dms=wt_dms_score)
    q2_df.to_csv(os.path.join(out_dir, "objective2_q2_per_position.csv"), index=False)

    # ---- Objectives 3-4: robustness / evolvability / CV discovery ----
    variants = _variants_from_df(df, wt_seq)
    adj = build_neighbor_graph(wt_seq, variants)
    all_scores = np.concatenate([[wt_dms_score], df["dms_score"].values])
    all_emb = np.vstack([wt_emb[None, :], var_emb])

    R_exp = experimental_robustness(all_scores, adj)
    Evo_exp = experimental_evolvability(all_scores, adj)
    R_latent = latent_robustness_proxy(all_emb, adj)
    Evo_latent = latent_neighbor_score_spread(all_emb, k=10)

    robustness_corr = correlate_ground_truth_vs_latent(R_exp, R_latent)
    evolvability_corr = correlate_ground_truth_vs_latent(Evo_exp, Evo_latent)
    tradeoff = robustness_evolvability_tradeoff(R_exp, Evo_exp)

    obj34_summary = {
        "analysis_layer": analysis_layer,
        "robustness_ground_truth_vs_latent_proxy": robustness_corr,
        "evolvability_ground_truth_vs_latent_proxy": evolvability_corr,
    }
    with open(os.path.join(out_dir, "objective34_summary.json"), "w") as f:
        json.dump(obj34_summary, f, indent=2, default=float)

    np.savez_compressed(
        os.path.join(out_dir, "objective34_arrays.npz"),
        R_exp=R_exp, Evo_exp=Evo_exp, R_latent=R_latent, Evo_latent=Evo_latent,
        tradeoff_bin_centers=tradeoff["robustness_bin_centers"],
        tradeoff_mean_evolvability=tradeoff["mean_evolvability"],
    )

    # ---- Objective 4b: diffusion-map collective variables ----
    n_dm = min(5, all_emb.shape[0] - 2)
    dm_embedding, dm_eigvals = diffusion_map(all_emb, n_components=n_dm)
    np.savez_compressed(os.path.join(out_dir, "objective4_diffusion_map.npz"),
                         embedding=dm_embedding, eigenvalues=dm_eigvals)
    # quick diagnostic: correlate each diffusion coordinate with DMS score,
    # so you immediately see if a low-dimensional CV tracks fitness.
    from scipy.stats import spearmanr
    dm_corrs = [float(spearmanr(dm_embedding[:, i], all_scores)[0]) for i in range(dm_embedding.shape[1])]

    summary = {
        "protein_name": protein_name,
        "n_variants": n_variants,
        "analysis_layer_used": analysis_layer,
        "intrinsic_dimension_by_layer": geom_df.to_dict(orient="records"),
        "q1_stats": {k: v for k, v in q1_out["stats"].items() if k != "interpretation"},
        "robustness_correlation": robustness_corr,
        "evolvability_correlation": evolvability_corr,
        "diffusion_coord_vs_dms_spearman": dm_corrs,
    }
    with open(os.path.join(out_dir, "SUMMARY.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)

    print(f"[{protein_name}] analysis complete -> {out_dir}")
    print(f"  plateau/analysis layer: {analysis_layer} (ID={geom_df.set_index('layer').loc[analysis_layer, 'id_mean']:.2f})")
    print(f"  robustness  ground-truth vs latent proxy: rho={robustness_corr['spearman_rho']:.3f} (n={robustness_corr['n']})")
    print(f"  evolvability ground-truth vs latent proxy: rho={evolvability_corr['spearman_rho']:.3f} (n={evolvability_corr['n']})")
    print(f"  diffusion coord 1 vs DMS score: rho={dm_corrs[0]:.3f}")

    return summary
