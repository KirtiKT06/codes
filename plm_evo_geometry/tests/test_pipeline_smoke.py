"""
test_pipeline_smoke.py

End-to-end smoke test: runs run_full_analysis() on a fully synthetic
protein + fake multi-layer embeddings, just to confirm the whole pipeline
(Objectives 1-4 wired together) executes without errors and produces the
expected output files. This does NOT validate scientific correctness of a
real result (that's what test_synthetic.py's targeted, known-ground-truth
tests are for) -- it validates plumbing.
"""
import sys, os, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from src.sequence_utils import generate_single_mutants, AMINO_ACIDS
from src.pipeline import run_full_analysis


def test_pipeline_runs_end_to_end():
    rng = np.random.default_rng(42)
    wt_seq = "".join(rng.choice(AMINO_ACIDS, size=15))
    positions = list(range(15))
    variants = generate_single_mutants(wt_seq, positions=positions)
    variants = list(rng.choice(variants, size=120, replace=False))

    dms_df = pd.DataFrame({
        "seq": [v.seq for v in variants],
        "position": [v.positions[0] for v in variants],
        "dms_score": rng.normal(1.0, 0.3, size=len(variants)),
    })

    n = len(dms_df) + 1
    # Fake 3-layer embeddings of different "hidden dims", none biologically
    # meaningful -- purely to exercise the pipeline's plumbing end to end.
    embeddings_by_layer = {
        0: rng.normal(size=(n, 32)),
        6: rng.normal(size=(n, 32)) * 0.1 + rng.normal(size=(1, 32)),  # tighter cluster -> lower ID
        12: rng.normal(size=(n, 32)),
    }

    out_dir = "/tmp/plm_pipeline_smoke_test"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    summary = run_full_analysis(wt_seq, dms_df, embeddings_by_layer,
                                 protein_name="SYNTH", out_dir=out_dir)

    expected_files = [
        "objective1_intrinsic_dimension.csv", "objective2_q1_table.csv",
        "objective2_q1_stats.json", "objective2_q2_per_position.csv",
        "objective34_summary.json", "objective34_arrays.npz",
        "objective4_diffusion_map.npz", "SUMMARY.json",
    ]
    for fname in expected_files:
        path = os.path.join(out_dir, fname)
        assert os.path.exists(path), f"missing output file: {fname}"

    assert summary["n_variants"] == 120
    assert summary["analysis_layer_used"] in embeddings_by_layer
    print("[pipeline smoke test] all expected outputs written; summary keys:",
          list(summary.keys()))


if __name__ == "__main__":
    test_pipeline_runs_end_to_end()
    print("PASS: pipeline smoke test")
