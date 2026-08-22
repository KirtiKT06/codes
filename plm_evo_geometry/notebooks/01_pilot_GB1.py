"""
01_pilot_GB1.py

Worked pilot for GB1 -- start here, since you already have GB1 DMS labels
(Nisthal et al. 2019) and an ESM-2 embedding pipeline from the active-
learning stability project. This script shows the two stages explicitly:

  STAGE A (run on your GPU machine, needs torch + network access to HF):
      extract multi-layer ESM-2 embeddings for the full GB1 single-mutant
      set and save them to disk.

  STAGE B (runs anywhere, pure numpy/scipy -- this is what's unit-tested):
      load the saved embeddings + DMS scores and run the full Objective 1-4
      analysis via src.pipeline.run_full_analysis().

Replace the placeholders marked TODO with your actual GB1 DMS CSV and WT
sequence. If you already have ESM-2 embeddings saved from the stability
project, skip straight to STAGE B and adapt the loader to whatever format
those are already in (just make sure row 0 = WT and rows 1..N match your
dms_df row order exactly -- pipeline.run_full_analysis() asserts this).
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# STAGE A -- ESM-2 embedding extraction (run on GPU machine)
# ---------------------------------------------------------------------------
def stage_a_extract_embeddings():
    from src.embeddings import extract_esm2_embeddings, save_embeddings

    # TODO: replace with your actual GB1 WT sequence and DMS dataframe.
    # Nisthal et al. 2019 (ddG) or the Olson et al. 2014 GB1 binding DMS --
    # whichever you used for the active-learning project -- both work here,
    # just be consistent about what "dms_score" means (ddG vs. fitness sign
    # conventions differ between the two datasets).
    wt_seq = "MTYKLILNGKTLKGETTTEAVDAATAEKVFKQYANDNGVDGEWTYDDATKTFTVTE"  # GB1 domain, example
    dms_df = pd.read_csv("data/GB1_dms.csv")  # TODO: columns ['seq', 'position', 'dms_score']

    all_seqs = [wt_seq] + list(dms_df["seq"])
    mutated_positions = [None] + list(dms_df["position"])  # None for WT (mean-pool WT fully)

    # Extract ALL layers on a subset first to find the plateau layer cheaply,
    # then re-run with layers=[plateau_layer] on the full set if you want to
    # save disk space -- for GB1 (~56 residues, ~1000 variants) keeping all
    # layers is fine even on a single GPU.
    embeddings = extract_esm2_embeddings(
        all_seqs, model_name="esm2_t33_650M", layers=None,
        device="cuda", batch_size=16, pooling="mean",
    )
    save_embeddings("data/GB1_esm2_650M_embeddings.npz", embeddings, all_seqs)
    print("Saved embeddings for", len(all_seqs), "sequences across",
          len(embeddings), "layers.")


# ---------------------------------------------------------------------------
# STAGE B -- geometry / robustness / evolvability / CV-discovery analysis
# ---------------------------------------------------------------------------
def stage_b_run_analysis():
    from src.embeddings import load_embeddings
    from src.pipeline import run_full_analysis

    sequences, embeddings_by_layer = load_embeddings("data/GB1_esm2_650M_embeddings.npz")
    wt_seq = sequences[0]

    dms_df = pd.read_csv("data/GB1_dms.csv")  # TODO: same file as Stage A
    assert list(dms_df["seq"]) == sequences[1:], (
        "Row order mismatch between dms_df and saved embeddings -- Stage A "
        "and Stage B must use the exact same sequence order."
    )

    summary = run_full_analysis(
        wt_seq=wt_seq,
        dms_df=dms_df,
        embeddings_by_layer=embeddings_by_layer,
        protein_name="GB1",
        out_dir="results/GB1",
        analysis_layer=None,   # let Objective 1's plateau-ID layer choose
        wt_dms_score=0.0,      # TODO: set to WT's actual score in your DMS convention
    )
    print(summary)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["a", "b"], help="which stage to run")
    args = parser.parse_args()
    if args.stage == "a":
        stage_a_extract_embeddings()
    else:
        stage_b_run_analysis()
