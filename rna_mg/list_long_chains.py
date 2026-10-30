#!/usr/bin/env python
"""Which test chains/structures carry the long-chain effect, and how much of the test set they are.
   python list_long_chains.py /data/rna_electrostats/controls_matched/pretrained/ernierna [min_len=250]"""
import sys, pandas as pd
d = pd.read_csv(sys.argv[1] + "/audit_chain_stats.csv.gz")
d = d[(d.target == d.target.iloc[0]) & (d.predictors == "lm_gbm_pca")][["pdb", "chain", "n"]]
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 250
long_ = d[d.n > lo].sort_values("n", ascending=False)
print(f"test chains: {len(d)}   residues: {d.n.sum()}")
print(f"chains with >{lo} residues: {len(long_)} in {long_.pdb.nunique()} structures, "
      f"{long_.n.sum()} residues = {100 * long_.n.sum() / d.n.sum():.1f}% of test residues")
print(long_.to_string(index=False))
