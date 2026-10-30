#!/bin/bash
L=/data/rna_electrostats
COMMON="--labels-dir $L/phi_res --device cuda --audit --structure-controls $L/structure_controls.csv.gz --dssr-controls $L/dssr_controls.csv.gz --coulomb-table $L/coulomb_baseline.csv.gz"
for M in ernierna rnafm splicebert rnabert; do
  for K in 0 1 2 3 4; do
    python centered_latent_manifold.py $COMMON --model $M --family-table $L/controls/families_kmer_t0.5.csv --family-folds 5 --family-fold $K --outdir $L/controls/family/$M/fold$K
  done
done
