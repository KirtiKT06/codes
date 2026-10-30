#!/bin/bash
cd ~/projects/codes/rna_mg
R=/data/rna_electrostats
OUT=$R/controls_matched

# 1. Chain-length analysis, now including the long-chain bins
RUNS=""
for m in rnabert rnafm splicebert ernierna; do
  RUNS="$RUNS $m/pretrained=$OUT/pretrained/$m"
  for s in 0 1 2; do RUNS="$RUNS $m/random_s$s=$OUT/random_s$s/$m"; done
done
python audit_extras.py length --runs $RUNS --controls controls_geom_coulomb \
    --bins 100 250 500 --min-structures 3 \
    --out $OUT/analysis/length_partition.csv > length_table.txt

# 2. Rebuild the length figure with the new bins
python make_figures_extra.py --root $R --out figures --only fig11

# 3. Fresh snapshot (warnings suppressed, shorter text sections)
python -W ignore collect_results.py --root $R --code-dir ~/projects/codes/rna_mg \
    --out project_snapshot.md --max-lines 60

# 4. Check what to upload
ls -la project_snapshot.md length_table.txt figures/fig8_controls_grid_micro.png \
   figures/fig8_controls_grid_macro.png figures/fig10_power.png figures/fig11_length.png \
   figures/fig12_paired.png