#!/usr/bin/env bash
# Pool the five family folds of each model and print the recovered unique-to-LM variance per injected fraction.
set -u
R=${R:-/data/rna_electrostats}
CODE=${CODE:-$HOME/projects/codes/rna_mg}
OUT=${OUT:-$R/controls_matched}
FAM_T=${FAM_T:-0.5}
cd "$CODE" || exit 1
rm -f "$OUT/analysis/spike_family_pooled.csv"
for m in rnabert rnafm splicebert ernierna; do
  F=""; for k in 0 1 2 3 4; do [ -f "$OUT/famspike_t${FAM_T}_fold$k/$m/spike_in_summary.csv" ] && F="$F $OUT/famspike_t${FAM_T}_fold$k/$m"; done
  [ -n "$F" ] || { echo "skip $m (no famspike runs)"; continue; }
  echo "=== pooled spike-in power, leave-family-out folds: $m"
  python audit_extras.py spikepool --tag "$m" --folds $F --controls controls_geom_coulomb \
      --out "$OUT/analysis/spike_family_pooled.csv"
done
