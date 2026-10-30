#!/usr/bin/env bash
# Pooled leave-family-out comparison, pretrained minus random-init on the identical held-out chains.
#   bash run_family_analysis.sh                 (models with both family and famrand runs are analysed)
# Needs:  RUN_FAMILY=1 RUN_FAMILY_RANDOM=1 bash run_controls_matched.sh ernierna rnafm
set -u
R=${R:-/data/rna_electrostats}
CODE=${CODE:-$HOME/projects/codes/rna_mg}
OUT=${OUT:-$R/controls_matched}
FAM_THRESHOLDS=${FAM_THRESHOLDS:-"0.5"}
cd "$CODE" || exit 1
rm -f "$OUT/analysis/family_paired_differences.csv"
for m in rnabert rnafm splicebert ernierna; do
  for t in $FAM_THRESHOLDS; do
    A=""; B=""
    for k in 0 1 2 3 4; do
      [ -f "$OUT/family_t${t}_fold$k/$m/audit_chain_stats.csv.gz" ] && A="$A $OUT/family_t${t}_fold$k/$m"
      [ -f "$OUT/famrand_t${t}_fold${k}_s0/$m/audit_chain_stats.csv.gz" ] && B="$B $OUT/famrand_t${t}_fold${k}_s0/$m"
    done
    [ -n "$A" ] && [ -n "$B" ] || { echo "skip $m t$t (need both family and famrand runs)"; continue; }
    echo "=== family-out, pooled over folds: $m t$t"
    python audit_extras.py famdiff --tag "${m}_t$t" --a $A --b $B \
        --controls controls_geom_coulomb controls_geom --out "$OUT/analysis/family_paired_differences.csv"
  done
done
