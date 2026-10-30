#!/usr/bin/env bash
# After run_controls_matched.sh: paired differences + chain-length stratification for every model.
#   bash run_analysis_matched.sh            (uses the same R / OUT / SEEDS variables as the run script)
set -u
R=${R:-/data/rna_electrostats}
CODE=${CODE:-$HOME/projects/codes/rna_mg}
OUT=${OUT:-$R/controls_matched}
SEEDS=${SEEDS:-"0 1 2"}
cd "$CODE" || exit 1
A=$OUT/analysis; mkdir -p "$A"
rm -f "$A/paired_differences.csv"
RUNS=""
for m in rnabert rnafm splicebert ernierna; do
  [ -f "$OUT/pretrained/$m/audit_chain_stats.csv.gz" ] || { echo "no pretrained run for $m, skipping"; continue; }
  B=""; for s in $SEEDS; do [ -f "$OUT/random_s$s/$m/audit_chain_stats.csv.gz" ] && B="$B $OUT/random_s$s/$m"; done
  [ -n "$B" ] || { echo "no random runs for $m, skipping"; continue; }
  echo "=== paired difference: $m (pretrained minus mean of random seeds)"
  python audit_extras.py diff --tag "$m" --a "$OUT/pretrained/$m" --b $B \
      --controls controls_geom_coulomb controls_geom --out "$A/paired_differences.csv"
  RUNS="$RUNS $m/pretrained=$OUT/pretrained/$m"
  for s in $SEEDS; do [ -f "$OUT/random_s$s/$m/audit_chain_stats.csv.gz" ] && RUNS="$RUNS $m/random_s$s=$OUT/random_s$s/$m"; done
done
echo "=== chain-length stratification"
python audit_extras.py length --runs $RUNS --controls controls_geom_coulomb --bins 100 250 --min-structures 3 --out "$A/length_partition.csv"
