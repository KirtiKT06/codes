#!/usr/bin/env bash
# Layer-matched controls, all with the SAME code path so they are directly comparable:
#   pretrained, random-init (3 seeds), optionally leave-family-out (5 folds x thresholds), + spike-in power analysis.
#
# Prerequisite (once):  python patch_audit.py ~/projects/codes/rna_mg/centered_latent_manifold.py
# Run inside the torch environment (the one you use for centered_latent_manifold.py):
#     bash run_controls_matched.sh                    # all four models
#     bash run_controls_matched.sh ernierna rnafm     # a subset
#
# Knobs (environment variables):
#   R=/data/rna_electrostats   CODE=~/projects/codes/rna_mg   OUT=$R/controls_matched
#   SEEDS="0 1 2"              random-init seeds
#   RUN_FAMILY=0|1             also run leave-family-out (5 folds each)   [default 0: it is the slow part]
#   RUN_FAMILY_RANDOM=0|1      random-init (seed 0) on the same family folds; needed for run_family_analysis.sh
#   FAM_THRESHOLDS="0.5"       which controls/families_kmer_t<thr>.csv tables to use (e.g. "0.5 0.7 0.9")
#   SPIKE=1|0                  spike-in power analysis on the pretrained run
#   SPIKE_TARGETS="potential_mean_kT_e_12A potential_mean_kT_e_4A"
#
# Layers are the ones the pretrained v3 runs selected (rnabert 3, rnafm 11, splicebert 2, ernierna 11).
# Every run writes LAYER.txt and log.txt into its folder, so which layer was used is never in doubt again.
# A finished run (audit_chain_stats.csv.gz present) is skipped, so the script can be restarted.
set -u
R=${R:-/data/rna_electrostats}
CODE=${CODE:-$HOME/projects/codes/rna_mg}
OUT=${OUT:-$R/controls_matched}
SEEDS=${SEEDS:-"0 1 2"}
RUN_FAMILY=${RUN_FAMILY:-0}
RUN_FAMILY_RANDOM=${RUN_FAMILY_RANDOM:-0}   # random-init on the SAME family folds (needed for a paired family-out comparison)
FAM_THRESHOLDS=${FAM_THRESHOLDS:-"0.5"}
SPIKE=${SPIKE:-1}
SPIKE_TARGETS=${SPIKE_TARGETS:-"potential_mean_kT_e_12A potential_mean_kT_e_4A"}
declare -A LAYER=( [rnabert]=3 [rnafm]=11 [splicebert]=2 [ernierna]=11 )
if [ "$#" -gt 0 ]; then MODELS="$*"; else MODELS="rnabert rnafm splicebert ernierna"; fi

cd "$CODE" || { echo "cannot cd to $CODE"; exit 1; }
grep -q "spike-fractions" centered_latent_manifold.py || {
  echo "centered_latent_manifold.py is not patched. Run: python patch_audit.py $CODE/centered_latent_manifold.py"; exit 1; }

COMMON="--labels-dir $R/phi_res --structure-controls $R/structure_controls.csv.gz \
--dssr-controls $R/dssr_controls.csv.gz --coulomb-table $R/coulomb_baseline.csv.gz \
--variants centered_base_resid --audit --device cuda"

run() {  # run <name> <model> [extra args...]
  local name=$1 model=$2; shift 2
  local out="$OUT/$name/$model"
  if [ -f "$out/audit_chain_stats.csv.gz" ]; then echo "[skip] $name/$model (done)"; return; fi
  mkdir -p "$out"
  echo "${LAYER[$model]}" > "$out/LAYER.txt"
  echo "[run ] $name/$model  layer ${LAYER[$model]}  $(date +%T)"
  python centered_latent_manifold.py $COMMON --model "$model" --layer "${LAYER[$model]}" \
      --outdir "$out" "$@" > "$out/log.txt" 2>&1 \
      || { echo "[FAIL] $name/$model -- see $out/log.txt"; tail -5 "$out/log.txt"; }
}

for m in $MODELS; do
  if [ "$SPIKE" = "1" ]; then
    run pretrained "$m" --spike-fractions 0.005 0.01 0.02 0.05 --spike-type linear interaction \
        --spike-targets $SPIKE_TARGETS
  else
    run pretrained "$m"
  fi
  for s in $SEEDS; do
    run "random_s$s" "$m" --random-init --random-seed "$s"
  done
  if [ "$RUN_FAMILY" = "1" ]; then
    for t in $FAM_THRESHOLDS; do
      tbl="$R/controls/families_kmer_t$t.csv"
      [ -f "$tbl" ] || { echo "missing family table $tbl"; continue; }
      for k in 0 1 2 3 4; do
        run "family_t${t}_fold$k" "$m" --family-table "$tbl" --family-folds 5 --family-fold "$k"
      done
    done
  fi
done

if [ "$RUN_FAMILY_RANDOM" = "1" ]; then
  for m in $MODELS; do
    for t in $FAM_THRESHOLDS; do
      tbl="$R/controls/families_kmer_t$t.csv"
      [ -f "$tbl" ] || { echo "missing family table $tbl"; continue; }
      for k in 0 1 2 3 4; do
        run "famrand_t${t}_fold${k}_s0" "$m" --random-init --random-seed 0 \
            --family-table "$tbl" --family-folds 5 --family-fold "$k"
      done
    done
  done
fi
echo "done. Next:"
echo "  python audit_extras.py diff   --tag <model> --a $OUT/pretrained/<model> --b $OUT/random_s0/<model> ... --out $OUT/analysis/paired_differences.csv"
echo "  python audit_extras.py length --runs <model>/pretrained=... <model>/random_s0=... --out $OUT/analysis/length_partition.csv"
