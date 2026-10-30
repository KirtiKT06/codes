#!/usr/bin/env bash
# FAIR relaxation test: relax native AND predicted chains with the same protocol, for chains with lDDT >= 0.9.
#     bash run_fair_relax_test.sh
# Needs fair_relax_test.py and predicted_structure_benchmark.py in the code folder.
# Stages and environments (same conda environments you used before):
#   1 select  (venv)     2 relax (rhofold)     3 score (venv, APBS: 4 solves per chain)     4 report (venv)
# Knobs: MIN_LDDT=0.9  STEPS=1000  WORKERS=8  MAX_CHAINS=0 (0 = all)  BENCH=bench_apbs_v2
# Time: relax is the slow stage on CPU; scoring is ~4 APBS solves per chain split over WORKERS.
# Resumable: finished relaxations and finished chains are skipped; use --overwrite on score to redo.
set -u
R=${R:-/data/rna_electrostats}
CODE=${CODE:-$HOME/projects/codes/rna_mg}
MIN_LDDT=${MIN_LDDT:-0.9}
STEPS=${STEPS:-1000}
WORKERS=${WORKERS:-8}
MAX_CHAINS=${MAX_CHAINS:-0}
BENCH=${BENCH:-bench_apbs_v2}
WORK=${WORK:-$R/fair_relax}   # use a different WORK for a different STEPS, e.g. WORK=$R/fair_relax_s100 STEPS=100
cd "$CODE" || exit 1
echo "[1/4] selecting chains with lDDT >= $MIN_LDDT"
conda run -n venv --no-capture-output python fair_relax_test.py select \
   --metrics "$R/$BENCH/chain_metrics.csv" --labels-dir "$R/phi_res" --pred-dir "$R/preds" \
   --work "$WORK" --min-lddt "$MIN_LDDT" --max-chains "$MAX_CHAINS" || exit 1
echo "[2/4] relaxing native and predicted chains ($STEPS steps each), same protocol"
conda run -n rhofold --no-capture-output python fair_relax_test.py relax \
   --work "$WORK" --rhofold-dir /data/RhoFold --steps "$STEPS"
echo "[3/4] scoring with APBS"
conda run -n venv --no-capture-output python fair_relax_test.py score \
   --work "$WORK" --pipeline-module "$CODE/rna_electrostatics_pipeline_v2.py" --workers "$WORKERS"
echo "[4/4] report"
conda run -n venv --no-capture-output python fair_relax_test.py report --work "$WORK" | tee "$WORK/fair_report.txt"
echo "report saved to $WORK/fair_report.txt ; per-chain table: $WORK/fair_chain_metrics.csv"
