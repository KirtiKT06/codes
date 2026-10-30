#!/usr/bin/env bash
# Spike-in POWER ANALYSIS on the leave-family-out folds (pretrained models, 5 folds each).
# Why: the pooled family folds hold ~3x more chains than the standard split, so unique-to-LM bounds are tighter
# (upper 95% bounds <= 0.006). This measures whether those folds could actually DETECT an effect of 0.5-5%.
#
# Prerequisite: python patch_audit.py already applied (it adds --spike-fractions).
# Run in the torch environment:
#     bash run_spike_family.sh                      # default models: ernierna rnafm
#     bash run_spike_family.sh ernierna rnafm splicebert rnabert
# About 13 min per run (family audit + 8 spike audits); 2 models = ~2.2 h. Finished runs are skipped.
# Then:  bash run_spike_family_analysis.sh
set -u
R=${R:-/data/rna_electrostats}
CODE=${CODE:-$HOME/projects/codes/rna_mg}
OUT=${OUT:-$R/controls_matched}
FAM_T=${FAM_T:-0.5}
FRACS=${FRACS:-"0.005 0.01 0.02 0.05"}
TARGETS=${TARGETS:-"potential_mean_kT_e_12A potential_mean_kT_e_4A"}
declare -A LAYER=( [rnabert]=3 [rnafm]=11 [splicebert]=2 [ernierna]=11 )
if [ "$#" -gt 0 ]; then MODELS="$*"; else MODELS="ernierna rnafm"; fi
cd "$CODE" || { echo "cannot cd to $CODE"; exit 1; }
grep -q "spike-fractions" centered_latent_manifold.py || { echo "run patch_audit.py first"; exit 1; }
tbl="$R/controls/families_kmer_t$FAM_T.csv"
[ -f "$tbl" ] || { echo "missing family table $tbl"; exit 1; }
COMMON="--labels-dir $R/phi_res --structure-controls $R/structure_controls.csv.gz \
--dssr-controls $R/dssr_controls.csv.gz --coulomb-table $R/coulomb_baseline.csv.gz \
--variants centered_base_resid --audit --device cuda"
for m in $MODELS; do
  for k in 0 1 2 3 4; do
    out="$OUT/famspike_t${FAM_T}_fold$k/$m"
    if [ -f "$out/spike_in_summary.csv" ]; then echo "[skip] famspike fold$k/$m (done)"; continue; fi
    mkdir -p "$out"; echo "${LAYER[$m]}" > "$out/LAYER.txt"
    echo "[run ] famspike_t${FAM_T}_fold$k/$m  layer ${LAYER[$m]}  $(date +%T)"
    python centered_latent_manifold.py $COMMON --model "$m" --layer "${LAYER[$m]}" --outdir "$out" \
        --family-table "$tbl" --family-folds 5 --family-fold "$k" \
        --spike-fractions $FRACS --spike-type linear interaction --spike-targets $TARGETS \
        > "$out/log.txt" 2>&1 || { echo "[FAIL] $out -- see log.txt"; tail -5 "$out/log.txt"; }
  done
done
echo "done. Next: bash run_spike_family_analysis.sh"
