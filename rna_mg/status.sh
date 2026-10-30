#!/bin/bash
L=/data/rna_electrostats
R=/home/feynman/projects/codes/rna_mg
n() { ls "$1" 2>/dev/null | wc -l; }
first() { [ -f "$1" ] && head -1 "$1" || echo "(no summary yet)"; }
when() { [ -e "$1" ] && stat -c %y "$1" | cut -c1-16 || echo "-"; }
echo "=== $(date '+%a %d %b %H:%M') ==="
echo "RUNNING (process counts):"
for s in predicted_structure_benchmark mg_sensitivity centered_latent_manifold run_rhofold_on_fasta; do
  echo "  $s: $(ps -eo args | grep "python.*$s" | grep -v grep | wc -l)"
done
echo "BENCH Coulomb : $(first $L/bench_coulomb_v2/summary.txt)  [summary $(when $L/bench_coulomb_v2/summary.txt)]"
echo "BENCH APBS    : $(first $L/bench_apbs_v2/summary.txt)  [summary $(when $L/bench_apbs_v2/summary.txt)]  chains done: $(n $L/bench_apbs_v2/per_residue)/189"
echo "MG pilot      : summary $(when $L/mg_pilot/mg_sensitivity_summary.txt)"
echo "MG full       : structures done $(n $L/mg_full/per_residue)/120  [summary $(when $L/mg_full/mg_sensitivity_summary.txt)]"
echo "RANDOM-INIT   : $(ls $L/controls/random_init/*/audit_variance_partition.csv 2>/dev/null | wc -l)/4"
echo "FAMILY-OUT    : $(ls $L/controls/family/*/fold*/audit_variance_partition.csv 2>/dev/null | wc -l)/20"
echo "MSA preds     : $(ls $R/preds_msa/*.pdb 2>/dev/null | wc -l) pdb files in preds_msa"
echo "LOGS WITH ERRORS: $(cd $R; grep -l -E 'Traceback' mg_full.log 2>/dev/null)"
