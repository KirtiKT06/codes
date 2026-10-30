#!/usr/bin/env bash
# RNA-class breakdown and pretraining-corpus overlap of the pretrained advantage.
#   bash run_class_overlap.sh                         # classes only (needs internet for RCSB)
#   REF=/data/rna_electrostats/refs/rnacentralDB bash run_class_overlap.sh      # classes + corpus overlap
# Run in the torch environment. Output: $R/chain_classes.csv, $R/corpus_overlap.csv,
# $O/analysis/bygroup.csv and a printed report (saved to class_overlap_report.txt).
#
# One-time reference build (RNAcentral: the pretraining corpus of RNA-FM, ERNIE-RNA and RNABERT; one search serves all three;
# add rnabert with MODELS="ernierna rnafm rnabert"):
#   mkdir -p $R/refs && cd $R/refs
#   wget https://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_active.fasta.gz
#   zcat rnacentral_active.fasta.gz | sed '/^>/!s/U/T/g' | mmseqs createdb stdin rnacentralDB --dbtype 2
#   (skip 'createindex' on a machine with < ~150 GB free RAM; instead search in memory-limited chunks: SPLITMEM=30G)
# (several GB to download, tens of GB on disk, and the index needs a lot of RAM: check free disk and memory first.)
set -u
R=${R:-/data/rna_electrostats}
CODE=${CODE:-$HOME/projects/codes/rna_mg}
O=${O:-$R/controls_matched}
MODELS=${MODELS:-"ernierna rnafm"}
THREADS=${THREADS:-16}
SPLITMEM=${SPLITMEM:-}     # e.g. SPLITMEM=30G caps MMseqs2 memory (needed for the full RNAcentral)
REF=${REF:-}
cd "$CODE" || exit 1
mkdir -p "$O/analysis"
{
echo "=== 1. RNA classes from RCSB"
python fetch_chain_classes_v2.py --labels-dir "$R/phi_res" --out "$R/chain_classes.csv" || echo "class annotation failed (internet?)."

echo "=== 1b. PDB release dates (to split chains released before / after the models were pretrained)"
python fetch_release_dates.py --classes "$R/chain_classes.csv" --out "$R/chain_release.csv" || echo "release dates failed (internet?)"

if [ -n "$REF" ]; then
  echo "=== 2. similarity to the reference corpus: $REF"
  RUNS=""
  for m in $MODELS; do RUNS="$RUNS $O/pretrained/$m"; for k in 0 1 2 3 4; do RUNS="$RUNS $O/family_t0.5_fold$k/$m"; done; done
  python corpus_overlap.py --labels-dir "$R/phi_res" --reference "$REF" --out "$R/corpus_overlap.csv" \
      --threads "$THREADS" ${SPLITMEM:+--split-memory-limit "$SPLITMEM"} --only-test-chains $RUNS
fi

for m in $MODELS; do
  echo; echo "=== 3. $m: pretrained minus random within RNA classes (standard test split)"
  python audit_extras.py bygroup --tag ${m}_std --a $O/pretrained/$m \
      --b $O/random_s0/$m $O/random_s1/$m $O/random_s2/$m --groups "$R/chain_classes.csv" --group-col rna_class \
      --targets potential_mean_kT_e_12A potential_mean_kT_e_4A --controls controls_geom_coulomb \
      --out "$O/analysis/bygroup.csv" --min-chains 5 --min-structures 3
  A=""; B=""
  for k in 0 1 2 3 4; do A="$A $O/family_t0.5_fold$k/$m"; B="$B $O/famrand_t0.5_fold${k}_s0/$m"; done
  if [ -f "$O/famrand_t0.5_fold0_s0/$m/audit_chain_stats.csv.gz" ]; then
    echo; echo "=== 3b. $m: within RNA classes, pooled held-out-family folds (more chains per class)"
    python audit_extras.py bygroup --tag ${m}_fam --a $A --b $B --b-mode pool --groups "$R/chain_classes.csv" \
        --group-col rna_class --targets potential_mean_kT_e_12A potential_mean_kT_e_4A --controls controls_geom_coulomb \
        --out "$O/analysis/bygroup.csv" --min-chains 8 --min-structures 4
  fi
  if [ -f "$R/chain_release.csv" ]; then
    echo; echo "=== 3c. $m: pretrained minus random by PDB release period (pooled held-out-family folds)"
    python audit_extras.py bygroup --tag ${m}_rel --a $A --b $B --b-mode pool --groups "$R/chain_release.csv" \
        --group-col release_bin --targets potential_mean_kT_e_12A potential_mean_kT_e_4A --controls controls_geom_coulomb \
        --out "$O/analysis/bygroup.csv" --min-chains 8 --min-structures 4
  fi
  if [ -n "$REF" ] && [ -f "$R/corpus_overlap.csv" ]; then
    echo; echo "=== 4a. $m: chains WITH vs WITHOUT a near-identical copy in the reference corpus"
    python audit_extras.py bygroup --tag ${m}_copy --a $A --b $B --b-mode pool --groups "$R/corpus_overlap.csv" \
        --group-col near_identical_copy --targets potential_mean_kT_e_12A potential_mean_kT_e_4A \
        --controls controls_geom_coulomb --out "$O/analysis/bygroup.csv" --min-chains 8 --min-structures 4
    echo; echo "=== 4b. $m: by best identity to the corpus (including identical copies)"
    python audit_extras.py bygroup --tag ${m}_ovl --a $A --b $B --b-mode pool --groups "$R/corpus_overlap.csv" \
        --group-col best_fident --bins 0 0.5 0.7 0.9 0.995 1.0001 \
        --targets potential_mean_kT_e_12A potential_mean_kT_e_4A --controls controls_geom_coulomb \
        --out "$O/analysis/bygroup.csv" --min-chains 8 --min-structures 4
  fi
done
} 2>&1 | tee class_overlap_report.txt
