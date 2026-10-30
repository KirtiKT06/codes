#!/usr/bin/env bash
# One-time build of the MMseqs2 nucleotide database from RNAcentral (U -> T).
# Run in the background so it survives a closed terminal:
#     nohup bash build_rnacentral_db.sh > build_db.log 2>&1 &
#     tail -n 5 build_db.log          # finished when it prints DONE
# Needs: mmseqs on PATH (torch env), ~100 GB free disk. Single core; can take an hour or more.
set -euo pipefail
REFDIR=${REFDIR:-/data/rna_electrostats/refs}
export TMPDIR=${TMPDIR:-/data/rna_electrostats/tmp}
FASTA=${FASTA:-rnacentral_active.fasta.gz}
mkdir -p "$TMPDIR"
cd "$REFDIR"
test -s "$FASTA" || { echo "ERROR: $REFDIR/$FASTA not found"; exit 1; }
command -v mmseqs >/dev/null || { echo "ERROR: mmseqs not on PATH (activate the torch env)"; exit 1; }
echo "start $(date)"
zcat "$FASTA" | sed '/^>/!s/U/T/g' | mmseqs createdb stdin rnacentralDB --dbtype 2
ls -la rnacentralDB* | head
echo "DONE $(date)"
