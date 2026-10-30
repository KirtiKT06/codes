#!/usr/bin/env bash
cd ~/projects/codes/rna_mg
R=/data/rna_electrostats

# 1. Random-init on the same family folds (10 runs, about 5 min each)
RUN_FAMILY=1 RUN_FAMILY_RANDOM=1 bash run_controls_matched.sh ernierna rnafm

# 2. Pooled pretrained-minus-random comparison on identical held-out chains
bash run_family_analysis.sh

# 3. Figures and a fresh snapshot
python make_figures_extra.py --root $R --out figures --only fig8
python -W ignore collect_results.py --root $R --code-dir ~/projects/codes/rna_mg \
    --out project_snapshot.md --max-lines 60