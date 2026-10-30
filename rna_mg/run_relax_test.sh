#!/usr/bin/env bash
# Relaxation test: do Amber-relaxed RhoFold structures remove the large systematic offset seen in the identity
# control (unrelaxed predictions: C1' / 4 A maps shifted by tens of kT/e)?
#   bash run_relax_test.sh            (N chains, default 40; uses conda envs `rhofold` and `venv` as in your logs)
# Stage 1 (rhofold env) writes relaxed models; stage 2 (venv env, has pdb2pqr/APBS) scores them with the SAME
# benchmark code; stage 3 compares with the unrelaxed results on the same chains.
# NOTE: --relax-steps needs the Amber/OpenMM relax dependencies of RhoFold; if stage 1 fails, that is the cause.
set -u
R=${R:-/data/rna_electrostats}
CODE=${CODE:-$HOME/projects/codes/rna_mg}
N=${N:-40}
STEPS=${STEPS:-1000}
ENGINE=${ENGINE:-apbs}
cd "$CODE" || exit 1
PRED=$R/preds_relaxed; BENCH=$R/bench_${ENGINE}_relaxed
echo "[1/3] relaxing $N chains ($STEPS steps)"
conda run -n rhofold --no-capture-output python run_rhofold_on_fasta.py --fasta test_chains.fasta \
   --rhofold-dir /data/RhoFold --ckpt /data/RhoFold/pretrained/RhoFold_pretrained.pt --out-dir "$PRED" \
   --device cuda:0 --single-seq --relax-steps "$STEPS" --max-chains "$N"
echo "[2/3] scoring relaxed models ($ENGINE engine)"
conda run -n venv --no-capture-output python predicted_structure_benchmark.py run \
   --labels-dir $R/phi_res --pred-dir "$PRED" --outdir "$BENCH" \
   --pipeline-module $CODE/rna_electrostatics_pipeline_v2.py --engine "$ENGINE" --workers 8 --overwrite
echo "[3/3] relaxed vs unrelaxed, same chains"
python - <<PY
import pandas as pd, numpy as np
rel = pd.read_csv("$BENCH/chain_metrics.csv"); rel = rel[rel.status == "ok"]
unr = pd.read_csv("$R/bench_${ENGINE}_v2/chain_metrics.csv"); unr = unr[unr.status == "ok"]
m = rel.merge(unr, on=["pdb_id", "chain"], suffixes=("_relaxed", "_unrelaxed"))
print(f"{len(m)} chains in common")
rows = []
for q in ("c1p", "m4", "m8", "m12"):
    rows.append({"quantity": q,
                 "median_pearson_unrelaxed": m[f"pred_pearson_{q}_unrelaxed"].median(),
                 "median_pearson_relaxed": m[f"pred_pearson_{q}_relaxed"].median(),
                 "median_offset_unrelaxed": m[f"pred_mean_offset_{q}_unrelaxed"].median(),
                 "median_offset_relaxed": m[f"pred_mean_offset_{q}_relaxed"].median()})
print(pd.DataFrame(rows).round(3).to_string(index=False))
print("median C1' lDDT / RMSD  unrelaxed:", round(m.lddt_c1p_unrelaxed.median(), 3), round(m.rmsd_c1p_unrelaxed.median(), 2),
      " relaxed:", round(m.lddt_c1p_relaxed.median(), 3), round(m.rmsd_c1p_relaxed.median(), 2))
PY
