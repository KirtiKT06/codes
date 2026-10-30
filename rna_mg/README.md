# Do RNA language models encode electrostatics? — code guide

An audit of pretrained RNA language models (`rnabert`, `rnafm`, `splicebert`, `ernierna`, loaded through
`multimolecule`) against per-residue **Poisson–Boltzmann (APBS) potentials**, with a physics baseline
(screened Coulomb), structure controls, and a predicted-structure benchmark.

This README says **what each script does, in which environment, in which order, and what it writes**.
The numbers and conclusions are in `REPORT.md`.

---------------------------------------------------------------------------------------------------

## 1. The idea in five lines

1. For ~1,873 BGSU non-redundant RNA structures, APBS gives a potential at every residue (labels).
2. We remove what is trivial (chain-level scale, letter identity) and ask whether LM embeddings predict
   what is left (the *within-chain* pattern).
3. We compare against controls built from the **real 3D structure** (pairing, packing, burial, DSSR
   features, screened Coulomb). The question is what the LM explains **beyond** them (`unique_lm`).
4. We stress-test the answer with a random-weights model and a leave-family-out split.
5. Separately, we ask whether an electrostatic map computed from a **predicted** structure
   (RhoFold+) matches the native map better than a geometric error of the same size would.

---------------------------------------------------------------------------------------------------

## 2. Environments (which one to use — most errors come from here)

| Environment (prompt shows) | Has | Use it for |
|---|---|---|
| `(venv)` | pdb2pqr, APBS, pandas | `rna_electrostatics_pipeline_v2.py`, `predicted_structure_benchmark.py`, `mg_sensitivity.py` |
| `(python_3.11_torch)` | torch + GPU, `multimolecule`, pandas, scikit-learn | `centered_latent_manifold.py`, `cross_model_latent_agreement.py`, `build_family_table.py`, `compare_audit_runs.py` |
| `(rhofold)` (conda, Python 3.7) | RhoFold+ | `run_rhofold_on_fasta.py` **only** |
| `(base)` | nothing useful (no pandas) | nothing |

Rule: **look at the prompt before every command.** A wrong environment gives either
`ModuleNotFoundError` or, worse, a run in which *every* chain "fails" quickly (the benchmark catches
per-chain errors and carries on, so check `summary.txt` says `failed: 0`).

## 3. Shell habits that avoid the errors we hit

* No `<placeholders>` in commands. Replace them first; `<` is a shell redirect.
* Do not put `$` in front of a path (`$/data/...` is wrong).
* `~` is **not** expanded inside quotes (`sh -c "... ~/x ..."` fails). Use full paths (`/home/feynman/...`).
* Shell variables (`P=...`, `L=...`) exist only in the terminal tab where you typed them.
* Long jobs: `nohup ... > name.log 2>&1 &`, then `tail -n 5 name.log`.
  (`tail -f` may say "Too many open files": harmless, use `tail -n`.)
* To wait for a job: `while kill -0 <PID> 2>/dev/null; do sleep 60; done; <next command>`.
  Do **not** use `pgrep -f <script>` inside `sh -c '...'`: the wait command contains the script name,
  finds itself and waits forever.

---------------------------------------------------------------------------------------------------

## 4. Data layout (feynman)

```
/data/rna_electrostats/
  phi_res/                         labels
    all_residue_potentials.csv     one row per residue: C1' point, P point, 4/8/12 A sphere means, pb_solver ...
    work/<ID>/                     <id>.fasta, <id>_complete_residues.pdb, PQR, ions.csv (ions stripped for the solve)
  structure_controls.csv.gz        pairing (ViennaRNA) + packing/burial        [build_structure_controls.py]
  dssr_controls.csv.gz             real pairs/stems/loops/multiplets/stacking  [build_dssr_controls.py]
  coulomb_baseline.csv.gz          screened-Coulomb baseline                   [coulomb_residual_test.py]
  centered_latent_v3/<model>/      pretrained audit outputs (audit_variance_partition.csv, ...)
  controls/random_init/<model>/    random-weights control
  controls/family/<model>/fold0..4 leave-family-out control
  controls/families_kmer_t0.5.csv  family table used for leave-family-out (6-mer similarity clusters)
  preds/                           189 single-sequence RhoFold+ predictions  <PDBID>_<chain>.pdb
  bench_coulomb_v2/, bench_apbs_v2/   predicted-structure benchmark outputs
  mg_pilot/, mg_full/              Mg2+ sensitivity outputs
/data/RhoFold/                     RhoFold+ repo, pretrained/RhoFold_pretrained.pt, database/ (MSA databases)
~/projects/codes/rna_mg/           all scripts
```

---------------------------------------------------------------------------------------------------

## 5. Pipeline at a glance

```
BGSU list ──> rna_electrostatics_pipeline_v2.py ──> phi_res/ (labels, PQR, ions.csv)
                                                       │
        ┌──────────────────────┬──────────────────────┼──────────────────────────┐
build_structure_controls   build_dssr_controls   coulomb_residual_test        (labels)
        └──────────────────────┴──────────┬───────────┴──────────────────────────┘
                                          ▼
                        centered_latent_manifold.py --audit   (per model)
                                          │
              ┌───────────────────────────┼─────────────────────────────┐
      --random-init                --family-table ...             cross_model_latent_agreement.py
              └──────────────┬────────────┘
                             ▼
                    compare_audit_runs.py  ──>  tables for the paper

phi_res ──> predicted_structure_benchmark.py fasta ──> run_rhofold_on_fasta.py ──> preds/
        └─> predicted_structure_benchmark.py run (coulomb | apbs) ──> summarize

phi_res ──> mg_sensitivity.py   (does keeping Mg2+ change the labels?)
```

---------------------------------------------------------------------------------------------------

## 6. Script reference

| Script | What it does | Main inputs | Main outputs |
|---|---|---|---|
| `rna_electrostatics_pipeline_v2.py` | BGSU list -> pdb2pqr -> APBS (nonlinear PB, 0.15 M, pdie 1, sdie 80, 300 K, srad 1.4, `chgm spl2`, adaptive grid). Ions stripped for the solve and saved to `ions.csv`. Records `pb_solver` (61 of 1,873 structures used the linearised fallback). Run `--help` for its options. | BGSU list, PDB/CIF files | `phi_res/` |
| `make_clean_subset.py`, `filter_dataset.py`, `repair_pdb_ids.py`, `mmseqs_cluster.py` | Dataset preparation and filtering used before/around the pipeline. | | |
| `build_structure_controls.py` | Per-residue **predicted pairing** (ViennaRNA) and **3D packing/burial** from the real coordinates. | `phi_res` | `structure_controls.csv.gz` |
| `build_dssr_controls.py` | Real base pairs, stems, loops, multiplets, stacking, sugar pucker from **DSSR** (JSON cached). | `phi_res`, deposited PDB/CIF files | `dssr_controls.csv.gz` |
| `coulomb_residual_test.py` | Point-charge screened-Coulomb baseline (`vac`, `dhS`, `dhP`) from the same PQR files; also reports how much of each PB label that arithmetic already explains. `dhP` (pdie 1) absolute values are uncalibrated: trust correlations, not kT/e values. | `phi_res` | `coulomb_baseline.csv.gz` |
| `centered_latent_manifold.py` (v4) | Removes chain scale and letter effect from the labels, then probes embeddings (PLS/CCA); with `--audit` gives macro/micro R², structure-level bootstrap CIs and the **variance partition** (`unique_lm`, `shared`, `unique_controls`, `unexplained`). Also holds the controls (`--random-init`, `--family-table`). | `phi_res`, control tables, `--model` (required) | `<outdir>/audit_variance_partition.csv`, `audit_probe_r2.csv`, per-variant folders |
| `latent_controls.py` | Library used by the manifold script: random-weights loader and leave-family-out split. Not run directly. | | |
| `build_family_table.py` | Assigns every chain to a family (`--method kmer` similarity clusters, or `--method cmscan` real Rfam). | `phi_res` | `families_*.csv` |
| `compare_audit_runs.py` | Puts several `audit_variance_partition.csv` files side by side; files sharing a label are pooled (mean, min..max). | audit CSVs | printed table (`--out` for CSV) |
| `cross_model_latent_agreement.py` | Do different LMs share the same electrostatic latent axes (CCA, within-chain)? | per-model manifold runs | `cross_model/` |
| `predicted_structure_benchmark.py` | Subcommands `fasta`, `run`, `summarize`. Compares the electrostatic map of a **predicted** chain with the **native chain recomputed in isolation**, against RMSD-matched and lDDT-matched geometric nulls. | `phi_res`, `preds/` | `chain_metrics.csv`, `per_residue/`, `summary.txt`, `excess_vs_null.csv` |
| `run_rhofold_on_fasta.py` | Runs RhoFold+ on every chain of a FASTA, resumable, writes `<out-dir>/<PDBID>_<chain>.pdb` and `failed.tsv`. | FASTA, checkpoint | `preds*/` |
| `mg_sensitivity.py` | Re-solves PB with structural Mg²⁺ as fixed +2 charges (plus a charge-0 control) for a sample of structures; compares with the ion-free labels. | `phi_res`, `ions.csv` | `mg_sensitivity_summary.txt`, `mg_effect_by_distance.csv`, drop-in `labels_r0.79/` |
| `make_figures.py`, `analyze_failure.py`, `rhofold_feature_hook.py`, `representation_characterization.py`, others | Figures and exploratory analyses from earlier stages. | | |

Older `centered_latent_manifold_v1/v2/v3_backup.py` are frozen copies; do not use them.

---------------------------------------------------------------------------------------------------

## 7. How to run (copy–paste recipes)

Set once per terminal tab (adjust if your paths differ):
```
cd /home/feynman/projects/codes/rna_mg
L=/data/rna_electrostats
P=/home/feynman/projects/codes/rna_mg/rna_electrostatics_pipeline_v2.py
```

### 7.1 Controls tables (once; skip if the three files exist)
```
python build_structure_controls.py --labels-dir $L/phi_res --out $L/structure_controls.csv.gz --workers 8
python build_dssr_controls.py --labels-dir $L/phi_res --pdb-dir <folder with deposited PDB/CIF files> \
       --json-dir $L/dssr_json --out $L/dssr_controls.csv.gz --workers 8      # edit --pdb-dir; needs DSSR on PATH or --dssr-bin
python coulomb_residual_test.py --labels-dir $L/phi_res --out $L/coulomb_baseline.csv.gz --workers 8
```

### 7.2 Main audit — pretrained model (`python_3.11_torch`)
```
python centered_latent_manifold.py --labels-dir $L/phi_res --model ernierna --device cuda \
   --structure-controls $L/structure_controls.csv.gz --dssr-controls $L/dssr_controls.csv.gz \
   --coulomb-table $L/coulomb_baseline.csv.gz --audit \
   --outdir $L/centered_latent_v3/ernierna
```
`--model` is required. Models: `rnabert rnafm splicebert ernierna`. Audit targets default to the C1′
point and the 4 Å and 12 Å sphere means (`--audit-targets` changes this). Key output:
`audit_variance_partition.csv` (columns include `target`, `controls`, `weighting`, `component`, `value`).

### 7.3 Random-weights control (`python_3.11_torch`)
Same command plus `--random-init --random-seed 0` and a different `--outdir`:
```
python centered_latent_manifold.py --labels-dir $L/phi_res --model ernierna --device cuda --audit \
   --structure-controls $L/structure_controls.csv.gz --dssr-controls $L/dssr_controls.csv.gz \
   --coulomb-table $L/coulomb_baseline.csv.gz --random-init --random-seed 0 \
   --outdir $L/controls/random_init/ernierna
```
Add `--layer N` to force the same layer as the pretrained run. Without it the random-init run picks its own
best layer, which may differ from the pretrained one (see REPORT, caveats).

### 7.4 Leave-family-out control
```
python build_family_table.py --labels-dir $L/phi_res --out $L/controls/families_kmer_t0.5.csv --threshold 0.5 --min-shared 30
for K in 0 1 2 3 4; do
  python centered_latent_manifold.py --labels-dir $L/phi_res --model ernierna --device cuda --audit \
     --structure-controls $L/structure_controls.csv.gz --dssr-controls $L/dssr_controls.csv.gz \
     --coulomb-table $L/coulomb_baseline.csv.gz \
     --family-table $L/controls/families_kmer_t0.5.csv --family-folds 5 --family-fold $K \
     --outdir $L/controls/family/ernierna/fold$K
done
```
Check the family table first: the largest group should be well under 30% of chains (it was 44% at the
default threshold, 10.5% at 0.5). `--method cmscan --rfam-cm /data/RhoFold/database/Rfam.cm` gives real Rfam
families (needs Infernal: `conda install -c bioconda infernal`).

### 7.5 Compare pretrained / random / family-out
```
M=ernierna
python compare_audit_runs.py \
  pretrained=$L/centered_latent_v3/$M/audit_variance_partition.csv \
  random=$L/controls/random_init/$M/audit_variance_partition.csv \
  family=$L/controls/family/$M/fold0/audit_variance_partition.csv \
  family=$L/controls/family/$M/fold1/audit_variance_partition.csv \
  family=$L/controls/family/$M/fold2/audit_variance_partition.csv \
  family=$L/controls/family/$M/fold3/audit_variance_partition.csv \
  family=$L/controls/family/$M/fold4/audit_variance_partition.csv
```
Defaults: `--controls controls_geom_coulomb --weighting micro`. Files sharing a label are pooled.

### 7.6 Do models share latent axes?
```
python cross_model_latent_agreement.py --root $L/centered_latent_v3 --models rnabert rnafm splicebert ernierna \
   --variant centered_base_resid --outdir $L/centered_latent_v3/cross_model
```
Run `centered_latent_manifold.py` once per model first (with `--variants` as you need).

### 7.7 Predicted-structure benchmark
```
# a) FASTA of test-split chains, 20-250 nt
python predicted_structure_benchmark.py fasta --labels-dir $L/phi_res --out test_chains.fasta --min-len 20 --max-len 250

# b) predict (rhofold env). Single-sequence mode:
conda activate rhofold
python run_rhofold_on_fasta.py --fasta test_chains.fasta --rhofold-dir /data/RhoFold \
   --ckpt /data/RhoFold/pretrained/RhoFold_pretrained.pt --out-dir $L/preds --device cuda:0 --single-seq
#    MSA mode: drop --single-seq (needs the databases, see 7.8) and use a different --out-dir, e.g. $L/preds_msa

# c) score (venv env). Coulomb engine is fast; APBS engine is the same physics as the labels, but slow:
python predicted_structure_benchmark.py run --labels-dir $L/phi_res --pred-dir $L/preds --outdir $L/bench_coulomb_v2 \
   --pipeline-module $P --engine coulomb --workers 8
python predicted_structure_benchmark.py run --labels-dir $L/phi_res --pred-dir $L/preds --outdir $L/bench_apbs_v2 \
   --pipeline-module $P --engine apbs --workers 4

# d) summarise
python predicted_structure_benchmark.py summarize --outdir $L/bench_coulomb_v2
```
Notes
* Predicted files must be named `<PDBID>_<chain>.pdb` and contain all heavy atoms.
* Null models: RMSD-matched `smooth`/`jitter` and lDDT-matched `smoothL`/`jitterL` (`--null-kinds`, default all four; `--n-null`).
* `--overwrite` redoes chains already listed (needed after a failed run, otherwise they are skipped).
* `summary.txt` first line must read `chains scored: 189 of 189 (failed: 0)`. If it says 0 of 189, read
  the `error` column of `chain_metrics.csv`; the usual cause is the wrong environment.
* The native chain is recomputed **in isolation** (predictors fold single chains; stored labels came
  from the whole complex), so the reference is "native, isolated, ion-free", never the stored label.
* Summary metrics: within-chain Pearson/Spearman, MAE, Wasserstein/sd, C1′-lDDT, Kabsch RMSD, and
  excess = predicted − null median (positive = prediction agrees better than a geometric error of equal size).

### 7.8 MSA databases for RhoFold+ (optional)
Built with `builddb.sh` into `/data/RhoFold/database` (~1.2 TB with `nt`). State of ours: the `nt` BLAST
database had 41 of 400 volumes incomplete or corrupt; `nt.nal` was rewritten to list only the 359 healthy
volumes (originals saved as `nt.nal.orig`, `nt.ndb` kept). Test:
```
/data/RhoFold/rhofold/data/bin/blastn -query /tmp/q.fa -db /data/RhoFold/database/nt -outfmt 6 | head -3
```
A segmentation fault means a bad volume; `Permission denied` means `chmod +x /data/RhoFold/rhofold/data/bin/*`.
State in the paper that the MSA database used a partial `nt`.

### 7.9 Mg²⁺ sensitivity (`venv`)
```
python mg_sensitivity.py --labels-dir $L/phi_res --outdir $L/mg_pilot --pipeline-module $P --n-structures 10 --workers 2   # pilot
python mg_sensitivity.py --labels-dir $L/phi_res --outdir $L/mg_full  --pipeline-module $P --workers 4                      # 120 structures
```
Ions kept only if: right element (MG), resolution OK (`low_resolution_flag` false), occupancy ≥ 0.5, within 5 Å
of a heavy RNA atom; structures need ≥ 3 such ions. The **charge-0 control** separates cavity/grid effects from
the pure charge effect. `labels_r0.79/` is a drop-in labels folder: `centered_latent_manifold.py --labels-dir
.../labels_r0.79` reruns the audit on Mg²⁺-aware labels.

---------------------------------------------------------------------------------------------------

## 8. Checking a day-long run

```
ps aux | grep -E "predicted_structure_benchmark|centered_latent_manifold|mg_sensitivity" | grep -v grep | wc -l   # 0 = none running
ls $L/controls/random_init/*/audit_variance_partition.csv        # want 4
ls $L/controls/family/*/fold*/audit_variance_partition.csv | wc -l   # want 20
head -1 $L/bench_coulomb_v2/summary.txt                           # chains scored: 189 of 189 (failed: 0)
ls $L/mg_full/mg_sensitivity_summary.txt
grep -l -E "Traceback|ERROR" *.log                                # any log with a problem
```

## 9. Reading `audit_variance_partition.csv`

For one target (e.g. the 12 Å sphere mean), all on held-out structures and the same learner:

| component | meaning |
|---|---|
| `unique_lm` | variance the LM explains that the structure/Coulomb controls do **not** (the quantity the paper is about) |
| `shared` | variance explained by both the LM and the controls |
| `unique_controls` | variance only the controls explain |
| `unexplained` | nobody explains it |

They add to 1. `controls_geom_coulomb` = geometry (pairing/packing/DSSR) + Coulomb baseline.
`micro` pools all residues; `macro` averages per chain.

## 10. Known limits (state them in the paper)

* Labels are **static, ion-free, single-structure** PB potentials.
* The `dhP` Coulomb family is uncalibrated in absolute units; use correlations and ratios only.
* Family groups from `build_family_table.py --method kmer` are similarity clusters, not curated families.
* Single-sequence RhoFold+ is less accurate than MSA mode (its authors say so).
* One random seed for the random-weights control so far.
