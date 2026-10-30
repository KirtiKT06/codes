# Do RNA language models encode electrostatics? — results report

Status as of 23 Oct 2026. Companion to `README.md` (how to run). All numbers below come from runs on feynman;
where something is still running or has not been checked, it says so.

---------------------------------------------------------------------------------------------------

## 1. Summary

1. **RNA LMs add nothing electrostatic beyond the structure/Coulomb controls.** In every model, target and
   control (pretrained, random weights, leave-family-out) the LM's *unique* explained variance is ≈ 0
   (at most +0.015, i.e. 1.5% of variance; almost all values are within ±0.01).
2. **The controls explain about 74–85% of the within-chain variance** (geometry + screened Coulomb); 15–26% is
   explained by nothing we have. Electrostatics here is largely a readout of 3D structure.
3. **The overlap between LM and controls ("shared" variance) is not general structural knowledge.**
   * At the C1′ point it is as large for a **random-weights** model as for the pretrained one
     (e.g. `rnabert` 0.631 vs 0.629): letter identity, position and architecture are enough.
   * At the 4 Å and 12 Å sphere means, pretrained models do carry more shared variance than random ones
     for three of the four models, but it **collapses under leave-family-out** (e.g. `ernierna` 12 Å: 0.45 → 0.12),
     so it mostly reflects recognising related sequences between train and test.
   * **The positive sentence we planned ("33–59% shared variance shows the LMs carry structural context")
     is not supported by these controls and should not go in the paper.**
4. **Predicted-structure benchmark (first run, 189 chains, single-sequence RhoFold+):** agreement of the
   predicted electrostatic map with the native map tracks lDDT/RMSD and does not beat the geometric
   expectation by much. The re-run with lDDT-matched nulls scored 0 chains because it was started in the
   wrong environment; it is being redone (§6).
5. **Mg²⁺ pilot (10 structures):** keeping Mg²⁺ barely reorders residues within a chain (median Pearson
   0.97–0.995) but changes residues within 6 Å of an ion a lot. Full run still in progress.

---------------------------------------------------------------------------------------------------

## 2. Question and data

* **Question.** Do pretrained RNA LMs (`rnabert`, `rnafm`, `splicebert`, `ernierna`) contain electrostatic
  information beyond what the sequence context and the 3D structure already give?
* **Labels.** Per-residue potentials from nonlinear PB (pdb2pqr + APBS; 0.15 M, pdie 1, sdie 80, 300 K,
  srad 1.4, `chgm spl2`, adaptive grid): C1′ point, phosphorus point, 4/8/12 Å sphere means.
  BGSU non-redundant set, 1,873 structures (4,342 chains). 1,812 solved with nonlinear PB, 61 with the
  linearised fallback. Ions are stripped for the solve (saved in `ions.csv`).
* **Within-chain view.** Chain scale and letter identity are removed from the targets, so the test is on
  the residue-to-residue pattern inside each chain.

## 3. Methods in brief

* **Audit / variance partition** (`centered_latent_manifold.py --audit`): same learner (gradient boosting)
  for controls only, LM only, and both, on held-out structures. Components: `unique_lm`, `shared`,
  `unique_controls`, `unexplained`. Controls = `controls_geom_coulomb` (pairing, packing/burial, DSSR
  features, screened-Coulomb baseline). Pooled over residues (`micro`). Targets: C1′ point, 4 Å, 12 Å.
* **Random-weights control**: same architecture/tokenizer/extraction, random weights, seed 0.
* **Leave-family-out**: 5 folds; families are 6-mer similarity clusters (threshold 0.5, ≥30 shared 6-mers;
  3,325 groups; largest 458 chains = 10.5%); structures straddling train/test families are dropped.
* **Predicted-structure benchmark**: native chain recomputed in isolation, predicted chain through the
  same route; compared within chain (Pearson, Spearman, MAE, Wasserstein/sd) and against RMSD-matched and
  lDDT-matched geometric nulls (smooth long-wavelength field, per-residue jitter).
* **Mg²⁺ check**: structural Mg²⁺ added as fixed +2 charges and PB re-solved; charge-0 control for cavity/grid.

---------------------------------------------------------------------------------------------------

## 4. Results

### 4.1 Pretrained audit and the two controls (micro weighting, `controls_geom_coulomb`)

Pretrained and random-init use the same test set; the family-out column averages 5 folds (range in brackets),
and each fold has a different test set, so compare it with the pretrained column only in direction, not in
exact value.

**Shared variance (LM ∩ controls)** — P = pretrained, R = random weights, F = leave-family-out

| target | model | P | R | F (mean [min..max]) |
|---|---|---|---|---|
| C1′ point | ernierna | 0.603 | 0.368 | 0.637 [0.574..0.742] |
| | rnafm | 0.560 | 0.508 | 0.590 [0.473..0.739] |
| | splicebert | 0.417 | 0.512 | 0.494 [0.442..0.562] |
| | rnabert | 0.629 | 0.631 | 0.660 [0.611..0.739] |
| 12 Å mean | ernierna | 0.448 | 0.069 | 0.121 [0.032..0.213] |
| | rnafm | 0.409 | 0.074 | 0.083 [0.022..0.142] |
| | splicebert | 0.206 | 0.074 | 0.041 [−0.008..0.111] |
| | rnabert | 0.111 | 0.077 | 0.011 [−0.011..0.038] |
| 4 Å mean | ernierna | 0.402 | 0.076 | 0.158 [0.095..0.204] |
| | rnafm | 0.351 | 0.085 | 0.126 [0.091..0.180] |
| | splicebert | 0.199 | 0.093 | 0.079 [0.045..0.133] |
| | rnabert | 0.138 | 0.117 | 0.070 [0.044..0.089] |

**Unique LM variance (what the LM adds beyond the controls)**

| target | model | P | R | F |
|---|---|---|---|---|
| C1′ point | ernierna | +0.003 | −0.002 | −0.003 |
| | rnafm | −0.003 | −0.003 | −0.003 |
| | splicebert | −0.003 | −0.006 | −0.002 |
| | rnabert | −0.002 | −0.001 | −0.002 |
| 12 Å mean | ernierna | +0.015 | +0.010 | −0.003 |
| | rnafm | −0.002 | +0.004 | −0.001 |
| | splicebert | +0.014 | +0.005 | −0.005 |
| | rnabert | +0.001 | +0.005 | −0.005 |
| 4 Å mean | ernierna | +0.001 | −0.006 | −0.004 |
| | rnafm | −0.003 | +0.009 | −0.004 |
| | splicebert | +0.004 | −0.001 | −0.002 |
| | rnabert | −0.001 | −0.002 | −0.005 |

**Other components.** `unexplained` is 0.15–0.26 in every cell. Controls alone (`shared + unique_controls`)
reach about 0.74–0.84 on the pretrained/random test set (e.g. `ernierna` 12 Å: 0.448 + 0.296 = 0.744; random:
0.069 + 0.674 = 0.743) and 0.77–0.85 on the family folds.

**What this shows**
* `unique_lm ≈ 0` is robust to every control: the headline audit result holds.
* In `ernierna` at 12 Å the pretrained LM recovers about 60% of what the controls explain (0.448/0.744);
  a random network recovers 9% (0.069/0.743); on unseen families 15% (0.121/0.800). The pretrained advantage
  is largely gone when families are held out.
* `rnabert` shows no pretraining effect at any target (P ≈ R).
* `splicebert` at the C1′ point is *higher* in the random model (0.512) than the pretrained one (0.417).
* Not yet reported here: bootstrap CIs on these differences (they exist in the audit outputs but were not
  collected), and the 8 Å target (not an audit target by default).

### 4.2 Predicted-structure benchmark

Design: 189 test-split chains of 20–250 nt, structures predicted by RhoFold+ in **single-sequence** mode
(no MSA), Coulomb engine, RMSD-matched smooth/jitter nulls. Numbers from that first run:

* Map fidelity tracks lDDT and RMSD: better structures give better electrostatic agreement.
* Excess over the RMSD-matched null (prediction minus null median; positive = better than a geometric
  error of equal size): about +0.04 to +0.05 at 8 Å, CI about +0.02 to +0.09.
* About 30% of chains fall below the null; all of them are chains with 13–26 Å RMSD (failed folds).
* Reading: the electrostatic map adds little beyond RMSD/lDDT, consistent with "electrostatics is a
  readout of geometry". This is a statement about these predictions, not about native folding.

**Pending.** The v2 re-run (adds lDDT-matched nulls `smoothL`/`jitterL`, Coulomb then APBS) reported
`chains scored: 0 of 189` for both engines: it was started in the `(rhofold)` environment instead of `(venv)`.
It is being re-run with `--overwrite`. The first-run numbers above should be replaced by the v2 numbers
once `summary.txt` reads `failed: 0`. The MSA-mode predictions are also still to do.

### 4.3 Mg²⁺ sensitivity (pilot: 10 structures, 23 chains, median 7 ions per structure)

Ion-free stored labels vs labels with Mg²⁺ (r = 0.79 Å), within-chain centred:

| quantity | median Pearson (IQR) | label variance changed (median) | global offset |
|---|---|---|---|
| C1′ point | 0.995 (0.986..0.999) | 1.1% | +1.07 kT/e |
| 4 Å mean | 0.988 (0.872..0.998) | 2.9% | +0.77 kT/e |
| 8 Å mean | 0.972 (0.789..0.993) | 7.2% | +0.98 kT/e |
| 12 Å mean | 0.990 (0.907..0.995) | 5.7% | +0.89 kT/e |

By distance to the nearest Mg²⁺ (C1′ point): ≤4 Å rms change 99.6 kT/e (306% of label sd, n = 8);
4–6 Å 19.9 kT/e (61%, n = 56); 6–8 Å 5.9 kT/e (18%); 8–12 Å 4.4 kT/e (14%); beyond 20 Å 2.7 kT/e (8%).
The charge-0 control gives changes ≤ 1.4 kT/e at 8 Å in every distance bin, so the change is the ion's
charge, not the cavity.

Reading: for the whole-chain audit ion-free labels are a reasonable simplification; for residues within
~6 Å of a resolved Mg²⁺ they are not. Only 64 of about 1,280 pilot residues are within 6 Å. Final numbers
need the 120-structure run (in progress).

### 4.4 Coulomb baseline
The screened-Coulomb baseline (`vac`, `dhS`, `dhP`) is part of the controls. `dhP` uses pdie 1, so its
absolute kT/e values are uncalibrated; only correlations and the Wasserstein ratio are trustworthy.

---------------------------------------------------------------------------------------------------

## 5. What the evidence supports

| Claim | Status |
|---|---|
| LMs do not encode electrostatics beyond structure/sequence-context controls | **Supported** (unique_lm ≈ 0 under all controls) |
| Electrostatics is largely a readout of 3D geometry | **Supported** (controls explain about 74–85%) |
| Predicted-structure map fidelity tracks lDDT/RMSD and adds little beyond them | **Supported by the first run**, to be confirmed with v2 and APBS |
| LMs carry structural context (shared variance) | **Not supported as stated**: at the point scale a random network matches it; at sphere scales it falls under leave-family-out |
| Pretraining adds structural information the architecture lacks | **Partly**: for `ernierna`, `rnafm`, `splicebert` at 4/12 Å on a random split; not for `rnabert`; largely gone on unseen families |
| A missing latent→field mapping / predicted structures are "not native" / LMs lack "crucial" information | **Do not claim** (not tested) |
| Ion-free labels are harmless | **True for within-chain pattern, false near Mg²⁺**; say so |

## 6. Caveats

* Layer choice: pretrained runs picked their layer on `centered_base_resid`; the random-init and family runs
  did not pass `--layer`, so each picked its own. Layers may differ between columns. Rerun with `--layer N`
  if a reviewer asks.
* Family-out test sets differ per fold and from the random-split test set.
* Families are 6-mer similarity clusters, not curated Rfam families (use `--method cmscan` for that).
* One random-weights seed (0). Seed-to-seed spread unknown.
* `rnabert` truncates at 438 nt; its test residues differ slightly from the others.
* Labels: static, ion-free, one structure per RNA; 61 structures solved with the linearised fallback.
* Single-sequence RhoFold+ is less accurate than MSA mode; the MSA database used a partial `nt`
  (359 of 400 volumes) once the MSA runs are made.
* The phosphorus point value is dropped from the paper (not trusted).
* Which earlier audit file belonged to which model was never confirmed by hand; the bundle now builds the
  pretrained path per model, so the tables above are correctly labelled.

---------------------------------------------------------------------------------------------------

## 7. Status

**Done**
- [x] Label pipeline; controls tables (structure, DSSR, Coulomb)
- [x] Pretrained audit for 4 models
- [x] Random-weights control (4 models)
- [x] Leave-family-out control (4 models × 5 folds), family table at threshold 0.5
- [x] First predicted-structure benchmark (189 chains, single-sequence, Coulomb, smooth/jitter nulls)
- [x] Mg²⁺ pilot
- [x] Control comparison tables

**Running or to redo**
- [ ] Benchmark v2 (Coulomb then APBS, with lDDT-matched nulls): rerun in `(venv)` with `--overwrite`
- [ ] Mg²⁺ full run (120 structures), started 23 Oct

**To do**
- [ ] MSA-mode predictions (3-chain test, then 189) and the benchmark on them
- [ ] Decide whether to rerun the audit on Mg²⁺-aware labels (`labels_r0.79/`) after the full Mg²⁺ run
- [ ] Optional controls: random-init with more seeds and `--layer` matched; curated Rfam families
- [ ] Cross-model latent agreement with controls (script ready, results not reviewed here)
- [ ] Bootstrap CIs for the control comparison table
- [ ] Rewrite the paper's positive framing (§1 point 3); drop the P-point result
- [ ] Figures and abstract/outline

## 8. Where the outputs are

| Result | Path |
|---|---|
| Pretrained audit | `/data/rna_electrostats/centered_latent_v3/<model>/audit_variance_partition.csv` |
| Random-weights | `/data/rna_electrostats/controls/random_init/<model>/` |
| Leave-family-out | `/data/rna_electrostats/controls/family/<model>/fold0..4/` |
| Control comparison text | `~/projects/codes/rna_mg/results_bundle.txt` |
| Benchmark | `/data/rna_electrostats/bench_coulomb_v2/`, `bench_apbs_v2/` (`summary.txt`, `excess_vs_null.csv`) |
| Mg²⁺ | `/data/rna_electrostats/mg_pilot/`, `mg_full/` |
