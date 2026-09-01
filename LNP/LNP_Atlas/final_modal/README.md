# LNP Encapsulation Efficiency (EE%) Classification

Multi-component molecular fingerprints and a classification-based framework for
predicting lipid nanoparticle (LNP) encapsulation efficiency, plus a
literature-grounded Formulation Quality Score (FQS) for ranking candidate
formulations.

**Author:** Kirti, IISc Bangalore
**Dataset:** LNP Atlas v1 (*Scientific Data*, December 2025)

---

## 1. Files

| File | Purpose |
|---|---|
| `lnp_pipeline.py` | Full research pipeline: cleaning → features → training/validation → interpretability → chemistry analysis → active-learning feedback loop. Reproduces every figure/table. |
| `lnp_model.py` | Standalone inference module — loads `lnp_classifier.pkl` and exposes `LNPModel.predict()` / `.predict_batch()` / `.optimize_ratio()`. No matplotlib/shap/umap dependency. |
| `lnp_classifier.pkl` | Trained model bundle produced by `lnp_pipeline.py` (`SaveAndPredictAPI`). |
| `LNP_Atlas_DB_202509_v1.csv` | Raw input dataset (required by `lnp_pipeline.py`). |

`lnp_pipeline.py` is organized as one class per section, all sharing a single `PipelineState` object; `main()` runs every section in order.

## 2. Requirements

```bash
pip install numpy pandas scipy scikit-learn statsmodels rdkit shap umap-learn matplotlib seaborn
pip install xgboost transformers torch   # optional — pipeline degrades gracefully without these
```
`lnp_model.py` only needs `numpy`, `pandas`, `rdkit`.

## 3. Data cleaning notes

- Rows missing EE% or any of the 4 SMILES are dropped; ambiguous EE% entries (`;`/`~`) dropped; SMILES canonicalized (fixes ~10/198 duplicate ionizable-lipid structures that were leaking across `GroupKFold` folds under different SMILES spellings).
- **Molar-ratio parsing fix:** the raw `lipid_molar_ratio` string does not follow one fixed component order across the dataset — different source papers report positions 2/4 as either `PEG:helper` or `helper:PEG` (verified directly: the identical MC3:DSPC:cholesterol:DMG-PEG2000 formulation appears as both `"50:10:38.5:1.5"` and `"50:1.5:38.5:10"` across papers). `DataCleaning` now assigns the **smaller** of positions 2/4 to PEG and the **larger** to helper (PEG-lipid is virtually always the minority component, 1–3 mol%), flagging affected rows in `ratio_was_reordered`. This is a heuristic, not a per-paper manual check — see the in-code comment for the empirical justification.

## 4. Running

```bash
python lnp_pipeline.py          # full run, regenerates all figures + lnp_classifier.pkl
```
Or run sections individually via `PipelineState` (see module docstring).

## 5. Pipeline sections (summary)

1. **Setup / DataCleaning / EDA** — style config, load+clean data, exploratory plots.
2. **NoiseFloor / RigorousNoiseFloor** — quantifies cross-lab EE% measurement noise; justifies classification over regression.
3. **FeatureEngineering** — count-Morgan ECFP6 (512-bit) fingerprints ×4 lipids + 8 RDKit descriptors + 4 molar fractions; assembles Feature Sets A–E.
4. **FeatureAblation** — 5-fold `GroupKFold`-by-ionizable-lipid comparison of A–E; Monte Carlo noise-injection robustness check (classification AUC vs. regression R²); Set D vs. E significance test.
5. **FingerprintBenchmark / BitWidthComparison** — representation choice and bit-width justification (incl. optional frozen ChemBERTa baseline).
6. **PrimaryModel** — main result: RF, Feature Set D, 5-fold `GroupKFold`, ROC/PR/confusion matrix.
7. **FQS** — Formulation Quality Score (geometric mean of EE/size/PDI desirability functions, literature-anchored thresholds).
8. **ModelComparison** — RF/ExtraTrees/XGBoost comparison, fold-level bootstrap CIs, data-leakage proof (random vs. `GroupKFold` split), baselines, leave-one-publication-out, temporal holdout + drift analysis, calibration comparison (uncalibrated/isotonic/sigmoid).
9. **SaveAndPredictAPI** — trains final model on all data, pickles `lnp_classifier.pkl`, defines `predict_lnp()`.
10. **ConformalPrediction** — split-conformal prediction sets with distribution-free coverage guarantee.
11. **ErrorAnalysis** — false positive/negative profiling + Applicability Domain (kNN distance in fingerprint space).
12. **LearningCurve** — tests whether the model is data-limited.
13. **SHAP** — bit-level and component-level SHAP, chemical interpretation of top bits, SMARTS enrichment, FDR/Bonferroni-corrected design rules, ester→ether in-silico case study.
14. **Chemistry** — UMAP chemical space, ratio optimizer, AD visualization, "universally robust" lipids, free LogP-calibrated pKa, cargo-type effects, data-driven lipid classes (k-means).
15. **FeedbackLoop** — human-in-the-loop retraining pipeline (not automatic online learning) + `suggest_next_candidates()` active-learning acquisition function (uncertainty + novelty + diversity-based shortlist for next synthesis targets).

## 6. Using the trained model

```python
from lnp_model import LNPModel

model = LNPModel.load("lnp_classifier.pkl")
result = model.predict(
    ionizable_smiles=MC3_SMILES, helper_smiles=DSPC_SMILES,
    sterol_smiles=CHOL_SMILES, peg_smiles=DMGPEG_SMILES,
    molar_ratio="50:10:38.5:1.5",   # ionizable:helper:sterol:peg
    size_nm=85.0, pdi=0.12,
)
print(result)
```

> **Note on molar-ratio order:** `lnp_model.py` expects `ionizable:helper:sterol:PEG`; `lnp_pipeline.py`'s `predict_lnp()` expects `ionizable:PEG:sterol:helper`. Both are internally consistent with their own demos and produce correct fractions — just don't copy a ratio string from one file's convention into the other without reordering PEG/helper.

## 7. Known limitations

- pKa is a transparent literature-anchored **heuristic**, not a calibrated predictor (no ChemAxon access assumed); the free LogP-calibrated alternative is validated on only 5 anchor points (leave-one-out RMSE reported).
- FQS size/PDI thresholds are drawn from specific cited literature/regulatory sources — verify these are the thresholds you want to defend in a manuscript.
- The molar-ratio PEG/helper fix is a **heuristic** (smaller value → PEG), not a manual per-paper audit; rare genuinely high-PEG formulations could be mislabeled. `ratio_was_reordered` flags affected rows for audit.
- `FeedbackLoop` retraining is deliberately human-in-the-loop, not automatic (see in-code rationale).
- N/P ratio was evaluated and excluded as a feature (available for only ~22% of curated data; encodes protocol, not molecular structure).

## 8. Citation

