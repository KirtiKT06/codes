# KUMUDLNP Pipeline — Mathematical and Logical Reference

This document walks through every stage of `lnp_ee_prediction_oop_fixed.py`, in the order the
pipeline runs, and explains **what each section computes, why, and the exact math behind it**.
It assumes you know basic statistics and linear algebra but not necessarily cheminformatics or
ML internals — those are explained from first principles where they appear.

Notation used throughout: $n$ = number of formulations (452 after cleaning); vectors are bold
lowercase ($\mathbf{x}$), matrices bold uppercase ($\mathbf{X}$); $\mathbb{1}[\cdot]$ is the
indicator function (1 if true, 0 if false).

---

## 0. Setup

No statistics here — this section only configures matplotlib's fonts, colors, and a
`save_pub_figure()` helper that rescales any figure to a fixed journal column width (88 mm
single-column, 180 mm double-column) before saving. Skip ahead.

---

## 1. DataCleaning

**Goal:** turn the raw, messy literature-aggregated CSV (1,092 rows, 63 papers) into 452 rows
you can trust.

### 1.1 Filtering
A row is kept only if it has a non-missing EE% value and all four lipid SMILES strings
(ionizable, helper, sterol, PEG) are present and parse as valid molecules under RDKit
(`Chem.MolFromSmiles(x) is not None`). Rows whose reported EE% string contains `;` or `~`
(ambiguous ranges/estimates, e.g. "70-80%" written informally) are dropped.

### 1.2 The molar-ratio reordering fix
Each formulation reports a 4-number ratio string like `50:1.5:38.5:10`. The problem: different
source papers order the middle two numbers differently — some write
`ionizable : PEG : sterol : helper`, others `ionizable : helper : sterol : PEG`. This is not a
guess; it was verified directly by finding the *same real formulation* (MC3:DSPC:cholesterol:
DMG-PEG2000, with true composition 50% ionizable / 10% helper / 38.5% sterol / 1.5% PEG)
reported as **both** `50:10:38.5:1.5` and `50:1.5:38.5:10` across two different papers for
identical chemistry.

The fix uses a simple, defensible heuristic grounded in real LNP chemistry: PEG-lipid is
*always* the minority component in practice (typically 1–3 mol%, essentially never more than
helper lipid). So, given the two ambiguous middle values $p_1$ (raw position 2) and $p_2$ (raw
position 4):

$$
\text{PEG\%} = \min(p_1, p_2), \qquad \text{Helper\%} = \max(p_1, p_2)
$$

Every fraction is then normalized by the row's own sum, since some rows' four numbers don't
sum to exactly 100:

$$
f_i = \frac{r_i}{\sum_{j \in \{\text{ion, peg, sterol, helper}\}} r_j}, \qquad i \in \{\text{ion, peg, sterol, helper}\}
$$

This isn't presented as certain — a `ratio_was_reordered` flag is kept on every affected row so
it can be manually audited.

### 1.3 SMILES canonicalization
Two different SMILES strings can represent the identical molecule (e.g. `CCO` and `OCC` are
both ethanol). Every SMILES is round-tripped through RDKit's canonical form:
$\text{SMILES} \to \text{Mol} \to \text{canonical SMILES}$, so that later steps (grouping by
lipid identity, deduplication) compare molecules, not strings.

---

## 2. EDA
Plain descriptive statistics and histograms — no new math. Skip ahead.

---

## 3. NoiseFloor — how much of the "signal" is actually measurement noise?

**The core question:** if you measured the *exact same* formulation twice, how different would
the two EE% readings be, just from lab-to-lab and assay-to-assay variation? This number is a
hard floor — no model, however good, can predict EE% more precisely than this, because the
target itself isn't reproducible below this level.

For every ionizable lipid with $\geq 2$ independent measurements (regardless of partner
lipids/ratio), compute the standard deviation of EE% across those measurements:

$$
\sigma_\ell = \sqrt{\frac{1}{n_\ell - 1}\sum_{i=1}^{n_\ell} (EE_{\ell,i} - \overline{EE_\ell})^2}
$$

for lipid $\ell$ with $n_\ell$ observations. The **noise floor** reported is the *median* of
$\sigma_\ell$ across all such lipids (9.9%) — median rather than mean because a handful of
lipids with wildly discordant measurements would otherwise dominate a mean and overstate the
typical noise level.

**Why this matters downstream:** this is the empirical justification for reframing the problem
as binary classification (High EE $\geq 80\%$ vs Low EE) instead of regression. A regression
model promising to predict EE% to within, say, 2 percentage points is promising something the
data itself cannot support, since two honest replicate measurements of the same formulation
already disagree by ~10 points on average.

---

## 4. RigorousNoiseFloor — a stricter noise estimate, and variance partitioning

### 4.1 The stricter between-lab estimate
Section 3's estimate mixes two things: genuine lab noise, *and* real chemistry differences
between formulations that happen to share an ionizable lipid but differ in helper/sterol/PEG
partner or ratio. To isolate pure measurement noise, this section restricts to formulations
that are **identical in all four SMILES and the molar ratio**, reported independently by
$\geq 2$ different papers (14 such cases). Since chemistry is now held perfectly fixed, any
remaining spread is attributable only to lab/protocol/assay variability. This gives the
**primary** noise-floor estimate used throughout the paper (median 11.3%).

### 4.2 Mixed-effects variance partitioning
This answers: "of all the variability in EE% across the whole dataset, how much is
systematically explained by X?" for X = cargo type, ionizable-lipid identity, and exact
formulation identity.

The model fit (via `statsmodels.mixedlm`) for a grouping variable $g$ is a **random-intercept
mixed-effects model**:

$$
EE_{ij} = \mu + u_j + \epsilon_{ij}, \qquad u_j \sim \mathcal{N}(0, \tau^2), \qquad \epsilon_{ij} \sim \mathcal{N}(0, \sigma^2)
$$

where $EE_{ij}$ is the $i$-th observation in group $j$ (e.g. the $j$-th ionizable lipid),
$\mu$ is the overall mean, $u_j$ is a random deviation specific to group $j$ (the "between-group"
component), and $\epsilon_{ij}$ is residual noise (the "within-group" component). $\tau^2$
(between-group variance) and $\sigma^2$ (residual variance) are estimated by REML
(restricted maximum likelihood).

The fraction of total variance attributable to the grouping variable is the
**intraclass correlation coefficient (ICC)**:

$$
\text{ICC} = \frac{\tau^2}{\tau^2 + \sigma^2}
$$

Three ICCs are computed, one per grouping variable:
- Cargo identity: ICC = 0.021 → only 2.1% of variance tracks with mRNA/siRNA/DNA
- Ionizable-lipid identity: ICC = 0.144 → 14.4% of variance tracks with which ionizable lipid was used
- Exact formulation identity (all 4 components + ratio): ICC = 0.496 → **49.6%** of variance is reproducibly linked to the *complete* formulation recipe

The interpretation: EE% is not a property you can attribute mostly to any single ingredient
(especially not the ionizable lipid alone, which the field conventionally treats as the main
design variable) — it's a property of the whole formulation, and even then, roughly half the
observed variance is *not* explained by formulation identity at all, consistent with the
measurement-noise floor established above.

---

## 5. FeatureEngineering — turning molecules into numbers

A machine-learning model needs fixed-length numeric vectors, not SMILES strings. This section
builds two kinds of features per lipid component (ionizable, helper, sterol, PEG), then
combines them with the molar-fraction features into five candidate feature sets.

### 5.1 Morgan (ECFP) fingerprints — the main representation

A Morgan fingerprint encodes a molecule's local substructures. The algorithm, in plain terms:

1. **Radius-0**: every atom gets an initial identifier based on its own properties (element,
   charge, degree, etc.), via an invariant hash.
2. **Radius-$r$ (iterative)**: each atom's identifier is updated by hashing together its own
   current identifier with the identifiers of its immediate neighbors:
   $$
   h_a^{(r)} = \text{Hash}\Big(h_a^{(r-1)}, \{h_b^{(r-1)} : b \in \mathcal{N}(a)\}\Big)
   $$
   This is exactly one round of message passing — after $r$ rounds, atom $a$'s identifier
   encodes everything within $r$ bonds of it. This pipeline uses $r=3$ ("ECFP6" — the 6 refers
   to the diameter, $2r$, of the considered neighborhood).
3. **Folding into a fixed-width vector**: each unique atom-environment identifier across all
   radii is hashed into one of $B$ bins ($B=512$ here, chosen empirically — see §8):
   $$
   \text{bit}(h) = h \bmod B
   $$
   The fingerprint vector $\mathbf{v} \in \mathbb{R}^{B}$ then holds, at each bin, the **count**
   of how many atom-environments hashed there (a count-based fingerprint, not just
   present/absent — this preserves more information, e.g. "this substructure appears 3 times").

Each of the 4 lipid components gets its own 512-dimensional fingerprint, giving up to
$4 \times 512 = 2048$ raw columns; columns that are all-zero across every formulation (bits no
molecule in the dataset ever activated) are dropped, leaving 820 "active" columns.

### 5.2 RDKit physicochemical descriptors
Eight standard descriptors per component: molecular weight (MolWt), calculated logP (a
measure of lipophilicity — how much a molecule prefers a fat/oil phase over water), topological
polar surface area (TPSA), hydrogen-bond acceptor/donor counts, rotatable-bond count, ring
count, and the fraction of sp$^3$ carbons. These are supplementary to the fingerprints, not a
replacement — a fingerprint bit says "this exact substructure is present," a descriptor gives a
smooth, continuous physical quantity.

### 5.3 Feature sets A–E
Five combinations are assembled for the ablation study (§6):

| Set | Contents |
|---|---|
| A | molar-ratio fractions only (4 features) |
| B | descriptors (all 4 components) + ratios (30) |
| C | fingerprints of the ionizable lipid only + ratios (474) |
| D | fingerprints of **all 4 components** + ratios (824) — **the main feature set used everywhere downstream** |
| E | fingerprints + descriptors + ratios, everything combined (850) |

---

## 6. FeatureAblation — which feature set actually helps, and by how much

### 6.1 Evaluation protocol: GroupKFold by ionizable lipid
Standard $k$-fold cross-validation splits *rows* randomly into folds. That would let two
formulations of the *same* (or a near-identical) ionizable lipid land in different folds — the
model could then partly "cheat" by memorizing that lipid's typical EE% from the training fold
and simply recognizing it in the test fold, rather than learning transferable structure-property
relationships. **GroupKFold** instead splits by *group* (here: ionizable lipid identity) — every
formulation sharing an ionizable lipid is guaranteed to land entirely in one fold. This is the
single most important methodological choice in the whole pipeline, and its effect is
quantified directly in §11.4.

### 6.2 Metrics: ROC-AUC and Average Precision

**ROC-AUC.** Rank every formulation by predicted probability of High EE. Pick one true-positive
formulation and one true-negative formulation at random. AUC is the probability the model
ranked the positive one higher:

$$
\text{AUC} = P\big(\hat{p}(x^+) > \hat{p}(x^-)\big)
$$

equivalently, the area under the curve of True Positive Rate vs. False Positive Rate as the
decision threshold sweeps from 1 to 0. AUC = 0.5 is random guessing; AUC = 1.0 is perfect
ranking. Crucially, **AUC does not depend on any decision threshold** — it measures the quality
of the model's ranking, independent of where you'd eventually draw the "High EE" cutoff line.

**Average Precision (AP).** The area under the Precision-Recall curve:
$$
\text{AP} = \sum_k (R_k - R_{k-1}) \, P_k
$$
where $P_k, R_k$ are precision and recall at the $k$-th threshold, sorted by decreasing
predicted probability. Unlike AUC, AP is sensitive to class imbalance in a useful way — with
these formulations roughly 53%/47% High/Low, imbalance isn't severe here, but AP is reported
alongside AUC as good practice.

### 6.3 The Set D vs Set E paired significance test
Set E (fingerprints + descriptors) scored marginally higher AUC than Set D (fingerprints
alone) on average, but is that difference real or just noise? A **paired Wilcoxon signed-rank
test** is used: for each of the 5 folds, compute $\Delta_i = \text{AUC}_E^{(i)} -
\text{AUC}_D^{(i)}$, then test whether the median of $\Delta$ is zero, using only the *signs and
ranks* of the differences (robust to the small sample size of $n=5$ folds, where a $t$-test's
normality assumption would be shaky). Result: $p = 1.0$ — no evidence the extra descriptors
help. Set D is kept as the primary feature set on the principle of preferring the simpler
representation when performance is statistically indistinguishable.

---

## 7. FingerprintBenchmark — is ECFP6 actually the best choice of representation?

Five fingerprint families are benchmarked under the identical GroupKFold protocol:

- **ECFP4/ECFP6**: as in §5.1, with radius 2 or 3.
- **MACCS keys**: a *fixed* dictionary of 166 hand-curated substructure patterns (e.g. "contains
  a ring," "contains N-O bond") — each bit means something a chemist chose in advance, unlike
  Morgan's arbitrary hashed bits.
- **Atom-pair fingerprint**: encodes every pair of atoms in the molecule together with the
  topological (bond-count) distance between them.
- **Topological torsion fingerprint**: encodes chains of 4 consecutive bonded atoms.
- **ChemBERTa embeddings**: a pretrained transformer language model (trained on millions of
  SMILES strings, the same architecture family as GPT/BERT) reads the SMILES text and outputs a
  continuous 768-dimensional vector per molecule via **mean pooling**:
  $$
  \mathbf{e} = \frac{\sum_{t=1}^{T} m_t \, \mathbf{h}_t}{\sum_{t=1}^{T} m_t}
  $$
  where $\mathbf{h}_t$ is the model's internal hidden-state vector for token $t$, and $m_t \in
  \{0,1\}$ masks out padding tokens. This is used *frozen* (no fine-tuning) — deliberately, since
  with only 452 formulations, fine-tuning a transformer risks memorizing rather than learning.

Result: ECFP6 (824-dim) wins outright (AUC 0.778–0.790 depending on run/environment), beating
even the descriptor-only baseline and, notably, beating the pretrained language-model embedding
(AUC ~0.72). This is a real, useful negative result for the field: bigger/fancier
representations aren't automatically better on small, chemically narrow datasets like this one.

---

## 8. BitWidthComparison — choosing the fingerprint's bit width

Recall the folding step from §5.1: `bit(h) = h mod B`. A smaller $B$ means more distinct
substructures get hashed into the same bin (a **collision**) — the fingerprint becomes less
informative, since two genuinely different substructures now look identical to the model. A
larger $B$ reduces collisions but adds more mostly-empty, sparse dimensions, which can hurt a
tree-based model's ability to find useful splits and inflates variance across folds. This is
swept at $B \in \{256, 512, 1024, 2048\}$ per component; $B=512$ wins (AUC 0.778), striking the
best balance — used as the standard fingerprint width for the rest of the pipeline.


## 9. PrimaryModel — the main classifier, and threshold selection

### 9.1 Random Forest, briefly
A Random Forest is an ensemble of $T=500$ decision trees. Each tree is grown on:
- a **bootstrap resample** of the training rows (sampling $n$ rows with replacement — each tree
  sees a different, overlapping subset of formulations), and
- at each split, only a **random subset of features** is considered (not all 824), which
  decorrelates the trees from one another.

Each tree makes a hard 0/1 prediction; the forest's predicted probability is the fraction of
trees voting "High EE":
$$
\hat{p}(x) = \frac{1}{T}\sum_{t=1}^{T} \mathbb{1}\big[\text{tree}_t(x) = \text{High EE}\big]
$$
`class_weight='balanced'` reweights the two classes inversely proportional to their frequency
during training, so the roughly-balanced-but-not-quite class split (52.7%/47.3%) doesn't bias
the trees toward the majority class.

### 9.2 Why GroupKFold matters here specifically (the leakage demonstration)
§11.4 makes this quantitative, but the logic is introduced here: every reported AUC/AP number in
this section comes from **out-of-fold (OOF) predictions** — for each formulation, its predicted
probability was produced by a model that never saw that formulation's ionizable lipid during
training. This is the strict, honest test of generalization to unseen chemistry.

### 9.3 Threshold selection — the corrected, pooled design
A probability like $\hat{p}=0.62$ isn't yet a decision. You need a threshold $t$: predict
"High EE" if $\hat{p} \geq t$. Naively picking $t$ to maximize a metric on the *same* predictions
being reported is a subtle form of leakage — you're optimizing against the answer key.

The fix here is **double (nested) cross-validation**, applied to threshold selection alone
(never to the reported AUC/AP, which don't need a threshold at all):

1. For each of the 5 outer folds, take *only the outer training data* (the ~360 formulations
   not in the test fold) and run an **inner** 5-fold GroupKFold on it, producing inner
   out-of-fold predictions $(\hat{p}_i, y_i)$ for those ~360 formulations — every inner
   prediction still comes from a model that never saw that formulation.
2. Pool the inner-OOF predictions from **all 5 outer folds together** (since each formulation
   sits in the training set of 4 of the 5 outer folds, this pools roughly $4 \times 452
   \approx 1{,}800$ observations — far more than any single fold's ~90 test points).
3. Choose **one** threshold $t^*$ by grid search over $t \in \{0.10, 0.11, \ldots, 0.90\}$
   ($k=80$ candidate values), maximizing **balanced accuracy**:
   $$
   \text{BalAcc}(t) = \tfrac{1}{2}\Big(\underbrace{\tfrac{TP(t)}{TP(t)+FN(t)}}_{\text{sensitivity}} + \underbrace{\tfrac{TN(t)}{TN(t)+FP(t)}}_{\text{specificity}}\Big), \qquad t^* = \arg\max_t \text{BalAcc}(t)
   $$
   Balanced accuracy (rather than plain accuracy) is used because it weights both classes
   equally regardless of their relative frequency.
4. **Uncertainty on $t^*$ itself** is estimated by the bootstrap: resample the pooled inner-OOF
   set with replacement 500 times, recompute $t^*$ each time, and take the 2.5th/97.5th
   percentiles of the resulting distribution as a 95% confidence interval.

This single $t^*$ (not five different per-fold values) is then applied uniformly to the outer
OOF predictions to compute the confusion matrix and derived rates:

$$
\text{Sensitivity} = \frac{TP}{TP+FN}, \quad
\text{Specificity} = \frac{TN}{TN+FP}, \quad
\text{PPV} = \frac{TP}{TP+FP}, \quad
\text{NPV} = \frac{TN}{TN+FN}
$$

where PPV (positive predictive value) is the probability a "High EE" call is actually correct,
and NPV (negative predictive value) is the same for "Low EE" calls.

---

## 10. FQS — Formulation Quality Score

**Goal:** combine three separate pieces of evidence (model confidence, particle size,
polydispersity) into a single ranking number, since a formulation with a great EE% prediction
but a terrible particle size isn't actually a good drug candidate.

### 10.1 Desirability functions (Derringer–Suich, 1980)
Each raw quantity is mapped to a **desirability score** $d \in [0,1]$, where 1 = ideal and 0 =
unacceptable, using a shape appropriate to that quantity:

**EE desirability** — "larger is better," with an exponent $r$ controlling how sharply the score
rewards high-confidence predictions over merely-above-chance ones:
$$
d_{EE}(P) = P^r, \qquad r = 3
$$
At $r=3$, $P=0.9$ maps to $d=0.73$ while $P=0.5$ maps to $d=0.125$ — steeper than linear
($r=1$), so the score doesn't reward marginal predictions much.

**Size desirability** — "target is best," a trapezoid: 0 below 30 nm, ramping linearly to 1 at
50 nm, flat at 1 across the FDA/USP-informed optimal window (50–120 nm), ramping back down to 0
by 150 nm:
$$
d_{\text{size}}(s) = \begin{cases}
0 & s \leq 30 \\
(s-30)/20 & 30 < s \leq 50 \\
1 & 50 < s \leq 120 \\
(150-s)/30 & 120 < s \leq 150 \\
0 & s > 150
\end{cases}
$$

**PDI desirability** — "smaller is better" (PDI, polydispersity index, measures how uniform the
particle sizes in a batch are — 0 is perfectly uniform):
$$
d_{\text{PDI}}(p) = \begin{cases} 1 & p \leq 0.20 \\ (0.30-p)/0.10 & 0.20 < p \leq 0.30 \\ 0 & p > 0.30 \end{cases}
$$

### 10.2 Combining scores: geometric mean
The three (or fewer, if size/PDI weren't measured) desirabilities are combined via **geometric
mean**, not arithmetic mean:
$$
\text{FQS} = 100 \times \Big(\prod_{i=1}^{k} d_i\Big)^{1/k}
$$
The geometric mean is deliberate: it's the standard choice for composite desirability indices
because it heavily penalizes a formulation that's unacceptable on *any single* criterion — if
$d_i = 0$ for even one component (e.g. wildly oversized particles), the entire FQS collapses to
0, regardless of how good the other two scores are. An arithmetic mean would let a great EE
score partially compensate for a disqualifying size, which is not the behavior you want from a
developability screen.


## 11. ModelComparison — the robustness suite

This section runs seven separate checks. Each answers a different "but what if...?" question a
skeptical reviewer would ask.

### 11.1 Model comparison: is Random Forest actually the best algorithm?
Random Forest, ExtraTrees, and XGBoost are compared under identical folds. **ExtraTrees**
differs from Random Forest only in how it picks split points: instead of searching for the
single best threshold on a randomly chosen feature, it picks the split threshold *randomly too*
and only chooses the best feature among those random splits — more randomness, usually faster,
sometimes better on noisy data. **XGBoost** is a gradient-boosted ensemble: trees are added
sequentially, each new tree fit to the *residual errors* of the ensemble so far (rather than
independently, as in Random Forest/ExtraTrees).

### 11.2 Bootstrap confidence intervals — on fold-level scores, not pooled probabilities
A subtlety flagged directly in the code's comments: naively pooling all 452 OOF probabilities
from ExtraTrees/XGBoost across folds and computing one AUC on the pooled set is *invalid* here,
because each fold's model has its own probability "scale" (especially with XGBoost's per-fold
`scale_pos_weight` recalibration) — concatenating and ranking probabilities from differently-
calibrated models collapses the apparent AUC toward chance even though each model discriminates
fine within its own fold. The fix: bootstrap over the 5 **fold-level AUC values** themselves
(each already a valid, self-contained statistic), not over pooled raw probabilities:
$$
\text{CI}_{95\%} = \Big[\text{percentile}_{2.5}\{\bar{a}^{(b)}\}, \; \text{percentile}_{97.5}\{\bar{a}^{(b)}\}\Big], \quad \bar{a}^{(b)} = \text{mean of a resample of the 5 fold AUCs}
$$
repeated for $b=1,\ldots,1000$ resamples.

### 11.3 McNemar's test — comparing two classifiers on the *same* predictions
Unlike the Wilcoxon test in §6.3 (which compares fold-level summary scores), McNemar's test
compares two models' **individual, paired predictions** directly. Build a 2×2 table of
disagreements: $b$ = formulations RF got right but ExtraTrees got wrong; $c$ = the reverse. The
test statistic:
$$
\chi^2 = \frac{(|b-c|-1)^2}{b+c}
$$
(the $-1$ is a continuity correction for the discreteness of counts) tests whether the two
models' *errors* are systematically different, not just whether their aggregate accuracy
differs.

### 11.4 The leakage proof — quantifying why GroupKFold matters
This is the direct, empirical demonstration behind the methodological claim: 20 repeats each of
(a) plain random 5-fold `StratifiedKFold` and (b) `GroupKFold` by ionizable lipid, each
producing a mean AUC. A two-sample $t$-test compares the two sets of 20 means:
$$
t = \frac{\bar{X}_{\text{random}} - \bar{X}_{\text{group}}}{\sqrt{s_{\text{random}}^2/n_1 + s_{\text{group}}^2/n_2}}
$$
Result: random splitting inflates AUC by ~0.087 (a ~11% relative overestimate), and the
difference is astronomically significant ($p \approx 10^{-35}$) — direct evidence that random
splitting lets the model partially "cheat" via near-duplicate lipids leaking across folds, and
that the lipid-grouped protocol used everywhere else in this pipeline is not an arbitrary
methodological preference but a measured necessity.

### 11.5 Leave-one-publication-out and the publication-grouped control
Repeats the same grouped-CV logic, but grouping by *publication* instead of *lipid*. If AUC
holds up under this stricter test too, it means the model isn't secretly learning lab-specific
or protocol-specific artifacts (e.g. a particular lab's typical particle-size measurement
convention) — it's learning something that transfers across labs. The modest AUC drop observed
(lipid-grouped ~0.78–0.79 vs. publication-grouped ~0.70) is interpreted honestly as evidence
that *some* lab/protocol-specific signal is present, layered on top of the dominant,
chemistry-driven signal.

A **publication-grouped control** additionally checks whether the temporal-drift results (§11.6)
are really about chemical *novelty over time*, or just an artifact of training-set size, by
holding training size roughly fixed (~122 samples) while varying which publications are
included — isolating the "how much distinct chemistry is in the training set" variable from the
"how much total data is in the training set" variable.

### 11.6 Calibration: Brier score and Expected Calibration Error (ECE)
A well-*calibrated* model's predicted probabilities should match observed frequencies — among
all formulations predicted at $\hat{p}=0.7$, about 70% should actually be High EE.

**Brier score** — mean squared error between predicted probability and the true 0/1 outcome:
$$
\text{Brier} = \frac{1}{n}\sum_{i=1}^n (\hat{p}_i - y_i)^2
$$
(0 = perfect, 0.25 = the score of always predicting 0.5 on a balanced problem.)

**ECE** — bins predictions into 10 probability buckets and averages, weighted by bucket size,
the gap between each bucket's mean predicted probability and its actual observed frequency:
$$
\text{ECE} = \sum_{b=1}^{10} \frac{|B_b|}{n} \,\Big|\, \hat{p}_{B_b} - y_{B_b} \,\Big|
$$
Two post-hoc recalibration methods are tried and compared against the native (uncalibrated) RF
output: **isotonic regression** (fits an arbitrary monotonic step function mapping raw
probability to calibrated probability) and **sigmoid/Platt scaling** (fits a 2-parameter
logistic curve for the same purpose), both via nested CV to avoid leakage. Neither improved
Brier score over the native RF output in this dataset, so uncalibrated probabilities are used
throughout — a deliberate, tested decision, not an oversight.


## 12. SaveAndPredictAPI — packaging the model

No new statistics. The final model is retrained on **all 452 formulations** (no held-out test
set — this is the deployable version, distinct from the 5 fold-specific models used for
evaluation above) and pickled together with everything needed to reproduce its exact feature
vector for a brand-new formulation: which 820 fingerprint bit-columns were "active" in training
(§5.1), the fingerprint radius/width, and the pooled decision threshold from §9.3.
`predict_lnp()` simply re-runs the exact same featurization steps from §5 on new SMILES/ratio
input and calls `.predict_proba()`.

---

## 13. ConformalPrediction — calibrated, distribution-free uncertainty

### 13.1 The guarantee being built
A raw probability, even a well-calibrated one, doesn't directly answer "should I trust *this*
specific prediction?" Conformal prediction constructs, for every new input, a **prediction set**
(not just a single label) with a formal coverage guarantee: over many predictions, the true
label will fall inside the predicted set at least $1-\alpha$ of the time (here $\alpha=0.10$,
i.e. 90% target coverage) — and this guarantee holds **regardless of what distribution the data
actually follows** ("distribution-free"), which is what makes it more trustworthy than an
assumption-laden confidence interval.

### 13.2 Nonconformity score
For a binary problem, the nonconformity score of a prediction is simply how far the model's
probability was from being certain and correct:
$$
s(x,y) = 1 - \hat{p}(y \mid x)
$$
i.e. if the true label is High EE and the model said $\hat{p}=0.9$ for High EE, nonconformity is
$0.1$ (very conforming); if the true label was Low EE but the model said $\hat p = 0.9$ for High
EE, nonconformity is $1-(1-0.9)=0.9$ (very non-conforming — the model was confidently wrong).

### 13.3 Calibration and the threshold $\hat{q}$
Using a held-out calibration set of $n_{\text{cal}}$ points, compute the nonconformity score for
each, then take the following empirical quantile (the $\lceil (n+1)(1-\alpha) \rceil / n$-th
smallest value, a finite-sample correction that makes the coverage guarantee mathematically
exact, not just approximate):
$$
\hat{q} = \text{Quantile}\Big(\{s_i\}_{i=1}^{n_{\text{cal}}}; \; \frac{\lceil (n_{\text{cal}}+1)(1-\alpha)\rceil}{n_{\text{cal}}}\Big)
$$
(this pipeline got $\hat{q} = 0.751$).

### 13.4 Constructing the prediction set for a new formulation
For a new formulation with predicted probability $\hat{p}$, include a class $y$ in the
prediction set if its nonconformity score doesn't exceed $\hat{q}$:
$$
\mathcal{C}(x) = \{ y : s(x,y) \leq \hat{q} \}
$$
Concretely: include "High EE" if $1-\hat p \le \hat q$ (i.e. $\hat p \ge 1-\hat q = 0.249$), and
include "Low EE" if $\hat p \le \hat q = 0.751$. Since $0.249 < 0.751$, there's a middle band
of probabilities (roughly $0.249$ to $0.751$) where **both** conditions hold and the set
contains both labels — the model honestly reporting "I can't rule out either answer" rather than
forcing a single guess. A probability far outside that band yields a **singleton** set — a
confident call. Empirically, 130/226 test formulations got singleton sets, and those singleton
predictions were right 79.2% of the time — notably better than the model's overall accuracy,
confirming the confident/uncertain split is doing real, useful work.

---

## 14. ErrorAnalysis — where the model fails, and the Applicability Domain

### 14.1 Error stratification
Standard confusion-matrix breakdown (TP/TN/FP/FN, §9.3), then compares the false positives
(model said High EE, was actually Low) and false negatives (model said Low EE, was actually
High) on covariates like true EE%, particle size, cargo type, and — critically — chemical
novelty (below), to see if errors cluster somewhere systematic rather than being random noise.

### 14.2 Applicability Domain (AD) — when should you trust a prediction at all?
Every model has a "comfort zone" defined by its training data; predictions on chemistry far
outside that zone are less trustworthy, however confident the raw probability looks. This is
quantified using the **Jaccard distance** in fingerprint space. For two binary fingerprint
vectors $\mathbf{a}, \mathbf{b} \in \{0,1\}^B$ (bit present/absent, ignoring counts here):
$$
J(\mathbf{a}, \mathbf{b}) = 1 - \frac{|\mathbf{a} \cap \mathbf{b}|}{|\mathbf{a} \cup \mathbf{b}|}
$$
i.e. 1 minus the fraction of "on" bits the two molecules share out of all bits either has on —
0 for identical fingerprints, 1 for fingerprints sharing no active bits at all.

For each formulation, compute its mean Jaccard distance to its $k=5$ nearest neighbors (by
ionizable-lipid fingerprint) in the training set. The **AD threshold** is the 95th percentile of
this distance across the training set itself — i.e., "how novel does a formulation have to be
to be more novel than 95% of what the model was trained on?" A formulation whose nearest-
neighbor distance exceeds this threshold is flagged as outside the applicability domain.

This is validated directly: AUC computed separately for in-AD (0.802) vs out-of-AD (0.598)
formulations shows a real, substantial performance drop outside the domain — confirming the
metric is measuring something that actually predicts reliability, not just an arbitrary
novelty score.

---

## 15. LearningCurve — is the model data-limited?

Repeatedly (10 times) subsample increasing fractions (10%–90%) of the *lipid groups* (not rows
— consistent with the GroupKFold logic throughout) from a fixed training pool, retrain, and
evaluate on one fixed held-out 20% lipid split. If AUC has plateaued by 90% of available data,
more data likely won't help much; if it's still rising (as observed here — AUC climbs from
0.660 at 10% to 0.815 at 90%, with no sign of flattening), the honest conclusion is that the
model is currently limited by dataset size, not by a ceiling in the chemistry-encoding approach
itself.


## 16. SHAP — interpretability, functional groups, and positional topology

This is the largest section of the pipeline; the math splits into several independent tools
applied to the same underlying question: *which chemical features drive EE%, and why does the
model believe that?*

### 16.1 SHAP values — attributing a prediction to its inputs
SHAP (SHapley Additive exPlanations) borrows a concept from cooperative game theory: if a
model's prediction is the "payout" of a game where each feature is a "player," how much did each
player individually contribute? The Shapley value for feature $i$ on a specific prediction is:
$$
\phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!} \Big[ v(S \cup \{i\}) - v(S) \Big]
$$
where $F$ is the full feature set, $S$ ranges over every possible subset not containing feature
$i$, and $v(S)$ is the model's expected output using only the features in $S$. In plain terms:
average, over every possible order in which features could be "added" to the model, how much
adding feature $i$ changed the prediction. This has a unique, mathematically guaranteed property
relevant here: the contributions sum exactly to the difference between the actual prediction and
the model's average prediction — $\sum_i \phi_i = \hat{p}(x) - \mathbb{E}[\hat{p}]$ — so nothing
is double-counted or left unexplained.

Computing this exactly is exponential in the number of features (infeasible for 824 features);
`shap.TreeExplainer` uses an exact, polynomial-time algorithm specific to tree ensembles (Lundberg
et al. 2020) that exploits the tree structure to compute the same Shapley values without brute-
force enumeration. **Mean absolute SHAP value** per feature, averaged across a sample of 200
formulations, ranks features by overall importance: $\overline{|\phi_i|} = \frac{1}{200}
\sum_{j=1}^{200} |\phi_i^{(j)}|$.

### 16.2 SMARTS substructure matching
A SMARTS string is a pattern-matching query language for molecular substructures (the molecular
equivalent of a regular expression). E.g. `[OX2H]` matches "an oxygen atom, exactly 2 connections,
with an attached hydrogen" — a hydroxyl group. `mol.GetSubstructMatches(pattern)` returns every
location in the molecule where the pattern matches; presence/absence and counts of these matches
become binary/count features tested against EE%.

### 16.3 Statistical association tests
Several different tests are used because they answer subtly different questions and have
different assumptions — using all of them and checking they agree is itself evidence the
findings aren't an artifact of one particular test's assumptions.

**Point-biserial correlation $r$** — the correlation between a binary variable (feature
present/absent) and a continuous one (EE%); mathematically identical to a standard Pearson
correlation where one variable happens to be 0/1:
$$
r = \frac{\overline{EE}_{1} - \overline{EE}_{0}}{s_{EE}} \sqrt{\frac{n_1 n_0}{n^2}}
$$

**Spearman rank correlation $\rho$** — like Pearson correlation, but computed on the *ranks* of
the values rather than the raw values, making it robust to outliers and to nonlinear-but-
monotonic relationships:
$$
\rho = 1 - \frac{6\sum_i d_i^2}{n(n^2-1)}, \qquad d_i = \text{rank}(x_i) - \text{rank}(y_i)
$$

**Mann-Whitney U test** — tests whether two groups' distributions differ, using only rank
information (no assumption that EE% is normally distributed within each group, which it isn't):
$$
U = \sum_{i=1}^{n_1}\sum_{j=1}^{n_2} \mathbb{1}[x_i > y_j]
$$
i.e., across every possible pairing of one "present" formulation with one "absent" formulation,
how often does the present one have higher EE%? This underlies every $\Delta EE$ figure in the
paper (e.g. "ether presence: +19.7 points, $p<0.001$").

**Chi-square test and $\phi$ coefficient** — for a 2×2 table of (feature present/absent) ×
(High/Low EE), $\chi^2$ tests whether the two are independent; $\phi = \sqrt{\chi^2/n}$ rescales
this into an effect-size measure comparable across different sample sizes (analogous to a
correlation coefficient for two binary variables).

### 16.4 Multiple-testing correction — Benjamini-Hochberg FDR and Bonferroni
Running many hypothesis tests inflates the chance that *some* test looks significant purely by
luck — if you run 100 independent tests all with no real effect, you'd expect about 5 to show
$p<0.05$ anyway. Two corrections are used:

**Bonferroni** — the strict, conservative option: multiply every raw $p$-value by the total
number of tests $m$ (capped at 1):
$$
p_{\text{Bonf}} = \min(m \cdot p, \; 1)
$$
This controls the probability of *even one* false positive across all $m$ tests, at the cost of
substantial statistical power.

**Benjamini-Hochberg FDR** — controls the *expected proportion* of false positives among tests
called significant (a less strict, higher-power standard, often more appropriate when many
tests are examined together as this section does). Sort the $m$ p-values ascending as $p_{(1)}
\le \ldots \le p_{(m)}$; find the largest $k$ such that
$$
p_{(k)} \le \frac{k}{m}\alpha
$$
then declare tests $1,\ldots,k$ significant. The corrected "q-value" reported for each test is
the smallest FDR threshold at which that specific test would be called significant.

The pipeline applies this correction **twice**: once per individual sub-analysis (local
correction, e.g. within just the point-biserial correlation table), and once **globally** across
all 140 tests run anywhere in this section (collected into one pooled list and corrected
jointly) — the global correction is the stricter, more defensible standard for deciding which
specific findings to state as confirmed in the manuscript, since it accounts for the true total
number of comparisons made across the whole section, not just within one table at a time.

### 16.5 Variance Inflation Factor (VIF) — detecting multicollinearity
When fitting a regression with several predictors, if two predictors are highly correlated with
each other, the model can't reliably separate their individual effects — the fitted coefficients
become unstable and their apparent magnitude/significance untrustworthy. VIF quantifies this for
predictor $j$ by regressing it on *all the other predictors* and checking how well they predict
it:
$$
\text{VIF}_j = \frac{1}{1-R_j^2}
$$
where $R_j^2$ is the $R^2$ of that auxiliary regression. VIF = 1 means no collinearity; VIF > 5
(some use >10) is the conventional rule-of-thumb warning sign. **Correct computation requires an
intercept term in the design matrix** — this was a genuine bug in an earlier version of this
script (VIFs of several hundred to tens of thousands, implying severe, paper-breaking
collinearity); with the intercept correctly included, the ether/ester/nitrogen adjusted models
all show VIF < 7, meaning the reported positional/architectural effects are not artifacts of
molecular-size/lipophilicity confounding.

### 16.6 Ordinary Least Squares (adjusted regression)
To ask "does this positional/structural effect survive after accounting for molecular weight
and lipophilicity?", an OLS model is fit:
$$
EE = \beta_0 + \beta_1 \cdot (\text{distance-to-nitrogen}) + \beta_2 \cdot \text{MolWt} + \beta_3 \cdot \text{LogP} + \varepsilon
$$
solved by minimizing $\sum_i (EE_i - \hat{EE}_i)^2$, i.e. $\hat{\boldsymbol\beta} = (X^\top
X)^{-1}X^\top y$. $\beta_1$'s p-value tests whether the distance effect remains statistically
distinguishable from zero *after* the other two variables have already explained what they can
— this is what "adjusted for MolWt and LogP" means throughout the topology results.

### 16.7 Graph-distance topology (the positional analysis)
For a SMARTS match anchored at atom $a$, and every nitrogen atom $n$ in the same molecule, RDKit's
`GetShortestPath(mol, a, n)` performs a breadth-first search over the molecular graph (atoms as
nodes, bonds as edges) and returns the shortest path; its length minus 1 is the bond-count
distance. The **minimum** over all nitrogens gives that occurrence's distance-to-headgroup,
which is then binned (headgroup $\leq4$, linker 5–8, tail $>8$ bonds) or used continuously in
the Spearman/OLS analyses of §16.3/§16.6. This is what lets the pipeline distinguish "an ether
group exists somewhere in this molecule" from "an ether group sits 9 bonds from the charged
nitrogen" — a materially more specific and more actionable claim.

### 16.8 K-means clustering of ionizable-lipid chemotypes
Unique ionizable lipids are described by 8 standard descriptors (LogP, MolWt, ring count, amine
counts, ester count), standardized ($z_j = (x_j - \mu_j)/\sigma_j$ so no single large-scale
descriptor like MolWt dominates purely due to its numeric range), then clustered into $k=4$
groups by minimizing within-cluster sum of squares:
$$
\arg\min_{\{C_1,\ldots,C_k\}} \sum_{j=1}^{k} \sum_{\mathbf{x} \in C_j} \|\mathbf{x} - \boldsymbol\mu_j\|^2
$$
solved by Lloyd's algorithm (alternating: assign each point to its nearest current centroid,
then recompute each centroid as the mean of its assigned points, repeat to convergence). $k=4$
was chosen by inspecting the **silhouette score** across $k=2,\ldots,8$ — for a point $i$,
$$
\text{sil}(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))}
$$
where $a(i)$ is the mean distance from $i$ to other points in its own cluster and $b(i)$ is the
mean distance to points in the nearest *other* cluster; averaged over all points, higher is
better-separated clusters. The resulting 4 classes showed significantly different EE%
distributions (Kruskal-Wallis, a rank-based generalization of the Mann-Whitney test to more than
two groups), evidence the clusters correspond to chemically and functionally real groupings, not
arbitrary partitions.


## 17. Chemistry — UMAP, robust-lipid analysis, and ratio optimization

### 17.1 UMAP — visualizing 512-dimensional fingerprint space in 2D
UMAP (Uniform Manifold Approximation and Projection) is a nonlinear dimensionality-reduction
technique. The core idea, in plain terms: build a graph in the original high-dimensional space
where each point connects to its $k=15$ nearest neighbors, weighted by a similarity measure that
accounts for locally varying density; then find a low-dimensional (2D) layout whose *own*
neighbor graph is as similar as possible to the high-dimensional one, by minimizing a
cross-entropy-like objective between the high-D and low-D fuzzy graph representations. Unlike
PCA (a linear projection), UMAP can preserve local neighborhood structure even when the true
underlying structure is curved/nonlinear — appropriate here since "chemical similarity" doesn't
live on a flat plane. This is a visualization tool only — no downstream numbers depend on the
UMAP coordinates themselves, only on the underlying fingerprints.

### 17.2 Universally-robust-lipid criterion
For each ionizable lipid with $\geq 3$ independent formulations, two numbers: mean EE% across
all its formulations, and the fraction of those formulations that are High EE. A lipid is
"universal" if that fraction $\geq 0.75$ *and* its standard deviation across formulations
$\leq 15\%$ (i.e. reliably good, not just occasionally good) — a plain frequency/threshold rule,
not a statistical test, but a direct operationalization of "does this lipid work well no matter
what you pair it with."

### 17.3 Ratio optimizer
A brute-force grid search: for a fixed set of 4 lipid identities, sweep ionizable-lipid fraction
and PEG fraction over a grid, derive sterol/helper fractions by a fixed empirical split
($\text{sterol} = 0.78 \times \text{remainder}$, consistent with the dataset's own median
sterol-to-(sterol+helper) ratio of 0.744 observed in §1), call `predict_lnp()` on every grid
point, and rank by predicted probability. This is exhaustive search over a small, interpretable
parameter space — no optimization algorithm needed given how few free parameters remain once
lipid identity is fixed.

---

## 18. FeedbackLoop — active-learning acquisition function

**Goal:** given hundreds of untested hypothetical formulations, rank them by how valuable it
would be to actually go test them in the lab — not simply by "which does the model predict best,"
since testing only the model's favorites teaches it nothing it doesn't already believe.

### 18.1 The three component scores
For each candidate formulation:

**Uncertainty score** — continuous distance from the least-confident possible prediction
($\hat p = 0.5$), rescaled to $[0,1]$ with 1 = maximal uncertainty:
$$
\text{uncertainty}(x) = 1 - 2\,|\hat{p}(x) - 0.5|
$$

**Novelty score** — the same applicability-domain kNN Jaccard distance from §14.2, rescaled by
the maximum distance observed in the current candidate pool:
$$
\text{novelty}(x) = \frac{d_{kNN}(x)}{\max_{x' \in \text{pool}} d_{kNN}(x')}
$$

**Promise score** — simply the raw predicted probability, $\text{promise}(x) = \hat{p}(x)$.

### 18.2 Combining into one acquisition score
A weighted sum, with an applicability-domain penalty multiplied in afterward:
$$
\text{score}(x) = \Big[0.5 \cdot \text{uncertainty}(x) + 0.3 \cdot \text{novelty}(x) + 0.2 \cdot \text{promise}(x)\Big] \times \text{ad\_penalty}(x)
$$
where $\text{ad\_penalty}=1.0$ if inside the applicability domain, $0.5$ if outside (down-
weighted, not excluded outright, unless the candidate is so far outside the domain
— beyond $2\times$ the AD threshold — that it's dropped from consideration entirely; the model's
uncertainty estimate itself isn't trustworthy that far outside its training distribution, so
"uncertain" stops meaning the same thing out there). The 0.5/0.3/0.2 weighting is a design
choice prioritizing genuine model uncertainty (the classic active-learning "query the examples
you're least sure about" heuristic) over pure novelty or pure optimism.

### 18.3 Greedy diversity selection (farthest-first)
Simply taking the top-10 by acquisition score risks picking 10 near-identical ratio-variants of
one lipid (exactly the failure mode discussed for the winners/losers/uncertain lists in an
earlier conversation). Instead, a **greedy farthest-first** selection is used:

1. Take the single highest-scoring candidate first.
2. For every remaining candidate, compute its **diversity** — the *minimum* fingerprint Jaccard
   distance (§14.2's formula) to anything already selected:
   $$
   \text{diversity}(x) = \min_{x' \in \text{Selected}} J\big(\text{fp}(x), \text{fp}(x')\big)
   $$
3. Pick the next candidate maximizing a 50/50 blend of its own acquisition score and its
   diversity from what's already chosen:
   $$
   \text{combined}(x) = 0.5\cdot\text{diversity}(x) + 0.5\cdot \frac{\text{score}(x)}{\max_{x'} \text{score}(x')}
   $$
4. Repeat until 10 are selected, preferentially picking from lipids not yet represented in the
   shortlist (falling back to allowing repeats only once every distinct lipid has been tried).

This greedy procedure is a heuristic approximation to the (computationally much harder) problem
of choosing the single *best* diverse subset of size 10 — it doesn't guarantee the globally
optimal shortlist, but is a standard, cheap, and effective approach used broadly in active
learning and experimental design.

---

## Summary table — every statistical tool used, and what question it answers

| Tool | Section | Question it answers |
|---|---|---|
| Standard deviation, ICC (mixed-effects) | 3, 4 | How much EE% variance is noise vs. real, and attributable to what? |
| Morgan fingerprints, RDKit descriptors | 5 | How do you turn a molecule into numbers? |
| ROC-AUC, Average Precision, Wilcoxon signed-rank | 6, 7 | Which feature representation predicts best, and is the difference real? |
| GroupKFold, leakage t-test | 6, 11.4 | Is the model actually learning transferable chemistry, not memorizing? |
| Random Forest, balanced accuracy, bootstrap CI | 9, 9.3 | The core classifier, and an honestly-derived decision threshold |
| Derringer-Suich desirability, geometric mean | 10 | Combining EE probability with size/PDI into one ranking score |
| ExtraTrees/XGBoost comparison, McNemar's test | 11.1–11.3 | Is Random Forest actually the best algorithm here? |
| Brier score, ECE, isotonic/Platt calibration | 11.6 | Are the predicted probabilities themselves trustworthy? |
| Conformal prediction (nonconformity, quantile) | 13 | Which individual predictions should you actually trust? |
| Jaccard distance, applicability domain | 14.2 | Is this formulation too far outside what the model has seen? |
| SHAP (Shapley values) | 16.1 | Which features drove this specific prediction? |
| SMARTS matching | 16.2 | Does this molecule contain a specific substructure? |
| Point-biserial $r$, Spearman $\rho$, Mann-Whitney U, $\chi^2$/$\phi$ | 16.3 | Is a chemical feature statistically associated with EE%? |
| Benjamini-Hochberg FDR, Bonferroni | 16.4 | How many "significant" findings survive correcting for the number of tests run? |
| VIF | 16.5 | Are the regression coefficients trustworthy, or confounded by collinear predictors? |
| OLS regression | 16.6 | Does an effect survive adjusting for other variables? |
| Graph shortest-path distance | 16.7 | Where in the molecule does a functional group sit, relative to the charged headgroup? |
| K-means, silhouette score | 16.8, 17 | Do ionizable lipids fall into distinct, EE%-relevant chemical classes? |
| UMAP | 17.1 | 2D visualization of chemical similarity (not used for any statistical claim) |
| Active-learning acquisition score, greedy diversity selection | 18 | Which untested formulations are most worth synthesizing next? |