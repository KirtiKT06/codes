# The Mathematics Behind the LNP EE% Pipeline
### A study chapter — formulas, worked numbers, assumptions, limitations

Each entry follows the same structure: **What it is → The math → A tiny worked example → Assumptions → Limitations / what a reviewer might attack.** Work through the numbers by hand at least once — that's what separates "I used scipy.stats.X" from "I understand X."

---

# CHAPTER 1 — Hypothesis Testing Toolkit

## 1.1 One-way ANOVA (F-test)

**What it is.** Tests whether the means of ≥2 groups are equal, by comparing between-group variance to within-group variance.

**The math.**
For $k$ groups, group $i$ has $n_i$ observations, mean $\bar{x}_i$, grand mean $\bar{x}$.

$$SS_{between} = \sum_{i=1}^{k} n_i(\bar{x}_i - \bar{x})^2 \qquad SS_{within} = \sum_{i=1}^{k}\sum_{j=1}^{n_i}(x_{ij}-\bar{x}_i)^2$$

$$F = \frac{SS_{between}/(k-1)}{SS_{within}/(N-k)} = \frac{MS_{between}}{MS_{within}}$$

$F$ follows an F-distribution with $(k-1, N-k)$ degrees of freedom under $H_0$: all means equal.

**Worked example.** Three ionizable lipids, EE% observations:
- Lipid A: 70, 74 (mean 72)
- Lipid B: 82, 86 (mean 84)
- Lipid C: 60, 64 (mean 62)

Grand mean $\bar{x} = 72$. $k=3$, $N=6$.

$SS_{between} = 2(72-72)^2 + 2(84-72)^2 + 2(62-72)^2 = 0 + 288 + 200 = 488$

$SS_{within}$: each pair has spread 4 around its mean → deviations $\pm2$, squared $=4$, per pair sum $=8$; three pairs → $SS_{within}=24$.

$MS_{between} = 488/(3-1) = 244$, $MS_{within} = 24/(6-3) = 8$

$F = 244/8 = 30.5$ → compare to F-table with (2,3) df. This is large → reject $H_0$, lipid identity matters.

**Assumptions.** (1) Independence of observations within/between groups, (2) normality of residuals within each group, (3) homogeneity of variance (homoscedasticity) across groups.

**Limitations / attack surface.** Sensitive to violations of normality and equal variance, especially with unequal $n_i$ (here group sizes vary a lot — some lipids have 2 obs, some have 20+). Doesn't tell you *which* groups differ (needs post-hoc tests, e.g. Tukey HSD — not used here). In your pipeline, this variance-decomposition use is really a random-effects framing, not a classical fixed-effects ANOVA — be ready to explain that distinction if pushed.

---

## 1.2 Variance Decomposition (σ² partitioning) — Rigorous Noise Floor

**What it is.** The formal analogue of ANOVA's SS decomposition, expressed as variances (method of moments), used to answer: "how much of total EE% variance is learnable (chemistry) vs. irreducible (cross-lab noise)?"

**The math.**
$$\sigma^2_{total} = \text{Var}(\text{all EE values})$$
$$\sigma^2_{within} = \text{mean over lipids of } \text{Var}(\text{EE} \mid \text{lipid})$$
$$\sigma^2_{between} = \sigma^2_{total} - \sigma^2_{within} \quad (\text{clipped at } 0)$$

This is the **intraclass correlation** framing: $ICC = \sigma^2_{between}/\sigma^2_{total}$ is the theoretical max $R^2$ a perfect regression model could achieve, because the rest is measurement noise that no amount of chemistry-aware features can predict.

**Worked example.** Suppose $\sigma^2_{total}=400$ (σ=20%), and across lipids with ≥2 obs the average within-lipid variance is $\sigma^2_{within}=100$ (σ=10%). Then $\sigma^2_{between}=300$, i.e. 75% of variance is "chemistry", 25% is "lab noise" → max theoretical $R^2 \approx 0.75$.

**Assumptions.** Within-lipid variance is estimated only from lipids with $n\ge2$ — assumes those lipids are representative of the full noise process for singleton lipids too. Also assumes noise is roughly homoscedastic across lipids (in reality, some lipids may be measured far more reproducibly than others).

**Limitations.** Small-$n$ groups give very unstable individual variance estimates (variance of a variance estimator shrinks slowly, $\propto 1/(n-1)$) — a lipid with only 2 replicates contributes a very noisy $\sigma^2$ to the average. This is exactly the kind of thing a stats-savvy reviewer will ask about: "how many of your multi-observation lipids actually have more than 3 replicates?"

---

## 1.3 Independent-samples t-test

**What it is.** Tests whether two independent samples have the same population mean.

**The math (Welch's, unequal variance form — safer default):**
$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$$

Degrees of freedom via Welch–Satterthwaite equation (not equal to $n_1+n_2-2$ unless variances/sizes are equal).

**Worked example.** Random-split AUCs (20 repeats): mean 0.82, sd 0.03. GroupKFold AUCs: mean 0.71, sd 0.04.

$$t = \frac{0.82-0.71}{\sqrt{0.03^2/20+0.04^2/20}} = \frac{0.11}{\sqrt{0.000045+0.00008}} = \frac{0.11}{0.0112} \approx 9.8$$

With df≈35, $p \ll 0.001$ → the AUC inflation from random splitting is statistically indisputable.

**Assumptions.** Independence of the two samples, approximately normal sampling distribution of the mean (CLT helps with $n=20$ repeats), and — for Welch's version — variances need *not* be equal (that's the whole point of Welch over Student's t).

**Limitations.** The 20 "repeats" here are correlated with each other in a subtle way (same underlying dataset, same folds structure across repeats within a method) — treating them as fully independent samples is an approximation. Also a t-test on only the *means* discards information about the shape/skew of the AUC distributions.

---

## 1.4 Wilcoxon Signed-Rank Test

**What it is.** Non-parametric paired test — tests whether the median of *paired differences* is zero. Used because your comparisons are paired-by-fold (same 5 folds, two models) and $n=5$ is far too small to trust a t-test's normality assumption.

**The math.**
1. Compute differences $d_i = x_i - y_i$ for each pair, drop zeros.
2. Rank $|d_i|$ from smallest to largest (average ranks for ties).
3. Sum ranks of positive differences ($W^+$) and negative differences ($W^-$).
4. Test statistic $W = \min(W^+, W^-)$; compare to the Wilcoxon distribution (exact for small $n$) or a normal approximation for larger $n$.

**Worked example.** RF vs ExtraTrees AUC per fold:

| Fold | RF | ET | d = RF−ET | \|d\| | rank |
|---|---|---|---|---|---|
| 1 | 0.75 | 0.70 | +0.05 | 0.05 | 3 |
| 2 | 0.80 | 0.78 | +0.02 | 0.02 | 1 |
| 3 | 0.68 | 0.71 | −0.03 | 0.03 | 2 |
| 4 | 0.90 | 0.83 | +0.07 | 0.07 | 4 |
| 5 | 0.77 | 0.68 | +0.09 | 0.09 | 5 |

$W^+ = 3+1+4+5 = 13$, $W^- = 2$. $W = 2$. For $n=5$, the critical value at $\alpha=0.05$ (two-sided) is 0 — so $W=2$ is *not* significant at this small sample size even though 4/5 folds favored RF. This is exactly why your printouts say "n=5 folds — treat as indicative, not high-powered."

**Assumptions.** Paired samples, differences are symmetric around the median (doesn't need normality), and — critically — treats the paired differences as exchangeable (i.i.d. under $H_0$).

**Limitations.** With $n=5$, statistical power is very low — you'll often fail to detect real differences. Ties reduce power further. Reviewers may ask why you didn't just run more CV repeats to get more paired samples (answer: you're constrained by only having 5 lipid-disjoint folds without re-randomizing the grouping — though you *could* have repeated GroupKFold with different random group orderings for more paired samples, similar to what you did for the leakage-proof).

---

## 1.5 Mann-Whitney U Test

**What it is.** Non-parametric test for two **independent** (unpaired) groups — used for the functional-group "presence vs absence" EE% design rules.

**The math.** Rank all $n_1+n_2$ observations together. $U_1 = R_1 - \frac{n_1(n_1+1)}{2}$ where $R_1$ = sum of ranks in group 1. Under $H_0$, $U$ is approximately normal for larger $n$ with known mean/variance.

**Worked example.** Ether-linkage present: EE = [85, 90], absent: EE = [60, 65, 70].
All 5 ranked: 60(1), 65(2), 70(3), 85(4), 90(5). $R_1$ (present group) $= 4+5=9$.
$U_1 = 9 - \frac{2\times3}{2} = 9-3=6$. Max possible $U_1 = n_1 n_2 = 6$ → this is the maximum separation possible (every "present" value beats every "absent" value) — strong effect, though $n$ is tiny so p-value will still be large.

**Assumptions.** Independent groups, but does NOT require normality — only requires the two distributions have the same *shape* if you want to interpret it strictly as a median-difference test; more generally it tests **stochastic dominance** (P(X>Y) ≠ 0.5).

**Limitations.** Loses information by converting to ranks (can't quantify effect size in original units directly — you also report the raw mean-difference δEE alongside it for that reason, which is good practice). Sensitive to ties when there are many repeated EE values.

---

## 1.6 Kruskal-Wallis Test

**What it is.** Extension of Mann-Whitney to >2 independent groups — the non-parametric analogue of one-way ANOVA. Used for cargo-type (3 groups) and lipid-class (4 groups) comparisons.

**The math.**
$$H = \frac{12}{N(N+1)}\sum_{i=1}^{k}\frac{R_i^2}{n_i} - 3(N+1)$$
where $R_i$ = sum of ranks in group $i$. Under $H_0$, $H \sim \chi^2_{k-1}$ approximately (for reasonably large $n_i$).

**Worked example.** With cargo groups mRNA (n=3), siRNA (n=3), DNA (n=2), $N=8$: you'd rank all 8 EE values together, sum ranks per group, plug into the formula. Your actual printed result was H with p=0.096 — not significant at α=0.05, meaning: no strong evidence cargo type itself shifts EE% (a useful negative result — supports "lipid chemistry dominates" narrative).

**Assumptions.** Independent groups, similarly-shaped distributions (for a clean median interpretation), and the chi-square approximation requires group sizes not too small (rule of thumb: ≥5 per group — DNA's n=2 group here is a real vulnerability if this is pushed on).

**Limitations.** Doesn't tell you *which* groups differ (would need Dunn's post-hoc test). Chi-square approximation is poor for very small/unbalanced groups — flag this explicitly if asked about the DNA cargo group.

---

## 1.7 Chi-Square Test of Independence

**What it is.** Tests whether two categorical variables are associated, using a contingency table. Used for SMARTS motif presence vs High/Low EE class.

**The math.** For a 2×2 table with observed counts $O_{ij}$, expected counts under independence:
$$E_{ij} = \frac{(\text{row}_i \text{ total})(\text{col}_j \text{ total})}{N}$$
$$\chi^2 = \sum_{i,j}\frac{(O_{ij}-E_{ij})^2}{E_{ij}}, \quad df=(r-1)(c-1)=1 \text{ for } 2\times2$$

**Worked example.** Motif present in 40 formulations (30 High-EE, 10 Low-EE); motif absent in 60 formulations (20 High-EE, 40 Low-EE). $N=100$, row totals 40/60, col totals 50 High/50 Low.

$E_{present,High} = 40\times50/100=20$, $E_{present,Low}=20$, $E_{absent,High}=30$, $E_{absent,Low}=30$.

$\chi^2 = \frac{(30-20)^2}{20}+\frac{(10-20)^2}{20}+\frac{(20-30)^2}{30}+\frac{(40-30)^2}{30} = 5+5+3.33+3.33=16.67$

$df=1$ → highly significant (critical value at α=0.05 is 3.84).

**Assumptions.** Observations independent, expected counts should generally be ≥5 in each cell (otherwise use Fisher's exact test instead) — your code filters out motifs with fewer than 5 present/absent, which is exactly guarding this assumption.

**Limitations.** Chi-square is an *approximation*; with small expected counts it's biased upward (inflates significance) — Fisher's exact test is the gold standard alternative for small samples. Also doesn't give you effect size directly — you supplement with ΔEE, which is the right instinct.

---

## 1.8 McNemar's Test

**What it is.** For comparing two **paired classifiers'** error rates on the *same* test instances (not two independent samples) using only the disagreement cases.

**The math.** Build a 2×2 table of correct/incorrect for classifier A vs B on the same instances:

| | B correct | B wrong |
|---|---|---|
| **A correct** | a | b |
| **A wrong** | c | d |

Only the discordant pairs $b,c$ matter. With continuity correction:
$$\chi^2 = \frac{(|b-c|-1)^2}{b+c}, \quad df=1$$

**Worked example.** Of 100 test cases: RF-correct/ET-wrong = 12, RF-wrong/ET-correct = 4.
$$\chi^2 = \frac{(|12-4|-1)^2}{12+4} = \frac{49}{16} = 3.06$$
$p\approx0.08$ — not significant at α=0.05, even though RF "won" 12 vs 4.

**Assumptions.** Paired binary outcomes on the *same* instances, discordant pair counts should not be too small (rule of thumb $b+c\ge25$, otherwise use the exact binomial version instead of the χ² approximation).

**Limitations.** Only uses the disagreement cells — throws away all the cases both models get right/wrong, so it can be underpowered when models are very similar. Tests thresholded (0/1) predictions only, ignoring probability calibration differences — your code explicitly notes this is a "robust to pooling issue" complement to the AUC-based tests, not a replacement.

---

## 1.9 Point-Biserial Correlation

**What it is.** Correlation between one continuous variable and one true binary variable — mathematically identical to Pearson's r when one variable is coded 0/1.

**The math.**
$$r_{pb} = \frac{\bar{Y}_1 - \bar{Y}_0}{s_Y}\sqrt{p(1-p)}$$
where $\bar{Y}_1,\bar{Y}_0$ are means of the continuous variable in the two binary groups, $s_Y$ is the pooled SD, $p$ is the proportion in group 1.

**Worked example.** Amide count in High-EE formulations: mean 1.2; in Low-EE: mean 0.8. Pooled SD = 0.9. $p$ (fraction High-EE) = 0.4.
$$r_{pb} = \frac{1.2-0.8}{0.9}\sqrt{0.4\times0.6} = 0.444\times0.49 \approx 0.22$$
A modest positive association.

**Assumptions.** The continuous variable should be roughly normally distributed within each group (a Pearson-family assumption); the binary variable is a true dichotomy, not an artificially binarized continuous variable (be careful — EE% itself was artificially binarized at 80%, so correlating a descriptor with that binary label vs. with raw EE is a meaningfully different question — you actually check both, good).

**Limitations.** Same limitation as Pearson r generally: only captures **linear** association; sensitive to outliers.

---

## 1.10 Spearman Rank Correlation (ρ)

**What it is.** Pearson correlation computed on the **ranks** instead of raw values — captures monotonic (not necessarily linear) relationships, robust to outliers and skew. Used everywhere in this pipeline (FQS validation, descriptor-EE correlations, pKa calibration) because EE% is skewed and relationships are often non-linear.

**The math.**
$$\rho = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$
where $d_i$ = difference in ranks of paired observations $i$ (this simplified formula assumes no ties; with ties, it reduces to Pearson's r computed on rank values).

**Worked example.** LogP vs EE% for 5 lipids:

| Lipid | LogP | rank(LogP) | EE% | rank(EE%) | d | d² |
|---|---|---|---|---|---|---|
| 1 | 5 | 1 | 60 | 1 | 0 | 0 |
| 2 | 8 | 2 | 65 | 2 | 0 | 0 |
| 3 | 12 | 4 | 80 | 3 | 1 | 1 |
| 4 | 10 | 3 | 85 | 4 | -1 | 1 |
| 5 | 15 | 5 | 90 | 5 | 0 | 0 |

$\sum d^2 = 2$, $n=5$: $\rho = 1 - \frac{6\times2}{5\times24} = 1-0.1 = 0.9$ — strong positive monotonic relationship.

**Assumptions.** Only requires a monotonic relationship, no linearity or normality assumption — much weaker assumptions than Pearson.

**Limitations.** Discards magnitude information (only order matters) — two very different-looking scatterplots can have identical ρ. Ties need correction terms (scipy handles this automatically, but know that the simplified formula above is technically only exact without ties). Doesn't capture non-monotonic (e.g. U-shaped) relationships at all — worth checking your pKa-vs-EE% relationship isn't secretly U-shaped, since a linear/monotonic-only tool would completely miss that.

---

## 1.11 Multiple Comparison Correction

**Why you need it at all.** If you run 20 independent tests at α=0.05, you expect ~1 false positive by chance alone even if nothing is real. Running many correlation/enrichment tests (functional groups, SMARTS motifs) without correction inflates the false-discovery rate.

### Bonferroni correction (used for SMARTS enrichment — small, fixed library of ~10 motifs)
$$\alpha_{adjusted} = \alpha / m \quad \text{(equivalently: } p_{adjusted}=\min(p\times m, 1)\text{)}$$
where $m$ = number of tests. **Controls the family-wise error rate (FWER)** — probability of *even one* false positive across all tests.

*Worked example*: 10 tests, raw p=0.008 for one. Bonferroni-adjusted: $p\times10 = 0.08$ → no longer significant at 0.05. Very conservative.

### Benjamini-Hochberg FDR (used for the larger functional-group correlation sweep)
Sort p-values ascending $p_{(1)}\le p_{(2)}\le\dots\le p_{(m)}$. Find the largest $k$ such that:
$$p_{(k)} \le \frac{k}{m}\alpha$$
Reject all hypotheses $1,\dots,k$. **Controls the expected proportion of false discoveries among rejected hypotheses**, not the probability of any false positive at all — a fundamentally more lenient criterion.

*Worked example*: 5 p-values sorted: 0.001, 0.01, 0.03, 0.04, 0.20. At α=0.05: thresholds are $\frac{1}{5}(.05)=.01$, $\frac{2}{5}(.05)=.02$, $\frac{3}{5}(.05)=.03$, $\frac{4}{5}(.05)=.04$, $\frac{5}{5}(.05)=.05$. Compare each: 0.001≤.01 ✓, 0.01≤.02 ✓, 0.03≤.03 ✓, 0.04≤.04 ✓, 0.20≤.05 ✗. Largest passing $k=4$ → reject the first 4 hypotheses (all but the last).

**Be ready to justify your choice, per-section.** SMARTS library = small, curated, fixed set → Bonferroni's conservatism is affordable and desirable (few tests, want strong guarantee). Functional-group/descriptor correlation sweep = many candidate features from an exploratory scan → FDR is standard practice (accepting a controlled fraction of false positives to retain power).

**Shared limitation.** Both corrections assume tests are either independent or positively dependent (Benjamini-Hochberg's simple form is proven valid under independence or positive regression dependence — PRDS). Molecular descriptors are often correlated with each other (e.g., LogP and ring count), technically violating strict independence — in practice this is usually treated as close enough, but it's a fair thing for a reviewer to raise.

---

# CHAPTER 2 — Resampling, Cross-Validation & the Leakage Story

## 2.1 The core problem: exchangeability vs. grouped data

Standard k-fold CV assumes each row is **exchangeable** — i.e., you could shuffle rows freely without changing the statistical meaning of a split. That assumption **breaks** when multiple rows share a dependency structure (e.g., the same ionizable lipid appears in 5 different formulations) — because then a "random" split lets near-duplicate rows leak across train/test, and the model can partly "memorize" the lipid rather than learn generalizable chemistry.

## 2.2 GroupKFold — the math

Partitions the **set of unique groups** (here: unique ionizable lipids) into $k$ folds, then assigns *all* rows belonging to a group to the same fold. So if lipid X has 6 formulations, all 6 rows go to exactly one of the $k$ folds — never split across train/test.

**Worked mini-example.** 3 lipids: A (4 rows), B (2 rows), C (2 rows). With $k=2$: Fold 1 might get {A} (4 rows), Fold 2 gets {B,C} (4 rows) — group sizes are balanced as evenly as possible in terms of *row count*, subject to the hard constraint that a group never splits.

**Contrast with StratifiedKFold**: stratification only balances the class label ratio (High/Low EE) per fold — it says nothing about which *lipids* end up where, so the same lipid's near-identical fingerprint rows can trivially land in both train and test.

## 2.3 Quantifying the leakage — what the paired t-test/boxplot actually shows

You empirically compare AUC under (a) StratifiedKFold with shuffling, repeated 20× with different random seeds, vs (b) GroupKFold, same 20 repeats. The **paired scatter plot** (Random AUC vs GroupKFold AUC per repeat, with a y=x reference line) is the most persuasive visual: if points systematically fall **below** the line, random splitting is systematically over-optimistic. This is essentially a **paired comparison design** — same underlying dataset and model, only the splitting strategy varies — which is exactly why the paired Wilcoxon/t-test is the statistically correct follow-up (not an unpaired test).

## 2.4 LeaveOneGroupOut (LOGO) — true leave-one-publication-out

Each of $m$ unique groups (publications) takes a turn as the sole test fold; trains on the rest. With $m$ publications, you get $m$ folds (this is *not* the same as choosing $k=m$ in GroupKFold — LOGO always uses **exactly** the natural group count, and importantly is only run when both classes are present in the training portion, hence your code's `if len(set(y[tr])) < 2: continue`).

**Statistical property.** This gives the least biased estimate of generalization to a *completely unseen* publication/lab, at the cost of very high variance per fold (some folds may have only a handful of test points) — which is exactly why you additionally bootstrap the *pooled* OOF predictions for a CI rather than trusting single-fold numbers.

## 2.5 GroupShuffleSplit

Randomly samples a **subset** of groups for train/test (with a specified train/test size ratio), rather than exhaustively partitioning into folds. Used for: (a) the learning curve (need many different-sized random training subsets), and (b) matching a target training-set size (~122 samples) for the publication-size-matched control. Because it's *not* exhaustive, repeated calls give different splits each time — hence 50 seeds tried and only a subset kept (train size within ±20 of target).

## 2.6 The Bootstrap — nonparametric resampling for confidence intervals

**The math.** Given $n$ observed values, draw $n$ samples **with replacement** to form one bootstrap resample; compute your statistic (e.g., mean AUC) on that resample; repeat $B$ times (e.g., $B=1000$); the empirical distribution of the $B$ statistic values approximates the sampling distribution. A $95\%$ CI = the [2.5th, 97.5th] percentiles of that distribution (the **percentile method**, which is what your code uses).

**Worked mini-example.** Original 5 fold-AUCs: [0.70, 0.75, 0.72, 0.80, 0.68]. One bootstrap resample (sampling 5 with replacement) might be [0.75, 0.75, 0.68, 0.72, 0.80] → mean 0.74. Repeat this process 1000 times, collect 1000 means, take the 2.5th/97.5th percentile of that list as your CI bounds.

**Why bootstrap over *fold-level* AUCs specifically (not pooled OOF probabilities).** Your code explicitly flags this: pooling raw probabilities from ExtraTrees/XGBoost across folds is invalid here because each fold's probability *scale* can drift (randomized splits, per-fold class-weight recalibration) — concatenating and re-ranking the pooled probabilities collapses discriminative power even though each fold discriminates fine on its own. Bootstrapping the 5 (or however many) **already-computed fold-level AUC summary statistics** sidesteps this scale problem entirely, at the cost of a much smaller effective sample size for the bootstrap ($n=5$ is a genuinely small resampling base — CIs from 5 points, even bootstrapped 1000 times, are inherently wide and somewhat unstable; this is a real limitation worth naming yourself before a reviewer does).

**Assumptions.** The observed sample is representative of the population it's drawn from (bootstrap approximates the true sampling distribution by resampling the *empirical* distribution) — this breaks down badly when $n$ is very small, since you can only ever see combinations of the original 5 values, never anything genuinely outside their range.

**Limitations.** Percentile-method bootstrap CIs can be biased for skewed statistics (bias-corrected-and-accelerated, BCa, is a refinement not used here). With $n=5$ underlying values, the bootstrap distribution has at most $\binom{2\times5-1}{5}\approx126$ truly distinct resamples, not the "smooth" distribution the method implicitly assumes — good to know as an honest limitation.

## 2.7 Monte Carlo Noise-Injection Simulation

**What it is.** Not a hypothesis test — a *simulation experiment*. Repeatedly perturb the true EE% labels with $\mathcal{N}(0,\sigma^2)$ noise, re-threshold at 80% for classification / re-fit regression, and track how AUC (classification) vs $R^2$ (regression) degrade as $\sigma$ increases, averaged over 20 random noise draws per level to smooth out single-draw variance.

**Why this design is convincing.** It's a controlled experiment where you know the *ground-truth* noise level you're injecting, so you can directly read off "how much added label noise does it take before performance craters" — and crucially, compare that empirical breaking point to the **independently measured** between-lab noise floor (from Chapter 1.2) as an external validity check: if real-world lab noise is around σ=15% and your simulation shows classification AUC stays reasonable up to σ≈15-20% while regression $R^2$ has already collapsed well before that point, that's a self-consistent, mutually reinforcing argument.

**Limitations.** Assumes injected Gaussian noise is a good model for real measurement noise (real inter-lab noise could be non-Gaussian, heteroscedastic by EE level, or have systematic lab-specific biases rather than pure random scatter — worth stating this as a modeling assumption, not a proven fact).

---

# CHAPTER 3 — Classification Metrics, Formally

## 3.1 ROC curve and AUC

**The math.** At each threshold $t$, classify positive if $\hat{p}\ge t$. Plot:
$$TPR(t) = \frac{TP(t)}{TP(t)+FN(t)} \quad \text{vs} \quad FPR(t)=\frac{FP(t)}{FP(t)+TN(t)}$$
as $t$ sweeps from 1 to 0. **AUC has an exact probabilistic interpretation:**
$$AUC = P(\hat{p}_{+} > \hat{p}_{-})$$
i.e., the probability that a randomly chosen true-positive example gets a higher predicted score than a randomly chosen true-negative example. This is mathematically identical to the **Mann-Whitney U statistic**, normalized:
$$AUC = \frac{U}{n_+ n_-}$$

**Worked example.** 2 positives with scores [0.9, 0.6], 2 negatives with scores [0.4, 0.7]. All 4 pairwise comparisons (pos vs neg):
(0.9 vs 0.4)✓, (0.9 vs 0.7)✓, (0.6 vs 0.4)✓, (0.6 vs 0.7)✗ → 3/4 correct orderings → AUC = 0.75.

**Assumptions.** None distributional — AUC is a rank-based, threshold-free, non-parametric statistic; that's exactly why it's the metric of choice here given how noisy/non-normal EE% is.

**Limitations.** AUC treats all misclassification costs as symmetric and all thresholds as equally important — in a real deployment you likely care much more about the region near your chosen operating threshold (0.5, or your balanced-accuracy-optimal cutoff) than the whole curve. AUC can also look deceptively good under class imbalance even when precision is poor — which is exactly why you *also* report Average Precision.

## 3.2 Precision-Recall Curve & Average Precision (AP)

**The math.** Precision $=\frac{TP}{TP+FP}$, Recall $=\frac{TP}{TP+FN}$ (=Sensitivity/TPR). AP summarizes the PR curve as a weighted mean of precision at each threshold, weighted by the *increase* in recall:
$$AP = \sum_n (R_n - R_{n-1})P_n$$

**Why it matters more than AUC under imbalance.** The PR curve's baseline (random classifier) is the positive class prevalence $\pi$ — **not** a flat 0.5 like ROC's. If only 30% of formulations are High-EE, a random classifier gets AP≈0.30 but ROC-AUC≈0.50 regardless of imbalance — meaning ROC can look "good" (e.g., 0.75) while PR reveals the model is barely better than the (already imbalance-adjusted) baseline. Your code explicitly reports the random-baseline AP line for exactly this reason.

**Limitations.** AP is sensitive to class imbalance in the *opposite* direction from AUC — very rare positive classes make even a strong model's AP look numerically unimpressive, which can be misleading if a reader compares your AP directly to an AUC-style intuition of "close to 1 = great."

## 3.3 Confusion-matrix-derived metrics

$$\text{Sensitivity (Recall)}=\frac{TP}{TP+FN} \qquad \text{Specificity}=\frac{TN}{TN+FP}$$
$$\text{PPV (Precision)}=\frac{TP}{TP+FP} \qquad \text{NPV}=\frac{TN}{TN+FN}$$
$$\text{Balanced Accuracy} = \frac{\text{Sensitivity}+\text{Specificity}}{2}$$

**Worked example.** TP=30, FN=10, TN=45, FP=15.
Sensitivity $=30/40=0.75$; Specificity $=45/60=0.75$; PPV $=30/45=0.667$; NPV$=45/55=0.818$; Balanced Acc $=0.75$.

**Why balanced accuracy over raw accuracy.** Raw accuracy $=(TP+TN)/N = 75/100=0.75$ here happens to match, but with class imbalance raw accuracy can be dominated by the majority class (e.g., 90% accuracy from always predicting "Low EE" if 90% of the data is Low EE) — balanced accuracy explicitly averages the *per-class* recall, immune to this.

**Limitations of the threshold-optimization step.** Sweeping thresholds to *maximize balanced accuracy on the same OOF set you're reporting performance on* is a mild form of overfitting the operating point (not the model itself) to your specific dataset — your code's own comment flags this ("treat as an operating point... not guaranteed-optimal on future data"). Good instinct; be ready to explain it if asked.

---

# CHAPTER 4 — Probability Calibration

## 4.1 Why raw classifier probabilities aren't automatically "true" probabilities

A Random Forest's `predict_proba` is the fraction of trees voting positive — a reasonable score, but not guaranteed to match the empirical frequency of the positive class among all instances receiving that score. E.g., among all formulations the model scores at 0.7, maybe only 55% are actually High-EE — the model is **overconfident**, and calibration fixes this mapping.

## 4.2 Platt Scaling (Sigmoid Calibration)

**The math.** Fits a 1D logistic regression from raw score $f(x)$ to calibrated probability:
$$P(y=1\mid f(x)) = \frac{1}{1+\exp(A\cdot f(x)+B)}$$
$A,B$ fit by maximum likelihood on a held-out calibration set.

**Assumptions.** Assumes the *miscalibration itself* has a sigmoidal shape — works well when the raw scores are monotonically related to true probability but systematically over/under-confident in a smooth, S-shaped way.

**Limitations.** Too rigid a parametric form if the true miscalibration pattern is more complex/non-monotonic-ish; needs relatively little data to fit (only 2 parameters) — an advantage when calibration data is scarce, as here.

## 4.3 Isotonic Regression

**The math.** A non-parametric, monotonic step-function fit that minimizes squared error subject to the constraint that the fitted function is non-decreasing:
$$\min \sum_i (y_i - g(x_i))^2 \quad \text{s.t. } g \text{ is non-decreasing}$$
Solved via the **Pool Adjacent Violators Algorithm (PAVA)**.

**Assumptions.** Only assumes monotonicity (higher raw score → higher or equal true probability) — no parametric shape assumption at all.

**Limitations.** More flexible than Platt but needs *more* calibration data to avoid overfitting (with few points, isotonic regression can produce a jagged step function that memorizes calibration-set noise) — exactly why your code uses **nested CV** for isotonic calibration specifically.

## 4.4 Brier Score

**The math.** Mean squared error between predicted probability and the binary outcome:
$$BS = \frac{1}{n}\sum_{i=1}^n(\hat{p}_i - y_i)^2$$
$BS=0$ is perfect; $BS=0.25$ is the score of a classifier that always predicts $\hat{p}=0.5$ on balanced data (a useful reference point).

**Property worth knowing.** Brier score is a **strictly proper scoring rule** — meaning a forecaster minimizes their expected Brier score *only* by reporting their true believed probability, never by hedging. This is a deep and important property to be able to explain if asked "why Brier score specifically."

**Decomposition (Murphy 1973)**: $BS = \text{Reliability} - \text{Resolution} + \text{Uncertainty}$ — reliability measures calibration error, resolution measures how much predictions vary from the base rate (sharpness), uncertainty is the irreducible variance of the outcome itself. Knowing this decomposition exists (even without deriving it live) signals real depth.

## 4.5 Expected Calibration Error (ECE)

**The math.** Bin predictions into $M$ bins by predicted probability; for each bin $B_m$:
$$ECE = \sum_{m=1}^{M}\frac{|B_m|}{n}\left|\text{acc}(B_m) - \text{conf}(B_m)\right|$$
where $\text{acc}(B_m)$ = actual fraction positive in that bin, $\text{conf}(B_m)$ = average predicted probability in that bin.

**Limitations.** Highly sensitive to the choice of $M$ (number of bins) and binning strategy (equal-width vs equal-count/quantile — your code uses quantile bins for the reliability diagram) — this is a well-known critique of ECE in the calibration literature and a great thing to preempt if your audience is ML-sophisticated.

## 4.6 Reliability Diagram

Plot of mean predicted probability (x-axis) vs observed frequency of positives (y-axis) per bin; the diagonal is perfect calibration. Points below the diagonal = overconfident in that probability range; above = underconfident.

## 4.7 Why nested CV for calibration specifically

If you calibrate on the same fold used to *evaluate* the calibrated probabilities, you leak information (the calibrator has effectively "seen" the answers). Your code's `oof_calibrated_probs` fits `CalibratedClassifierCV(..., cv=3)` **inside** each outer GroupKFold training fold, so the final evaluation fold never touches calibration fitting — a second layer of leakage-avoidance on top of the outer group-based split.

---

# CHAPTER 5 — Conformal Prediction

## 5.1 The core idea

Instead of "the model is 73% confident," conformal prediction gives you a **prediction set** — e.g. {High EE} or {High EE, Low EE} — with a mathematically guaranteed coverage probability, using only a held-out calibration set and no distributional assumptions about the model itself.

## 5.2 The math, step by step

**Nonconformity score** for a labeled example: how "strange" is this true label under the model?
$$s(x,y) = 1 - \hat{p}(y\mid x)$$
i.e., $1$ minus the model's predicted probability of the *true* class. High score = model was surprised by this label.

**Calibration.** Compute $s_i$ for all $n$ calibration examples. Find the $\lceil (n+1)(1-\alpha)\rceil / n$ empirical quantile of $\{s_i\}$, call it $\hat{q}$.

**Why the $(n+1)$ correction, not just $n$.** This specific finite-sample correction is what makes the coverage guarantee **exact** (not just asymptotic) under the exchangeability assumption — it accounts for the fact that the test point itself is "one more" exchangeable draw alongside the $n$ calibration points. Omitting the $+1$ would give you a guarantee that only holds in the limit $n\to\infty$.

**Prediction set for a new point** $x$: include label $y$ in the set if $s(x,y)\le\hat{q}$, i.e.:
$$\mathcal{C}(x) = \{y : 1-\hat{p}(y\mid x) \le \hat{q}\}$$

**Worked example.** Suppose $\hat{q}=0.35$ (calibrated for 90% target coverage, α=0.10). New point has $\hat{p}(\text{High})=0.8$.
- Nonconformity of "High" $=1-0.8=0.2 \le 0.35$ → include High.
- Nonconformity of "Low" $=1-(1-0.8)=1-0.2=0.8 \not\le0.35$ → exclude Low.
- Result: singleton set {High} — confident prediction.

If instead $\hat{p}(\text{High})=0.55$: nonconformity of High $=0.45>0.35$ (excluded!), nonconformity of Low $=0.55>0.35$ (also excluded) — can happen at boundary cases, and by construction of $\hat q$ this should be rare; typically near 50/50 you instead get **both** labels included (ambiguous set), which is the more common outcome to expect near the decision boundary.

## 5.3 The coverage guarantee — precisely stated

$$P\big(y_{test} \in \mathcal{C}(x_{test})\big) \ge 1-\alpha$$

This holds **marginally** (averaged over the randomness of calibration set draw and test point), **not conditionally** (it does NOT guarantee 90% coverage *within* every sub-population, e.g. within just the High-EE class, or just novel chemotypes) — a crucial nuance.

## 5.4 Assumptions — the one to really know

**Exchangeability**: calibration and test data must be draws from the same underlying distribution in a way where their joint distribution is invariant to reordering. This is *weaker* than i.i.d. but still a real assumption. In this pipeline, calibration/test come from the same random split of OOF predictions — reasonably exchangeable *within* the training chemistry distribution.

**Where this genuinely breaks (good thing to raise proactively):** when you later apply `predict_lnp_conformal` to a **brand-new, never-before-seen ionizable lipid** (exactly the active-learning use case in Chapter 12), that new point is *not* exchangeable with the calibration set if it's chemically novel (outside the applicability domain) — the coverage guarantee is **not** rigorously valid there. This is precisely why the pipeline additionally gates conformal-based candidate selection with the separate Applicability Domain check — conformal prediction alone doesn't protect you against distribution shift.

## 5.5 Limitations, summarized

- Marginal, not conditional/class-wise, coverage.
- Coverage guarantee requires exchangeability, which degrades under covariate shift (novel chemistry) — exactly the applicability-domain caveat above.
- Set size (informativeness) depends entirely on how well-separated the underlying probabilities are — a poorly discriminating model will just give you wide/ambiguous sets very often, which is *honest* but not necessarily *useful*.

---

# CHAPTER 6 — Cheminformatics Math

## 6.1 Morgan / ECFP Circular Fingerprints

**The algorithm (Morgan algorithm, extended-connectivity form).**
1. Initialize each atom with an integer identifier (based on atom properties: element, degree, charge, H-count, etc.)
2. For radius $r=1,2,\dots,R$: each atom's new identifier is a hash of (its current identifier, sorted list of its neighbors' identifiers, bond types) — this is one "round" of neighborhood aggregation, closely analogous conceptually to one layer of message-passing in a graph neural network.
3. Collect all unique identifiers generated across all atoms and all radii $\le R$ as the fingerprint's "on" features.
4. **Folding/hashing into fixed length**: each identifier is hashed mod $N_{bits}$ (e.g., 512) to fit into a fixed-size vector — this can cause **hash collisions** (two different substructures mapping to the same bit), which is a real, known information-loss source, especially at small bit widths. Your bit-width sweep (256→2048) is directly probing this collision-rate tradeoff.

**Count vs. bit vectors.** A bit vector just records presence/absence (0/1) per position; a **count** vector (what your code uses — `GetCountFingerprint`) records how many times each substructure occurred, retaining more information (e.g., a lipid with 3 ester groups vs. 1 gets different count vectors, but identical bit vectors).

**Assumptions/limitations.** Radius controls how "far" structural context extends (radius 3 ≈ ECFP6, capturing neighborhoods roughly 6 bonds in diameter) — too small misses pharmacophore-level context, too large risks every atom in a smallish molecule producing near-identical, non-discriminating giant substructures. Folding/hashing collisions are a genuine, irreducible tradeoff against dimensionality — there is no folding scheme that eliminates them entirely at fixed bit width, only makes them statistically rarer.

## 6.2 MACCS Keys

166 (sometimes reported as 167 with a padding bit) pre-defined, human-curated structural keys/patterns (e.g., "has a ring," "has ≥2 nitrogens"), each a fixed yes/no SMARTS-style check — unlike Morgan fingerprints, the "features" are fixed and interpretable by a chemist by name/index, at the cost of far lower structural resolution and no ability to capture patterns outside the predefined set.

## 6.3 Atom-Pair and Topological Torsion Fingerprints

**Atom pairs** encode (atom-type-i, shortest-path-distance, atom-type-j) triples for every pair of atoms in the molecule. **Topological torsions** encode 4-atom contiguous paths (like a dihedral/torsion angle pattern, but purely topological/2D, not 3D geometry) — both are alternative, complementary 2D descriptors to circular fingerprints, useful as an ablation/robustness check on representation choice (exactly how your `FingerprintBenchmark` section uses them).

## 6.4 Jaccard / Tanimoto Distance — the workhorse metric

**The math.** For two binary fingerprint vectors $A,B$:
$$T(A,B) = \frac{|A\cap B|}{|A\cup B|} = \frac{c}{a+b-c}$$
where $c$ = number of bits on in both, $a$ = bits on in A, $b$ = bits on in B. **Tanimoto distance** $=1-T(A,B)$.

**Worked example.** Fingerprint A has bits {2,5,9,14}, B has bits {5,9,14,20,21}. Intersection = {5,9,14} → $c=3$. Union = {2,5,9,14,20,21} → 6 elements.
$$T = 3/6 = 0.5 \implies \text{distance} = 0.5$$

**Why Tanimoto and not Euclidean/cosine on binary chemical fingerprints.** Euclidean distance on sparse high-dimensional binary vectors is dominated by the vectors' *sizes* (number of "on" bits) rather than their overlap pattern — two large, mostly-different fingerprints can have similar Euclidean distance to two small, mostly-identical ones. Tanimoto directly normalizes by the union, making it the field-standard similarity metric for binary molecular fingerprints, robust to that size effect.

**Limitations.** Tanimoto similarity has a known "size dependence" bias of its own — very small fingerprints (few bits on) tend to produce more extreme similarity values by chance; and Tanimoto is not a true metric in the sense some optimizations assume (though Tanimoto *distance* $1-T$ does satisfy the triangle inequality, which is why it's safe to use inside a kNN model as you do).

## 6.5 RDKit Physicochemical Descriptors

- **LogP (Crippen/Wildman method)**: sum of atomic contribution values (empirically fit per atom-type from a large calibration set) — an atom-additive model, not a first-principles computation. Limitation: atom-typing schemes can misclassify unusual functional groups, and the additive model ignores 3D conformational/intramolecular effects entirely.
- **TPSA (Topological Polar Surface Area)**: sum of pre-tabulated fragment contributions for polar atoms (N, O, and their attached H's) — again a fragment-additive empirical model, computed from 2D topology only (no 3D surface calculation involved despite the name "surface area" — a great gotcha fact to have ready).
- **HBA/HBD, rotatable bonds, ring count, fraction Csp3**: simple substructure/count-based rule definitions (e.g., rotatable bond = any single, non-ring, non-terminal bond between two non-terminal heavy atoms) — deterministic and interpretable, but rule-based cutoffs mean edge cases (e.g., amide bonds, which are technically single bonds but have partial double-bond character/restricted rotation) require special-casing that different toolkits sometimes handle inconsistently.

## 6.6 ChemBERTa Frozen Embeddings

**Architecture, briefly.** A BERT-style transformer pretrained via masked-language-modeling on SMILES strings (predicting randomly masked tokens from context) — this pretraining objective forces the model to learn a representation that captures which substructures/tokens are "chemically plausible" in context.

**Mean pooling, formally.** Given per-token hidden states $h_1,\dots,h_L$ (with an attention mask $m_i\in\{0,1\}$ for real vs. padding tokens):
$$\text{embedding} = \frac{\sum_i m_i h_i}{\sum_i m_i}$$
i.e., average the non-padding token representations — the standard frozen-feature-extraction choice absent a fine-tuning objective that would otherwise justify using just the [CLS] token.

**Why frozen, not fine-tuned — the bias-variance argument.** Fine-tuning a transformer with millions of parameters on $n\approx452$ labeled examples risks catastrophic overfitting (very high model capacity relative to data) — freezing the pretrained weights and only training a much lower-capacity downstream model (Random Forest, ~hundreds of trees on a fixed embedding) is a much better-calibrated bias-variance tradeoff at this sample size. This is a textbook example of matching model capacity to data availability, and it's worth stating in exactly that language.

---

# CHAPTER 7 — Ensemble Machine Learning

## 7.1 Random Forest

**The math.** For $b=1,\dots,B$ trees: draw a bootstrap sample of size $n$ (with replacement) from the training data; grow a decision tree, but at each split only consider a random subset of $m<p$ features (default $\sqrt{p}$ for classification); grow to full depth (or until a stopping criterion) without pruning. Final prediction:
$$\hat{p}(y=1\mid x) = \frac{1}{B}\sum_{b=1}^{B}\mathbb{1}[\hat{y}_b(x)=1]$$
— literally the fraction of trees voting positive, which is why RF probabilities are well-behaved but not automatically calibrated (Chapter 4).

**Why the two sources of randomness (bootstrap rows + random feature subsets) matter.** Bagging (bootstrap aggregating) primarily reduces **variance** by averaging over many high-variance, low-bias trees (each individual deep tree overfits badly, but their *average* has much lower variance while retaining low bias) — the classic **bias-variance decomposition** argument:
$$\text{Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible noise}$$
Averaging $B$ *identically distributed but correlated* estimators with pairwise correlation $\rho$ and individual variance $\sigma^2$ gives:
$$\text{Var(average)} = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$$
As $B\to\infty$, the second term vanishes, but the **first term (driven by tree-to-tree correlation) does not** — this is *exactly why* the random feature-subset step exists: decorrelating the trees (by preventing them from always splitting on the same dominant feature) lowers $\rho$ and thus lowers the *floor* on achievable variance reduction, beyond what bagging alone could do.

**`class_weight='balanced'`**: reweights the Gini/entropy split criterion and the vote-aggregation inversely proportional to class frequency — important here given High/Low EE imbalance.

**Limitations.** RF probabilities cluster away from 0 and 1 (the "averaging" mechanism structurally prevents extreme confident scores unless nearly all trees agree) — one specific, well-known reason RF output benefits from post-hoc calibration. Also, RF importances (and by extension SHAP on RF) can be biased toward high-cardinality features (many fingerprint bits) vs. low-cardinality ones (the 4 molar-fraction features) purely due to how many possible split points each offers — worth flagging as an interpretability caveat when discussing the SHAP component comparison.

## 7.2 Extra Trees (Extremely Randomized Trees)

Same ensemble-averaging logic as RF, but at each candidate split, thresholds are chosen **randomly** (not the locally-optimal Gini-minimizing threshold) among the random feature subset, and typically the *whole* training set (not a bootstrap resample) is used per tree. This further decorrelates trees ($\rho$ lower still) at the cost of slightly higher per-tree bias — another concrete instantiation of the bias-variance tradeoff in the same family of models, useful framing if asked "why compare against Extra Trees specifically."

## 7.3 XGBoost (Gradient Boosting)

**The math, briefly.** Unlike bagging (parallel, independent trees averaged together), boosting builds trees **sequentially**, each new tree fit to the *negative gradient* of the loss function with respect to the current ensemble's predictions (for log-loss, this residual is closely related to $y-\hat p$). Each tree's contribution is scaled by a learning rate $\eta$:
$$F_m(x) = F_{m-1}(x) + \eta \cdot h_m(x)$$
This is fundamentally a **bias-reduction** strategy (each new tree explicitly corrects previous errors) as opposed to RF/ExtraTrees' **variance-reduction** strategy via averaging — an important conceptual contrast to articulate.

**`scale_pos_weight`**: multiplies the gradient/loss contribution of positive-class examples by $\frac{n_{neg}}{n_{pos}}$, the boosting-specific analogue of RF's `class_weight='balanced'`.

**Limitations.** More hyperparameter-sensitive than RF (learning rate, depth, number of rounds all interact); more prone to overfitting with a small dataset like $n\approx452$ if not carefully regularized — which is likely part of why RF ends up the primary/deployed model here rather than XGBoost, worth having that framing ready.

---

# CHAPTER 8 — Dimensionality Reduction & Clustering

## 8.1 Variance Threshold (feature selection)

Drops any feature whose variance across all samples is exactly (or below) some cutoff — here, exactly 0 for descriptors (a truly constant feature literally cannot help any split-based model, since there's nothing to split on). This is the simplest possible unsupervised feature filter — it uses no label information at all, so it can never accidentally leak label information into feature selection (a nice property, but also means it does nothing to remove *redundant-but-varying* features).

## 8.2 UMAP (Uniform Manifold Approximation and Projection)

**Conceptual math (know this at the explain-to-a-committee level, not full derivation).**
1. Build a weighted k-nearest-neighbor graph in the original high-dimensional space, where edge weights approximate the probability that two points are "neighbors," using a *locally adaptive* distance normalization (each point gets its own effective radius based on its local density — this is what lets UMAP handle regions of varying density well, unlike a fixed-bandwidth method).
2. Construct an analogous graph in a low-dimensional (2D) embedding space, initialized (e.g., via spectral embedding).
3. Optimize the low-dimensional coordinates via stochastic gradient descent to minimize a cross-entropy-like objective between the high-D and low-D neighbor-probability graphs — pulling true neighbors together and pushing non-neighbors apart.

**Key parameters.** `n_neighbors`: controls the local/global structure tradeoff — small values (e.g., 5-15) emphasize preserving fine local neighborhoods (can fragment global structure into many small islands); larger values (30-100) emphasize preserving broader/global structure at the cost of local detail. `min_dist`: how tightly points are allowed to pack in the embedding — smaller values give tighter, more clumped clusters; larger values give a more evenly-spread-out embedding.

**Limitations — the big one.** UMAP embeddings are for **visualization/qualitative exploration only** — inter-cluster *distances* and even cluster *sizes* in a UMAP plot are **not** reliably meaningful (unlike, say, PCA, where axis distances retain a direct variance interpretation) — never use UMAP coordinates themselves as model input features or claim quantitative distance meaning from the 2D plot. This is a very commonly-asked "gotcha" — be ready with exactly this caveat.

## 8.3 K-Means Clustering

**The math.** Minimize within-cluster sum of squares:
$$\arg\min_{S} \sum_{i=1}^{k}\sum_{x\in S_i} \|x-\mu_i\|^2$$
solved iteratively (Lloyd's algorithm): assign each point to its nearest current centroid, recompute centroids as the mean of assigned points, repeat until convergence. **Not guaranteed to find the global optimum** — the objective is non-convex, so results depend on initialization (hence `n_init=10`, running the whole procedure 10 times with different random centroid starts and keeping the best).

**Assumptions.** Implicitly assumes roughly spherical, similarly-sized clusters in **Euclidean** space (which is why descriptors were standardized first via `StandardScaler` — without standardizing, a feature like MolWt with a much larger numeric range than, say, ring count would dominate the Euclidean distance purely due to units/scale, not genuine chemical importance).

## 8.4 Silhouette Score

**The math.** For point $i$: $a(i)$ = mean distance to other points in its own cluster; $b(i)$ = mean distance to points in the nearest *other* cluster.
$$s(i) = \frac{b(i)-a(i)}{\max(a(i),b(i))} \in [-1,1]$$
Average across all points for the overall score. $s\approx1$: well-clustered; $s\approx0$: on a cluster boundary; $s<0$: likely misassigned.

**Combined with the elbow method (inertia vs. $k$)** — inertia (within-cluster SS) *always* decreases monotonically as $k$ increases (more clusters can only reduce or maintain within-cluster spread), so you look for the "elbow" where the *rate* of decrease sharply slows — this is why silhouette (which does NOT monotonically improve with more clusters) is used *alongside* inertia, not instead of it, to pick $k=4$: silhouette can actually get worse with more clusters even as inertia keeps dropping, giving you a genuine tension to resolve rather than a naive "always add more clusters" trap.

---

# CHAPTER 9 — SHAP: Game-Theoretic Interpretability

## 9.1 The Shapley value, from cooperative game theory

**The original problem (Shapley, 1953).** $n$ players cooperate in a game with payoff function $v(S)$ for any coalition $S\subseteq\{1,\dots,n\}$. How do you fairly split the total payoff $v(\{1,\dots,n\})$ among individual players, accounting for their marginal contribution across *every possible order* they could join the coalition?

**The formula.**
$$\phi_i = \sum_{S\subseteq N\setminus\{i\}} \frac{|S|!\,(n-|S|-1)!}{n!}\Big[v(S\cup\{i\}) - v(S)\big]$$

This averages player $i$'s marginal contribution $v(S\cup\{i\})-v(S)$ over **every possible subset** $S$ of the *other* players that could have already joined before $i$, weighted by how many orderings produce that particular subset.

**Mapping to SHAP for ML.** "Players" = features; "payoff" $v(S)$ = the model's expected prediction given only features in $S$ are "known" (others marginalized out); $\phi_i$ = feature $i$'s SHAP value for one specific prediction — its fair share of the difference between the model's output for this instance and the average model output.

## 9.2 Worked mini-example (2 features, by hand)

Model: $f(x_1,x_2) = 2x_1 + 3x_2 + x_1 x_2$ (has an interaction term, so simple additive attribution isn't obvious). Baseline (both features "absent"/at reference value 0): $v(\emptyset)=0$. Instance: $x_1=1, x_2=1$, so $v(\{1,2\})=2+3+1=6$.

$v(\{1\})$ (only $x_1$ known, $x_2$ at baseline 0) $=2(1)+3(0)+1\cdot0=2$
$v(\{2\})$ (only $x_2$ known) $=2(0)+3(1)+0=3$

Orderings for feature 1: (arrives first) marginal $= v(\{1\})-v(\emptyset)=2-0=2$; (arrives second, after 2) marginal $=v(\{1,2\})-v(\{2\})=6-3=3$. Average (equal weight for $n=2$): $\phi_1 = (2+3)/2 = 2.5$

Similarly for feature 2: (first) $=v(\{2\})-v(\emptyset)=3$; (second) $=v(\{1,2\})-v(\{1\})=6-2=4$. $\phi_2=(3+4)/2=3.5$

**Check (local accuracy property):** $\phi_1+\phi_2 = 2.5+3.5=6 = v(\{1,2\})-v(\emptyset)$ ✓ — the SHAP values exactly sum to the prediction's deviation from baseline, *including* fair credit for the interaction term (split evenly between the two features here, since both are "equally responsible" for it by symmetry in this toy example) — this exact-decomposition property is the mathematical guarantee Shapley values uniquely provide (Shapley proved it's the *only* attribution scheme satisfying four fairness axioms simultaneously: efficiency/local accuracy, symmetry, dummy/missingness, and additivity).

## 9.3 TreeExplainer — why it's tractable at all

Naively, computing exact Shapley values requires evaluating $v(S)$ for all $2^n$ subsets — utterly intractable for hundreds of fingerprint-bit features. **TreeExplainer** (Lundberg et al.) exploits tree structure: since a tree's prediction only depends on which branch each feature's value routes you down, the exact expected-value computation over feature subsets can be done by tracking multiple paths through the tree simultaneously with a clever combinatorial bookkeeping scheme — reducing complexity from exponential to **low-order polynomial** in the number of features and tree depth, making exact (not approximate/sampled) Shapley values computationally feasible for a Random Forest.

## 9.4 The four defining properties (know these by name)

1. **Local accuracy (efficiency)**: SHAP values sum exactly to (prediction − expected baseline prediction) — demonstrated numerically above.
2. **Missingness**: a feature that's genuinely absent/has no effect gets $\phi_i=0$.
3. **Consistency (monotonicity)**: if a model changes so that a feature's marginal contribution increases (weakly) for every possible subset, that feature's SHAP value cannot decrease — this is the property that rules out simpler heuristic attribution schemes (like raw feature importances from impurity decrease), which can violate it.

## 9.5 Limitations, honestly stated

- **Correlated features split credit ambiguously.** If two fingerprint bits are highly correlated (e.g., both flag "presence of a tertiary amine" via slightly different substructure matches), Shapley values will split "credit" between them in a way that can understate either one's importance individually, even though the *combined* effect is large — a very fair thing to flag proactively given how correlated overlapping circular-fingerprint bits often are.
- **SHAP explains the *model*, not necessarily the *true underlying chemistry*** — if the model itself has learned a spurious/confounded pattern (e.g., a lab-specific batch effect masquerading as a chemistry signal, which your leave-one-publication-out analysis explicitly checks for), SHAP will faithfully and confidently explain *that* spurious pattern too. SHAP importance is not causal evidence.
- Computing SHAP on a **sample** of 200 rows (not the full dataset, for speed) means the summary plot reflects that subsample's specific distribution — worth being explicit that it's a subsample if asked.

---

# CHAPTER 10 — Composite Scoring (FQS) & Optimization

## 10.1 Derringer-Suich Desirability Functions (1980)

**The general idea.** Convert each raw response variable into a unitless "desirability" $d\in[0,1]$, using a shape appropriate to whether you want that variable large, small, or within a target range — then combine.

**Larger-the-better** (used for $d_{EE}$):
$$d = \left(\frac{Y-Y_{min}}{Y_{max}-Y_{min}}\right)^r$$
In your simplified version (already a 0-1 probability), $d_{EE}=P^r$ directly. The exponent $r$ controls curvature: $r=1$ is linear, $r>1$ pushes desirability down more steeply for anything less than fully confident (bows the curve *below* the diagonal, penalizing uncertainty harder), $r<1$ would do the opposite (reward moderate confidence more generously).

**Target-is-best / trapezoidal** (used for $d_{size}$): rises linearly from 0 to 1 over a lower ramp, plateaus at 1 over the ideal range, falls linearly back to 0 over an upper ramp — a piecewise-linear function.

**Smaller-the-better** (used for $d_{PDI}$): linear decreasing from 1 (at or below the ideal cutoff) to 0 (at the unacceptable cutoff).

## 10.2 Combining via Geometric Mean — the crucial design choice

$$FQS = 100 \times \left(\prod_{i=1}^{k} d_i\right)^{1/k}$$

**Why geometric, not arithmetic, mean — the math of why this matters.** Consider $d_{EE}=0.9$, $d_{size}=0.9$, $d_{PDI}=0.01$ (a formulation that's terrible on PDI, great on everything else).

- Arithmetic mean: $(0.9+0.9+0.01)/3 = 0.603$ → looks like a "decent, 60%-good" formulation.
- Geometric mean: $(0.9\times0.9\times0.01)^{1/3} = (0.0081)^{1/3} \approx 0.201$ → correctly flags this as a poor formulation overall.

The geometric mean's key mathematical property: **it goes to zero if any single factor goes to zero**, and more generally it penalizes imbalance across components far more than the arithmetic mean does (this follows from the AM-GM inequality: geometric mean ≤ arithmetic mean always, with equality only when all components are equal) — exactly matching the real-world requirement that a formulation failing badly on *any one* critical QC axis (size, PDI, encapsulation) should NOT be rescued by doing well on the others. This is the same design logic behind **QED** (Quantitative Estimate of Drug-likeness, Bickerton et al. 2012, Nature Chemistry) which this FQS is explicitly modeled after.

**Limitations.** Geometric mean is sensitive to how you handle missing components (your `compute_FQS` handles this by only including *available* components in the product/root — reasonable, but be aware it means FQS values aren't directly comparable between a 2-component and 3-component computation, since they're geometric means over different numbers of factors). Also, the choice of exponent $r=3$ for $d_{EE}$ was validated by trying multiple $r$ values (your code sweeps $r=2$ to $9$) rather than derived from first principles — an empirical/pragmatic choice, worth naming as such rather than implying deep theoretical justification.

---

# CHAPTER 11 — Regression Tools Used for Calibration

## 11.1 Ordinary Least Squares (Linear Regression)

**The math.** Fit $y=\beta_0+\beta_1x$ minimizing $\sum(y_i-\hat y_i)^2$. Closed-form solution:
$$\hat\beta_1 = \frac{\sum(x_i-\bar x)(y_i-\bar y)}{\sum(x_i-\bar x)^2}, \qquad \hat\beta_0=\bar y - \hat\beta_1\bar x$$

Used for the LogP → apparent-pKa calibration, fit on only 5 literature anchor points.

**Assumptions.** Linearity of the true relationship, homoscedastic and normally distributed residuals (for valid inference/CIs, though here mainly used as a point predictor), and — critically with $n=5$ — very little room to detect violations of any of these assumptions at all.

## 11.2 Leave-One-Out Cross-Validation (LOOCV) — why here specifically

With only 5 calibration data points, a train/test split would leave almost nothing to train *or* test on. LOOCV trains on $n-1=4$ points and predicts the 1 left out, repeated 5 times (once per point) — this **maximizes** the training data used in each fold (a real advantage when $n$ is this small) while still giving an honest, unbiased-ish estimate of out-of-sample error, since each held-out point was never used to fit the model that predicts it.

**Limitations.** With $n=5$, LOOCV error estimates themselves have very high variance (you're averaging only 5 residuals) — your code's own printed caveat ("small n=5 means this error estimate is itself noisy") is exactly the right level of honesty to carry into your talk. Also, LOOCV is known to have a *general* limitation for model comparison/selection purposes (its variance as an estimator of true test error can be higher than k-fold CV in some settings) — though here it's used purely for error estimation on a fixed model form, not model selection, which somewhat sidesteps that particular critique.

## 11.3 LOWESS (Locally Weighted Scatterplot Smoothing)

**The math, conceptually.** For each x-value where you want a fitted point, take a local window of nearby points (window size controlled by `frac`, the fraction of all points included), fit a **weighted** linear (or low-order polynomial) regression within that window — points closer to the query x-value get higher weight (typically via a tricube weight function) — then move to the next query point and repeat. The result is a smooth, flexible, non-parametric curve that adapts to local structure rather than imposing one global functional form.

**Why used here specifically (vs. a single global linear fit).** For the pKa-vs-EE% panels, LOWESS is explicitly chosen so that reviewers can visually assess whether the relationship implied by the reported Spearman ρ is genuinely monotonic-and-smooth throughout, or is being driven by a nonlinear/non-monotonic pattern that a single ρ number could mask — a good, honest visualization choice.

**Limitations.** `frac` is a bias-variance tradeoff knob (small frac = flexible but noisy/wiggly; large frac = smooth but can wash out real local structure) chosen somewhat by eye rather than cross-validated — worth being upfront that it's a visualization aid, not a formally-validated regression model.

---

# CHAPTER 12 — Applicability Domain & Active Learning

## 12.1 kNN-Distance Applicability Domain (AD)

**The math.** For each training point, compute the mean Jaccard distance to its $k=5$ nearest neighbors (Tanimoto/Jaccard distance on the ionizable-lipid fingerprint, since SHAP showed that's the dominant signal). The AD threshold is set at the **95th percentile** of this distribution of training-set self-distances:
$$\tau_{AD} = P_{95}\big(\{\bar{d}_{kNN}(x_i) : x_i \in \text{training set}\}\big)$$
A new point is "in-domain" if its mean kNN distance to the (fixed) training set is $\le\tau_{AD}$.

**Why 95th percentile specifically.** This is a deliberate design choice, not a law of nature — it says "we accept that up to 5% of our *own training data* would nominally count as borderline-novel relative to the rest of training," giving a threshold calibrated to the actual density/spread of the training chemistry, rather than an arbitrary fixed distance cutoff that wouldn't adapt if you used a different fingerprint or different lipid library.

**Empirical validation, not just definition.** You don't just define the threshold — you verify it *works*, by checking AUC(inside AD) vs AUC(outside AD) actually differ substantially. This turns AD from an assumption into a tested claim.

**Limitations.** The threshold is entirely a property of the *training set's own internal density* — if your whole training set happens to sit in a narrow region of chemical space, the AD threshold will be small and *everything* outside that narrow region gets flagged "novel," even things a chemist might consider only mildly unusual. AD, as defined here, also only looks at the ionizable-lipid fingerprint — it doesn't directly account for novel *combinations* of otherwise-familiar helper/sterol/PEG lipids or novel molar ratios.

## 12.2 Active Learning — Acquisition Function

**The general active-learning idea.** Rather than randomly choosing what to label/synthesize/test next, rank candidates by how much labeling *this specific one* would improve the model — typically balancing **uncertainty** (the model doesn't know the answer, so a label here is maximally informative) against **representativeness/novelty** (this candidate isn't just a copy of something you already know) and, often, **usefulness of the outcome itself** (don't only chase pure uncertainty at the expense of practically promising candidates).

**Your specific acquisition score:**
$$a(x) = \big(0.5\cdot u(x) + 0.3\cdot \nu(x) + 0.2\cdot \pi(x)\big)\times \text{AD-penalty}(x)$$
where $u(x)=1-2|\hat p(x)-0.5|$ (continuous uncertainty, max at $\hat p=0.5$), $\nu(x)$ = normalized kNN distance (novelty), $\pi(x)=\hat p(x)$ (promise — favors likely-High-EE candidates).

**Why a *continuous* uncertainty score, not a binary "ambiguous/not" flag.** A candidate at $\hat p=0.50$ and one at $\hat p=0.75$ might both technically fall inside a conformal-ambiguous set (if $\hat q$ happens to be large), but they are **not** equally uninformative — 0.50 is maximally uncertain by construction, 0.75 much less so. Using the continuous score $u(x)$ instead of the binary conformal-ambiguous flag lets the ranking discriminate meaningfully *within* the "ambiguous" bucket rather than treating them as tied.

**The AD penalty as a gate, not a smooth term.** Rather than blending "novelty" and "in-domain-ness" into one continuous tradeoff, your pipeline applies novelty as a *reward* (weight 0.3) but caps genuinely out-of-domain candidates with a hard multiplicative penalty (0.5×) and an outright exclusion beyond `ad_multiplier_cutoff × τ_AD` — reflecting the reasoning from Chapter 5.4: **too far out-of-domain, and the model's own uncertainty estimate can no longer be trusted**, so novelty stops being a virtue and starts being a red flag.

## 12.3 Greedy Farthest-First Diversity Selection

**The problem.** Simply taking the top-$N$ candidates by acquisition score alone risks a shortlist that's all minor ratio-variants of one or two chemotypes (since a genuinely promising/uncertain lipid will have many similar high-scoring neighbors in the sweep).

**The greedy algorithm (a classic, well-known heuristic — related to the k-center problem).**
1. Start with the single highest-scoring candidate.
2. Repeatedly add whichever remaining candidate maximizes:
$$\text{combined}(x) = 0.5 \times \min_{j\in\text{selected}}\big(\text{Jaccard-dist}(x,j)\big) + 0.5\times\frac{a(x)}{\max(a)}$$
i.e., a candidate must be both far from everything already picked (its minimum distance to the selected set, not average — this is the "farthest-first" / maximin logic) *and* still reasonably good on its own acquisition score.
3. Repeat until $N$ candidates are selected.

**Why "minimum distance to selected set," not average.** Using the *minimum* enforces that a new candidate is genuinely distinct from its single closest already-picked neighbor — using an *average* would let a candidate slip through by being far from most selected points even if it's a near-duplicate of just one of them.

**Limitations.** This greedy heuristic is **not guaranteed globally optimal** — it's a fast, reasonable approximation to a combinatorially hard "best diverse subset" selection problem (true optimal subset selection over all $\binom{|pool|}{N}$ combinations is generally NP-hard for these kinds of objectives). Also, the 50/50 weighting between diversity and acquisition score is a manual hyperparameter choice, not learned or cross-validated.

---

# Master Cheat-Sheet: One Formula, One Sentence, Per Topic

| Topic | Core formula/idea in one line |
|---|---|
| F-test | Between-group variance ÷ within-group variance |
| Wilcoxon signed-rank | Sum ranks of signed paired differences; small n → low power |
| Mann-Whitney U | Rank-sum comparison of two independent groups |
| Kruskal-Wallis | Rank-based ANOVA for ≥2 groups |
| Chi-square | Σ(observed−expected)²/expected on a contingency table |
| McNemar | Compares only the *discordant* paired-classifier predictions |
| Spearman ρ | Pearson correlation computed on ranks |
| Bonferroni | α/m — controls family-wise error, conservative |
| Benjamini-Hochberg | Controls expected false discovery *proportion*, less conservative |
| GroupKFold | Whole groups (lipids) assigned to one fold, never split |
| Bootstrap CI | Resample with replacement, take percentiles of the resampled statistic |
| ROC-AUC | P(random positive scores higher than random negative) |
| Brier score | Mean squared error of predicted probabilities — a proper scoring rule |
| ECE | Weighted average gap between predicted confidence and observed accuracy, per bin |
| Conformal prediction | Include label if its nonconformity ≤ a calibration-set quantile — guarantees marginal coverage under exchangeability |
| Tanimoto/Jaccard | Intersection over union of fingerprint bits |
| Random Forest | Bagged, decorrelated trees — averaging reduces variance |
| XGBoost | Sequential trees fit to residual gradients — reduces bias |
| UMAP | Preserve local-neighbor graph structure in low dimensions — NOT for quantitative distances |
| Silhouette score | (between-cluster dist − within-cluster dist) / max of the two |
| Shapley value | Fair credit-splitting via averaging marginal contributions over all possible feature orderings |
| Desirability (geometric mean) | Any single failed component → whole score collapses toward 0 |
| LOOCV | Train on n−1, test on the 1 left out, repeated n times |
| Applicability Domain | 95th-percentile kNN distance in training set defines "in-domain" |
| Active learning acquisition | Rank by (uncertainty + novelty + promise), gated by domain membership |