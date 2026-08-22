# PLM Latent Geometry as a State Space for Protein Evolution — Execution Plan

This repo turns the tightened framework into runnable code. The pure-numpy
analysis modules (`intrinsic_dim`, `sequence_utils`, `geometry_analysis`,
`robustness_evolvability`, `cv_discovery`, `pipeline`) are unit-tested against
synthetic data with known ground truth in `tests/` — run those first, on
this machine or yours, before touching real data:

```bash
pip install -r requirements.txt
python tests/test_synthetic.py        # 8 targeted correctness checks
python tests/test_pipeline_smoke.py   # end-to-end plumbing check
```

Both currently pass (8/8 and 1/1). The ESM-2 extraction module
(`src/embeddings.py`) needs `torch` + network access to HuggingFace and is
**not** covered by these tests — it needs to be run and checked on your own
GPU machine. Everything downstream of it (loaded from a saved `.npz`) is
fully testable offline, which is why the split exists.

---

## Phase 0 — Environment + toolkit validation (you are here)

- [x] Core modules implemented and unit-tested (this repo).
- [ ] `pip install -r requirements.txt` on your own machine/cluster.
- [ ] Confirm `transformers` can load an ESM-2 checkpoint on your GPU:
  ```python
  from src.embeddings import extract_esm2_embeddings
  extract_esm2_embeddings(["MTYKLILNGK"], model_name="esm2_t12_35M", device="cuda")
  ```
  Use the smallest model (35M) for this smoke check — it's fast and rules
  out environment problems before you commit GPU time to the real run.

**Estimated time: <1 day.**

---

## Phase 1 — GB1 pilot (start here, not with a new protein)

You already have GB1 DMS labels and an ESM-2 embedding pipeline from the
active-learning stability project. Use that head start:

1. Decide explicitly which GB1 dataset you're using — **Olson et al. 2014**
   (binding fitness, near-complete double mutants) or **Nisthal et al. 2019**
   (ddG stability). They are different phenotypes; don't mix them. Note this
   choice in `config/proteins.yaml`.
2. Adapt `notebooks/01_pilot_GB1.py` Stage A to either reuse your existing
   embeddings or re-extract with `src/embeddings.py` (recommended even if
   you have old embeddings, so every layer is saved — most existing
   pipelines only keep the last layer, which Section 3.4 of the framework
   doc argues is probably not the layer you want).
3. Run Stage B:
   ```bash
   python notebooks/01_pilot_GB1.py b
   ```
   This runs the full Objective 1–4 pipeline via `src/pipeline.run_full_analysis`
   and writes everything to `results/GB1/`.
4. **Checkpoint — read `results/GB1/SUMMARY.json` before proceeding.**
   Specifically look at:
   - `intrinsic_dimension_by_layer`: does ID show the peak→plateau→ascent
     pattern from Valeriani et al., and does the plateau land near ID≈6-7,
     or somewhere clearly different for a single-protein mutational
     neighborhood (itself an interesting result either way)?
   - `q1_stats['embed_dist_resid']['spearman_rho']` vs.
     `q1_stats['hamming_dist']['spearman_rho']`: if the BLOSUM-controlled
     residual correlation is close to zero while raw Hamming/BLOSUM
     correlation is strong, most of what looks like "PLM semantic structure"
     in Q1 is actually substitution chemistry — this materially changes how
     much weight you can put on Objectives 3–4 with this embedding/layer.

**Estimated time: 3–5 days**, mostly GPU embedding extraction + sanity-checking
outputs, not new coding — reuse what's here.

---

## Phase 2 — Objective 1: geometry characterization (multi-protein)

Run `pipeline.run_geometry_objective1` across all four proteins once you
have multi-layer embeddings for each. Compare:
- Is the peak → plateau → ascent pattern universal, or protein-specific?
- Does plateau ID differ between GFP (large, saturating single-domain),
  TEM-1 (enzyme, catalytic constraints), GB1 (small, well-packed), and RBD
  (interface-heavy, immune-pressure-shaped)? A systematic difference here
  would itself be a genuine, reportable finding.

**Checkpoint:** if plateau ID is essentially flat and uninformative across
all four proteins (no discernible three-phase structure), Objective 1 as
framed may not be reproducing on single-protein mutational landscapes what
Valeriani et al. found on generic protein corpora — worth knowing early,
since Objective 4's CV-discovery step assumes there's structure to find.

**Estimated time: 1 week** (mostly embedding extraction for 3 more proteins;
GFP and RBD are larger so budget more GPU time / consider `esm2_t30_150M`
first before committing to 650M+ for all four).

---

## Phase 3 — Objective 2: Q1/Q2 with proper controls

Already implemented in `geometry_analysis.q1_analysis` / `q2_per_position`
and wired into the pipeline. Per protein, look at:
- `objective2_q1_stats.json` for the BLOSUM-residual comparison (see Phase 1
  checkpoint above).
- `objective2_q2_per_position.csv` for per-position correlation heterogeneity
  — plot rho against solvent accessibility or binding-interface annotation
  if you have structural data (you already work with structures via
  MDAnalysis/RDKit, so this is a natural add: pull relative SASA from a
  static structure or an MD ensemble you already have for these proteins).

**Add grammaticality** (pLM pseudo-perplexity of each mutant) alongside
embedding distance here — it's one extra forward-pass quantity from the same
model call, and gives you the Hie et al. CSCS-style 2D (grammaticality ×
semantic change) coordinate for free.

**Estimated time: 3–4 days per protein** once embeddings exist, mostly
analysis/plotting rather than new code.

---

## Phase 4 — Objectives 3: robustness / evolvability ground truth vs. latent proxy

Already implemented (`robustness_evolvability.py`) and wired into the
pipeline. Key outputs per protein, in `objective34_summary.json`:
- `robustness_ground_truth_vs_latent_proxy` / `evolvability_...`: the actual
  test of the central hypothesis. A near-zero correlation is a real,
  reportable negative result, not a bug — see the module docstring.
- `objective34_arrays.npz` → `tradeoff_bin_centers` / `tradeoff_mean_evolvability`:
  plot these against each other to check for the Draghi et al. (2010)
  intermediate-robustness evolvability peak.

**Important limitation to flag in any writeup:** if you only have
single-mutant DMS data (no doubles), non-WT genotypes have thin
neighborhoods (≤19 possible neighbors via same-position substitutions +
WT) — robustness/evolvability estimates for those genotypes are sparse
lower-bound estimates, not full neutral-network degree. This is exactly why
GB1 (Olson et al., near-complete doubles) and GB1 (4-combo) are valuable —
prioritize double-mutant-available proteins for the strongest version of
this analysis, and treat single-mutant-only proteins as a secondary,
caveated comparison.

**Estimated time: 1 week per protein** for the full ground-truth/latent
comparison plus the tradeoff-shape check, most of it interpretation rather
than compute.

---

## Phase 5 — Objective 4: collective-variable discovery + statistical mechanics

`cv_discovery.diffusion_map` is already run automatically in the pipeline
(`objective4_diffusion_map.npz`, with a quick Spearman correlation against
DMS score already computed as `diffusion_coord_vs_dms_spearman` in
`SUMMARY.json`). To go further:

1. If diffusion coordinate 1 correlates well with DMS score, that IS a
   candidate "macroscopic variable" — characterize it (which
   positions/substitutions drive it, does it track a known structural axis
   like core-packing vs. surface).
2. For a genuine slow-mode / statistical-mechanics treatment (closer to
   DiffEvol's framing), try `cv_discovery.random_walk_trajectory` to
   generate pseudo-trajectories over the Hamming-1 graph, then feed those to
   a proper TICA implementation (e.g. `deeptime`) exactly as you would MD
   frames — this reuses tooling from your own CV-discovery experience
   directly, just pointed at a mutational graph instead of a time series.
3. Compare against a DCA/Potts-model baseline wherever an MSA is available
   for the protein (all four candidates have deep MSAs) — if the "learned
   macroscopic variables" don't at least correlate with Potts coupling
   structure, that's an important caveat before claiming they reflect a
   general genotype-phenotype coarse-graining rather than something
   ESM-2-specific. (Not implemented here — `pydca` or `EVcouplings` are the
   standard tools; flagged as a to-do rather than built, since it's a
   genuinely separate modeling effort.)

**Estimated time: 2 weeks**, the least templated part of the project —
budget accordingly.

---

## Phase 6 — Cross-protein synthesis + model-scale / pooling controls

Once all four proteins have gone through Phases 2–5:
- Repeat the GB1 pilot at 2+ ESM-2 sizes (e.g. 35M and 650M) to check
  whether findings are scale-invariant (per Valeriani et al.) or emergent
  only at scale — Section 3.5 of the framework doc.
- Compare mean-pooled vs. position-specific embeddings, especially for GB1
  and RBD (interface-localized function) vs. GFP/TEM-1.
- Build the cross-protein comparison table: plateau ID, Q1 residual
  correlation, robustness/evolvability correlation, diffusion-coordinate–DMS
  correlation, all side by side. This table is your Objective-1-through-4
  results table.

**Estimated time: 1–2 weeks.**

---

## Phase 7 — Writeup

Deliverables per the original proposal: evolutionary latent atlas (per-
protein diffusion-map + ID visualizations), learned coarse variables (with
characterization from Phase 5), graph geometry (neighbor-graph structure
from `sequence_utils.build_neighbor_graph`), and the statistical-mechanics
formulation from Phase 5 — write this up explicitly against the DiffEvol
(2026) and Draghi et al. (2010) results as the two nearest points of
comparison (Section 2 of the framework doc).

---

## Repository map

```
src/
  sequence_utils.py           Hamming/BLOSUM distance, mutant generation, neighbor graph
  intrinsic_dim.py            TwoNN + local MLE intrinsic dimension estimators
  geometry_analysis.py        Q1 (BLOSUM-controlled) / Q2 (per-position) tests
  robustness_evolvability.py  Ground-truth (Wagner/Draghi) + latent proxies + correlation
  cv_discovery.py             Diffusion maps + random-walk trajectories for TICA
  embeddings.py                ESM-2 extraction (run on your GPU machine, not tested here)
  pipeline.py                  Orchestrates Objectives 1-4 end to end
tests/
  test_synthetic.py           8 targeted correctness checks vs. known ground truth
  test_pipeline_smoke.py      End-to-end plumbing check
notebooks/
  01_pilot_GB1.py              Worked GB1 example (Stage A: embed, Stage B: analyze)
config/
  proteins.yaml                 Data-source notes for GFP / TEM-1 / GB1 / RBD
```

## Known limitations to keep in mind while running this

- Robustness/evolvability on single-mutant-only data is sparse (Phase 4).
- `diffusion_map`'s epsilon (kernel bandwidth) uses a median-distance
  heuristic — sensitivity-check it before trusting a specific number of
  "macroscopic variables."
- No DCA/Potts baseline is implemented yet (Phase 5, item 3) — flagged, not
  built, since it's a genuinely separate modeling effort worth scoping
  on its own.
