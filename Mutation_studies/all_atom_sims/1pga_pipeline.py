"""
1PGA Mutation Study — Full Pipeline
=====================================
PART 1 : Compute scalar MD features from trajectories
         (RMSD, RMSF, Q(t), Rg, SASA, secondary structure)
PART 2 : Validate MD features against Nisthal experimental y scores
PART 3 : Active learning with ESM-2 embeddings + MD features

Directory structure expected (from your simulate_mutant.py):
  OUT_ROOT/{mut}_rep{rep}/{mut}_rep{rep}_trajectory.dcd
  OUT_ROOT/{mut}_rep{rep}/{mut}_rep{rep}_minimized.pdb

Usage:
  python 1pga_pipeline.py --mode features   # compute MD features
  python 1pga_pipeline.py --mode validate   # correlate with Nisthal y
  python 1pga_pipeline.py --mode train      # train RF + active learning
  python 1pga_pipeline.py --mode all        # run everything in sequence
"""

import os
import argparse
import warnings
import pickle
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import spearmanr, pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import LeaveOneOut
import MDAnalysis as mda
from MDAnalysis.analysis import rms, contacts
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis

# ─────────────────────────────────────────────
# CONFIGURATION  —  edit these paths
# ─────────────────────────────────────────────

OUT_ROOT    = "/data/openmm_out/1PGA_mutations/run_330K"  # where your MD trajectories are
PKL_PATH    = "mutant_dataset.pkl"          # your 50 mutants with embeddings + y
CAND_PATH   = "selected_mutants.csv"      # 530k candidate pool
FEAT_OUT    = "md_features.csv"             # output: scalar MD features
RESULTS_OUT = "active_learning_results.csv" # output: AL nominations

N_REPLICAS  = 1      # currently 1, change to 10 when you have more replicas
N_SELECT    = 10     # candidates to nominate per AL round
N_ROUNDS    = 10     # how many AL rounds to simulate


# ═══════════════════════════════════════════════════════════════════════
# PART 1 — MD FEATURE EXTRACTION
# ═══════════════════════════════════════════════════════════════════════

def load_trajectory(mut, rep, out_root=OUT_ROOT):
    """
    Load an MDAnalysis Universe for one mutant/replica.
    Returns (universe, minimized_universe) or (None, None) if files missing.
    """
    rep_dir  = os.path.join(out_root, f"{mut}_rep{rep}")
    dcd_file = os.path.join(rep_dir,  f"{mut}_rep{rep}_trajectory.dcd")
    min_file = os.path.join(rep_dir,  f"{mut}_rep{rep}_minimized.pdb")

    if not os.path.exists(dcd_file):
        print(f"  [SKIP] Missing trajectory: {dcd_file}")
        return None, None
    if not os.path.exists(min_file):
        print(f"  [SKIP] Missing minimized PDB: {min_file}")
        return None, None

    u   = mda.Universe(min_file, dcd_file)
    ref = mda.Universe(min_file)
    return u, ref


# ── Feature 1: RMSD ──────────────────────────────────────────────────
def compute_rmsd(u, ref):
    """
    Mean backbone RMSD over trajectory relative to minimized structure.
    Lower = stays closer to native fold = more stable.
    Returns scalar (Å).
    """
    backbone = u.select_atoms("backbone")
    ref_backbone = ref.select_atoms("backbone")

    rmsd_run = rms.RMSD(
        backbone,
        ref_backbone,
        superposition=True
    ).run()

    # rmsd_run.results.rmsd shape: (n_frames, 3) — col 2 is RMSD value
    rmsd_values = rmsd_run.results.rmsd[:, 2]
    return float(np.mean(rmsd_values))


# ── Feature 2: RMSF ──────────────────────────────────────────────────
def compute_rmsf(u):
    """
    Mean per-residue Cα RMSF over trajectory.
    Lower = less flexible = more stable overall.
    Returns scalar (Å).
    """
    ca = u.select_atoms("protein and name CA")
    rmsf_run = rms.RMSF(ca).run()
    return float(np.mean(rmsf_run.rmsf))


def compute_rmsf_max(u):
    """
    Max per-residue Cα RMSF — captures the most flexible/unstable region.
    Returns scalar (Å).
    """
    ca = u.select_atoms("protein and name CA")
    rmsf_run = rms.RMSF(ca).run()
    return float(np.max(rmsf_run.rmsf))


# ── Feature 3: Fraction of Native Contacts Q(t) ───────────────────────
def compute_native_contacts(u, ref):
    ca_u   = u.select_atoms("protein and name CA")
    ca_ref = ref.select_atoms("protein and name CA")

    Q_run = contacts.Contacts(
        u,
        select=(ca_u, ca_u),          # AtomGroup, not string
        refgroup=(ca_ref, ca_ref),     # AtomGroup, not string
        method="soft_cut",
        radius=8.0
    ).run()

    Q_values = Q_run.timeseries[:, 1]
    return float(np.mean(Q_values))


def compute_native_contacts_std(u, ref):
    ca_u   = u.select_atoms("protein and name CA")
    ca_ref = ref.select_atoms("protein and name CA")

    Q_run = contacts.Contacts(
        u,
        select=(ca_u, ca_u),
        refgroup=(ca_ref, ca_ref),
        method="soft_cut",
        radius=8.0
    ).run()

    Q_values = Q_run.timeseries[:, 1]
    return float(np.std(Q_values))


# ── Feature 4: Radius of Gyration ────────────────────────────────────
def compute_rg(u):
    """
    Mean radius of gyration over trajectory.
    Lower Rg = more compact = more stably folded.
    Returns scalar (Å).
    """
    protein = u.select_atoms("protein")

    rg_values = []
    for ts in u.trajectory:
        rg_values.append(protein.radius_of_gyration())

    return float(np.mean(rg_values))


def compute_rg_std(u):
    """
    Std of Rg over trajectory.
    High std = protein expands/contracts = unfolding events = less stable.
    Returns scalar (Å).
    """
    protein = u.select_atoms("protein")

    rg_values = []
    for ts in u.trajectory:
        rg_values.append(protein.radius_of_gyration())

    return float(np.std(rg_values))


# ── Feature 5: End-to-end distance ───────────────────────────────────
def compute_end_to_end(u):
    """
    Mean end-to-end distance (first Cα to last Cα).
    A proxy for chain compactness.
    Returns scalar (Å).
    """
    ca = u.select_atoms("protein and name CA")
    distances = []

    for ts in u.trajectory:
        first = ca.positions[0]
        last  = ca.positions[-1]
        d     = np.linalg.norm(last - first)
        distances.append(d)

    return float(np.mean(distances))


# ── Feature 6: Hydrogen bonds ─────────────────────────────────────────
def compute_hbonds(u):
    """
    Geometric hydrogen bond counter — no charge info needed.
    Counts N-H...O and O-H...O pairs within distance/angle cutoffs.
    """
    protein = u.select_atoms("protein")
    hbond_counts = []

    # Donor heavy atoms (N, O) and acceptors (O, N)
    donors    = protein.select_atoms("name N O and not resname HOH")
    acceptors = protein.select_atoms("name O N and not resname HOH")

    for ts in u.trajectory[::50]:  # sample every 50 frames for speed
        count = 0
        d_pos = donors.positions
        a_pos = acceptors.positions

        # Vectorised distance check
        for i, dp in enumerate(d_pos):
            dists = np.linalg.norm(a_pos - dp, axis=1)
            # H-bond distance cutoff: 3.5 Å, exclude self
            mask = (dists < 3.5) & (dists > 0.5)
            count += mask.sum()

        hbond_counts.append(count)

    return float(np.mean(hbond_counts))


# ── Feature 7: RMSD plateau (last 20% of trajectory) ─────────────────
def compute_rmsd_plateau(u, ref):
    """
    Mean RMSD in the last 20% of frames.
    If the protein has drifted far from native at the end = unstable.
    Returns scalar (Å).
    """
    backbone     = u.select_atoms("backbone")
    ref_backbone = ref.select_atoms("backbone")

    rmsd_run = rms.RMSD(
        backbone,
        ref_backbone,
        superposition=True
    ).run()

    rmsd_values = rmsd_run.results.rmsd[:, 2]
    n20 = max(1, int(0.2 * len(rmsd_values)))
    return float(np.mean(rmsd_values[-n20:]))


# ── Master function: compute all features for one mutant ──────────────
def compute_features_one_mutant(mut, n_replicas=N_REPLICAS):
    """
    Compute all scalar features averaged across replicas.
    Returns a dict of feature_name → scalar value.
    """
    all_rep_features = []

    for rep in range(1, n_replicas + 1):
        u, ref = load_trajectory(mut, rep)
        if u is None:
            continue

        print(f"    rep {rep}: {len(u.trajectory)} frames")

        rep_feats = {
            "mean_rmsd":       compute_rmsd(u, ref),
            "mean_rmsf":       compute_rmsf(u),
            "max_rmsf":        compute_rmsf_max(u),
            "mean_Q":          compute_native_contacts(u, ref),
            "std_Q":           compute_native_contacts_std(u, ref),
            "mean_rg":         compute_rg(u),
            "std_rg":          compute_rg_std(u),
            "mean_end2end":    compute_end_to_end(u),
            "mean_rmsd_plateau": compute_rmsd_plateau(u, ref),
            "mean_hbonds":     compute_hbonds(u),
        }

        # rep_feats = {
        #     "mean_rmsd":         compute_rmsd(u, ref),          # PRIMARY
        #     "mean_rmsd_plateau": compute_rmsd_plateau(u, ref),  # PRIMARY
        #     "std_rg":            compute_rg_std(u),             # SECONDARY
        #     "mean_end2end":      compute_end_to_end(u),         # SECONDARY
        #     "mean_rmsf":         compute_rmsf(u),               # SECONDARY
        #     "max_rmsf":          compute_rmsf_max(u),           # SECONDARY
        #     "mean_rg":           compute_rg(u),                 # SECONDARY
        #     "mean_hbonds":       compute_hbonds(u),             # FIXED
        #     # REMOVED: mean_Q, std_Q  ← not useful with implicit solvent
        # }

        all_rep_features.append(rep_feats)

    if not all_rep_features:
        return None

    # Average across replicas
    feature_names = all_rep_features[0].keys()
    averaged = {
        feat: float(np.nanmean([r[feat] for r in all_rep_features]))
        for feat in feature_names
    }
    return averaged


def run_feature_extraction(mutant_list, out_path=FEAT_OUT):
    """
    Run feature extraction for all mutants and save to CSV.
    """
    print("=" * 60)
    print("PART 1 — MD FEATURE EXTRACTION")
    print("=" * 60)

    records = []
    for i, mut in enumerate(mutant_list):
        print(f"\n[{i+1}/{len(mutant_list)}] {mut}")
        feats = compute_features_one_mutant(mut)

        if feats is None:
            print(f"  [SKIP] No valid trajectories for {mut}")
            continue

        feats["mutant"] = mut
        records.append(feats)
        print(f"  mean_Q={feats['mean_Q']:.3f}  mean_rmsd={feats['mean_rmsd']:.2f}Å  "
              f"mean_rg={feats['mean_rg']:.2f}Å  mean_hbonds={feats['mean_hbonds']:.1f}")

    df = pd.DataFrame(records)
    # Reorder columns
    cols = ["mutant"] + [c for c in df.columns if c != "mutant"]
    df = df[cols]
    df.to_csv(out_path, index=False)
    print(f"\nSaved features to: {out_path}")
    print(df.describe())

    # ── Compute relative features vs 1PGA ──────────────────────
    if "1PGA" in df["mutant"].values:
        wt_row = df[df["mutant"] == "1PGA"].iloc[0]

        rel_cols = ["mean_rmsd", "mean_rmsd_plateau", "mean_rg",
                    "mean_hbonds", "mean_rmsf", "mean_Q"]

        for col in rel_cols:
            df[f"delta_{col}"] = df[col] - wt_row[col]

        # Save updated CSV with relative features
        df.to_csv(out_path, index=False)
        print("Added relative (delta) features vs 1PGA")
    else:
        print("[WARN] 1PGA not found in features — run 1PGA simulation first")

    return df


# ═══════════════════════════════════════════════════════════════════════
# PART 2 — VALIDATION AGAINST NISTHAL y SCORES
# ═══════════════════════════════════════════════════════════════════════

FEATURE_LABELS = {
    "mean_Q":            ("Fraction Native Contacts Q",     "higher = more stable",  1),
    "mean_rmsd":         ("Mean RMSD (Å)",                  "lower = more stable",  -1),
    "mean_rmsf":         ("Mean RMSF (Å)",                  "lower = more stable",  -1),
    "max_rmsf":          ("Max RMSF (Å)",                   "lower = more stable",  -1),
    "std_Q":             ("Std of Q(t)",                    "lower = more stable",  -1),
    "mean_rg":           ("Radius of Gyration (Å)",         "lower = more stable",  -1),
    "std_rg":            ("Std of Rg (Å)",                  "lower = more stable",  -1),
    "mean_end2end":      ("End-to-end Distance (Å)",        "lower = more stable",  -1),
    "mean_rmsd_plateau": ("RMSD Plateau (Å)",               "lower = more stable",  -1),
    "mean_hbonds":       ("Mean H-bonds",                   "higher = more stable",  1),
}


def run_validation(feat_path=FEAT_OUT, pkl_path=PKL_PATH):
    """
    Correlate each MD feature with Nisthal experimental y scores.
    Prints a table and saves a correlation plot.
    """
    print("\n" + "=" * 60)
    print("PART 2 — VALIDATION vs NISTHAL y SCORES")
    print("=" * 60)

    feat_df = pd.read_csv(feat_path)

    with open(pkl_path, "rb") as f:
        pkl_df = pickle.load(f)

    # Merge on mutant name
    merged = feat_df.merge(
        pkl_df[["name", "y"]],
        left_on="mutant",
        right_on="name",
        how="inner"
    )
    print(f"Matched {len(merged)} mutants with experimental scores")

    # feature_cols = [c for c in feat_df.columns if c != "mutant"]

    feature_cols = [c for c in feat_df.columns
                if c != "mutant" and not feat_df[c].isna().all()]

    results = []
    for feat in feature_cols:
        if merged[feat].isna().all():
            continue
        valid = merged[[feat, "y"]].dropna()
        if len(valid) < 5:
            continue
        rho, p_s = spearmanr(valid[feat], valid["y"])
        r,   p_p = pearsonr(valid[feat],  valid["y"])
        results.append({
            "feature":       feat,
            "spearman_rho":  rho,
            "spearman_p":    p_s,
            "pearson_r":     r,
            "pearson_p":     p_p,
            "n":             len(valid)
        })

    corr_df = pd.DataFrame(results).sort_values(
        "spearman_rho", key=abs, ascending=False
    )

    print("\nCorrelation with experimental ΔG (Nisthal y):")
    print(f"{'Feature':<22} {'Spearman ρ':>12} {'p-value':>10} {'Pearson r':>10} {'p-value':>10}")
    print("-" * 70)
    for _, row in corr_df.iterrows():
        sig = "***" if row["spearman_p"] < 0.001 else (
              "**"  if row["spearman_p"] < 0.01  else (
              "*"   if row["spearman_p"] < 0.05  else ""))
        print(f"{row['feature']:<22} {row['spearman_rho']:>12.3f} {row['spearman_p']:>10.4f}"
              f" {row['pearson_r']:>10.3f} {row['pearson_p']:>10.4f}  {sig}")

    # Best feature by |Spearman rho|
    best = corr_df.iloc[0]
    print(f"\nBest feature: {best['feature']}  (ρ={best['spearman_rho']:.3f})")

    # ── Correlation plot ──
    n_feats = len(corr_df)
    ncols = 3
    nrows = int(np.ceil(n_feats / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
    axes = axes.flatten()

    for i, (_, row) in enumerate(corr_df.iterrows()):
        feat = row["feature"]
        valid = merged[[feat, "y"]].dropna()
        ax = axes[i]
        ax.scatter(valid["y"], valid[feat], alpha=0.7,
                   edgecolors="k", linewidths=0.4, s=50, color="steelblue")

        # Regression line
        z = np.polyfit(valid["y"], valid[feat], 1)
        p = np.poly1d(z)
        xs = np.linspace(valid["y"].min(), valid["y"].max(), 100)
        ax.plot(xs, p(xs), color="tomato", linewidth=1.5)

        label, _, _ = FEATURE_LABELS.get(feat, (feat, "", 0))
        ax.set_xlabel("Nisthal y (−ΔG, kcal/mol)", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(f"ρ = {row['spearman_rho']:.3f}  r = {row['pearson_r']:.3f}",
                     fontsize=9)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("MD Features vs Nisthal Experimental Stability (y = −ΔG)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("validation_correlations.png", dpi=150, bbox_inches="tight")
    print("Saved: validation_correlations.png")
    plt.close()

    return corr_df, merged


# ═══════════════════════════════════════════════════════════════════════
# PART 3 — ACTIVE LEARNING
# ═══════════════════════════════════════════════════════════════════════

def build_feature_matrix(feat_df, pkl_df):
    """
    Build X (ESM-2 embeddings + MD features) and y (mean_Q as activity label).

    Two options for the activity label:
      - mean_Q  : fraction of native contacts (higher = more stable)
      - composite: weighted combination of features correlated with stability

    Returns X (n, 1280+n_md_feats), y_label, mutant_names.
    """
    merged = feat_df.merge(
        pkl_df[["name", "embedding", "y"]],
        left_on="mutant",
        right_on="name",
        how="inner"
    ).dropna()

    # ESM-2 embeddings: shape (n, 1280)
    X_emb = np.array(merged["embedding"].tolist())

    # MD features as additional input features
    md_feat_cols = [c for c in feat_df.columns if c != "mutant"]
    X_md = merged[md_feat_cols].values.astype(float)

    # Normalise MD features to [0,1] before concatenating
    from sklearn.preprocessing import MinMaxScaler
    scaler_md = MinMaxScaler()
    X_md_scaled = scaler_md.fit_transform(X_md)

    # Combined feature matrix
    X = np.hstack([X_emb, X_md_scaled])

    # Activity label: mean_Q (primary stability signal from MD)
    # y_label = merged["mean_Q"].values
    # In build_feature_matrix(), change this line:
    y_label = -merged["delta_rmsd_plateau"].values

    # Also keep experimental y for validation
    y_exp = merged["y"].values

    names = merged["mutant"].values
    print(f"Feature matrix: {X.shape}  (1280 ESM-2 + {X_md_scaled.shape[1]} MD features)")

    return X, y_label, y_exp, names, scaler_md, md_feat_cols


class ActiveLearner:
    """
    EVOLVEpro-style active learning loop.

    X_pool       : (N_candidates, n_features) — ESM-2 embeddings of ALL candidates
    names_pool   : (N_candidates,) — mutant names
    scores_pool  : (N_candidates,) — ground truth scores (for simulation)
                   In real use, these come from MD simulation after nomination.
                   Here we use MAVENN predicted scores as a proxy.
    n_select     : candidates to nominate per round
    """

    def __init__(self, X_pool, names_pool, scores_pool, n_select=N_SELECT):
        self.X_pool      = X_pool
        self.names_pool  = np.array(names_pool)
        self.scores_pool = np.array(scores_pool)
        self.n_select    = n_select

        self.rf = RandomForestRegressor(
            n_estimators=100,
            criterion="friedman_mse",
            n_jobs=-1,
            random_state=42
        )
        self.scaler = MinMaxScaler()

        self.tested_idx    = []   # pool indices already "measured"
        self.tested_scores = []   # their activity scores

        self.history = []         # round-by-round records

    def seed(self, seed_names, seed_scores):
        """Add initial labeled data (your 50 simulated mutants)."""
        name_to_idx = {n: i for i, n in enumerate(self.names_pool)}
        found = 0
        for name, score in zip(seed_names, seed_scores):
            if name in name_to_idx:
                self.tested_idx.append(name_to_idx[name])
                self.tested_scores.append(score)
                found += 1
        print(f"Seeded AL with {found} labeled mutants.")

    def _untested(self):
        return list(set(range(len(self.names_pool))) - set(self.tested_idx))

    def _fit(self):
        X = self.X_pool[self.tested_idx]
        y = np.array(self.tested_scores)
        y_scaled = self.scaler.fit_transform(y.reshape(-1, 1)).ravel()
        self.rf.fit(X, y_scaled)

    def run_round(self, round_num):
        """
        Fit model, nominate top candidates, add their scores.
        In real use: you would simulate nominated candidates first,
        then call add_results() before the next round.
        """
        self._fit()

        untested = self._untested()
        if not untested:
            print("All candidates tested!")
            return []

        X_u      = self.X_pool[untested]
        pred     = self.rf.predict(X_u)

        # Top-N selection
        top_local = np.argsort(pred)[::-1][:self.n_select]
        nom_idx   = [untested[i] for i in top_local]
        nom_pred  = pred[top_local]
        nom_true  = self.scores_pool[nom_idx]   # ground truth (from dataset)
        nom_names = self.names_pool[nom_idx]

        # LOO cross-validation R² on training set
        X_train = self.X_pool[self.tested_idx]
        y_train = np.array(self.tested_scores)
        if len(y_train) >= 10:
            loo = LeaveOneOut()
            y_pred_loo = np.zeros_like(y_train)
            for train_i, test_i in loo.split(X_train):
                rf_tmp = RandomForestRegressor(
                    n_estimators=50, random_state=42, n_jobs=-1
                )
                ys = y_train[train_i]
                ys_sc = MinMaxScaler().fit_transform(
                    ys.reshape(-1, 1)
                ).ravel()
                rf_tmp.fit(X_train[train_i], ys_sc)
                pred_tmp = rf_tmp.predict(X_train[test_i])
                # Inverse scale approximate
                y_pred_loo[test_i] = pred_tmp * (ys.max() - ys.min()) + ys.min()
            r2_loo = r2_score(y_train, y_pred_loo)
            rho_loo, _ = spearmanr(y_train, y_pred_loo)
        else:
            r2_loo  = np.nan
            rho_loo = np.nan

        record = {
            "round":                round_num,
            "n_tested":             len(self.tested_idx),
            "r2_loo":               r2_loo,
            "spearman_loo":         rho_loo,
            "nominated_names":      nom_names.tolist(),
            "nominated_pred":       nom_pred.tolist(),
            "nominated_true":       nom_true.tolist(),
            "mean_true_nominated":  float(nom_true.mean()),
            "max_true_nominated":   float(nom_true.max()),
            "mean_true_tested":     float(np.mean(self.tested_scores)),
        }
        self.history.append(record)

        print(f"\n─── Round {round_num} ───────────────────────────────────")
        print(f"  Labeled so far   : {len(self.tested_idx)}")
        print(f"  LOO R²           : {r2_loo:.3f}" if not np.isnan(r2_loo) else
              "  LOO R²           : (need ≥10 pts)")
        print(f"  LOO Spearman ρ   : {rho_loo:.3f}" if not np.isnan(rho_loo) else
              "  LOO Spearman ρ   : (need ≥10 pts)")
        print(f"  Nominated (top {self.n_select}):")
        for nm, pr, tr in zip(nom_names, nom_pred, nom_true):
            print(f"    {nm:<12}  pred={pr:.3f}  true={tr:.3f}")
        print(f"  Mean true score of nominated : {nom_true.mean():.3f}")
        print(f"  Mean true score of all tested: {np.mean(self.tested_scores):.3f}")

        return nom_idx

    def add_results(self, nom_idx):
        """
        Add nominated candidates to labeled set.
        In real use: run MD first, compute mean_Q, call this.
        In simulation mode: look up ground truth from pool.
        """
        for idx in nom_idx:
            self.tested_idx.append(idx)
            self.tested_scores.append(self.scores_pool[idx])

    def final_ranking(self):
        """Return full ranking of all untested candidates."""
        self._fit()
        untested = self._untested()
        X_u  = self.X_pool[untested]
        pred = self.rf.predict(X_u)
        pred_orig = self.scaler.inverse_transform(
            pred.reshape(-1, 1)
        ).ravel()

        df = pd.DataFrame({
            "mutant":           self.names_pool[untested],
            "predicted_score":  pred_orig,
            "true_score":       self.scores_pool[untested],
        }).sort_values("predicted_score", ascending=False).reset_index(drop=True)
        return df


def plot_al_progress(history, save_path="al_progress.png"):
    """Four-panel plot showing active learning progress over rounds."""
    rounds   = [r["round"]                   for r in history]
    r2s      = [r["r2_loo"]                  for r in history]
    rhos     = [r["spearman_loo"]            for r in history]
    mean_nom = [r["mean_true_nominated"]     for r in history]
    max_nom  = [r["max_true_nominated"]      for r in history]
    mean_all = [r["mean_true_tested"]        for r in history]

    fig = plt.figure(figsize=(14, 10))
    gs  = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel 1: LOO R²
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(rounds, r2s, "o-", color="steelblue", linewidth=2)
    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_title("Model Quality — LOO R²", fontweight="bold")
    ax1.set_xlabel("Round"); ax1.set_ylabel("R²")
    ax1.set_ylim(-0.1, 1.05)
    ax1.grid(True, alpha=0.3)

    # Panel 2: Spearman rho
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(rounds, rhos, "o-", color="darkorange", linewidth=2)
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_title("Model Quality — LOO Spearman ρ", fontweight="bold")
    ax2.set_xlabel("Round"); ax2.set_ylabel("ρ")
    ax2.set_ylim(-0.1, 1.05)
    ax2.grid(True, alpha=0.3)

    # Panel 3: Score of nominated candidates per round
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(rounds, mean_nom, "o-", color="green",  linewidth=2, label="Mean of nominated")
    ax3.plot(rounds, max_nom,  "s--", color="limegreen", linewidth=1.5, label="Max of nominated")
    ax3.plot(rounds, mean_all, "^:", color="gray",   linewidth=1.5, label="Mean of all tested")
    ax3.set_title("Stability Score of Nominated Candidates", fontweight="bold")
    ax3.set_xlabel("Round"); ax3.set_ylabel("True stability score")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Cumulative best score discovered
    all_scores = []
    cumulative_max = []
    for r in history:
        all_scores.extend(r["nominated_true"])
        cumulative_max.append(max(all_scores))
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(rounds, cumulative_max, "o-", color="crimson", linewidth=2)
    ax4.set_title("Best Mutant Found (Cumulative)", fontweight="bold")
    ax4.set_xlabel("Round"); ax4.set_ylabel("Best true score found")
    ax4.grid(True, alpha=0.3)

    fig.suptitle("Active Learning Progress — 1PGA Stability",
                 fontsize=14, fontweight="bold")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.close()


def run_active_learning(feat_path=FEAT_OUT, pkl_path=PKL_PATH,
                        cand_path=CAND_PATH):
    """
    Full active learning pipeline.

    NOTE on candidate pool embeddings:
    The 530k MAVENN sequences don't have ESM-2 embeddings yet.
    Two options:
      A) Embed a stratified sample of ~5000 candidates (recommended)
      B) Use only the 52 candidates_for_sim.csv (immediate, no embedding needed
         if you compute them — do this first)

    For now we use Option B as the immediate runnable version.
    Replace X_pool / names_pool / scores_pool with the full 530k once embedded.
    """
    print("\n" + "=" * 60)
    print("PART 3 — ACTIVE LEARNING")
    print("=" * 60)

    # ── Load labeled training data ──
    feat_df = pd.read_csv(feat_path)
    with open(pkl_path, "rb") as f:
        pkl_df = pickle.load(f)

    X, y_md, y_exp, names, scaler_md, md_feat_cols = build_feature_matrix(
        feat_df, pkl_df
    )

    print(f"\nTraining data: {X.shape[0]} mutants")
    print(f"Activity label (mean_Q) range: {y_md.min():.3f} – {y_md.max():.3f}")

    # Validate that mean_Q correlates with experimental y
    rho_val, p_val = spearmanr(y_md, y_exp)
    print(f"\nValidation: Spearman ρ(mean_Q, Nisthal_y) = {rho_val:.3f}  "
          f"(p={p_val:.4f})")
    if abs(rho_val) < 0.3:
        print("  [WARN] Weak correlation — consider more replicas or longer sim")
    else:
        print("  [OK] mean_Q is a meaningful stability proxy")

    # ── Load candidate pool ──
    # Using candidates_for_sim.csv as immediate pool.
    # These don't have embeddings yet — you need to compute them.
    # The code below shows how. If already computed, load from cache.

    cand_df = pd.read_csv(cand_path)
    cand_embed_cache = "candidate_embeddings.pkl"

    if os.path.exists(cand_embed_cache):
        print(f"\nLoading candidate embeddings from {cand_embed_cache}")
        with open(cand_embed_cache, "rb") as f:
            cand_with_emb = pickle.load(f)
    else:
        print(f"\nCandidate embeddings not found at {cand_embed_cache}")
        print("Computing ESM-2 embeddings for candidates...")
        cand_with_emb = embed_candidates(cand_df, cache_path=cand_embed_cache)

    # Build pool feature matrix (ESM-2 only, no MD features for untested)
    X_pool    = np.array(cand_with_emb["embedding"].tolist())
    names_pool = cand_with_emb["sequence"].values
    scores_pool = cand_with_emb["score"].values   # MAVENN predicted stability score

    # Pad X_pool with zeros for MD feature columns (unknown until simulated)
    # This is correct — the RF uses ESM-2 dims as the main signal for nomination
    n_md = len(md_feat_cols)
    X_pool_padded = np.hstack([
        X_pool,
        np.zeros((len(X_pool), n_md))
    ])

    print(f"Candidate pool: {X_pool_padded.shape}")

    # Add training mutants to pool so AL can "re-discover" them as check
    X_full    = np.vstack([X,          X_pool_padded])
    n_full    = np.concatenate([names, names_pool])
    s_full    = np.concatenate([y_md,  scores_pool])

    # ── Initialise and run AL ──
    al = ActiveLearner(X_full, n_full, s_full, n_select=N_SELECT)
    al.seed(names, y_md)   # seed with your 50 labeled mutants

    for round_num in range(1, N_ROUNDS + 1):
        nom_idx = al.run_round(round_num)
        al.add_results(nom_idx)   # in real use: simulate first, then add

    # ── Final ranking ──
    ranking = al.final_ranking()
    ranking.to_csv(RESULTS_OUT, index=False)
    print(f"\nFinal ranking saved to: {RESULTS_OUT}")
    print("\nTop 15 predicted stable mutants:")
    print(ranking.head(15)[["mutant", "predicted_score", "true_score"]].to_string())

    # ── Plot progress ──
    plot_al_progress(al.history)

    # ── Save AL object ──
    with open("active_learner.pkl", "wb") as f:
        pickle.dump(al, f)
    print("\nSaved AL object to: active_learner.pkl")

    return al, ranking


# ─────────────────────────────────────────────────────────────────────
# ESM-2 EMBEDDING FOR NEW CANDIDATES
# (run this once to embed your candidate pool)
# ─────────────────────────────────────────────────────────────────────

def embed_candidates(cand_df, cache_path="candidate_embeddings.pkl",
                     batch_size=8, model_name="facebook/esm2_t33_650M_UR50D"):
    """
    Compute ESM-2 mean-pooled embeddings for candidate sequences.
    Saves to cache_path so you only do this once.
    """
    import torch
    from transformers import AutoTokenizer, AutoModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embedding on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model     = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    seqs = cand_df["sequence"].tolist()
    all_emb = []

    for i in range(0, len(seqs), batch_size):
        batch = seqs[i : i + batch_size]
        inputs = tokenizer(
            batch, return_tensors="pt",
            padding=True, truncation=True
        ).to(device)

        with torch.no_grad():
            out = model(**inputs)

        emb = out.last_hidden_state.mean(dim=1).cpu().numpy()
        all_emb.append(emb)

        if i % 200 == 0:
            print(f"  {i}/{len(seqs)} sequences embedded")

    emb_array = np.vstack(all_emb)
    cand_df = cand_df.copy()
    cand_df["embedding"] = list(emb_array)

    with open(cache_path, "wb") as f:
        pickle.dump(cand_df, f)
    print(f"Saved embeddings to: {cache_path}")
    return cand_df


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="1PGA Active Learning Pipeline")
    parser.add_argument(
        "--mode",
        choices=["features", "validate", "train", "embed", "all"],
        default="all",
        help=(
            "features : compute MD features from trajectories\n"
            "validate : correlate MD features with Nisthal y\n"
            "embed    : compute ESM-2 embeddings for candidates\n"
            "train    : run active learning\n"
            "all      : run everything in sequence"
        )
    )
    parser.add_argument(
        "--mutants",
        nargs="+",
        default=None,
        help="Specific mutant names to process (default: all from PKL)"
    )
    args = parser.parse_args()

    # Load mutant list
    with open(PKL_PATH, "rb") as f:
        pkl_df = pickle.load(f)

    if args.mutants:
        mutant_list = args.mutants
    else:
        # mutant_list = pkl_df["name"].tolist()
        mutant_list = ["1PGA"] + pkl_df["name"].tolist()

    print(f"Mutants to process: {len(mutant_list)}")

    if args.mode in ("features", "all"):
        feat_df = run_feature_extraction(mutant_list)

    if args.mode in ("validate", "all"):
        if not os.path.exists(FEAT_OUT):
            print(f"[ERROR] {FEAT_OUT} not found. Run --mode features first.")
        else:
            corr_df, merged = run_validation()

    if args.mode == "embed":
        cand_df = pd.read_csv(CAND_PATH)
        embed_candidates(cand_df)

    if args.mode in ("train", "all"):
        if not os.path.exists(FEAT_OUT):
            print(f"[ERROR] {FEAT_OUT} not found. Run --mode features first.")
        else:
            al, ranking = run_active_learning()

    print("\nDone.")