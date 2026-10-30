"""
Knowledge-guided Unified Model for Understanding Design and encapsulation efficiency of Lipid Nanoparticles (KUMUDLNP)

Author: Kirti | IISc Bangalore | June 2026
Dataset: LNP Atlas v1 (Scientific Data, December 2025)

Class-based code; state that is genuinely used by a later section is carried on a shared PipelineState
object, everything else stays a local variable. 

Run end-to-end:
    python lnp_ee_prediction_oop.py

Run/inspect one section only, e.g. just the primary model:
    state = PipelineState()
    for cls in [Setup, DataCleaning, FeatureEngineering, FeatureAblation]:
        cls(state).run()
    PrimaryModel(state).run()

    python -c "from lnp_ee_prediction_oop import *; state = PipelineState(); [cls(state).run() for cls in [Setup, DataCleaning, FeatureEngineering, FeatureAblation]]; PrimaryModel(state).run()"F
"""

# ============================================================================
# imports
# ============================================================================
import pandas as pd
import numpy as np
import json, os
from datetime import datetime
import re
import pickle
import warnings
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import (
    rdFingerprintGenerator, Fragments, Descriptors, Lipinski, 
    rdMolDescriptors, rdFingerprintGenerator, MACCSkeys, AllChem)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, RandomForestRegressor
from sklearn.model_selection import (GroupKFold, StratifiedKFold, LeaveOneOut, 
                                     LeaveOneGroupOut, GroupShuffleSplit, train_test_split)
from sklearn.metrics import (roc_auc_score, average_precision_score, balanced_accuracy_score, r2_score,
    roc_curve, precision_recall_curve, confusion_matrix, brier_score_loss, ConfusionMatrixDisplay)
from sklearn.feature_selection import VarianceThreshold
from sklearn.dummy import DummyClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
from sklearn.metrics import mean_squared_error
import shap
from scipy import stats
from scipy.stats import wilcoxon, kruskal
from scipy.stats import chi2_contingency, spearmanr, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib
matplotlib.use('Agg')  # headless/non-interactive: script only saves figures, never plt.show()
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import font_manager
import seaborn as sns


# ============================================================================
# Shared pipeline state
# ============================================================================
class PipelineState:
    """
    Everything one Section produces that a LATER Section actually reads.
    Purely section-local variables (plot axes, loop temporaries, etc.) are
    never stored here -- they stay local to the method that uses them.
    """
    def __init__(self):
        self.PUB_PALETTE = None
        self.save_pub_figure = None
        self.clean_df = None
        self.df = None
        self.variability = None
        self.bl_df = None
        self.LIPID_COLS = None
        self.count_fp_array = None
        self.desc_matrix = None
        self.active_cols = None
        self.fp_matrix = None
        self.ratio_feats = None
        self.X_A = None
        self.X_B = None
        self.X_C = None
        self.X_D = None
        self.X_E = None
        self.y = None
        self.groups = None
        self.gkf = None
        self.ecfp_count_array = None
        self.build_fp_feature_set = None
        self.results_df = None
        self.all_y_true = None
        self.all_y_prob = None
        self.best_t = None
        self.compute_FQS = None
        self.MC3 = None
        self.DSPC = None
        self.CHOLESTEROL = None
        self.DMG_PEG = None
        self.bundle = None
        self.predict_lnp = None
        self.X_ion = None
        self.ion_fp_cols = None
        self.knn = None
        self.ad_threshold = None
        self.q_hat = None
        self.conformal_predict_set = None

# ============================================================================
# Section00 Setup
# ============================================================================
class Setup:
    """
    Global configuration and the publication figure style helper (imports live at module level).
    Writes to state:   PUB_PALETTE, save_pub_figure
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        #------------------------------ Setup (style/config) --------------------------------------
        warnings.filterwarnings('ignore')
        RDLogger.DisableLog('rdApp.*')
        print('All imports OK')
        print(f'shap version: {shap.__version__}')

        font_manager.fontManager.addfont("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-regular.otf")
        font_manager.fontManager.addfont("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-bold.otf")
        font_manager.fontManager.addfont("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-italic.otf")
        font_manager.fontManager.addfont("/usr/share/texmf/fonts/opentype/public/tex-gyre/texgyreheros-bolditalic.otf")

        MM_PER_INCH = 25.4
        COLW_SINGLE_MM = 88
        COLW_DOUBLE_MM = 180
        MAX_HEIGHT_MM = 170
        PUB_PALETTE = [
            "#6DB0D6",
            '#D55E00',
            '#009E73',
            "#D271A6",
            "#E2250425",
            "#075989",
            '#F0E442',
            '#000000',
            '#FF0000',
            '#1F77B4',
            '#008000',
            '#00FFFF',
            '#800080',
            "#E6AA39",
            '#FF00FF',
            "#F0700E"
        ]

        mpl.rcParams['text.usetex'] = False
        mpl.rcParams['text.latex.preamble'] = (
            r'\usepackage[cm]{sfmath}'
            r'\usepackage{amsmath}'
            r'\usepackage{amssymb}'
        )
        mpl.rcParams.update({

            # Fonts
            'font.family': 'sans-serif',
            'font.sans-serif': ['TeX Gyre Heros'],
            'font.size': 10,

            # Axes
            'axes.titlesize': 10,
            'axes.labelsize': 10,
            'axes.labelcolor': '#202020',
            'axes.edgecolor': '#6F6F6C',
            'axes.linewidth': 0.75,
            'axes.spines.top': False,
            'axes.spines.right': False,

            # Ticks
            'xtick.labelsize': 8,
            'ytick.labelsize': 8,
            'xtick.direction': 'out',
            'ytick.direction': 'out',
            'xtick.color': '#202020',
            'ytick.color': '#202020',
            'xtick.major.width': 0.75,
            'ytick.major.width': 0.75,

            # Lines
            'lines.linewidth': 1.5,

            # Legend
            'legend.fontsize': 8,
            'legend.frameon': False,

            # Colors
            'text.color': '#202020',
            'axes.prop_cycle': mpl.cycler(color=PUB_PALETTE),

            # Backgrounds
            'figure.facecolor': 'white',
            'axes.facecolor': 'white',
            'savefig.facecolor': 'white',

            # Output
            'savefig.dpi': 600,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })

        def save_pub_figure(
            fig,
            name,
            width='single',
            height_mm=None,
            formats=('png', 'svg')
        ):
            """
            Resize figure to journal column width and export.
            Parameters
            ----------
            fig : matplotlib.figure.Figure
                Figure handle.
            name : str
                Output filename stem (without extension).
            width : {'single', 'double'} or float
                Width in journal columns or custom mm.
            height_mm : float, optional
                Explicit height in mm. If None, aspect ratio is preserved.
            formats : tuple
                Output formats, e.g. ('pdf', 'png').
            """
            if width == 'single':
                width_mm = COLW_SINGLE_MM
            elif width == 'double':
                width_mm = COLW_DOUBLE_MM
            else:
                width_mm = float(width)

            cur_w_in, cur_h_in = fig.get_size_inches()
            aspect = cur_h_in / cur_w_in
            width_in = width_mm / MM_PER_INCH

            if height_mm is None:
                height_in = min(
                    width_in * aspect,
                    MAX_HEIGHT_MM / MM_PER_INCH
                )
            else:
                height_in = height_mm / MM_PER_INCH

            fig.tight_layout(pad=0.2)
            for fmt in formats:
                filename = f"{name}.{fmt}"
                fig.savefig(
                    filename,
                    format=fmt,
                    bbox_inches='tight',
                    dpi=600
                )
                print(
                    f"Saved {filename} "
                    f"({width_mm:.0f} mm * {height_in * MM_PER_INCH:.0f} mm)"
                )
            return fig
        print(
            "Publication figure style applied.\n"
            "Use save_pub_figure(fig, 'figure_name', width='single' or 'double')"
        )

        # expose to later sections
        self.state.PUB_PALETTE = PUB_PALETTE
        self.state.save_pub_figure = save_pub_figure
        return self.state

# ============================================================================
# Section01 DataCleaning
# ============================================================================
class DataCleaning:
    """
    Load the LNP Atlas v1 dataset and apply the data-cleaning rules used throughout the paper.
    Writes to state:   clean_df, df
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        # ── Load raw data ────────────────────────────────────────────────────────────
        df = pd.read_csv('LNP_Atlas_DB_202509_v1.csv', encoding='latin1')
        SMILES_COLS = [
            'ionizable_lipid_smiles', 'peg_lipid_smiles',
            'sterol_lipid_smiles',    'helper_lipid_smiles',
        ]
        # Replace blank strings with NaN
        for col in SMILES_COLS:
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
        # Keep rows with EE% and all 4 SMILES
        mask = (
            df['encapsulation_efficiency_percent_std'].notna()
            & df[SMILES_COLS].notna().all(axis=1)
        )
        ee_df = df[mask].copy()
        # Parse EE% — extract first numeric value
        def extract_ee(x):
            if pd.isna(x): return np.nan
            m = re.search(r'\d+\.?\d*', str(x))
            return float(m.group()) if m else np.nan
        ee_df['EE'] = ee_df['encapsulation_efficiency_percent_std'].apply(extract_ee)

        # Drop ambiguous rows
        clean_df = ee_df[~ee_df['encapsulation_efficiency_percent_std'].astype(str).str.contains(';',regex=False)].copy()
        clean_df = clean_df[~clean_df['encapsulation_efficiency_percent_std'].astype(str).str.contains('~',regex=False)].copy()

        # Robust ratio parsing — extract leading numeric:numeric:numeric:numeric
        clean_df['lipid_molar_ratio'] = clean_df['lipid_molar_ratio'].str.extract(r'^([0-9.:]+)')
        clean_df = clean_df[clean_df['lipid_molar_ratio'].str.count(':') == 3].copy()
        clean_df = clean_df[~clean_df['lipid_molar_ratio'].str.contains('Estimated', na=False)].copy()

        # Parse molar fractions
        split = clean_df['lipid_molar_ratio'].str.split(':', expand=True)
        clean_df['ion_ratio']    = split[0].astype(float)
        raw_pos2                 = split[1].astype(float)
        clean_df['sterol_ratio'] = split[2].astype(float)
        raw_pos4                 = split[3].astype(float)

        # FIX: `lipid_molar_ratio` does NOT follow one fixed component order
        # across the dataset -- different source papers (aggregated from ~63
        # papers into LNP Atlas) report positions 2/4 in different orders,
        # some as IL:PEG:sterol:helper, others as IL:helper:sterol:PEG.
        # Verified directly: the identical real formulation (DLin-MC3-DMA :
        # DSPC : cholesterol : DMG-PEG2000, IL 50% / helper 10% / sterol
        # 38.5% / PEG 1.5%) appears as BOTH "50:10:38.5:1.5" (paper
        # 10.1021/jacs.2c12893) and "50:1.5:38.5:10" (paper
        # 10.1038/s41467-022-33157-4) for the exact same components -- i.e.
        # positions 2 and 4 are swapped depending on source paper convention,
        # not a real compositional difference. Naively parsing position 2 as
        # PEG% mislabels ~39% of rows (positions where "PEG%" > "helper%" --
        # physically backwards, since PEG-lipid is virtually always the
        # minority component in real LNP formulations, typically 1-3 mol%).

        # Heuristic fix: assign the SMALLER of positions 2/4 to PEG and the
        # LARGER to helper. Not a substitute for manually re-checking all 63
        # source papers, but strongly supported empirically: after this fix
        # the PEG% distribution collapses from an implausible bimodal spread
        # (mean 10.7%, max 60%) to a tight, chemically sensible one (median
        # ~2.3%, IQR 1.5-2.5%, max 15%). Rare real formulations with
        # genuinely high PEG content could be mislabeled by this heuristic --
        # `ratio_was_reordered` flags every row it affected, for audit.
        clean_df['peg_ratio']    = np.minimum(raw_pos2, raw_pos4)
        clean_df['helper_ratio'] = np.maximum(raw_pos2, raw_pos4)
        clean_df['ratio_was_reordered'] = raw_pos2 > raw_pos4

        ratio_sum = clean_df[['ion_ratio','peg_ratio','sterol_ratio','helper_ratio']].sum(axis=1)
        for name in ['ion','peg','sterol','helper']:
            clean_df[f'{name}_frac'] = clean_df[f'{name}_ratio'] / ratio_sum

        # Validate all 4 SMILES with RDKit
        for col in SMILES_COLS:
            clean_df = clean_df[
                clean_df[col].apply(lambda x: Chem.MolFromSmiles(str(x)) is not None)
            ].copy()

        for col in SMILES_COLS:
            clean_df[col] = clean_df[col].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(str(x))))

        # Parse particle size and PDI (optional, used in FQS)
        def parse_first_number(x):
            if pd.isna(x): return np.nan
            m = re.search(r'(\d+\.?\d*)', str(x))
            return float(m.group(1)) if m else np.nan

        clean_df['size_nm'] = clean_df['particle_size_nm_std'].apply(parse_first_number)
        clean_df['PDI_val'] = clean_df['pdi_std'].apply(parse_first_number)

        clean_df = clean_df[clean_df['EE'].notna()].copy().reset_index(drop=True)

        print(f'Final dataset: {len(clean_df)} formulations')
        print(f'Unique ionizable lipid structures: {clean_df["ionizable_lipid_smiles"].nunique()}')
        print(f'Cargo type distribution:')
        print(clean_df['target_type'].value_counts(dropna=False))
        print(f'\nSize available: {clean_df["size_nm"].notna().sum()} rows')
        print(f'PDI available:  {clean_df["PDI_val"].notna().sum()} rows')
        print(f'\nEE% summary:')
        print(clean_df['EE'].describe().round(1))

        sterol_of_rem = clean_df['sterol_frac'] / (clean_df['sterol_frac'] + clean_df['helper_frac'])
        print(f"Sterol/(sterol+helper) median: {sterol_of_rem.median():.3f}")
        print(f"Sterol/(sterol+helper) mean:   {sterol_of_rem.mean():.3f}")

        # expose to later sections
        self.state.clean_df = clean_df
        self.state.df = df
        return self.state

# ============================================================================
# Section02 EDA
# ============================================================================
class EDA:
    """
    Exploratory data analysis figures.
    Reads from state:  PUB_PALETTE, save_pub_figure, clean_df
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        PUB_PALETTE = self.state.PUB_PALETTE
        save_pub_figure = self.state.save_pub_figure
        clean_df = self.state.clean_df

        fig, axes = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True)
        colors = PUB_PALETTE[:16]
        # EE% distribution
        axes[0].hist(clean_df['EE'], bins=20, color=colors[15], edgecolor='white', alpha=0.6)
        axes[0].axvline(clean_df['EE'].mean(),   color=colors[1], lw=1.5, ls='-', label=f'Mean={clean_df["EE"].mean():.1f}%')
        axes[0].axvline(clean_df['EE'].median(), color=colors[7], lw=1.5, ls='--', label=f'Median={clean_df["EE"].median():.1f}%')
        axes[0].axvline(80, color=colors[3], lw=1.5, ls=':', label='Threshold=80%')
        axes[0].set_xlabel('Encapsulation Efficiency (%)')
        axes[0].set_ylabel('Count')
        # axes[0].set_title('EE% Distribution')
        axes[0].legend()

        # EE% by cargo type
        cargo_order = clean_df['target_type'].value_counts().index.tolist()
        sns.boxplot(data=clean_df.dropna(subset=['target_type']), x='target_type', y='EE', order=cargo_order, ax=axes[1],
                    palette=PUB_PALETTE[:3], fliersize=4, linewidth=0.8)
        axes[1].set_xlabel('Cargo Type'); axes[1].set_ylabel('EE%')
        # axes[1].set_title('EE% by Cargo Type')

        # Molar fraction distributions
        for frac, color, label in [
            ('ion_frac',colors[0],'Ionizable'), ('peg_frac',colors[6],'PEG'),
            ('sterol_frac',colors[12],'Sterol'),  ('helper_frac',colors[15],'Helper')
        ]:
            axes[2].hist(clean_df[frac], bins=20, alpha=0.55, color=color, label=label, edgecolor='white')
        axes[2].set_xlabel('Molar Fraction'); axes[2].set_ylabel('Count')
        # axes[2].set_title('Molar Fraction Distributions'); 
        axes[2].legend()

        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec2_EDA', width='double')
        plt.show()

        return self.state


# ============================================================================
# Section03 NoiseFloor
# ============================================================================
class NoiseFloor:
    """
    Noise-floor analysis: how much of the EE% variance is even learnable given label noise.
    Reads from state:  PUB_PALETTE, save_pub_figure, clean_df
    Writes to state:   variability
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        PUB_PALETTE = self.state.PUB_PALETTE
        save_pub_figure = self.state.save_pub_figure
        clean_df = self.state.clean_df

        variability = (
            clean_df.groupby('ionizable_lipid_smiles')['EE']
            .agg(['count','mean','std'])
            .rename(columns={'count':'n_obs','mean':'mean_EE','std':'std_EE'})
            .sort_values('n_obs', ascending=False)
        )
        multi = variability[variability['n_obs'] >= 2]

        print(f'Lipids with >= 2 observations: {len(multi)}')
        print(f'Median intra-lipid EE sigma:   {multi["std_EE"].median():.1f}%  ← the noise floor')
        print(f'Mean   intra-lipid EE sigma:   {multi["std_EE"].mean():.1f}%')
        print(f'\nConclusion: no regression model can achieve MAE < ~{multi["std_EE"].median():.1f}% on this data.')
        print(f'Classification (high/low) is the correct framing.')
        colors = PUB_PALETTE[:16]
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
        axes[0].hist(multi['std_EE'].dropna(), bins=20, color=colors[0], edgecolor='white')
        axes[0].axvline(multi['std_EE'].median(), color=colors[1], lw=1.5, ls='--',
                        label=f'Median σ={multi["std_EE"].median():.1f}%')
        axes[0].set_xlabel('Intra-lipid EE $\\sigma$ (%) — for lipids with $n\\geq2$ labs')
        axes[0].set_ylabel('Count')
        # axes[0].set_title('Within-Lipid Variability = Noise Floor'); 
        axes[0].legend()

        axes[1].scatter(multi['n_obs'], multi['std_EE'], alpha=0.5, s=20, color=colors[2])
        axes[1].set_xlabel('Number of observations per ionizable lipid')
        axes[1].set_ylabel('Intra-lipid EE $\\sigma$ (%)') 
        # axes[1].set_title('Variability vs Observations')

        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec3_intra_lipid_noise_analysis', width='double')
        plt.show()

        # expose to later sections
        self.state.variability = variability
        return self.state


# ============================================================================
# Section04 RigorousNoiseFloor
# ============================================================================
class RigorousNoiseFloor:
    """
    Rigorous (replicate-based) noise-floor analysis, and the leave-one-publication-out baseline data.
    Reads from state:  PUB_PALETTE, save_pub_figure, clean_df, variability
    Writes to state:   bl_df
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        """Execute this section (notebook cells [5, 6, 7]) and update shared state."""
        PUB_PALETTE = self.state.PUB_PALETTE
        save_pub_figure = self.state.save_pub_figure
        clean_df = self.state.clean_df
        variability = self.state.variability

        # ── Find formulations that appear in multiple publications ──────────────────
        # Group by ALL 4 SMILES + molar ratio (same exact formulation)
        form_key = clean_df[[
            'ionizable_lipid_smiles','helper_lipid_smiles',
            'sterol_lipid_smiles','peg_lipid_smiles','lipid_molar_ratio'
        ]].apply(lambda r: '|'.join(r.astype(str)), axis=1)
        clean_df['formulation_key'] = form_key

        # Group by key + DOI (proxy for paper)
        form_paper = clean_df.groupby(['formulation_key','paper_doi'])['EE'].mean().reset_index()
        form_count = form_paper.groupby('formulation_key').size().reset_index(name='n_papers')
        multi_form = form_count[form_count['n_papers'] >= 2]['formulation_key']

        # For each multi-paper formulation compute between-lab stats
        between_lab = []
        for key in multi_form:
            subset = form_paper[form_paper['formulation_key'] == key]
            ee_vals = subset['EE'].values
            between_lab.append({
                'formulation_key': key[:60] + '...',
                'n_papers': len(subset),
                'mean_EE':  ee_vals.mean(),
                'std_EE':   ee_vals.std(),
                'cv_pct':   (ee_vals.std() / ee_vals.mean() * 100) if ee_vals.mean() > 0 else np.nan,
                'range':    ee_vals.max() - ee_vals.min(),
                'values':   list(np.round(ee_vals, 1)),
            })

        bl_df = pd.DataFrame(between_lab).sort_values('std_EE', ascending=False)

        print(f'Formulations measured in ≥2 papers: {len(bl_df)}')
        print(bl_df[['n_papers', 'mean_EE', 'std_EE']])
        print(f'\nBetween-lab EE% variability:')
        print(f'  Median std:   {bl_df["std_EE"].median():.1f}%')
        print(f'  Median CV:    {bl_df["cv_pct"].median():.1f}%')
        print(f'  Median range: {bl_df["range"].median():.1f}%')
        print(f'\nWorst offenders (highest between-lab std):')
        print(bl_df.head(8)[['n_papers','mean_EE','std_EE','cv_pct','values']].to_string())


        # --------------------------- Mixed-effects variance partitioning ---------------------------
        print("\n" + "="*80)
        print("MIXED-EFFECTS VARIANCE PARTITIONING")
        print("="*80)


        # --------------------------- Ionizable lipid identity ---------------------------
        ion_model = smf.mixedlm("EE ~ 1", data=clean_df, groups=clean_df["ionizable_lipid_smiles"])
        ion_res = ion_model.fit(reml=True)
        ion_var = float(ion_res.cov_re.iloc[0, 0])
        ion_resid = float(ion_res.scale)
        ion_icc = ion_var / (ion_var + ion_resid)

        print("\nIonizable lipid identity")
        print(f"Between-lipid variance   = {ion_var:.2f}")
        print(f"Residual variance        = {ion_resid:.2f}")
        print(f"ICC                      = {ion_icc:.3f}")
        print(f"Variance explained       = {ion_icc*100:.1f}%")

        # --------------------------- Cargo identity ---------------------------
        cargo_model = smf.mixedlm("EE ~ 1", data=clean_df, groups=clean_df["target_type"])
        cargo_res = cargo_model.fit(reml=True)
        cargo_var = float(cargo_res.cov_re.iloc[0, 0])
        cargo_resid = float(cargo_res.scale)
        cargo_icc = cargo_var / (cargo_var + cargo_resid)

        print("\nCargo identity")
        print(f"Cargo variance           = {cargo_var:.2f}")
        print(f"Residual variance        = {cargo_resid:.2f}")
        print(f"ICC                      = {cargo_icc:.3f}")
        print(f"Variance explained       = {cargo_icc*100:.1f}%")

        # --------------------------- Exact formulation identity ---------------------------
        clean_df["formulation_id"] = (
            clean_df["ionizable_lipid_smiles"].astype(str) + "|" +
            clean_df["helper_lipid_smiles"].astype(str) + "|" +
            clean_df["sterol_lipid_smiles"].astype(str) + "|" +
            clean_df["peg_lipid_smiles"].astype(str) + "|" +
            clean_df["lipid_molar_ratio"].astype(str))

        form_model = smf.mixedlm("EE ~ 1", data=clean_df, groups=clean_df["formulation_id"])
        form_res = form_model.fit(reml=True)
        form_var = float(form_res.cov_re.iloc[0, 0])
        form_resid = float(form_res.scale)
        form_icc = form_var / (form_var + form_resid)

        print("\nExact formulation identity")
        print(f"Formulation variance     = {form_var:.2f}")
        print(f"Residual variance        = {form_resid:.2f}")
        print(f"ICC                      = {form_icc:.3f}")
        print(f"Variance explained       = {form_icc*100:.1f}%")

        intra_lipid_sigma = variability.loc[variability["n_obs"] >= 2, "std_EE"].median()
        between_lab_sigma = bl_df["std_EE"].median()
        cargo_pct = 100 * cargo_icc
        ionizable_pct = 100 * ion_icc
        formulation_pct = 100 * form_icc

        print("\nFigure values")
        print(f"Intra-lipid sigma      : {intra_lipid_sigma:.1f}%")
        print(f"Between-lab sigma      : {between_lab_sigma:.1f}%")
        print(f"Cargo ICC             : {cargo_pct:.1f}%")
        print(f"Ionizable ICC         : {ionizable_pct:.1f}%")
        print(f"Formulation ICC       : {formulation_pct:.1f}%")

        # --------------------------- Figure: Noise floor and variance partitioning ---------------------------
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
        colors = PUB_PALETTE[:16]

        # Panel A
        noise_labels = ['Same ionizable\nlipid', 'Exact formulation\n(across papers)']
        noise_vals = [intra_lipid_sigma, between_lab_sigma]
        bars = axes[0].bar(noise_labels, noise_vals, color=[colors[0], colors[3]], edgecolor='white', alpha=0.7)
        for b, v in zip(bars, noise_vals):
            axes[0].text(b.get_x() + b.get_width()/2, v + 0.3, f'{v:.1f}%',ha='center')
        axes[0].set_ylabel(f'EE variability ($\\sigma$, %)')
        # axes[0].set_title('Empirical measurement variability')

        # Panel B
        labels = ["Cargo identity", "Ionizable lipid identity", "Exact formulation identity"]
        vals = [cargo_pct, ionizable_pct, formulation_pct]
        axes[1].barh(labels,vals,color=[colors[6], colors[1], colors[2]],edgecolor='white',alpha=0.75)
        for y, v in enumerate(vals):
            axes[1].text(v + 1, y, f"{v:.1f}%", va="center")
        axes[1].set_xlim(0, 60)
        axes[1].set_xlabel("Variance explained (%)")
        axes[1].set_title("Random-effects variance partitioning")

        save_pub_figure(fig, 'fig_variance_partition_and_noise_floor', width='double')
        plt.show()

        # --------------------------- Plot the distribution of between-lab std per formulation ---------------------------
        fig, axes = plt.subplots(figsize=(4, 3))
        colors = PUB_PALETTE[:8]
        # Right: distribution of between-lab std per formulation
        axes.hist(bl_df['std_EE'].dropna(), bins=10, color=colors[5], edgecolor='white', alpha=0.6)
        axes.axvline(bl_df['std_EE'].median(), color=colors[3], lw=1.5, ls='--',
                        label=f'Median $\\sigma$ = {bl_df["std_EE"].median():.1f}%')
        axes.set_xlabel('Between-lab EE% $\\sigma$ (same formulation, different papers)')
        axes.set_ylabel('Count')
        # axes[1].set_title('Between-Lab Reproducibility\n(formulations measured in ≥2 papers)', fontsize=12)
        axes.legend()
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec4_variance_decomposition_across_lab', width='single')
        plt.show()

        # expose to later sections
        self.state.bl_df = bl_df
        return self.state


# ============================================================================
# Section05 FeatureEngineering
# ============================================================================
class FeatureEngineering:
    """
    Build molecular fingerprints, physicochemical descriptors, and molar-fraction features; assemble Feature Sets A-E.
    Reads from state:  clean_df
    Writes to state:   LIPID_COLS, count_fp_array, desc_matrix, active_cols, fp_matrix, ratio_feats, X_A, X_B, X_C, X_D, X_E, y, groups
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        clean_df = self.state.clean_df

        LIPID_COLS = {
            'ionizable': 'ionizable_lipid_smiles',
            'helper':    'helper_lipid_smiles',
            'sterol':    'sterol_lipid_smiles',
            'peg':       'peg_lipid_smiles',
        }

        # Initialise fingerprint generator once
        fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=512)

        def count_fp_array(smiles, nbits=512):
            """Count-based Morgan fingerprint as numpy array."""
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None:
                return np.zeros(nbits, dtype=np.float32)
            fp = fpgen.GetCountFingerprint(mol)
            arr = np.zeros(nbits, dtype=np.float32)
            for idx, cnt in fp.GetNonzeroElements().items():
                arr[idx % nbits] += cnt  # fold into nbits
            return arr

        def calc_desc(smiles):
            """8 physicochemical RDKit descriptors."""
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None: return {}
            return {
                'MolWt': Descriptors.MolWt(mol), 'LogP': Descriptors.MolLogP(mol),
                'TPSA': rdMolDescriptors.CalcTPSA(mol), 'HBA': Lipinski.NumHAcceptors(mol),
                'HBD': Lipinski.NumHDonors(mol), 'RotBonds': Lipinski.NumRotatableBonds(mol),
                'RingCount': Lipinski.RingCount(mol), 'FractionCSP3': Lipinski.FractionCSP3(mol)
            }

        # Build fingerprint matrix
        fp_parts, desc_parts = [], []
        for prefix, col in LIPID_COLS.items():
            fps = np.vstack(clean_df[col].apply(count_fp_array).values)
            fp_parts.append(pd.DataFrame(fps, columns=[f'{prefix}_fp{i}' for i in range(512)]))
            desc = clean_df[col].apply(calc_desc).apply(pd.Series)
            desc.columns = [f'{prefix}_{c}' for c in desc.columns]
            desc_parts.append(desc)
            print(f'  {prefix}: FP shape={fps.shape}')

        fp_matrix = pd.concat(fp_parts, axis=1).reset_index(drop=True)
        active_cols = fp_matrix.columns[fp_matrix.sum(axis=0) > 0].tolist()
        fp_matrix   = fp_matrix[active_cols]

        desc_matrix = pd.concat(desc_parts, axis=1).reset_index(drop=True)
        vt = VarianceThreshold(0.0); vt.fit(desc_matrix)
        desc_matrix = desc_matrix.loc[:, vt.get_support()]

        ratio_feats = clean_df[['ion_frac','peg_frac','sterol_frac','helper_frac']].reset_index(drop=True)

        # ── Assemble feature sets for ablation ────────────────────────────────────────
        ion_fp = fp_matrix[[c for c in fp_matrix.columns if c.startswith('ionizable_')]]

        X_A = ratio_feats.copy()                                                        # ratios only
        X_B = pd.concat([desc_matrix, ratio_feats], axis=1)                             # desc + ratios
        X_C = pd.concat([ion_fp, ratio_feats], axis=1)                                  # ionizable FP only + ratios
        X_D = pd.concat([fp_matrix, ratio_feats], axis=1)                               # all-4 FP + ratios [MAIN]
        X_E = pd.concat([fp_matrix, desc_matrix, ratio_feats], axis=1)                  # full

        y      = (clean_df['EE'].values >= 80).astype(int)
        y_reg  = clean_df['EE'].values
        groups = clean_df['ionizable_lipid_smiles'].values

        print(f'\nActive FP bits: {len(active_cols)} / 2048')
        print(f'Class balance: {y.mean():.1%} High EE (>=80%)')
        for label, X in [('A ratios only',X_A),('B desc+ratios',X_B),('C ion-FP only + ratios',X_C),
                          ('D all4-FP+ratios [MAIN]',X_D),('E full',X_E)]:
            print(f'  {label:<30} {X.shape[1]:>5} features')

        # expose to later sections
        self.state.LIPID_COLS = LIPID_COLS
        self.state.count_fp_array = count_fp_array
        self.state.desc_matrix = desc_matrix
        self.state.active_cols = active_cols
        self.state.fp_matrix = fp_matrix
        self.state.ratio_feats = ratio_feats
        self.state.X_A = X_A
        self.state.X_B = X_B
        self.state.X_C = X_C
        self.state.X_D = X_D
        self.state.X_E = X_E
        self.state.y = y
        self.state.groups = groups
        return self.state


# ============================================================================
# Section06 FeatureAblation
# ============================================================================
class FeatureAblation:
    """
    Feature-set ablation (A-E) and Monte Carlo label-noise-injection robustness check.
    Reads from state:  PUB_PALETTE, save_pub_figure, clean_df, bl_df, X_A, X_B, X_C, X_D, X_E, y, groups
    Writes to state:   gkf
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        """Execute this section (notebook cells [9, 10, 11]) and update shared state."""
        PUB_PALETTE = self.state.PUB_PALETTE
        save_pub_figure = self.state.save_pub_figure
        clean_df = self.state.clean_df
        bl_df = self.state.bl_df
        X_A = self.state.X_A
        X_B = self.state.X_B
        X_C = self.state.X_C
        X_D = self.state.X_D
        X_E = self.state.X_E
        y = self.state.y
        groups = self.state.groups

        gkf = GroupKFold(n_splits=5)
        rfc_abl = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)

        ablation_results = []
        print(f'{"Feature Set":<38}  n_feat  AUC (mean±std)     AP (mean±std)')
        print('-'*85)

        for label, Xabl in [
            ('A: Molar ratios only',           X_A),
            ('B: RDKit desc (all 4) + ratios', X_B),
            ('C: Morgan FP (ionizable only) + ratios',  X_C),
            ('D: Morgan FP (all 4) + ratios', X_D),
            ('E: FP + desc + ratios (full)',   X_E),
        ]:
            aucs, aps = [], []
            for tr, te in gkf.split(Xabl, y, groups):
                rfc_abl.fit(Xabl.iloc[tr], y[tr])
                prob = rfc_abl.predict_proba(Xabl.iloc[te])[:,1]
                aucs.append(roc_auc_score(y[te], prob))
                aps.append(average_precision_score(y[te], prob))
            ablation_results.append({'label':label,'n':Xabl.shape[1],
                                      'auc_m':np.mean(aucs),'auc_s':np.std(aucs),
                                      'ap_m':np.mean(aps),'ap_s':np.std(aps)})
            print(f'{label:<38}  {Xabl.shape[1]:<5}  {np.mean(aucs):.3f}±{np.std(aucs):.3f}     {np.mean(aps):.3f}±{np.std(aps):.3f}')

        # --------------------------- Statistical comparison: Set D vs Set E ---------------------------

        D_fold_aucs = []
        E_fold_aucs = []
        rf_compare = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)

        for tr, te in gkf.split(X_D, y, groups):
            # Set D
            rf_compare.fit(X_D.iloc[tr], y[tr])
            prob_D = rf_compare.predict_proba(X_D.iloc[te])[:, 1]
            D_fold_aucs.append(roc_auc_score(y[te], prob_D))

            # Set E
            rf_compare.fit(X_E.iloc[tr], y[tr])
            prob_E = rf_compare.predict_proba(X_E.iloc[te])[:, 1]
            E_fold_aucs.append(roc_auc_score(y[te], prob_E))

        D_fold_aucs = np.array(D_fold_aucs)
        E_fold_aucs = np.array(E_fold_aucs)
        delta = E_fold_aucs - D_fold_aucs

        try:
            stat, p = wilcoxon(delta)
        except ValueError:
            stat, p = np.nan, np.nan

        print("\n=== Set D vs Set E Comparison ===")
        print(f"D fold AUCs: {np.round(D_fold_aucs, 3)}")
        print(f"E fold AUCs: {np.round(E_fold_aucs, 3)}")
        print(f"Mean $\\Delta$ AUC (E-D): {delta.mean():+.4f}")
        print(f"Wilcoxon p-value: {p:.3f}")

        if p < 0.05:
            print("Difference is statistically significant.")
        else:
            print("Difference is NOT statistically significant.")

        # --------------------------- Plot ablation ---------------------------
        fig, ax = plt.subplots(figsize=(7, 3), constrained_layout=True)
        labels_short = [r['label'].split(':')[0] for r in ablation_results]
        aucs_m = [r['auc_m'] for r in ablation_results]
        aucs_s = [r['auc_s'] for r in ablation_results]
        colors = PUB_PALETTE[:16]
        bars = ax.bar(labels_short, aucs_m, yerr=aucs_s, capsize=5, color=colors, edgecolor='white', alpha=0.6)
        for bar, val in zip(bars, aucs_m):
            ax.text(bar.get_x()+bar.get_width()/2+0.13, bar.get_height()+0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        ax.axhline(0.5, ls='--', color=colors[7], lw=1, label='Random classifier (AUC=0.5)')
        ax.set_ylabel('ROC-AUC (5-Fold GroupKFold, lipid-held-out)')
        # ax.set_title('Feature Set Ablation — Binary EE% Classification (High EE ≥ 80%)', fontsize=12)
        ax.set_ylim([0.35, 0.95]); ax.legend()
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec6_ablation', width='double')
        plt.show()

        # --------------------------- Inject Gaussian noise into EE% labels ---------------------------
        # Regression R² collapses; classification AUC stays
        noise_levels = [0, 2, 4, 6, 8, 10, 11, 12, 14, 16, 18, 20]     # % noise sigma to add
        n_monte_carlo = 20                                             # repeats per noise level

        mc_auc_means, mc_auc_stds = [], []
        mc_r2_means,  mc_r2_stds  = [], []

        np.random.seed(42)
        for sigma in noise_levels:
            aucs_mc, r2s_mc = [], []
            for trial in range(n_monte_carlo):
                # Add Gaussian noise to EE%
                noisy_ee = clean_df['EE'].values + np.random.normal(0, sigma, len(clean_df))
                noisy_ee = np.clip(noisy_ee, 0, 100)
                y_noisy  = (noisy_ee >= 80).astype(int)

                # Classification AUC
                rfc_mc = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                                 random_state=42, n_jobs=-1)
                aucs_fold = []
                for tr, te in gkf.split(X_D, y_noisy, groups):
                    rfc_mc.fit(X_D.iloc[tr], y_noisy[tr])
                    prob_mc = rfc_mc.predict_proba(X_D.iloc[te])[:,1]
                    try:
                        aucs_fold.append(roc_auc_score(y_noisy[te], prob_mc))
                    except:
                        pass
                aucs_mc.append(np.mean(aucs_fold))

                # Regression R²
                rfr_mc = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
                r2s_fold = []
                for tr, te in gkf.split(X_D, noisy_ee, groups):
                    rfr_mc.fit(X_D.iloc[tr], noisy_ee[tr])
                    pred_mc = rfr_mc.predict(X_D.iloc[te])
                    r2s_fold.append(r2_score(noisy_ee[te], pred_mc))
                r2s_mc.append(np.mean(r2s_fold))

            mc_auc_means.append(np.mean(aucs_mc))
            mc_auc_stds.append(np.std(aucs_mc))
            mc_r2_means.append(np.mean(r2s_mc))
            mc_r2_stds.append(np.std(r2s_mc))
            print(f"$\\sigma$={sigma:>2}%  AUC={np.mean(aucs_mc):.3f}±{np.std(aucs_mc):.3f}  R²={np.mean(r2s_mc):.3f}±{np.std(r2s_mc):.3f}")

        mc_auc_means = np.array(mc_auc_means); mc_auc_stds = np.array(mc_auc_stds)
        mc_r2_means  = np.array(mc_r2_means);  mc_r2_stds  = np.array(mc_r2_stds)

        # --------------------------- Plot Monte Carlo noise injection results ---------------------------
        fig, ax = plt.subplots(figsize=(7, 3))
        colors = PUB_PALETTE[8:16]
        ax2 = ax.twinx()

        ax.plot(noise_levels, mc_auc_means, '^--', color=colors[8], lw=1.5, ms=4, label='Classification AUC (left)')
        ax.fill_between(noise_levels, mc_auc_means-mc_auc_stds, mc_auc_means+mc_auc_stds, alpha=0.08, color=colors[8])
        ax.axhline(0.5, ls='--', color=colors[5], lw=1, alpha=0.8)
        ax.set_ylabel('ROC-AUC (classification)', color=colors[8])
        ax.tick_params(axis='y', labelcolor=colors[8])
        ax.set_ylim([0.4, 0.80])

        ax2.plot(noise_levels, mc_r2_means, 's--', color=colors[4], lw=1.5, ms=4, label='Regression $R^2$ (right)')
        ax2.fill_between(noise_levels, mc_r2_means-mc_r2_stds, mc_r2_means+mc_r2_stds, alpha=0.08, color=colors[4])
        ax2.axhline(0.0, ls='--', color=colors[4], lw=1, alpha=0.8)
        ax2.set_ylabel('$R^2$ (regression)', color=colors[4])
        ax2.tick_params(axis='y', labelcolor=colors[4])

        # Mark the empirical noise floor
        ax.axvline(bl_df['std_EE'].median(), ls=':', color=colors[7], lw=1,
                   label=f'Median between-lab EE standard deviation ({bl_df['std_EE'].median():.1f}%)')

        ax.set_xlabel('Added Gaussian noise $\\sigma$ (% EE)')
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, loc='upper right')
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec4_monte_carlo_noise', width='double')
        plt.show()
    
        # expose to later sections
        self.state.gkf = gkf
        return self.state


# ============================================================================
# Section07 FingerprintBenchmark
# ============================================================================
class FingerprintBenchmark:
    """
    Benchmark fingerprint / molecular representation choices against each other.
    Also benchmarks a frozen, pretrained ChemBERTa embedding under the same
    protocol, if `transformers`/`torch` are installed and a HuggingFace Hub
    checkpoint download succeeds -- this directly answers the "why fixed
    fingerprints instead of learned embeddings?" question with an empirical
    row in the same table, rather than by argument alone. Optional: this
    section runs fine without it, just without that one extra row.

    Reads from state:  PUB_PALETTE, save_pub_figure, clean_df, LIPID_COLS, desc_matrix, ratio_feats, X_E, y, groups, gkf
    Writes to state:   ecfp_count_array, build_fp_feature_set
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        PUB_PALETTE = self.state.PUB_PALETTE
        save_pub_figure = self.state.save_pub_figure
        clean_df = self.state.clean_df
        LIPID_COLS = self.state.LIPID_COLS
        desc_matrix = self.state.desc_matrix
        ratio_feats = self.state.ratio_feats
        X_E = self.state.X_E
        y = self.state.y
        groups = self.state.groups
        gkf = self.state.gkf

        def bitvect_to_array(fp, nbits):
            arr = np.zeros((nbits,), dtype=np.float32)
            DataStructs.ConvertToNumpyArray(fp, arr)
            return arr

        def ecfp_count_array(smiles, radius, nbits=512):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None: return np.zeros(nbits, dtype=np.float32)
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
            fp = gen.GetCountFingerprint(mol)
            arr = np.zeros(nbits, dtype=np.float32)
            for idx, cnt in fp.GetNonzeroElements().items():
                arr[idx % nbits] += cnt
            return arr

        def maccs_array(smiles):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None: return np.zeros(167, dtype=np.float32)
            return bitvect_to_array(MACCSkeys.GenMACCSKeys(mol), 167)

        def atompair_array(smiles, nbits=512):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None: return np.zeros(nbits, dtype=np.float32)
            fp = rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, nBits=nbits)
            return bitvect_to_array(fp, nbits)

        def torsion_array(smiles, nbits=512):
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None: return np.zeros(nbits, dtype=np.float32)
            fp = rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(mol, nBits=nbits)
            return bitvect_to_array(fp, nbits)

        FP_BUILDERS = {
            'ECFP4 (r=2)':                lambda s: ecfp_count_array(s, radius=2, nbits=512),
            'ECFP6 (r=3)':                lambda s: ecfp_count_array(s, radius=3, nbits=512),
            'MACCS (167 keys)':           maccs_array,
            'Atom-pair (512b)':           atompair_array,
            'Topological torsion (512b)': torsion_array,
        }

        def build_fp_feature_set(builder_fn):
            parts = []
            for prefix, col in LIPID_COLS.items():
                arrs = np.vstack(clean_df[col].apply(builder_fn).values)
                cols = [f'{prefix}_fp{i}' for i in range(arrs.shape[1])]
                parts.append(pd.DataFrame(arrs, columns=cols))
            mat = pd.concat(parts, axis=1).reset_index(drop=True)
            active = mat.columns[mat.sum(axis=0) > 0].tolist()
            return mat[active]

        fp_bench_results = []
        gkf_bench = GroupKFold(n_splits=5)

        for fp_name, builder in FP_BUILDERS.items():
            print(f'Building {fp_name} ...')
            fp_mat = build_fp_feature_set(builder)
            X_fp   = pd.concat([fp_mat, ratio_feats], axis=1)
            aucs, aps = [], []
            rfc_bench = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                                random_state=42, n_jobs=-1)
            for tr, te in gkf_bench.split(X_fp, y, groups):
                rfc_bench.fit(X_fp.iloc[tr], y[tr])
                prob = rfc_bench.predict_proba(X_fp.iloc[te])[:, 1]
                aucs.append(roc_auc_score(y[te], prob))
                aps.append(average_precision_score(y[te], prob))
            fp_bench_results.append({'Fingerprint': fp_name, 'n_features': X_fp.shape[1],
                                      'AUC_mean': np.mean(aucs), 'AUC_std': np.std(aucs),
                                      'AP_mean': np.mean(aps), 'AP_std': np.std(aps)})

        # Reference points already computed above: descriptors-only, and ECFP6+descriptors (Set E)
        for label, X_ref in [('RDKit descriptors only', pd.concat([desc_matrix, ratio_feats], axis=1)),
                              ('ECFP6 + RDKit desc (Set E)', X_E)]:
            aucs, aps = [], []
            rfc_bench = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                                random_state=42, n_jobs=-1)
            for tr, te in gkf_bench.split(X_ref, y, groups):
                rfc_bench.fit(X_ref.iloc[tr], y[tr])
                prob = rfc_bench.predict_proba(X_ref.iloc[te])[:, 1]
                aucs.append(roc_auc_score(y[te], prob))
                aps.append(average_precision_score(y[te], prob))
            fp_bench_results.append({'Fingerprint': label, 'n_features': X_ref.shape[1],
                                      'AUC_mean': np.mean(aucs), 'AUC_std': np.std(aucs),
                                      'AP_mean': np.mean(aps), 'AP_std': np.std(aps)})

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel

            CHEMBERTA_MODEL = 'seyonec/ChemBERTa-zinc-base-v1'
            print(f'\nLoading pretrained {CHEMBERTA_MODEL} (frozen, no fine-tuning)...')
            _tokenizer = AutoTokenizer.from_pretrained(CHEMBERTA_MODEL)
            _chemberta = AutoModel.from_pretrained(CHEMBERTA_MODEL)
            _chemberta.eval()

            @torch.no_grad()
            def chemberta_embed_array(smiles):
                """Mean-pooled last-hidden-state embedding for one SMILES string."""
                mol = Chem.MolFromSmiles(str(smiles))
                if mol is None:
                    return np.zeros(_chemberta.config.hidden_size, dtype=np.float32)
                inputs = _tokenizer(str(smiles), return_tensors='pt', truncation=True, max_length=128)
                outputs = _chemberta(**inputs)
                mask = inputs['attention_mask'].unsqueeze(-1).float()
                pooled = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1).clamp(min=1)
                return pooled.squeeze(0).numpy().astype(np.float32)

            print('Embedding all 4 lipid components with ChemBERTa (this takes a while, no GPU needed for this dataset size)...')
            embed_parts = []
            for prefix, col in LIPID_COLS.items():
                arrs = np.vstack(clean_df[col].apply(chemberta_embed_array).values)
                cols = [f'{prefix}_emb{i}' for i in range(arrs.shape[1])]
                embed_parts.append(pd.DataFrame(arrs, columns=cols))
            X_embed = pd.concat(embed_parts + [ratio_feats.reset_index(drop=True)], axis=1)

            aucs, aps = [], []
            rfc_embed = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                                random_state=42, n_jobs=-1)
            for tr, te in gkf_bench.split(X_embed, y, groups):
                rfc_embed.fit(X_embed.iloc[tr], y[tr])
                prob = rfc_embed.predict_proba(X_embed.iloc[te])[:, 1]
                aucs.append(roc_auc_score(y[te], prob))
                aps.append(average_precision_score(y[te], prob))
            fp_bench_results.append({'Fingerprint': 'ChemBERTa embeddings (frozen) + ratios',
                                      'n_features': X_embed.shape[1],
                                      'AUC_mean': np.mean(aucs), 'AUC_std': np.std(aucs),
                                      'AP_mean': np.mean(aps), 'AP_std': np.std(aps)})
            print(f'ChemBERTa embeddings + ratios: AUC={np.mean(aucs):.3f}±{np.std(aucs):.3f}  '
                  f'AP={np.mean(aps):.3f}±{np.std(aps):.3f}')

        except ImportError:
            print('\ntransformers/torch not installed -- skipping the pretrained-embedding '
                  'benchmark row (pip install transformers torch).')
        except Exception as e:
            print(f'\nChemBERTa embedding benchmark failed ({type(e).__name__}: {e}) -- skipping this row.')

        fp_bench_df = pd.DataFrame(fp_bench_results).sort_values('AUC_mean', ascending=False).reset_index(drop=True)
        print('\n' + fp_bench_df.round(3).to_string(index=False))

        fig, ax = plt.subplots(figsize=(7, 3))
        color = PUB_PALETTE[8:16]
        order = fp_bench_df.sort_values('AUC_mean')
        bars = ax.barh(order['Fingerprint'], order['AUC_mean'], xerr=order['AUC_std'], color=color, edgecolor='white', alpha=0.6)
        for bar, val in zip(bars, order['AUC_mean']):
            ax.annotate(f'{val:.3f}', xy=(bar.get_width(), bar.get_y() + bar.get_height()/2+0.08),
                        xytext=(5,0),
                        textcoords='offset points',
                        va='center',
                        fontsize=7)
        ax.set_xlabel('ROC-AUC (5-fold GroupKFold by ionizable lipid)')
        ax.axvline(0.5, color='grey', ls='--', lw=1.5)
        ax.set_xlim(0, 1.0)
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec7_fingerprint_benchmark', width='single')  
        plt.show()

        # expose to later sections
        self.state.ecfp_count_array = ecfp_count_array
        self.state.build_fp_feature_set = build_fp_feature_set
        return self.state


# ============================================================================
# Section08 BitWidthComparison
# ============================================================================
class BitWidthComparison:
    """
    ECFP6 bit-width comparison used to justify the 512-bit choice in the primary model.
    Reads from state:  PUB_PALETTE, save_pub_figure, ratio_feats, y, groups, ecfp_count_array, build_fp_feature_set, gkf
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        PUB_PALETTE = self.state.PUB_PALETTE
        save_pub_figure = self.state.save_pub_figure
        ratio_feats = self.state.ratio_feats
        y = self.state.y
        groups = self.state.groups
        ecfp_count_array = self.state.ecfp_count_array
        build_fp_feature_set = self.state.build_fp_feature_set
        gkf = self.state.gkf

        BIT_WIDTHS = [256, 512, 1024, 2048]
        bitwidth_results = []

        for nbits in BIT_WIDTHS:
            print(f'Building ECFP6 (r=3) at {nbits} bits/component ...')
            builder = lambda s, nbits=nbits: ecfp_count_array(s, radius=3, nbits=nbits)
            fp_mat = build_fp_feature_set(builder)
            X_bw = pd.concat([fp_mat, ratio_feats], axis=1)
            aucs, aps = [], []
            rfc_bw = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                             random_state=42, n_jobs=-1)
            for tr, te in gkf.split(X_bw, y, groups):
                rfc_bw.fit(X_bw.iloc[tr], y[tr])
                prob = rfc_bw.predict_proba(X_bw.iloc[te])[:, 1]
                aucs.append(roc_auc_score(y[te], prob))
                aps.append(average_precision_score(y[te], prob))
            bitwidth_results.append({'nbits_per_component': nbits, 'n_features': X_bw.shape[1],
                                      'AUC_mean': np.mean(aucs), 'AUC_std': np.std(aucs),
                                      'AP_mean': np.mean(aps), 'AP_std': np.std(aps)})

        bitwidth_df = pd.DataFrame(bitwidth_results)
        print('\n' + bitwidth_df.round(3).to_string(index=False))
        best_row = bitwidth_df.loc[bitwidth_df['AUC_mean'].idxmax()]
        print(f"\nBest: {int(best_row['nbits_per_component'])} bits/component "
              f"(AUC={best_row['AUC_mean']:.3f})")

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.errorbar(bitwidth_df['nbits_per_component'], bitwidth_df['AUC_mean'],
                    yerr=bitwidth_df['AUC_std'], marker='o', markersize=4, capsize=3,
                    color=PUB_PALETTE[0], linewidth=1)
        ax.set_xscale('log', base=2)
        ax.set_xticks(BIT_WIDTHS)
        ax.set_xticklabels(BIT_WIDTHS)
        ax.set_ylim([0.0, 1.0])
        ax.set_xlabel('Fingerprint bits per lipid component')
        ax.set_ylabel('ROC-AUC (5-fold GroupKFold, lipid-held-out)')
        ax.scatter([best_row['nbits_per_component']], [best_row['AUC_mean']],
                   s=70, facecolors='none', edgecolors=PUB_PALETTE[1], linewidths=1.5, zorder=5)
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec8_ecfp6_bitwidth_comparison', width='single')
        plt.show()

        return self.state


# ============================================================================
# Section09 PrimaryModel
# ============================================================================
class PrimaryModel:
    """
    Primary model: 5-fold GroupKFold-by-lipid classification, ROC/PR curves, and the confusion matrix at the optimal threshold.

    THRESHOLD SELECTION (double/nested cross-validation, pooled): the decision
    threshold used for the confusion matrix / sensitivity / specificity / PPV / NPV
    is selected without ever looking at any sample's own held-out test-fold label.
    For every outer fold, an inner 5-fold GroupKFold is run on that fold's TRAINING
    data only, producing inner out-of-fold (OOF) probabilities for the samples in
    that training set. These inner-OOF (probability, true-label) pairs are pooled
    across all 5 outer folds -- since each sample sits in the training set of 4 of
    the 5 outer folds, this pools roughly 4x452 inner-OOF observations, i.e. far
    more data than any single fold's ~90 test samples -- and ONE global
    balanced-accuracy-optimal threshold is chosen from that pooled set. This single
    threshold is then applied uniformly to the outer OOF test predictions to compute
    the confusion matrix. A per-fold diagnostic threshold is also reported (kept
    from the previous version of this analysis) purely to illustrate how much a
    threshold estimated from ~90 samples alone would vary fold-to-fold; it is NOT
    used for the reported operating point. AUC/AP are unaffected by any of this --
    they are threshold-free and were already computed on genuinely out-of-fold
    probabilities in the original version of this pipeline.

    Reads from state:  PUB_PALETTE, save_pub_figure, X_D, y, groups, gkf
    Writes to state:   results_df, all_y_true, all_y_prob, best_t
    """
    def __init__(self, state: PipelineState):
        self.state = state

    @staticmethod
    def _balanced_acc_optimal_threshold(y_true, y_prob, thresholds=None):
        if thresholds is None:
            thresholds = np.linspace(0.1, 0.9, 80)
        baccs = [balanced_accuracy_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
        return thresholds[np.argmax(baccs)]

    def run(self):
        PUB_PALETTE = self.state.PUB_PALETTE
        save_pub_figure = self.state.save_pub_figure
        X_D = self.state.X_D
        y = self.state.y
        groups = self.state.groups
        gkf = self.state.gkf

        rfc = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)

        fold_results = []
        # Store OOF predictions in ORIGINAL dataframe order
        all_y_true = np.zeros(len(y), dtype=int)
        all_y_prob = np.zeros(len(y), dtype=float)
        # Per-fold diagnostic thresholds (small-sample, reported only to show
        # instability -- the actual operating point is the pooled threshold below)
        fold_thresholds = []
        pooled_inner_true = []
        pooled_inner_prob = []

        for fold, (tr, te) in enumerate(gkf.split(X_D, y, groups)):
            rfc.fit(X_D.iloc[tr], y[tr])
            prob = rfc.predict_proba(X_D.iloc[te])[:,1]
            pred = (prob >= 0.5).astype(int)
            auc  = roc_auc_score(y[te], prob)
            ap   = average_precision_score(y[te], prob)
            bacc = balanced_accuracy_score(y[te], pred)
            n_lip_tr = len(np.unique(groups[tr]))
            n_lip_te = len(np.unique(groups[te]))

            # --- Inner CV on the TRAINING split only, feeds the pooled threshold below ---
            inner_gkf = GroupKFold(n_splits=min(5, len(np.unique(groups[tr]))))
            inner_y_true = np.zeros(len(tr), dtype=int)
            inner_y_prob = np.zeros(len(tr), dtype=float)
            rfc_inner = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
            for itr, ite in inner_gkf.split(X_D.iloc[tr], y[tr], groups[tr]):
                rfc_inner.fit(X_D.iloc[tr].iloc[itr], y[tr][itr])
                inner_y_prob[ite] = rfc_inner.predict_proba(X_D.iloc[tr].iloc[ite])[:, 1]
                inner_y_true[ite] = y[tr][ite]
            fold_threshold = self._balanced_acc_optimal_threshold(inner_y_true, inner_y_prob)
            fold_thresholds.append(fold_threshold)
            pooled_inner_true.append(inner_y_true)
            pooled_inner_prob.append(inner_y_prob)

            fold_results.append({'fold':fold+1,'AUC':auc,'AP':ap,'BalAcc':bacc,
                                 'n_train':len(tr),'n_test':len(te),
                                 'lips_train':n_lip_tr,'lips_test':n_lip_te,
                                 'fold_diagnostic_threshold':round(float(fold_threshold),3)})
            all_y_true[te] = y[te]
            all_y_prob[te] = prob
            print(f'  Fold {fold+1}: AUC={auc:.3f} AP={ap:.3f} BalAcc={bacc:.3f} '
                  f'| train={len(tr)}({n_lip_tr} lipids) test={len(te)}({n_lip_te} lipids) '
                  f'| fold-only diagnostic threshold={fold_threshold:.3f} (small-sample; not used for reporting)')

        results_df   = pd.DataFrame(fold_results)
        assert len(all_y_true) == len(y)
        assert len(all_y_prob) == len(y)
        print(
            f"OOF prediction alignment check OK "
            f"({len(all_y_prob)} predictions for {len(y)} samples)")
        print(f'\n===  5-Fold GroupKFold Summary  ===')
        print(f'ROC-AUC:        {results_df["AUC"].mean():.3f} ± {results_df["AUC"].std():.3f}')
        print(f'Avg Precision:  {results_df["AP"].mean():.3f}  ± {results_df["AP"].std():.3f}')
        print(f'Balanced Acc:   {results_df["BalAcc"].mean():.3f} ± {results_df["BalAcc"].std():.3f}')

        fig, axes = plt.subplots(1, 2, figsize=(7, 3), constrained_layout=True)
        colors = PUB_PALETTE[:8]
        # ROC
        fpr, tpr, _ = roc_curve(all_y_true, all_y_prob)
        oof_auc = roc_auc_score(all_y_true, all_y_prob)
        axes[0].plot(fpr, tpr, lw=2.5, color=colors[0], label=f'RF (AUC={oof_auc:.3f})')
        axes[0].plot([0,1],[0,1],'k--',lw=1,label='Random (AUC=0.500)')
        axes[0].fill_between(fpr, tpr, alpha=0.08, color=colors[0])
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate')
        axes[0].legend()
        # PR
        prec, rec, _ = precision_recall_curve(all_y_true, all_y_prob)
        oof_ap = average_precision_score(all_y_true, all_y_prob)
        baseline = all_y_true.mean()
        axes[1].plot(rec, prec, lw=2.5, color=colors[1], label=f'RF (AP={oof_ap:.3f})')
        axes[1].axhline(baseline, ls='--', color='k', lw=1, label=f'Random (AP={baseline:.3f})')
        axes[1].fill_between(rec, prec, alpha=0.08, color=colors[1])
        axes[1].set_xlabel('Recall') 
        axes[1].set_ylabel('Precision')
        axes[1].legend()
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec9_roc_pr', width='double')
        plt.show()

        # --- Single pooled threshold, chosen on inner-CV OOF predictions only ---
        # (see class docstring). This replaces two earlier, less satisfactory designs:
        # (1) a threshold picked on the same pooled test predictions being reported
        #     (optimistic, leaked); (2) five separate per-fold thresholds applied
        #     sample-by-sample (leak-free, but noisy and unstable: ~90 samples per
        #     fold produced a 0.45-0.69 range that overstated how uncertain the
        #     "true" operating point actually is). Pooling every fold's inner-CV OOF
        #     predictions into one set before choosing the threshold uses ~4x more
        #     data per estimate and yields a single, honestly-derived operating point.
        pooled_inner_true = np.concatenate(pooled_inner_true)
        pooled_inner_prob = np.concatenate(pooled_inner_prob)
        best_t = float(self._balanced_acc_optimal_threshold(pooled_inner_true, pooled_inner_prob))

        # Bootstrap 95% CI on the pooled threshold itself, so its own uncertainty is
        # reported honestly rather than presented as a single exact-looking decimal.
        rng = np.random.RandomState(42)
        n_pool = len(pooled_inner_true)
        boot_thresholds = []
        for _ in range(500):
            idx = rng.choice(n_pool, n_pool, replace=True)
            if len(np.unique(pooled_inner_true[idx])) < 2:
                continue
            boot_thresholds.append(self._balanced_acc_optimal_threshold(pooled_inner_true[idx], pooled_inner_prob[idx]))
        boot_thresholds = np.array(boot_thresholds)
        t_lo, t_hi = np.percentile(boot_thresholds, [2.5, 97.5])

        print(f'\nFold-only diagnostic thresholds (each from ~90 test-fold-sized samples, '
              f'shown to illustrate small-sample instability): {[round(t,3) for t in fold_thresholds]}')
        print(f'Pooled nested threshold (selected on {n_pool} inner-CV out-of-fold predictions '
              f'pooled across all 5 outer folds): {best_t:.3f}')
        print(f'Bootstrap 95% CI on the pooled threshold ({len(boot_thresholds)} resamples): '
              f'[{t_lo:.3f}, {t_hi:.3f}]')

        y_pred_opt = (all_y_prob >= best_t).astype(int)
        cm = confusion_matrix(all_y_true, y_pred_opt)

        TP,TN,FP,FN = cm[1,1],cm[0,0],cm[0,1],cm[1,0]
        print(f'Sensitivity:  {TP/(TP+FN):.3f}')
        print(f'Specificity:  {TN/(TN+FP):.3f}')
        print(f'PPV:          {TP/(TP+FP):.3f}')
        print(f'NPV:          {TN/(TN+FN):.3f}')

        fig, ax = plt.subplots(figsize=(5, 4))
        colors = PUB_PALETTE[:8]
        ConfusionMatrixDisplay(cm, display_labels=['Low EE','High EE']).plot(ax=ax, colorbar=False, cmap='Purples', values_format='d')
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec9_confusion_matrix', width='double')
        plt.show()

        # expose to later sections
        self.state.results_df = results_df
        self.state.all_y_true = all_y_true
        self.state.all_y_prob = all_y_prob
        self.state.best_t = best_t
        return self.state


# ============================================================================
# Section10 FQS
# ============================================================================
class FQS:
    """
    LNP Formulation Quality Score (FQS): combine classifier output with physicochemical QC thresholds.
    A QED-inspired composite score (Bickerton et al. 2012, Nat. Chem.) combining three clinically grounded desirability functions:
        | Component | Clinical basis | Optimal range |
        | d_EE | P(EE ≥ 80%) from classifier | 1.0 = certain high EE |
        | d_size | FDA/USP guidance on LNP size | 50-120 nm |
        | d_PDI | ICH Q1A(R2) standard | PDI ≤ 0.2 |
            
        Formula: FQS = 100 * (d_EE * d_size * d_PDI)^(1/3)
    Reads from state:  PUB_PALETTE, save_pub_figure, clean_df, all_y_prob
    Writes to state:   compute_FQS
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        """Execute this section (notebook cells [17, 18, 19, 20, 21]) and update shared state."""
        PUB_PALETTE = self.state.PUB_PALETTE
        save_pub_figure = self.state.save_pub_figure
        clean_df = self.state.clean_df
        all_y_prob = self.state.all_y_prob

        def d_EE(P, r=3):
            P = np.clip(P, 0, 1)
            return P ** r

        def d_size(size_nm):
            if pd.isna(size_nm):
                return np.nan
            s = float(size_nm)
            if s <= 30:   return 0.0
            if s <= 50:   return (s - 30) / 20.0
            if s <= 120:  return 1.0
            if s <= 150:  return (150 - s) / 30.0
            return 0.0

        def d_PDI(pdi):
            if pd.isna(pdi):
                return np.nan
            p = float(pdi)
            if p <= 0.20:  return 1.0
            if p <= 0.30:  return (0.30 - p) / 0.10
            return 0.0

        def compute_FQS(prob_ee, size_nm=None, pdi=None):
            comps, labels = [d_EE(prob_ee)], ['EE']
            ds = d_size(size_nm) if (size_nm is not None and not pd.isna(size_nm)) else None
            dp = d_PDI(pdi)      if (pdi is not None and not pd.isna(pdi))          else None
            if ds is not None: comps.append(ds); labels.append('size')
            if dp is not None: comps.append(dp); labels.append('PDI')
            geo = float(np.prod(comps) ** (1/len(comps)))
            return {'FQS': round(geo*100,2), 'components': '+'.join(labels),
                    'd_EE': round(d_EE(prob_ee),4),
                    'd_size': round(ds,4) if ds is not None else None,
                    'd_PDI':  round(dp,4) if dp is not None else None}

        fig, axes = plt.subplots(1, 3, figsize=(8, 3))
        colors = PUB_PALETTE[:8]
        sizes = np.linspace(0, 200, 300)
        pdis  = np.linspace(0, 0.5, 300)
        probs = np.linspace(0, 1, 300)

        axes[0].plot(probs, [d_EE(p) for p in probs], lw=2, color=colors[0])
        axes[0].fill_between(probs, [d_EE(p) for p in probs], alpha=0.08, color=colors[0])
        axes[0].set_xlabel('P(EE≥80%)')
        axes[0].set_ylabel('Desirability Score')
        # axes[0].set_title('$d_{EE}$', fontsize=11)

        axes[1].plot(sizes, [d_size(s) for s in sizes], lw=2, color=colors[1])
        axes[1].axvspan(50, 120, alpha=0.08, color=colors[1], label='Optimal: 50-120 nm')
        axes[1].axvline(150, ls='--', color=colors[2], lw=1.5, label='Upper limit: 150 nm')
        axes[1].set_xlabel('Particle Size (nm)')
        axes[1].set_ylabel('Desirability Score')
        axes[1].set_xlim(25, 175)
        # axes[1].set_title('$d_{size}$', fontsize=11)
        axes[1].legend()

        axes[2].plot(pdis, [d_PDI(p) for p in pdis], lw=2, color=colors[3])
        axes[2].axvline(0.2, ls='--', color=colors[4], lw=1.5, label='Target $\\leq$0.2')
        axes[2].axvline(0.3, ls=':', color=colors[5],  lw=1.5, label='FDA limit 0.3')
        axes[2].set_xlabel('PDI')
        axes[2].set_ylabel('Desirability Score')
        axes[2].set_xlim(0.0, 0.4)
        # axes[2].set_title('$d_{PDI}$', fontsize=11)
        axes[2].legend()

        for ax in axes:
            ax.set_ylim([-0.05, 1.05])
            ax.grid(alpha=0.2)

        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec10_desirability_functions', width='double')
        plt.show()

        # Reuse OOF probabilities already computed in Section 9 (PrimaryModel) --
        # consistent with the ROC/PR curves plotted there.
        clean_df['prob_high_EE'] = all_y_prob

        fqs_records = [compute_FQS(row['prob_high_EE'], row['size_nm'], row['PDI_val'])
                       for _, row in clean_df.iterrows()]
        fqs_df = pd.DataFrame(fqs_records)
        clean_df['FQS']        = fqs_df['FQS']
        clean_df['FQS_comps']  = fqs_df['components']
        clean_df['d_EE_val']   = fqs_df['d_EE']
        clean_df['d_size_val'] = fqs_df['d_size']
        clean_df['d_PDI_val']  = fqs_df['d_PDI']

        full3 = clean_df[clean_df['FQS_comps']=='EE+size+PDI'].copy()
        print(f'3-component FQS formulations: {len(full3)}')
        print(f'FQS distribution:')
        print(full3['FQS'].describe().round(1))

        rho_EE,  p_EE   = stats.spearmanr(full3['FQS'], full3['EE'])
        rho_sz,  p_sz   = stats.spearmanr(full3['FQS'], -full3['size_nm'])
        rho_PDI, p_PDI  = stats.spearmanr(full3['FQS'], -full3['PDI_val'])
        print(f'\nFQS vs EE%:    rho={rho_EE:.3f} (p={p_EE:.2e})')
        print(f'FQS vs -size:  rho={rho_sz:.3f} (p={p_sz:.2e})')
        print(f'FQS vs -PDI:   rho={rho_PDI:.3f} (p={p_PDI:.2e})')

        fig, ax = plt.subplots(figsize=(4, 3))
        P_range = np.linspace(0, 1, 200)

        ax.plot(P_range, P_range, ls='--', color='grey', lw=1, label='r=1 (linear)')
        ax.plot(P_range, d_EE(P_range, r=3), color=PUB_PALETTE[0], lw=1.5, label='r=3 (current)')
        ax.fill_between(P_range, 0, d_EE(P_range, r=3), color=PUB_PALETTE[0], alpha=0.15)

        ax.set_xlabel(f'P(EE$\\geq$80%)')
        ax.set_ylabel('Desirability Score')
        ax.legend(fontsize=6, loc='upper left')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])

        plt.tight_layout()
        save_pub_figure(fig, 'fig_FQS_dEE_desirability', width='single')
        plt.show()

        fig, ax = plt.subplots(figsize=(3, 3))
        P_range = np.linspace(0, 1, 200)

        ax.plot(P_range, P_range, ls='--', color='grey', lw=1, label='r=1 (linear)')
        for i in range(2,10):
            ax.plot(P_range, d_EE(P_range, r=i), color=PUB_PALETTE[i], lw=1.5, label=f'r={i}')

        ax.set_xlabel(f'P(EE$\\geq$80%)')
        ax.set_ylabel('Desirability Score')
        ax.legend(fontsize=6, loc='upper left')
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])

        plt.tight_layout()
        save_pub_figure(fig, 'fig_FQS_dEE_desirability__', width='single')
        plt.show()

        fig, axes = plt.subplots(1, 3, figsize=(8, 3))
        colors = PUB_PALETTE[:8]
        axes[0].hist(full3.loc[full3['EE']<80,'FQS'],  bins=20, color=colors[3], alpha=0.75, label='Low EE', edgecolor='white')
        axes[0].hist(full3.loc[full3['EE']>=80,'FQS'], bins=20, color=colors[5], alpha=0.75, label='High EE', edgecolor='white')
        axes[0].set_xlabel('FQS'); axes[0].set_ylabel('Count')
        axes[0].legend()

        sc = axes[1].scatter(full3['EE'], full3['FQS'], c=full3['size_nm'],
                              cmap='RdYlGn_r', s=25, alpha=0.6, vmin=50, vmax=200)
        axes[1].axhline(50, ls='--', color=colors[4], lw=1)
        axes[1].axvline(80, ls='--', color=colors[7], lw=1)
        plt.colorbar(sc, ax=axes[1], label='Size (nm)')
        axes[1].set_xlabel('True EE%'); axes[1].set_ylabel('FQS')

        full3['FQS_q'] = pd.qcut(full3['FQS'], 4, duplicates='drop')
        n_cats = full3['FQS_q'].cat.categories
        label_map = {c: f'Q{i+1}' for i, c in enumerate(n_cats)}
        full3['FQS_q'] = full3['FQS_q'].map(label_map)

        qdf = full3.groupby('FQS_q', observed=True)[['EE','size_nm','PDI_val']].mean().round(1)
        qdf.plot(kind='bar', ax=axes[2], color=colors[:3])
        axes[2].set_xlabel('FQS Quartile')
        axes[2].legend()
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec10_fqs_analysis', width='double')
        plt.show()

        # expose to later sections
        self.state.compute_FQS = compute_FQS
        return self.state


# ============================================================================
# Section11 ModelComparison
# ============================================================================
class ModelComparison:
    """
    Model comparison and robustness suite: significance testing, leakage proof, baselines, leave-one-publication-out, temporal holdout, calibration.
    Reads from state:  PUB_PALETTE, save_pub_figure, clean_df, X_D, y, groups, results_df, all_y_true, all_y_prob, gkf
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):

        PUB_PALETTE = self.state.PUB_PALETTE
        save_pub_figure = self.state.save_pub_figure
        clean_df = self.state.clean_df
        X_D = self.state.X_D
        y = self.state.y
        groups = self.state.groups
        results_df = self.state.results_df
        all_y_true = self.state.all_y_true
        all_y_prob = self.state.all_y_prob
        gkf = self.state.gkf

        models = {
            'Random Forest (n=500)': RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1),
            'Extra Trees (n=500)':   ExtraTreesClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1),
        }
        try:
            from xgboost import XGBClassifier
            models['XGBoost'] = XGBClassifier(
                n_estimators=500, max_depth=4, learning_rate=0.05,
                scale_pos_weight=(y==0).sum()/(y==1).sum(),
                random_state=42, n_jobs=-1, eval_metric='logloss'
            )
        except ImportError:
            print('XGBoost not installed — skip (pip install xgboost)')

        print(f'{"Model":<32}  AUC (mean±std)     AP (mean±std)')
        print('-'*72)
        for name, model in models.items():
            aucs, aps = [], []
            for tr, te in gkf.split(X_D, y, groups):
                model.fit(X_D.iloc[tr], y[tr])
                prob = model.predict_proba(X_D.iloc[te])[:,1]
                aucs.append(roc_auc_score(y[te], prob))
                aps.append(average_precision_score(y[te], prob))
            print(f'{name:<32}  {np.mean(aucs):.3f}±{np.std(aucs):.3f}     {np.mean(aps):.3f}±{np.std(aps):.3f}')

        # NOTE: an unused `bootstrap_ci()` pooled-probability helper previously lived
        # here. It has been removed: pooling raw probabilities across GroupKFold folds
        # with different per-fold calibration is invalid (see the note below), and the
        # function was dead code -- `bootstrap_ci_from_folds` below is what's actually used.

        # Get XGBoost OOF probs for comparison
        try:
            from xgboost import XGBClassifier
            xgb = XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
                                 scale_pos_weight=(y==0).sum()/(y==1).sum(),
                                 random_state=42, n_jobs=-1, eval_metric='logloss')
            xgb_probs = np.zeros(len(y))
            for tr, te in gkf.split(X_D, y, groups):
                xgb.fit(X_D.iloc[tr], y[tr])
                xgb_probs[te] = xgb.predict_proba(X_D.iloc[te])[:,1]
            has_xgb = True
        except ImportError:
            has_xgb = False
            xgb_probs = all_y_prob  # fallback
            print('XGBoost not installed — using RF as comparison')

        et = ExtraTreesClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
        et_probs = np.zeros(len(y))
        for tr, te in gkf.split(X_D, y, groups):
            et.fit(X_D.iloc[tr], y[tr])
            et_probs[te] = et.predict_proba(X_D.iloc[te])[:,1]

        # --------------------------- Bootstrap CIs: per-fold (NOT pooled) ---------------------------
        def fold_aucs(model, X, y, groups, gkf):
            aucs = []
            for tr, te in gkf.split(X, y, groups):
                model.fit(X.iloc[tr], y[tr])
                prob = model.predict_proba(X.iloc[te])[:, 1]
                aucs.append(roc_auc_score(y[te], prob))
            return np.array(aucs)

        def bootstrap_ci_from_folds(fold_scores, n_boot=1000, ci=95, seed=42):
            rng = np.random.RandomState(seed)
            means = [rng.choice(fold_scores, len(fold_scores), replace=True).mean() for _ in range(n_boot)]
            lo, hi = np.percentile(means, [(100-ci)/2, 100-(100-ci)/2])
            return np.mean(fold_scores), lo, hi

        rf_fold_aucs  = fold_aucs(RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                                          random_state=42, n_jobs=-1), X_D, y, groups, gkf)
        et_fold_aucs  = fold_aucs(et, X_D, y, groups, gkf)
        xgb_fold_aucs = fold_aucs(xgb, X_D, y, groups, gkf) if has_xgb else rf_fold_aucs

        print('=== Bootstrap 95% CIs over the 5 per-fold AUCs (n=1000 resamples) ===')
        for name, scores in [('RF (primary)', rf_fold_aucs), ('ExtraTrees', et_fold_aucs),
                              ('XGBoost' if has_xgb else 'XGB/RF', xgb_fold_aucs)]:
            m, lo, hi = bootstrap_ci_from_folds(scores)
            print(f'{name:<20}  AUC={m:.3f} [{lo:.3f}–{hi:.3f}]  (per-fold: {np.round(scores,3).tolist()})')

        print('\n=== Paired Wilcoxon signed-rank test on matched per-fold AUCs ===')
        for name, scores in [('ExtraTrees', et_fold_aucs), ('XGBoost' if has_xgb else 'XGB/RF', xgb_fold_aucs)]:
            diff = rf_fold_aucs - scores
            try:
                stat, p = wilcoxon(diff)
            except ValueError:
                stat, p = np.nan, np.nan
            print(f'RF vs {name}: mean $\\Delta$AUC={diff.mean():+.3f}, Wilcoxon p={p:.3f}  '
                  f'(n=5 folds -- treat as indicative, not a high-powered test)')
        print('(p<0.05 = statistically significant difference; p>0.05 = not significant)')

        pred_rf = (all_y_prob >= 0.5).astype(int)
        pred_et = (et_probs   >= 0.5).astype(int)
        b = ((pred_rf==1) & (pred_et==0)).sum()
        c = ((pred_rf==0) & (pred_et==1)).sum()
        mcnemar_stat = (abs(b-c)-1)**2 / (b+c) if (b+c) > 0 else 0
        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, df=1)
        print(f'\n=== McNemar Test (RF vs ExtraTrees at threshold=0.5, pooled thresholded predictions) ===')
        print(f'Disagreements: RF-correct/ET-wrong={b}, RF-wrong/ET-correct={c}')
        print(f'McNemar χ²={mcnemar_stat:.2f}, p={mcnemar_p:.3f}')

        # --------------------------- Data leakage proof ---------------------------
        n_repeats_leak = 20
        random_aucs, group_aucs, group_shuffle_aucs = [], [], []
        rfc_leak = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
        for rep in range(n_repeats_leak):
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=rep)
            aucs_r = []
            for tr, te in skf.split(X_D, y):
                rfc_leak.fit(X_D.iloc[tr], y[tr])
                aucs_r.append(roc_auc_score(y[te], rfc_leak.predict_proba(X_D.iloc[te])[:,1]))
            random_aucs.append(np.mean(aucs_r))
            aucs_g = []
            for tr, te in gkf.split(X_D, y, groups):
                rfc_leak.fit(X_D.iloc[tr], y[tr])
                aucs_g.append(roc_auc_score(y[te], rfc_leak.predict_proba(X_D.iloc[te])[:,1]))
            group_aucs.append(np.mean(aucs_g))
        random_aucs = np.array(random_aucs)
        group_aucs  = np.array(group_aucs)

        print('=== Data Leakage Analysis: Random vs Group-Based Splitting ===')
        print(f'Random (stratified) KFold AUC: {random_aucs.mean():.3f} ± {random_aucs.std():.3f}')
        print(f'GroupKFold AUC:                {group_aucs.mean():.3f} ± {group_aucs.std():.3f}')
        print(f'Inflation from random split:   +{random_aucs.mean()-group_aucs.mean():.3f} AUC points')
        print(f'That is a {(random_aucs.mean()-group_aucs.mean())/group_aucs.mean()*100:.1f}% relative overestimate')
        t_stat, p_val = stats.ttest_ind(random_aucs, group_aucs)
        print(f't-test: t={t_stat:.2f}, p={p_val:.2e} — difference is {"significant" if p_val<0.05 else "not significant"}')

        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        color = PUB_PALETTE[:8]
        data_box = [random_aucs, group_aucs]
        bp = axes[0].boxplot(data_box, labels=['Random\n(StratifiedKFold)\n(WRONG)','GroupKFold\n(lipid-held-out)\n(CORRECT)'],
                             patch_artist=True, notch=True)
        bp['boxes'][0].set_facecolor(color[5]); bp['boxes'][0].set_alpha(0.6)
        bp['boxes'][1].set_facecolor(color[1]); bp['boxes'][1].set_alpha(0.6)
        axes[0].set_ylabel('ROC-AUC (20 repeats)')
        axes[1].scatter(random_aucs, group_aucs, alpha=0.7, s=40, color=color[4], zorder=3)
        lims = [min(min(random_aucs),min(group_aucs))-0.02, max(max(random_aucs),max(group_aucs))+0.02]
        axes[1].plot(lims, lims, 'k--', lw=1.5, label='y=x (no inflation)')
        axes[1].set_xlabel('Random Split AUC')
        axes[1].set_ylabel('GroupKFold AUC')
        axes[1].legend()

        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec11B_leakage_proof', width='double')
        plt.show()

        # --------------------------- Baseline Classifiers ---------------------------
        def cv_eval(X, y, groups, model):
            aucs, aps = [], []
            for tr, te in gkf.split(X, y, groups):
                model.fit(X.iloc[tr], y[tr])
                prob = model.predict_proba(X.iloc[te])[:, 1]
                aucs.append(roc_auc_score(y[te], prob)); aps.append(average_precision_score(y[te], prob))
            return np.mean(aucs), np.std(aucs), np.mean(aps), np.std(aps)

        print('=== Baseline classifiers (context for AUC/AP) ===')
        baseline_rows = []
        for name, model in [('Majority-class baseline', DummyClassifier(strategy='most_frequent')),
                             ('Stratified-random baseline', DummyClassifier(strategy='stratified', random_state=42)),
                             ('RandomForest (X_D, chemistry only) [MAIN MODEL]',
                              RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1))]:
            am, asd, apm, apsd = cv_eval(X_D, y, groups, model)
            baseline_rows.append({'Model': name, 'AUC_mean': am, 'AUC_std': asd, 'AP_mean': apm, 'AP_std': apsd})
            print(f'{name:<48} AUC={am:.3f}±{asd:.3f}  AP={apm:.3f}±{apsd:.3f}')
        baseline_df = pd.DataFrame(baseline_rows)
        print(f"\nThe main model's AUC exceeds the majority-class baseline by "
              f"{baseline_df.loc[baseline_df['Model'].str.contains('MAIN'), 'AUC_mean'].values[0] - 0.5:.3f} "
              f"(baseline AUC is 0.5 by construction).")

        # --------------------------- Leave-One-Publication-Out Validation ---------------------------
        paper_key = clean_df['paper_doi'].fillna(clean_df['paper_title']).values
        n_papers = len(set(paper_key))
        print(f'{n_papers} unique publications in the curated dataset')

        gkf_paper = GroupKFold(n_splits=5)
        rfc_paper = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
        aucs_p, aps_p = [], []
        for tr, te in gkf_paper.split(X_D, y, paper_key):
            rfc_paper.fit(X_D.iloc[tr], y[tr])
            prob = rfc_paper.predict_proba(X_D.iloc[te])[:, 1]
            aucs_p.append(roc_auc_score(y[te], prob)); aps_p.append(average_precision_score(y[te], prob))
        print(f'\nGroupKFold by PUBLICATION (5-fold): AUC={np.mean(aucs_p):.3f}±{np.std(aucs_p):.3f}  '
              f'AP={np.mean(aps_p):.3f}±{np.std(aps_p):.3f}')
        print(f'(Main model, GroupKFold by LIPID):   AUC={results_df["AUC"].mean():.3f}±{results_df["AUC"].std():.3f}  '
              f'AP={results_df["AP"].mean():.3f}±{results_df["AP"].std():.3f}')

        logo = LeaveOneGroupOut()
        all_t2, all_p2, n_folds_used = [], [], 0
        for tr, te in logo.split(X_D, y, paper_key):
            if len(set(y[tr])) < 2:
                continue
            rfc_paper.fit(X_D.iloc[tr], y[tr])
            prob = rfc_paper.predict_proba(X_D.iloc[te])[:, 1]
            all_t2.append(y[te]); all_p2.append(prob); n_folds_used += 1
        all_t2 = np.concatenate(all_t2); all_p2 = np.concatenate(all_p2)
        auc_logo = roc_auc_score(all_t2, all_p2); ap_logo = average_precision_score(all_t2, all_p2)
        print(f'\nTrue leave-one-publication-out ({n_folds_used}/{n_papers} publications used): '
              f'pooled AUC={auc_logo:.3f}, AP={ap_logo:.3f}')

        n_boot = 2000
        boot_aucs = []
        rng = np.random.RandomState(42)
        for _ in range(n_boot):
            idx = rng.choice(len(all_t2), len(all_t2), replace=True)
            if len(np.unique(all_t2[idx])) < 2:
                continue
            boot_aucs.append(roc_auc_score(all_t2[idx], all_p2[idx]))

        auc_ci = np.percentile(boot_aucs, [2.5, 97.5])
        ci_low, ci_high = auc_ci
        print(
            f"Under leave-one-publication-out validation, the model achieved "
            f"a pooled ROC-AUC of {auc_logo:.3f} "
            f"(95% CI: {ci_low:.3f}-{ci_high:.3f}).")

        fig, ax = plt.subplots(figsize=(7, 3))
        labels = ['By ionizable\nlipid (main)', 'By publication\n(5-fold)', 'By publication\n(true LOO)']
        vals = [results_df['AUC'].mean(), np.mean(aucs_p), auc_logo]
        errs = [results_df['AUC'].std(), np.std(aucs_p), 0]
        colors = PUB_PALETTE[:3]
        bars = ax.bar(labels, vals, yerr=errs, capsize=6, color=colors, edgecolor='white', alpha=0.6)
        ax.axhline(0.5, ls='--', color='grey', lw=1, label='Random (AUC=0.5)')
        ax.set_ylabel('ROC-AUC'); ax.set_ylim([0.4, 0.95])
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2+0.13, bar.get_height()+0.01,
                    f'{val:.3f}', ha='center', va='bottom')
        ax.legend()
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec11D_leave_one_paper_out', width='double')
        plt.show()

        print(f"\nInterpretation: a drop from the lipid-grouped AUC to the publication-grouped AUC indicates "
            f"lab/protocol-specific signal (e.g. size/PDI measurement conventions, batch effects) that the "
            f"model is partly relying on, beyond pure lipid chemistry. Report both numbers in the manuscript "
            f"and discuss the gap honestly rather than only citing the more favorable lipid-grouped AUC.")

        # --------------------------- Temporal Holdout Validation ---------------------------
        # FIX: sanity-bound any year parsed via regex fallback. A DOI or title can
        # contain an incidental 4-digit substring (article/journal IDs, etc.) that
        # looks like a year but isn't -- e.g. matching a DOI suffix. We restrict
        # accepted years to a plausible publication window and flag/drop anything
        # outside it rather than silently feeding a wrong year into the temporal
        # drift analysis.
        YEAR_MIN, YEAR_MAX = 1990, 2026
        YEAR_CANDIDATES = ['publication_year', 'pub_year', 'year', 'paper_year']
        year_col = next((c for c in YEAR_CANDIDATES if c in clean_df.columns), None)
        if year_col is None:
            def extract_year(row):
                for field in ['paper_title', 'paper_doi']:
                    if field in clean_df.columns and pd.notna(row.get(field)):
                        for m in re.finditer(r'(19|20)\d{2}', str(row[field])):
                            candidate = int(m.group(0))
                            if YEAR_MIN <= candidate <= YEAR_MAX:
                                return candidate
                return np.nan
            clean_df['pub_year_parsed'] = clean_df.apply(extract_year, axis=1)
            year_col = 'pub_year_parsed'
            n_recovered = clean_df[year_col].notna().sum()
            print(f'No explicit year column found -- parsed {n_recovered}/{len(clean_df)} years from title/DOI text '
                  f'(regex matches outside [{YEAR_MIN}, {YEAR_MAX}] were rejected as likely DOI/ID artifacts, not years).')
            print('If this recovers too few rows, point year_col at the correct column from the raw LNP_Atlas_DB CSV.')
        else:
            n_out_of_range = (~clean_df[year_col].between(YEAR_MIN, YEAR_MAX) & clean_df[year_col].notna()).sum()
            if n_out_of_range > 0:
                print(f'WARNING: {n_out_of_range} rows in "{year_col}" fall outside [{YEAR_MIN}, {YEAR_MAX}] -- '
                      f'these are set to NaN before the temporal analysis rather than trusted as-is.')
                clean_df.loc[~clean_df[year_col].between(YEAR_MIN, YEAR_MAX), year_col] = np.nan

        print(clean_df[year_col].describe())

        valid = clean_df[year_col].notna()
        cutoff = clean_df.loc[valid, year_col].quantile(0.50)
        print(f'\nTemporal cutoff (50th percentile of publication year): {cutoff:.0f}')
        train_idx = clean_df[valid & (clean_df[year_col] <= cutoff)].index
        test_idx  = clean_df[valid & (clean_df[year_col]  > cutoff)].index
        print(f'Train: {len(train_idx)} formulations (<= {cutoff:.0f})')
        print(f'Test:  {len(test_idx)} formulations (> {cutoff:.0f})')
        if len(test_idx) > 10 and 0 < y[test_idx].sum() < len(test_idx):
            rfc_temporal = RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                                   random_state=42, n_jobs=-1)
            rfc_temporal.fit(X_D.loc[train_idx], y[train_idx])
            prob_temporal = rfc_temporal.predict_proba(X_D.loc[test_idx])[:, 1]
            auc_temporal = roc_auc_score(y[test_idx], prob_temporal)
            ap_temporal  = average_precision_score(y[test_idx], prob_temporal)
            print(f'\nTemporal holdout AUC: {auc_temporal:.3f}')
            print(f'Temporal holdout AP:  {ap_temporal:.3f}')
        else:
            print('\nTest set too small or single-class after filtering -- widen the cutoff quantile or check year parsing.')

        print("\n=== Temporal Drift Analysis ===")
        quantiles = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]
        temporal_results = []
        for q in quantiles:
            cutoff = clean_df.loc[valid, year_col].quantile(q)
            train_idx = clean_df[valid & (clean_df[year_col] <= cutoff)].index
            test_idx = clean_df[valid & (clean_df[year_col] > cutoff)].index
            if len(test_idx) < 20:
                continue
            if not (0 < y[test_idx].sum() < len(test_idx)):
                continue

            rf_temp = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
            rf_temp.fit(X_D.loc[train_idx], y[train_idx])
            prob_temp = rf_temp.predict_proba(X_D.loc[test_idx])[:, 1]
            auc_temp = roc_auc_score(y[test_idx], prob_temp)
            ap_temp = average_precision_score(y[test_idx], prob_temp)

            train_lipids = clean_df.loc[train_idx, 'ionizable_lipid_smiles'].unique()
            test_lipids = clean_df.loc[test_idx, 'ionizable_lipid_smiles'].unique()
            train_fp = np.vstack([self.state.count_fp_array(x) for x in train_lipids])
            test_fp = np.vstack([self.state.count_fp_array(x) for x in test_lipids])
            nn = NearestNeighbors(n_neighbors=1, metric='jaccard')
            nn.fit(train_fp > 0)
            dist, _ = nn.kneighbors(test_fp > 0)
            mean_novelty = float(dist.mean())

            train_n_lipids = (clean_df.loc[train_idx, 'ionizable_lipid_smiles'].nunique())
            test_n_lipids = (clean_df.loc[test_idx, 'ionizable_lipid_smiles'].nunique())
            temporal_results.append({
                'quantile': q,
                'cutoff_year': int(round(cutoff)),
                'train_n': len(train_idx),
                'test_n': len(test_idx),
                'train_lipids': train_n_lipids,
                'test_lipids': test_n_lipids,
                'AUC': auc_temp,
                'AP': ap_temp,
                'mean_novelty': mean_novelty,})
        temporal_df = pd.DataFrame(temporal_results)
        print("\nTemporal performance by training-window size:")
        print(temporal_df.round(3).to_string(index=False))
        rho, p = spearmanr(temporal_df["mean_novelty"], temporal_df["AUC"])
        print(f"Novelty-AUC correlation: "
            f"rho={rho:.3f}, p={p:.3f}")

        fig, axes = plt.subplots(1, 2, figsize=(10,3))
        axes[0].plot(temporal_df['cutoff_year'], temporal_df['AUC'], marker='o', lw=1.5, color=PUB_PALETTE[0])
        axes[0].set_xlabel('Latest publication year in training set')
        axes[0].set_ylabel('Temporal holdout ROC-AUC')
        axes[0].set_ylim([0.5, 1.0])
        axes[1].plot(temporal_df['cutoff_year'], temporal_df['mean_novelty'], marker='s', lw=1.5, color=PUB_PALETTE[1])
        axes[1].set_xlabel('Latest publication year in training set')
        axes[1].set_ylabel('Mean chemical novelty')

        plt.tight_layout()
        save_pub_figure(fig, 'fig_temporal_drift', width='double')
        plt.show()

        # --------------------------- Publication-grouped control ---------------------------
        target_train_size = 122
        paper_groups = clean_df['paper_doi'].fillna(clean_df['paper_title'])
        aucs = []
        aps = []
        for seed in range(50):
            gss = GroupShuffleSplit(n_splits=1, train_size=target_train_size / len(clean_df), random_state=seed)
            train_idx, test_idx = next(gss.split(X_D, y, paper_groups))
            if abs(len(train_idx) - target_train_size) > 20:
                continue
            rf = RandomForestClassifier(n_estimators=500,class_weight='balanced',random_state=42,n_jobs=-1)
            rf.fit(X_D.iloc[train_idx], y[train_idx])
            prob = rf.predict_proba(X_D.iloc[test_idx])[:, 1]
            aucs.append(roc_auc_score(y[test_idx], prob))
            aps.append(average_precision_score(y[test_idx], prob))

        if len(aucs) == 0:
            print(f'No GroupShuffleSplit seed (of 50 tried) produced a train size within '
                  f'$\\pm$ 20 of {target_train_size} -- widen the tolerance in this cell or check '
                  f'publication group sizes before trusting this control.')
        else:
            print(f'Publication-grouped control (~{target_train_size} train samples, '
                  f'{len(aucs)}/50 seeds used)')
            print(f'AUC = {np.mean(aucs):.3f} ± {np.std(aucs):.3f}')
            print(f'AP  = {np.mean(aps):.3f} ± {np.std(aps):.3f}')

        # --------------------------- Calibration ---------------------------
        def ece_score(y_true, y_prob, n_bins=10):
            bins = np.linspace(0, 1, n_bins + 1)
            ece, n_total = 0.0, len(y_true)
            for i in range(n_bins):
                mask = (y_prob >= bins[i]) & (y_prob < bins[i+1])
                if mask.sum() == 0: continue
                ece += (mask.sum() / n_total) * abs(y_prob[mask].mean() - y_true[mask].mean())
            return ece

        def oof_calibrated_probs(method):
            probs = np.zeros(len(y))
            for tr, te in gkf.split(X_D, y, groups):
                cal_clf = CalibratedClassifierCV(
                    RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                            random_state=42, n_jobs=-1),
                    method=method, cv=3)
                cal_clf.fit(X_D.iloc[tr], y[tr])
                probs[te] = cal_clf.predict_proba(X_D.iloc[te])[:, 1]
            return probs

        brier_uncal = brier_score_loss(all_y_true, all_y_prob)
        ece_uncal   = ece_score(all_y_true, all_y_prob)
        ptrue_uncal, ppred_uncal = calibration_curve(all_y_true, all_y_prob, n_bins=10, strategy='quantile')

        print('Fitting isotonic calibration (nested CV) ...')
        prob_isotonic = oof_calibrated_probs('isotonic')
        brier_iso = brier_score_loss(y, prob_isotonic)
        ece_iso   = ece_score(y, prob_isotonic)
        ptrue_iso, ppred_iso = calibration_curve(y, prob_isotonic, n_bins=10, strategy='quantile')

        print('Fitting sigmoid (Platt) calibration (nested CV) ...')
        prob_sigmoid = oof_calibrated_probs('sigmoid')
        brier_sig = brier_score_loss(y, prob_sigmoid)
        ece_sig   = ece_score(y, prob_sigmoid)
        ptrue_sig, ppred_sig = calibration_curve(y, prob_sigmoid, n_bins=10, strategy='quantile')

        calib_summary = pd.DataFrame([
            {'Method': 'Uncalibrated (native RF)', 'Brier': brier_uncal, 'ECE': ece_uncal},
            {'Method': 'Isotonic',                  'Brier': brier_iso,  'ECE': ece_iso},
            {'Method': 'Sigmoid (Platt)',           'Brier': brier_sig,  'ECE': ece_sig},
        ]).sort_values('Brier').reset_index(drop=True)
        print('\n' + calib_summary.round(4).to_string(index=False))
        print('\nLower is better for both metrics (Brier: 0=perfect/0.25=random; ECE: 0=perfect).')
        best = calib_summary.iloc[0]['Method']
        print(f'Best by Brier score: {best}')

        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect calibration')
        axes[0].plot(ppred_uncal, ptrue_uncal, 'd', color=PUB_PALETTE[0], ms=4, label=f'Uncalibrated (ECE={ece_uncal:.3f})')
        axes[0].plot(ppred_iso, ptrue_iso, 's-', color=PUB_PALETTE[1], ms=4, label=f'Isotonic (ECE={ece_iso:.3f})')
        axes[0].plot(ppred_sig, ptrue_sig, '^-', color=PUB_PALETTE[2], ms=4, label=f'Sigmoid (ECE={ece_sig:.3f})')
        axes[0].set_xlabel('Mean predicted probability')
        axes[0].set_ylabel('Fraction of positives')
        axes[0].legend(fontsize=6, loc='upper left')
        axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1])
        axes[1].bar(calib_summary['Method'], calib_summary['Brier'], color=PUB_PALETTE[:3])
        axes[1].set_ylabel('Brier score (lower = better)')
        axes[1].tick_params(axis='x')

        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec11F_calibration_method_comparison', width='double')
        plt.show()

        return self.state


# ============================================================================
# SectionG SaveAndPredictAPI
# ============================================================================
class SaveAndPredictAPI:
    """
    Train the final deployable classifier, save it, and expose predict_lnp() for scoring new formulations.
    Reads from state:  active_cols, X_D, y, all_y_true, all_y_prob, best_t, compute_FQS
    Writes to state:   MC3, DSPC, CHOLESTEROL, DMG_PEG, bundle, predict_lnp
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        active_cols = self.state.active_cols
        X_D = self.state.X_D
        y = self.state.y
        all_y_true = self.state.all_y_true
        all_y_prob = self.state.all_y_prob
        best_t = self.state.best_t
        compute_FQS = self.state.compute_FQS

        clf_final = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
        print('Training final model on full dataset...')
        clf_final.fit(X_D, y)
        print('Done.')
        print("\n=== Training Feature Configuration ===")
        print("Fingerprint radius:", 3)
        print("Fingerprint bits: 512")
        print("EE threshold:", 80)

        model_bundle = {
            'model':            clf_final,
            'active_fp_cols':   active_cols,
            'feature_names':    X_D.columns.tolist(),
            'fp_radius':        3,
            'fp_nbits':         512,
            'ee_threshold':     80,
            # Nested-CV-selected balanced-accuracy-optimal cutoff (Section 9), not an
            # arbitrary default and not picked on the same pooled test set it's reported
            # against (see PrimaryModel docstring). Still an operating point, not a
            # guaranteed-optimal threshold on future data -- AUC/AP remain primary.
            'decision_threshold': round(float(best_t), 3),
            'lipid_order':      ['ionizable','helper','sterol','peg'],
            'training_n':       len(X_D),
            'training_auc':     round(roc_auc_score(all_y_true, all_y_prob), 3),
            'training_ap':      round(average_precision_score(all_y_true, all_y_prob), 3),
        }
        with open('lnp_classifier.pkl','wb') as f:
            pickle.dump(model_bundle, f)
        print(f'Saved lnp_classifier.pkl')
        print(f'AUC={model_bundle["training_auc"]}, AP={model_bundle["training_ap"]}')

        def predict_lnp(ionizable_smiles, helper_smiles, sterol_smiles, peg_smiles,
                        molar_ratio, size_nm=None, pdi=None,
                        model_bundle=None, model_path='lnp_classifier.pkl'):
            '''
            Predict EE% class and Formulation Quality Score for a new LNP.
            '''
            if model_bundle is None:
                with open(model_path,'rb') as f: model_bundle = pickle.load(f)

            clf       = model_bundle['model']
            active_fp = model_bundle['active_fp_cols']

            smiles_map = {'ionizable':ionizable_smiles,'helper':helper_smiles,
                          'sterol':sterol_smiles,'peg':peg_smiles}
            mols = {}
            for name, smi in smiles_map.items():
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None: return {'valid':False,'error':f'Invalid SMILES for {name}: {smi}'}
                mols[name] = mol

            try:
                parts = [float(x) for x in str(molar_ratio).strip().split(':')]
                assert len(parts)==4
                total = sum(parts)
                ion_f,peg_f,ste_f,hel_f = [p/total for p in parts]
            except Exception as e:
                return {'valid':False,'error':f'Invalid molar ratio: {e}'}

            fpgen_local = rdFingerprintGenerator.GetMorganGenerator(
                radius=model_bundle['fp_radius'], fpSize=model_bundle['fp_nbits'])

            def mol_fp(mol, prefix, nbits=512):
                fp  = fpgen_local.GetCountFingerprint(mol)
                arr = np.zeros(nbits, dtype=np.float32)
                for idx, cnt in fp.GetNonzeroElements().items(): arr[idx%nbits] += cnt
                return {f'{prefix}_fp{i}': arr[i] for i in range(nbits)}

            fp_row = {}
            for name in ['ionizable','helper','sterol','peg']:
                fp_row.update(mol_fp(mols[name], name))

            feat = {col: fp_row.get(col,0.0) for col in active_fp}
            feat.update({'ion_frac':ion_f,'peg_frac':peg_f,'sterol_frac':ste_f,'helper_frac':hel_f})
            X_new = pd.DataFrame([feat])[model_bundle['feature_names']]

            prob = float(clf.predict_proba(X_new)[0,1])
            is_high = prob >= model_bundle['decision_threshold']
            dist = abs(prob - 0.5)
            conf = 'High' if dist>0.25 else ('Moderate' if dist>0.10 else 'Low')

            fqs_r = compute_FQS(prob, size_nm=size_nm, pdi=pdi)
            fqs   = fqs_r['FQS']
            grade = 'Excellent' if fqs>=75 else ('Good' if fqs>=55 else ('Moderate' if fqs>=35 else 'Poor'))

            return {
                'prob_high_EE': round(prob,4),
                'prediction':   f'High EE (>={model_bundle["ee_threshold"]}%)' if is_high else f'Low EE',
                'confidence':   conf,
                'FQS':          fqs,
                'FQS_grade':    grade,
                'd_EE':         fqs_r['d_EE'],
                'd_size':       fqs_r['d_size'],
                'd_PDI':        fqs_r['d_PDI'],
                'FQS_components': fqs_r['components'],
                'valid':        True, 'error': None,
            }

        CHOLESTEROL = 'OC1CCC2(C)C(CCC3C2CC=C2C3(C)CCC(C(C)CCCC(C)C)C2)C1'
        DSPC        = 'CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[NH3+])OC(=O)CCCCCCCCCCCCCCCC'
        DMG_PEG     = 'CCCCCCCCCCCCCCCCCC(=O)OCC(OC(=O)CCCCCCCCCCCCCCCCC)COC(=O)OCC(O)COCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCC'
        MC3         = 'O=C(OCCC(OC(=O)CCCCCCC/C=C\\CCCCCCCC)COCCN(CC)CC)CCCCCCC/C=C\\CCCCCCCC'

        bundle = model_bundle
        print('Demo Prediction')
        print('=== MC3 — standard mRNA-LNP formulation ===')
        r = predict_lnp(MC3, DSPC, CHOLESTEROL, DMG_PEG, '50:1.5:38.5:10',
                        size_nm=85.0, pdi=0.12, model_bundle=bundle)
        for k,v in r.items(): print(f'  {k}: {v}')
        print()
        print('=== Same lipids, poor formulation (large size, high PDI) ===')
        r2 = predict_lnp(MC3, DSPC, CHOLESTEROL, DMG_PEG, '50:1.5:38.5:10',
                         size_nm=220.0, pdi=0.45, model_bundle=bundle)
        for k,v in r2.items(): print(f'  {k}: {v}')

        self.state.MC3 = MC3
        self.state.DSPC = DSPC
        self.state.CHOLESTEROL = CHOLESTEROL
        self.state.DMG_PEG = DMG_PEG
        self.state.bundle = bundle
        self.state.predict_lnp = predict_lnp
        return self.state

# ============================================================================
# SectionH ConformalPrediction
# ============================================================================
class ConformalPrediction:
    """
    Conformal prediction for calibrated, distribution-free uncertainty on new predictions.
    Reads from state:  all_y_true, all_y_prob, predict_lnp
    Writes to state:   q_hat, conformal_predict_set
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        all_y_true = self.state.all_y_true
        all_y_prob = self.state.all_y_prob
        predict_lnp = self.state.predict_lnp
        alpha = 0.10

        oof_idx = np.arange(len(all_y_true))
        calib_idx, conftest_idx = train_test_split(oof_idx, test_size=0.5, random_state=42, stratify=all_y_true)

        def nonconformity(y_true, prob_pos):
            p_true = np.where(y_true == 1, prob_pos, 1 - prob_pos)
            return 1 - p_true

        calib_scores = nonconformity(all_y_true[calib_idx], all_y_prob[calib_idx])
        n_calib = len(calib_scores)
        q_level = np.ceil((n_calib + 1) * (1 - alpha)) / n_calib
        q_hat = np.quantile(calib_scores, min(q_level, 1.0), method='higher')
        print(f'Conformal threshold q_hat = {q_hat:.3f} at alpha={alpha} (target coverage {1-alpha:.0%})')

        def conformal_predict_set(prob_pos, q_hat):
            labels = []
            if (1 - prob_pos) <= q_hat: labels.append(1)
            if prob_pos <= q_hat:       labels.append(0)
            return labels

        test_probs = all_y_prob[conftest_idx]
        test_true  = all_y_true[conftest_idx]
        pred_sets  = [conformal_predict_set(p, q_hat) for p in test_probs]
        set_sizes  = [len(s) for s in pred_sets]
        covered    = [t in s for t, s in zip(test_true, pred_sets)]

        print(f'\nEmpirical coverage on held-out conformal test split: {np.mean(covered):.3f} (target {1-alpha:.0%})')
        print(f'Prediction set sizes -- singleton (confident): {sum(1 for s in set_sizes if s==1)}/{len(set_sizes)}'
              f'  |  ambiguous (both labels): {sum(1 for s in set_sizes if s==2)}/{len(set_sizes)}')

        singleton_mask = np.array(set_sizes) == 1
        if singleton_mask.sum() > 0:
            singleton_acc = np.mean([t == s[0] for t, s, m in zip(test_true, pred_sets, singleton_mask) if m])
            print(f'Accuracy on confident (singleton) predictions: {singleton_acc:.3f}')

        def predict_lnp_conformal(*args, q_hat=q_hat, **kwargs):
            result = predict_lnp(*args, **kwargs)
            if not result.get('valid', True):
                return result
            pset = conformal_predict_set(result['prob_high_EE'], q_hat)
            label_map = {1: 'High EE', 0: 'Low EE'}
            result['conformal_set'] = [label_map[l] for l in pset]
            result['conformal_confident'] = len(pset) == 1
            return result

        self.state.q_hat = q_hat
        self.state.conformal_predict_set = conformal_predict_set
        return self.state


# ============================================================================
# SectionI ErrorAnalysis
# ============================================================================
class ErrorAnalysis:
    """
    Error analysis: where the model fails, and the Applicability Domain (AD) check.
    Reads from state:  PUB_PALETTE, save_pub_figure, clean_df, fp_matrix, all_y_true, all_y_prob, best_t
    Writes to state:   X_ion, ion_fp_cols, knn, ad_threshold
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        PUB_PALETTE = self.state.PUB_PALETTE
        save_pub_figure = self.state.save_pub_figure
        clean_df = self.state.clean_df
        fp_matrix = self.state.fp_matrix
        all_y_true = self.state.all_y_true
        all_y_prob = self.state.all_y_prob
        best_t = self.state.best_t

        ion_fp_cols = [c for c in fp_matrix.columns if c.startswith('ionizable_')]
        X_ion = fp_matrix[ion_fp_cols].values

        k = 5
        knn = NearestNeighbors(n_neighbors=k+1, metric='jaccard', n_jobs=-1)
        knn.fit(X_ion > 0)
        dists, _ = knn.kneighbors(X_ion > 0)
        knn_dists = dists[:, 1:].mean(axis=1)
        ad_threshold = np.percentile(knn_dists, 95)

        clean_df['knn_dist'] = knn_dists

        clean_df['pred_label']   = (all_y_prob >= best_t).astype(int)
        clean_df['true_label']   = all_y_true
        clean_df['pred_prob']    = all_y_prob

        TP = clean_df[(clean_df['true_label']==1) & (clean_df['pred_label']==1)]
        TN = clean_df[(clean_df['true_label']==0) & (clean_df['pred_label']==0)]
        FP = clean_df[(clean_df['true_label']==0) & (clean_df['pred_label']==1)]
        FN = clean_df[(clean_df['true_label']==1) & (clean_df['pred_label']==0)]

        print(f'TP: {len(TP)}  TN: {len(TN)}  FP: {len(FP)}  FN: {len(FN)}')

        print('\n=== FALSE POSITIVES (model says High EE, actually Low EE) ===')
        print(f'n={len(FP)}')
        if len(FP) > 0:
            print(f'  Mean true EE%:     {FP["EE"].mean():.1f}% (range {FP["EE"].min():.1f}-{FP["EE"].max():.1f})')
            print(f'  Mean pred prob:    {FP["pred_prob"].mean():.3f}')
            print(f'  Cargo types:       {FP["target_type"].value_counts().to_dict()}')
            if 'size_nm' in FP.columns:
                print(f'  Mean size (nm):    {FP["size_nm"].mean():.1f}')
            if 'knn_dist' in FP.columns:
                print(f'  Mean kNN dist:     {FP["knn_dist"].mean():.4f}  (higher = more novel)')
            print(f'  Unique ionizable lipids: {FP["ionizable_lipid_smiles"].nunique()}')

        print('\n=== FALSE NEGATIVES (model says Low EE, actually High EE) ===')
        print(f'n={len(FN)}')
        if len(FN) > 0:
            print(f'  Mean true EE%:     {FN["EE"].mean():.1f}% (range {FN["EE"].min():.1f}-{FN["EE"].max():.1f})')
            print(f'  Mean pred prob:    {FN["pred_prob"].mean():.3f}')
            print(f'  Cargo types:       {FN["target_type"].value_counts().to_dict()}')
            if 'size_nm' in FN.columns:
                print(f'  Mean size (nm):    {FN["size_nm"].mean():.1f}')
            if 'knn_dist' in FN.columns:
                print(f'  Mean kNN dist:     {FN["knn_dist"].mean():.4f}')
            print(f'  Unique ionizable lipids: {FN["ionizable_lipid_smiles"].nunique()}')

        fig, axes = plt.subplots(1, 3, figsize=(8, 3))
        colors = PUB_PALETTE[:16]
        axes[0].hist(TP['pred_prob'], bins=20, alpha=0.6, color=colors[0], edgecolor='white', label=f'TP (n={len(TP)})', density=True)
        axes[0].hist(FP['pred_prob'], bins=20, alpha=0.6, color=colors[1], edgecolor='white', label=f'FP (n={len(FP)})', density=True)
        axes[0].hist(TN['pred_prob'], bins=20, alpha=0.6, color=colors[2], edgecolor='white', label=f'TN (n={len(TN)})', density=True)
        axes[0].hist(FN['pred_prob'], bins=20, alpha=0.6, color=colors[3], edgecolor='white', label=f'FN (n={len(FN)})', density=True)
        axes[0].axvline(best_t, color='black', lw=1.5, ls='--', label=f'threshold={best_t:.2f}')
        axes[0].set_xlabel('Predicted Probability'); axes[0].set_ylabel('Density')
        axes[0].legend()
        axes[1].hist(FP['EE'], bins=20, alpha=0.6, color=colors[4], edgecolor='white', label=f'False Positives (n={len(FP)})')
        axes[1].hist(FN['EE'], bins=20, alpha=0.6, color=colors[5], edgecolor='white', label=f'False Negatives (n={len(FN)})')
        axes[1].axvline(80, color='black', lw=1.5, ls='--', label='EE threshold')
        axes[1].set_xlabel('True EE%'); axes[1].set_ylabel('Count')
        axes[1].legend()
        error_mask = clean_df['true_label'] != clean_df['pred_label']
        axes[2].hist(clean_df.loc[~error_mask, 'knn_dist'], bins=20, alpha=0.6, color=colors[6], label='Correct', density=True, edgecolor='white')
        axes[2].hist(clean_df.loc[error_mask, 'knn_dist'], bins=20, alpha=0.6, color=colors[5], label='Misclassified', density=True, edgecolor='white')
        axes[2].axvline(ad_threshold, color='black', lw=1.5, ls='--', label='AD threshold')
        axes[2].set_xlabel('kNN Distance (novelty)'); axes[2].set_ylabel('Density')
        axes[2].legend()
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec11I_error_analysis', width='double')
        plt.show()

        self.state.X_ion = X_ion
        self.state.ion_fp_cols = ion_fp_cols
        self.state.knn = knn
        self.state.ad_threshold = ad_threshold
        return self.state


# ============================================================================
# SectionJ LearningCurve
# ============================================================================
class LearningCurve:
    """
    Learning curve: is the model data-limited?
    Reads from state:  PUB_PALETTE, save_pub_figure, X_D, y, groups
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        PUB_PALETTE = self.state.PUB_PALETTE
        save_pub_figure = self.state.save_pub_figure
        X_D = self.state.X_D
        y = self.state.y
        groups = self.state.groups

        train_sizes_pct = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        n_repeats = 10
        lc_results = {sz: [] for sz in train_sizes_pct}
        rfc_lc = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)

        gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        train_pool_idx, test_idx = next(gss_outer.split(X_D, y, groups))
        X_pool, y_pool, g_pool = X_D.iloc[train_pool_idx], y[train_pool_idx], groups[train_pool_idx]
        X_test, y_test = X_D.iloc[test_idx], y[test_idx]

        unique_groups_pool = np.unique(g_pool)

        for rep in range(n_repeats):
            rng = np.random.RandomState(rep)
            rng.shuffle(unique_groups_pool)
            for sz in train_sizes_pct:
                n_groups = max(5, int(len(unique_groups_pool) * sz))
                sel_groups = unique_groups_pool[:n_groups]
                tr_mask = np.isin(g_pool, sel_groups)
                if tr_mask.sum() < 10: continue
                if len(np.unique(y_pool[tr_mask])) < 2: continue
                try:
                    rfc_lc.fit(X_pool.iloc[tr_mask], y_pool[tr_mask])
                    prob_lc = rfc_lc.predict_proba(X_test)[:,1]
                    lc_results[sz].append(roc_auc_score(y_test, prob_lc))
                except:
                    pass

        lc_means = [np.mean(lc_results[sz]) for sz in train_sizes_pct]
        lc_stds  = [np.std(lc_results[sz])  for sz in train_sizes_pct]
        n_samples_approx = [int(sz * len(train_pool_idx)) for sz in train_sizes_pct]

        print('Learning Curve Results:')
        for sz, n, m, s in zip(train_sizes_pct, n_samples_approx, lc_means, lc_stds):
            print(f'  {sz*100:.0f}% train ({n:>3} samples): AUC = {m:.3f} ± {s:.3f}')

        fig, ax = plt.subplots(figsize=(7, 3))
        color = PUB_PALETTE[:8]
        lc_means = np.array(lc_means); lc_stds = np.array(lc_stds)
        ax.plot(n_samples_approx, lc_means, 'd-', color=color[0], lw=1.5, ms=5)
        ax.fill_between(n_samples_approx, lc_means-lc_stds, lc_means+lc_stds, alpha=0.1, color=color[0])
        ax.axhline(0.5, ls='--', color='grey', lw=1, label='Random classifier AUC=0.5')
        ax.axvline(len(train_pool_idx), ls=':', color=color[1], lw=2, label=f'Full training set (n$\\approx${len(train_pool_idx)})')
        ax.set_xlabel('Number of Training Formulations')
        ax.set_ylabel('ROC-AUC (held-out 20\\% lipids)')
        ax.legend(loc='upper left')
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec11J_learning_curve', width='double')
        plt.show()

        return self.state


# ============================================================================
# Section12 SHAP
# ============================================================================
# FIX: one shared ester SMARTS pattern, used everywhere in this section instead of
# three slightly different ad hoc patterns ('[CX3](=O)[OX2H0][#6]' in the SMARTS
# enrichment library, 'C(=O)O[#6]' in the headgroup/linker/tail GROUPS dict, and the
# bare 'C(=O)O' used by the SHAP case-study edit). All three should be counting the
# same functional group.
ESTER_SMARTS = 'C(=O)O[#6]'

class SHAP:
    """
    SHAP interpretability: important bits, their chemistry, functional-group correlations, SMARTS enrichment, and a SHAP-guided design case study.
    Reads from state:  PUB_PALETTE, save_pub_figure, clean_df, fp_matrix, X_D, y, predict_lnp
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        PUB_PALETTE = self.state.PUB_PALETTE
        save_pub_figure = self.state.save_pub_figure
        clean_df = self.state.clean_df
        fp_matrix = self.state.fp_matrix
        X_D = self.state.X_D
        y = self.state.y
        predict_lnp = self.state.predict_lnp

        # Running collector for the GLOBAL multiple-testing correction at the end of
        # this section (FIX: previously each sub-analysis table corrected its own
        # p-values in isolation, which understates the true false-discovery rate when
        # dozens of overlapping tests are run across the section as a whole).
        global_pvals = []   # list of dicts: {'analysis': ..., 'feature': ..., 'p': ...}

        rf_final = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
        rf_final.fit(X_D, y)

        explainer = shap.TreeExplainer(rf_final)
        np.random.seed(42)
        sample_idx = np.random.choice(len(X_D), min(200, len(X_D)), replace=False)
        X_sample   = X_D.iloc[sample_idx]

        print('Computing SHAP values...')
        shap_values = explainer.shap_values(X_sample)

        if isinstance(shap_values, list):
            shap_high = np.array(shap_values[1])
        elif shap_values.ndim == 3:
            shap_high = shap_values[:, :, 1]
        else:
            shap_high = shap_values

        print(f'SHAP matrix: {shap_high.shape}. Done.')

        def assign_component(col):
            if col == 'ion_frac':
                return 'Ionizable fraction'
            elif col == 'helper_frac':
                return 'Helper fraction'
            elif col == 'sterol_frac':
                return 'Sterol fraction'
            elif col == 'peg_frac':
                return 'PEG fraction'
            for prefix in ['ionizable','helper','sterol','peg']:
                if col.startswith(f'{prefix}_fp'):
                    return f'{prefix} FP'
            return 'Other'

        mean_abs_shap = pd.Series(np.abs(shap_high).mean(axis=0), index=X_D.columns)
        comp_shap = mean_abs_shap.groupby(mean_abs_shap.index.map(assign_component)).sum().sort_values(ascending=False)
        print('\nMean |SHAP| by component:')
        print(comp_shap.round(4))

        top20_idx  = mean_abs_shap.sort_values(ascending=False).head(20).index
        X_top20    = X_sample[top20_idx]
        shap_top20 = shap_high[:, [X_D.columns.get_loc(c) for c in top20_idx]]

        fig = plt.figure(figsize=(5, 10))
        shap.summary_plot(shap_top20, X_top20, feature_names=list(top20_idx), show=False, plot_type='violin', color=plt.get_cmap('viridis'))

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#cccccc')
        ax.spines['bottom'].set_color('#cccccc')
        plt.xlabel('SHAP value | impact on model output', fontsize=7)
        plt.ylabel('Feature', fontsize=7)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec12_shap_summary', width='double')
        plt.show()

        fig, ax = plt.subplots(figsize=(7, 3))
        colors = PUB_PALETTE[1:5]
        comp_shap.sort_values().plot.barh(ax=ax, color=colors, edgecolor='white', alpha=0.6)
        ax.set_xlabel('Mean |SHAP value|')
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec12_shap_components', width='double')
        plt.show()

        top_ion_shap = (
            mean_abs_shap[mean_abs_shap.index.str.startswith('ionizable_fp')]
            .sort_values(ascending=False)
            .head(20)
        )

        def characterise_bit(bit_col, df, lipid_col='ionizable_lipid_smiles', top_n=5):
            bit_idx = int(bit_col.split('fp')[1])
            ion_fp_col = fp_matrix[[c for c in fp_matrix.columns if c.startswith('ionizable_')]]
            bit_vals = ion_fp_col[bit_col] if bit_col in ion_fp_col.columns else pd.Series(np.zeros(len(df)))
            active_mask = bit_vals > 0
            n_active = active_mask.sum()
            active_ee_mean = df.loc[active_mask.values[:len(df)], 'EE'].mean() if n_active > 0 else np.nan
            inactive_ee_mean = df.loc[~active_mask.values[:len(df)], 'EE'].mean()
            active_smiles = df.loc[active_mask.values[:len(df)], lipid_col].unique()[:top_n]
            fragment_counts = {}
            frag_names = [
                ('fr_NH2', 'Primary amine'), ('fr_NH1', 'Secondary amine'), ('fr_NH0', 'Tertiary amine'),
                ('fr_ester', 'Ester'), ('fr_ether', 'Ether'), ('fr_amide', 'Amide'),
                ('fr_alkyl_halide', 'Alkyl halide'), ('fr_ArN', 'Aromatic N'),
                ('fr_C_O', 'Carbonyl'), ('fr_Al_OH', 'Alcohol'),]
            for smi in active_smiles:
                mol = Chem.MolFromSmiles(str(smi))
                if mol is None: continue
                for attr, name in frag_names:
                    val = getattr(Fragments, attr)(mol)
                    fragment_counts[name] = fragment_counts.get(name, 0) + val

            top_frags = sorted(fragment_counts.items(), key=lambda x: -x[1])
            top_frag_str = ', '.join([f'{n}({v})' for n, v in top_frags[:4] if v > 0])

            return {
                'bit': bit_col,
                'n_active': n_active,
                'pct_active': f'{n_active/len(df)*100:.1f}%',
                'active_mean_EE': round(active_ee_mean, 1) if not np.isnan(active_ee_mean) else 'N/A',
                'inactive_mean_EE': round(inactive_ee_mean, 1),
                'delta_EE': round(active_ee_mean - inactive_ee_mean, 1) if not np.isnan(active_ee_mean) else 'N/A',
                'top_fragments': top_frag_str if top_frag_str else 'No common fragments',
            }

        print('Characterising top SHAP fingerprint bits...')
        char_results = []
        for bit_col in top_ion_shap.index[:15]:
            if bit_col in fp_matrix.columns:
                result = characterise_bit(bit_col, clean_df)
                result['shap'] = round(top_ion_shap[bit_col], 5)
                char_results.append(result)

        char_df = pd.DataFrame(char_results)
        print('\n=== Top Ionizable Lipid SHAP Bits — Chemical Interpretation ===')
        print(char_df[['bit','shap','pct_active','active_mean_EE','inactive_mean_EE','delta_EE','top_fragments']].to_string(index=False))
        print('\n\\Delta EE = mean EE when bit active MINUS mean EE when inactive')
        print('Positive \\Delta EE = this substructure correlates with higher EE%')

        frag_defs = [
            ('fr_NH0',       'Tertiary amine'),
            ('fr_NH1',       'Secondary amine'),
            ('fr_NH2',       'Primary amine'),
            ('fr_ester',     'Ester linkage'),
            ('fr_ether',     'Ether linkage'),
            ('fr_amide',     'Amide'),
            ('fr_C_O',       'Carbonyl (C=O)'),
            ('fr_Al_OH',     'Aliphatic OH'),
            ('fr_ArN',       'Aromatic N'),
            ('fr_unbrch_alkane', 'Unbranched alkyl chain'),
        ]

        frag_data = []
        for _, row in clean_df.iterrows():
            mol = Chem.MolFromSmiles(str(row['ionizable_lipid_smiles']))
            if mol is None: continue
            entry = {'EE': row['EE'], 'label': int(row['EE'] >= 80)}
            for attr, name in frag_defs:
                entry[name] = getattr(Fragments, attr)(mol)
            entry['RotBonds'] = Lipinski.NumRotatableBonds(mol)
            entry['RingCount'] = Lipinski.RingCount(mol)
            entry['LogP'] = Descriptors.MolLogP(mol)
            entry['MolWt'] = Descriptors.MolWt(mol)
            frag_data.append(entry)

        frag_df = pd.DataFrame(frag_data)

        feature_cols = [name for _, name in frag_defs] + ['RotBonds', 'RingCount', 'LogP', 'MolWt']
        corr_results = []
        for col in feature_cols:
            if frag_df[col].std() == 0: continue
            r, p = stats.pointbiserialr(frag_df['label'], frag_df[col])
            corr_results.append({'Feature': col, 'r': r, 'p': p,
                                  'high_EE_mean': frag_df.loc[frag_df['label']==1, col].mean(),
                                  'low_EE_mean':  frag_df.loc[frag_df['label']==0, col].mean()})
        corr_df = pd.DataFrame(corr_results).sort_values('r', ascending=False)

        corr_df['q_fdr'] = multipletests(corr_df['p'], method='fdr_bh')[1]
        corr_df['FDR_sig'] = corr_df['q_fdr'] < 0.05
        print('=== Functional Group Correlation with High EE% (point-biserial r) ===')
        print(corr_df[['Feature', 'r', 'p', 'q_fdr', 'FDR_sig', 'high_EE_mean', 'low_EE_mean']].sort_values('q_fdr').to_string(index=False))
        n_sig = corr_df['FDR_sig'].sum()
        print(
            f"\n{n_sig} of {len(corr_df)} features remain "
            f"significant after FDR correction.")
        global_pvals += [{'analysis': 'point_biserial_functional_group', 'feature': r['Feature'], 'p': r['p']}
                          for _, r in corr_df.iterrows()]

        fig, ax = plt.subplots(figsize=(7, 3))
        color = PUB_PALETTE[:8]
        colors = [color[0] if r > 0 else color[1] for r in corr_df['r']]
        bars = ax.barh(range(len(corr_df)), corr_df['r'], color=colors, alpha=0.6, edgecolor='white')
        ax.set_yticks(range(len(corr_df)))
        ax.set_yticklabels(corr_df['Feature'])
        ax.axvline(0, color='black', lw=2)
        ax.set_xlabel('Point-Biserial Correlation with High EE% (r)')
        for i, (_, row) in enumerate(corr_df.iterrows()):
            q = row['q_fdr']
            sig = (
                '***' if q < 0.001 else
                '**'  if q < 0.01 else
                '*'   if q < 0.05 else
                ''
            )
            if sig:
                x = row['r'] + (0.01 if row['r'] >= 0 else -0.03)
                ax.text(x, i, sig, va='center', color='black')

        ax.text(0.5, -0.18, '* q<0.05   ** q<0.01   *** q<0.001 (FDR corrected)', transform=ax.transAxes, ha='center', fontsize=8, color='dimgray', style='italic')
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec12A1_chemistry_correlation', width='double')
        plt.show()

        spearman_results = []
        for col in feature_cols:
            if frag_df[col].std() == 0:
                continue
            rho, p = spearmanr(frag_df['EE'], frag_df[col])
            spearman_results.append({
                'Feature': col,
                'rho': rho,
                'p': p,
                'mean_value': frag_df[col].mean()
            })
        spearman_df = (pd.DataFrame(spearman_results).sort_values('rho', ascending=False))
        spearman_df['q_fdr'] = multipletests(spearman_df['p'], method='fdr_bh')[1]
        spearman_df['FDR_sig'] = spearman_df['q_fdr'] < 0.05
        global_pvals += [{'analysis': 'spearman_functional_group', 'feature': r['Feature'], 'p': r['p']}
                          for _, r in spearman_df.iterrows()]

        print(f'\n=== Functional Group Correlation with Continuous EE (Spearman $\\rho$) ===')
        print(spearman_df[['Feature','rho','p','q_fdr','FDR_sig']].sort_values('q_fdr').to_string(index=False))

        fig, ax = plt.subplots(figsize=(7, 3))
        colors = [PUB_PALETTE[0] if rho > 0 else PUB_PALETTE[1] for rho in spearman_df['rho']]
        ax.barh(range(len(spearman_df)), spearman_df['rho'], color=colors, alpha=0.6, edgecolor='white')
        ax.set_yticks(range(len(spearman_df)))
        ax.set_yticklabels(spearman_df['Feature'])
        ax.axvline(0, color='black', lw=2)
        ax.set_xlabel(f'Spearman Correlation with EE ($\\rho$)')

        for i, (_, row) in enumerate(spearman_df.iterrows()):
            q = row['q_fdr']
            sig = (
                    '***' if q < 0.001 else
                    '**'  if q < 0.01 else
                    '*'   if q < 0.05 else
                    ''
                    )
            if sig:
                x = row['rho'] + (0.01 if row['rho'] >= 0 else -0.03)
                ax.text(x, i, sig, va='center')
        ax.text(0.5, -0.18, '* q<0.05   ** q<0.01   *** q<0.001 (FDR corrected)', transform=ax.transAxes, ha='center', fontsize=8, color='dimgray', style='italic')
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec12A2_chemistry_spearman', width='double')
        plt.show()

        combined_df = corr_df[['Feature', 'r', 'p']].merge(spearman_df[['Feature', 'rho']], on='Feature')
        print(combined_df.sort_values('rho', ascending=False))

        print('=== Chemical Design Rules for High EE% LNP Formulation ===\n')

        rules = []
        for col in feature_cols:
            if col not in frag_df.columns: continue
            pos = frag_df.loc[frag_df[col] > 0, 'EE']
            neg = frag_df.loc[frag_df[col] == 0, 'EE']
            if len(pos) < 5 or len(neg) < 5: continue
            stat, p = stats.mannwhitneyu(pos, neg, alternative='two-sided')
            direction = 'increases' if pos.mean() > neg.mean() else 'decreases'
            rules.append({
                'Feature': col,
                'direction': direction,
                'present_mean': round(pos.mean(), 1),
                'absent_mean':  round(neg.mean(), 1),
                'delta':  round(pos.mean() - neg.mean(), 1),
                'p_mwu':  round(p, 4),
                'n_present': len(pos),
            })

        rules_df = pd.DataFrame(rules).sort_values('delta', ascending=False)
        rules_df['q_fdr'] = multipletests(rules_df['p_mwu'], method='fdr_bh')[1]
        rules_df['FDR_sig'] = rules_df['q_fdr'] < 0.05
        global_pvals += [{'analysis': 'mwu_design_rule', 'feature': r['Feature'], 'p': r['p_mwu']}
                          for _, r in rules_df.iterrows()]
        print("\n=== MWU Design Rules After FDR Correction ===")
        print(rules_df[['Feature','delta','p_mwu','q_fdr','FDR_sig']].sort_values('q_fdr').to_string(index=False))
        print(
            f"\n{rules_df['FDR_sig'].sum()} of "
            f"{len(rules_df)} rules remain significant after FDR correction."
        )

        print('POSITIVE RULES (presence → higher EE%):')
        pos_rules = rules_df[(rules_df['delta'] > 2) & (rules_df['FDR_sig'])].head(6)
        for _, r in pos_rules.iterrows():
            sig = '(q={:.3f})'.format(r['q_fdr'])
            print(f'  ✓ Presence of {r["Feature"]:22s}: mean EE {r["present_mean"]}% vs {r["absent_mean"]}%  \\Delta ={r["delta"]:+.1f}%  {sig}')

        print('\nNEGATIVE RULES (presence → lower EE%):')
        neg_rules = rules_df[(rules_df['delta'] < -2) & (rules_df['FDR_sig'])].tail(6)
        for _, r in neg_rules.iterrows():
            sig = '(q={:.3f})'.format(r['q_fdr'])
            print(f'  ✗ Presence of {r["Feature"]:22s}: mean EE {r["present_mean"]}% vs {r["absent_mean"]}%  \\Delta ={r["delta"]:+.1f}%  {sig}')

        print('\n=== Molar Ratio Rules ===')
        for frac in ['ion_frac', 'peg_frac', 'sterol_frac', 'helper_frac']:
            r, p = stats.spearmanr(clean_df[frac], clean_df['EE'])
            direction = 'higher' if r > 0 else 'lower'
            print(f'  {frac:<15}: $\\rho$={r:+.3f} (p={p}) — {direction} fraction → {direction} EE%')
            global_pvals.append({'analysis': 'molar_ratio_spearman', 'feature': frac, 'p': p})

        plot_df = rules_df.copy()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = [PUB_PALETTE[0] if x > 0 else PUB_PALETTE[1] for x in plot_df['delta']]
        bars = ax.barh(plot_df['Feature'], plot_df['delta'], color=colors, alpha=0.6, edgecolor='white')
        ax.axvline(0, color='black', lw=1.5)
        ax.set_xlabel(r'$\Delta$EE (%) = EE$_{present}$ - EE$_{absent}$')
        ax.set_ylabel('Functional Group')

        for i, (_, row) in enumerate(plot_df.iterrows()):
            q = row['q_fdr']
            sig = (
                '***' if q < 0.001 else
                '**'  if q < 0.01 else
                '*'   if q < 0.05 else
                ''
            )
            if sig:
                x = row['delta']
                offset = 0.5 if x >= 0 else -1.5
                ax.text(x + offset, i, sig, fontsize=10, va='center')
        ax.text(0.5, -0.14, '* q<0.05 ** q<0.01 *** q<0.001 (FDR corrected)', transform=ax.transAxes, ha='center', fontsize=8, color='dimgray', style='italic')
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec12_mwu_deltaEE', width='double')
        plt.show()

        plot_df = rules_df.copy()
        plot_df['neglog10p'] = -np.log10(plot_df['p_mwu'])
        fig, ax = plt.subplots(figsize=(6,4))
        colors = [PUB_PALETTE[0] if d > 0 else PUB_PALETTE[1] for d in plot_df['delta']]
        ax.scatter(plot_df['delta'], plot_df['neglog10p'], c=colors, s=80, alpha=0.8)
        ax.axvline(0, color='black', lw=1)
        ax.axhline(-np.log10(0.05), color='grey', ls='--')
        for _, row in plot_df.iterrows():
            if row['p_mwu'] < 0.05:
                ax.text(row['delta'], row['neglog10p'] + 0.05, row['Feature'], fontsize=8)
        ax.set_xlabel(r'$\Delta$EE (%)')
        ax.set_ylabel(r'$-\log_{10}(p_{MWU})$')
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec12_volcano_mwu', width='double')
        plt.show()

        # --------------------------- Systematic SMARTS Substructure Enrichment ---------------------------
        SMARTS_LIB = {
            'Tertiary amine (aliphatic)':          '[NX3;H0;!$(N=*);!$(N-a)]([#6])([#6])[#6]',
            'Secondary amine (aliphatic)':         '[NX3;H1;!$(N=*);!$(N-a)]',
            'Ester linker':                        ESTER_SMARTS,
            'Amide linker':                        '[CX3](=O)[NX3]',
            'Ether linker':                        '[OD2]([#6])[#6]',
            'Branched sp3 carbon':                 '[CX4;H0,H1;!R]([#6])([#6])[#6]',
            'Cyclic headgroup (N in ring)':        '[NX3]1[#6][#6][#6][#6][#6]1',
            'Hydroxyl group':                      '[OX2H]',
            'Disulfide/thioether':                 '[#16X2]',
            'Terminal alkene (unsaturated tail)':  '[CX3]=[CX3]',
        }

        def match_present(smi, smarts):
            mol = Chem.MolFromSmiles(str(smi)); patt = Chem.MolFromSmarts(smarts)
            if mol is None or patt is None: return False
            return len(mol.GetSubstructMatches(patt)) > 0

        rows = []
        for name, smarts in SMARTS_LIB.items():
            present = clean_df['ionizable_lipid_smiles'].apply(lambda s: match_present(s, smarts))
            n_present = present.sum()
            if n_present < 5 or n_present > len(clean_df) - 5: continue
            high = clean_df['EE'] >= 80
            ct = pd.crosstab(present, high)
            if ct.shape != (2, 2): continue
            chi2, p, dof, exp = chi2_contingency(ct)
            rows.append({'Motif': name, 'n_present': int(n_present), 'pct_present': round(n_present/len(clean_df)*100, 1),
                         'EE_present': round(clean_df.loc[present, 'EE'].mean(), 1),
                         'EE_absent': round(clean_df.loc[~present, 'EE'].mean(), 1),
                         'delta_EE': round(clean_df.loc[present, 'EE'].mean() - clean_df.loc[~present, 'EE'].mean(), 1),
                         'chi2': round(chi2, 2), 'p': p})
        smarts_df = pd.DataFrame(rows).sort_values('p').reset_index(drop=True)
        smarts_df['p_bonferroni'] = np.minimum(smarts_df['p'] * len(smarts_df), 1.0)
        global_pvals += [{'analysis': 'smarts_enrichment', 'feature': r['Motif'], 'p': r['p']}
                          for _, r in smarts_df.iterrows()]
        smarts_df['p'] = smarts_df['p'].round(5); smarts_df['p_bonferroni'] = smarts_df['p_bonferroni'].round(4)
        print('=== SMARTS Substructure Enrichment in High-EE (>=80%) Ionizable Lipids ===')
        print(smarts_df.to_string(index=False))

        fig, ax = plt.subplots(figsize=(7, 3))
        color = PUB_PALETTE[:8]
        sig = smarts_df['p_bonferroni'] < 0.05
        colors = [color[0] if (d > 0 and s) else (color[1] if (d < 0 and s) else color[6])
                  for d, s in zip(smarts_df['delta_EE'], sig)]
        ax.barh(smarts_df['Motif'], smarts_df['delta_EE'], color=colors, edgecolor='white', alpha=0.6)
        ax.axvline(0, color=color[7], lw=1.5, ls='--')
        ax.set_xlabel(r'$\Delta$EE (%) = EE$_{present}$ − EE$_{absent}$')
        plt.tight_layout()
        plt.legend(handles=[plt.Line2D([0], [0], color=color[0], lw=6, label='Positive $\\Delta$EE (Bonferroni p<0.05)'),
                            plt.Line2D([0], [0], color=color[1], lw=6, label='Negative $\\Delta$EE (Bonferroni p<0.05)'),
                            plt.Line2D([0], [0], color=color[6], lw=6, label='Not significant')], frameon=False)
        save_pub_figure(fig, 'fig_sec12C_smarts_enrichment', width='double')
        plt.show()

        # --------------------------- HEADGROUP / LINKER / TAIL ANALYSIS ---------------------------
        GROUPS = {
            "OH": "[OX2H]",
            "Ether": "[#6]-O-[#6]",
            "Ester": ESTER_SMARTS,
            "Amide": "C(=O)N",
        }

        def classify_distance(dist):
            if dist <= 4:
                return "Headgroup"
            elif dist <= 8:
                return "Linker"
            else:
                return "Tail"

        def count_group_locations(smiles, smarts):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return {"Headgroup": 0, "Linker": 0, "Tail": 0}
            patt = Chem.MolFromSmarts(smarts)
            matches = mol.GetSubstructMatches(patt)
            if len(matches) == 0:
                return {"Headgroup": 0, "Linker": 0, "Tail": 0}
            nitrogen_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7]
            if len(nitrogen_atoms) == 0:
                return {"Headgroup": 0, "Linker": 0, "Tail": 0}
            result = {"Headgroup": 0, "Linker": 0, "Tail": 0}
            for match in matches:
                anchor = match[0]
                nearest = 999
                for n in nitrogen_atoms:
                    if n == anchor:
                        dist = 0
                    else:
                        try:
                            path = Chem.rdmolops.GetShortestPath(mol, n, anchor)
                            dist = len(path) - 1
                        except:
                            continue
                    nearest = min(nearest,dist)
                region = classify_distance(nearest)
                result[region] += 1
            return result

        loc_df = pd.DataFrame()
        loc_df["EE"] = clean_df["EE"]
        loc_df["HIGH_EE"] = (clean_df["EE"] >= 80).astype(int)
        for group_name, smarts in GROUPS.items():
            head_vals = []
            linker_vals = []
            tail_vals = []
            for smi in clean_df["ionizable_lipid_smiles"]:
                locations = count_group_locations(smi, smarts)
                head_vals.append(int(locations["Headgroup"] > 0))
                linker_vals.append(int(locations["Linker"] > 0))
                tail_vals.append(int(locations["Tail"] > 0))
            loc_df[f"{group_name}_Headgroup"] = head_vals
            loc_df[f"{group_name}_Linker"] = linker_vals
            loc_df[f"{group_name}_Tail"] = tail_vals

        EE = loc_df["EE"].values
        HIGH_EE = loc_df["HIGH_EE"].values
        results = []
        feature_cols = [c for c in loc_df.columns if c not in ["EE", "HIGH_EE"]]
        for feature in feature_cols:
            x = loc_df[feature].values
            if len(np.unique(x)) < 2:
                continue
            n_present = int(np.sum(x))
            prevalence = 100 * np.mean(x)
            try:
                rho, p_rho = spearmanr(x, EE)
            except:
                rho = np.nan
                p_rho = np.nan
            ee_present = EE[x == 1]
            ee_absent = EE[x == 0]
            if (len(ee_present) < 3 or len(ee_absent) < 3):
                continue
            delta_EE = (np.mean(ee_present) - np.mean(ee_absent))
            _, p_mwu = mannwhitneyu(ee_present, ee_absent, alternative="two-sided")
            contingency = pd.crosstab(x, HIGH_EE)
            try:
                chi2, p_chi2, _, _ = chi2_contingency(contingency)
                n = contingency.values.sum()
                phi = np.sqrt(chi2 / n)
            except:
                chi2 = np.nan
                p_chi2 = np.nan
                phi = np.nan

            results.append({
                "Feature": feature, "n_present": n_present, "Prevalence_%": prevalence,
                "Phi": phi, "Chi2": chi2, "p_chi2": p_chi2,
                "rho_spearman": rho, "p_spearman":p_rho, "Delta_EE": delta_EE, "p_mwu": p_mwu
            })

        results_df = pd.DataFrame(results)
        MIN_COUNT = 10
        results_df = results_df[results_df["n_present"] >= MIN_COUNT].copy()
        results_df["q_fdr"] = multipletests(results_df["p_mwu"], method="fdr_bh")[1]
        results_df["p_bonf"] = multipletests(results_df["p_mwu"], method="bonferroni")[1]
        results_df["FDR_sig"] = (results_df["q_fdr"] < 0.05)
        results_df["Bonf_sig"] = (results_df["p_bonf"] < 0.05)
        global_pvals += [{'analysis': 'headgroup_linker_tail_presence', 'feature': r['Feature'], 'p': r['p_mwu']}
                          for _, r in results_df.iterrows()]
        results_df = results_df.sort_values("Delta_EE", ascending=False)

        print("\n")
        print("=" * 120)
        print("HEADGROUP / LINKER / TAIL ANALYSIS")
        print("=" * 120)
        display_cols = ["Feature","n_present","Prevalence_%","Phi","rho_spearman","Delta_EE","q_fdr","p_bonf","FDR_sig","Bonf_sig"]
        print(results_df[display_cols].round(3).to_string(index=False))
        results_df.to_csv("table_headgroup_linker_tail_analysis.csv", index=False)
        print("\nSaved: table_headgroup_linker_tail_analysis.csv")

        # --------------------------- MECHANISTIC TOPOLOGY ANALYSIS ---------------------------
        HIGH_EE_THRESHOLD = 80
        MIN_COUNT = 10
        GROUPS = {"OH": "[OX2H]", "Ether": "[#6]-O-[#6]", "Ester": ESTER_SMARTS, "Amide": "C(=O)N"}

        def nearest_distance(smiles, smarts):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return np.nan
            patt = Chem.MolFromSmarts(smarts)
            matches = mol.GetSubstructMatches(patt)
            if len(matches) == 0: return np.nan
            N_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
            if len(N_atoms) == 0: return np.nan
            best = 999
            for match in matches:
                anchor = match[0]
                for n in N_atoms:
                    if n == anchor:
                        dist = 0
                    else:
                        try:
                            path = Chem.rdmolops.GetShortestPath(mol, n, anchor)
                            dist = len(path) - 1
                        except:
                            continue
                    best = min(best, dist)
            return best

        def distance_bin(x):
            if pd.isna(x): return np.nan
            if x <= 2: return "0-2"
            elif x <= 5: return "3-5"
            elif x <= 8: return "6-8"
            else: return ">8"

        def region(x):
            if pd.isna(x): return np.nan
            if x <= 4: return "Headgroup"
            elif x <= 8: return "Linker"
            else: return "Tail"

        topo_df = pd.DataFrame()
        topo_df["EE"] = clean_df["EE"]
        topo_df["HIGH_EE"] = (clean_df["EE"] >= HIGH_EE_THRESHOLD).astype(int)
        topo_df["MolWt"] = (clean_df["ionizable_lipid_smiles"].apply(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s))))
        topo_df["LogP"] = (clean_df["ionizable_lipid_smiles"].apply(lambda s: Descriptors.MolLogP(Chem.MolFromSmiles(s))))
        for group, smarts in GROUPS.items():
            topo_df[f"{group}_dist"] = (clean_df["ionizable_lipid_smiles"].apply(lambda x: nearest_distance(x, smarts)))
            topo_df[f"{group}_bin"] = (topo_df[f"{group}_dist"].apply(distance_bin))
            topo_df[f"{group}_region"] = (topo_df[f"{group}_dist"].apply(region))

        corr_rows = []
        for group in GROUPS:
            col = f"{group}_dist"
            subset = topo_df[["EE", col]].dropna()
            if len(subset) < 15: continue
            rho,p = spearmanr(subset[col], subset["EE"])
            corr_rows.append({"Feature":group, "n":len(subset), "rho":rho, "p":p})
        corr_df = pd.DataFrame(corr_rows)
        global_pvals += [{'analysis': 'topology_distance_spearman', 'feature': r['Feature'], 'p': r['p']}
                          for _, r in corr_df.iterrows()]
        corr_df.to_csv("table_graph_distance_correlations.csv", index=False)
        print("\n=== CONTINUOUS DISTANCE CORRELATIONS ===")
        print(corr_df.round(3))

        bin_results = []
        for group in GROUPS:
            for b in ["0-2","3-5","6-8",">8"]:
                indicator = (topo_df[f"{group}_bin"] == b).astype(int)
                n_present = int(indicator.sum())
                if n_present < MIN_COUNT: continue
                ee_present = topo_df.loc[indicator==1, "EE"]
                ee_absent = topo_df.loc[indicator==0, "EE"]
                delta = (ee_present.mean() - ee_absent.mean())
                _, p_mwu = mannwhitneyu(ee_present, ee_absent)
                contingency = pd.crosstab(indicator, topo_df["HIGH_EE"])
                chi2,p_chi2,_,_ = (chi2_contingency(contingency))
                phi = np.sqrt(chi2 / contingency.values.sum())
                bin_results.append({"Feature": group, "Distance_bin": b, "n_present": n_present,
                                     "Mean_EE": ee_present.mean(), "Median_EE": ee_present.median(),
                                     "Delta_EE": delta, "Phi": phi, "p_chi2": p_chi2, "p_mwu": p_mwu})

        bin_df = pd.DataFrame(bin_results)
        bin_df["q_fdr"] = multipletests(bin_df["p_mwu"], method="fdr_bh")[1]
        bin_df["p_bonf"] = multipletests(bin_df["p_mwu"], method="bonferroni")[1]
        bin_df["FDR_sig"] = (bin_df["q_fdr"] < 0.05)
        bin_df["Bonf_sig"] = (bin_df["p_bonf"] < 0.05)
        global_pvals += [{'analysis': 'topology_distance_bin', 'feature': f"{r['Feature']}_{r['Distance_bin']}", 'p': r['p_mwu']}
                          for _, r in bin_df.iterrows()]
        bin_df.to_csv("table_distance_bins.csv", index=False)
        print("\n=== DISTANCE BIN ANALYSIS ===")
        print(bin_df.sort_values("Delta_EE", ascending=False).round(3))

        region_results = []
        for group in GROUPS:
            for reg in ["Headgroup", "Linker", "Tail"]:
                indicator = (topo_df[f"{group}_region"] == reg).astype(int)
                n_present = int(indicator.sum())
                if n_present < MIN_COUNT: continue
                ee_present = topo_df.loc[indicator==1, "EE"]
                ee_absent = topo_df.loc[indicator==0, "EE"]
                delta = (ee_present.mean() - ee_absent.mean())
                _, p_mwu = mannwhitneyu(ee_present, ee_absent)
                contingency = pd.crosstab(indicator, topo_df["HIGH_EE"])
                chi2,p_chi2,_,_ = (chi2_contingency(contingency))
                phi = np.sqrt(chi2 / contingency.values.sum())
                rho,p_rho = spearmanr(indicator, topo_df["EE"])
                region_results.append({"Feature": f"{group}_{reg}", "n_present": n_present,
                                        "Prevalence_%": 100*np.mean(indicator), "Phi": phi,
                                        "rho_spearman": rho, "Delta_EE": delta, "p_mwu": p_mwu})

        region_df = pd.DataFrame(region_results)
        region_df["q_fdr"] = multipletests(region_df["p_mwu"], method="fdr_bh")[1]
        region_df["p_bonf"] = multipletests(region_df["p_mwu"], method="bonferroni")[1]
        global_pvals += [{'analysis': 'headgroup_linker_tail_region', 'feature': r['Feature'], 'p': r['p_mwu']}
                          for _, r in region_df.iterrows()]
        region_df.to_csv("table_headgroup_linker_tail.csv", index=False)

        confound_rows = []
        for group in GROUPS:
            dcol = f"{group}_dist"
            subset = topo_df[[dcol,"MolWt"]].dropna()
            if len(subset) > 15:
                rho,p = spearmanr(subset[dcol], subset["MolWt"])
                confound_rows.append({"Feature":group, "Confounder":"MolWt", "rho":rho, "p":p})
            subset = topo_df[[dcol,"LogP"]].dropna()
            if len(subset) > 15:
                rho,p = spearmanr(subset[dcol],subset["LogP"])
                confound_rows.append({"Feature":group, "Confounder":"LogP", "rho":rho, "p":p})

        confound_df = pd.DataFrame(confound_rows)
        confound_df.to_csv("table_distance_confounding.csv", index=False)
        print("\n=== DISTANCE CONFOUNDING ===")
        print(confound_df.round(3))

        # --------------------------- Adjusted Regression Analysis ---------------------------
        reg_results = []
        for group in GROUPS:
            dcol = f"{group}_dist"
            tmp = topo_df[["EE", dcol, "MolWt", "LogP"]].dropna()
            if len(tmp) < 20: continue
            model = smf.ols(f"EE ~ {dcol} + MolWt + LogP", data=tmp).fit()
            reg_results.append({"Feature": group, "Beta_distance": model.params[dcol],
                                 "p_distance": model.pvalues[dcol], "R2": model.rsquared})

        reg_df = pd.DataFrame(reg_results)
        reg_df["q_fdr"] = multipletests(reg_df["p_distance"],method="fdr_bh")[1]
        global_pvals += [{'analysis': 'topology_adjusted_regression', 'feature': r['Feature'], 'p': r['p_distance']}
                          for _, r in reg_df.iterrows()]
        reg_df.to_csv("table_adjusted_regression.csv",index=False)
        print("\n=== ADJUSTED REGRESSION ===")
        print(reg_df.round(4))

        order = ["0-2", "3-5", "6-8", ">8"]
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.barplot(data=bin_df, x="Distance_bin", y="Delta_EE", hue="Feature", order=order)
        ax.axhline(0, color="black", linestyle="--")
        ax.set_ylabel("\\Delta EE (%)")
        plt.tight_layout()
        save_pub_figure(fig, "fig_distance_bin_analysis", width="double")
        plt.show()
        print("\nSaved:")
        print(" table_graph_distance_correlations.csv")
        print(" table_distance_bins.csv")
        print(" table_headgroup_linker_tail.csv")
        print(" table_distance_confounding.csv")
        print(" table_adjusted_regression.csv")
        print(" fig_distance_bin_analysis.png")

        # --------------------------- RIGOROUS OH MECHANISTIC ANALYSIS ---------------------------
        OH_PATTERN = Chem.MolFromSmarts("[OX2H]")

        def count_oh(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return np.nan
            return len(mol.GetSubstructMatches(OH_PATTERN))

        def nearest_oh_distance(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return np.nan
            oh_matches = mol.GetSubstructMatches(OH_PATTERN)
            if len(oh_matches) == 0: return np.nan
            N_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
            if len(N_atoms) == 0: return np.nan
            best = 999
            for match in oh_matches:
                O_idx = match[0]
                for n in N_atoms:
                    try:
                        path = Chem.rdmolops.GetShortestPath(mol, O_idx, n)
                        dist = len(path) - 1
                        best = min(best, dist)
                    except:
                        continue
            return best

        def headgroup_oh_count(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return np.nan
            oh_matches = mol.GetSubstructMatches(OH_PATTERN)
            N_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
            if len(oh_matches) == 0: return 0
            if len(N_atoms) == 0: return 0
            count = 0
            for match in oh_matches:
                O_idx = match[0]
                nearest = 999
                for n in N_atoms:
                    try:
                        path = Chem.rdmolops.GetShortestPath(mol, O_idx, n)
                        dist = len(path) - 1
                        nearest = min(nearest, dist)
                    except:
                        continue
                if nearest <= 4:
                    count += 1
            return count

        oh_df = pd.DataFrame()
        oh_df["EE"] = clean_df["EE"]
        oh_df["MolWt"] = (clean_df["ionizable_lipid_smiles"].apply(lambda s: Descriptors.MolWt(Chem.MolFromSmiles(s))))
        oh_df["LogP"] = (clean_df["ionizable_lipid_smiles"].apply(lambda s: Descriptors.MolLogP(Chem.MolFromSmiles(s))))
        oh_df["OH_count"] = (clean_df["ionizable_lipid_smiles"].apply(count_oh))
        oh_df["OH_distance"] = (clean_df["ionizable_lipid_smiles"].apply(nearest_oh_distance))
        oh_df["Headgroup_OH_count"] = (clean_df["ionizable_lipid_smiles"].apply(headgroup_oh_count))

        oh_df = oh_df[oh_df["OH_count"] > 0].copy()

        print("\n"); print("="*80); print("OH CORRELATIONS"); print("="*80)
        for col in ["OH_count", "OH_distance", "Headgroup_OH_count"]:
            rho,p = spearmanr(oh_df[col], oh_df["EE"])
            print(f"{col:20s} rho={rho:+.3f} p={p:.5f}")
            global_pvals.append({'analysis': 'OH_correlation', 'feature': col, 'p': p})

        print("\n"); print("="*80); print("OH COUNT EFFECT"); print("="*80)
        summary = (oh_df.groupby("OH_count")["EE"].agg(["count","mean","median","std"]))
        print(summary.round(2))

        groups_kw = [g["EE"].values for _,g in oh_df.groupby("OH_count") if len(g) >= 3]
        if len(groups_kw) >= 2:
            H,p = kruskal(*groups_kw)
            print(f"\nKruskal-Wallis: H={H:.3f} p={p:.5g}")
            global_pvals.append({'analysis': 'OH_count_kruskal', 'feature': 'OH_count', 'p': p})

        print("\n"); print("="*80); print("DISTANCE EFFECT"); print("="*80)
        print(oh_df.groupby(pd.cut(oh_df["OH_distance"], bins=[0,2,5,8,20]))["EE"].agg(["count","mean","median","std"]))

        print("\n"); print("="*80); print("MODEL 1"); print("EE ~ OH_COUNT"); print("="*80)
        m1 = smf.ols("EE ~ OH_count", data=oh_df).fit()
        print(m1.summary())
        global_pvals.append({'analysis': 'OH_model1_ols', 'feature': 'OH_count', 'p': m1.pvalues.get('OH_count', np.nan)})

        print("\n"); print("="*80); print("MODEL 2"); print("EE ~ OH_DISTANCE"); print("="*80)
        m2 = smf.ols("EE ~ OH_distance", data=oh_df).fit()
        print(m2.summary())
        global_pvals.append({'analysis': 'OH_model2_ols', 'feature': 'OH_distance', 'p': m2.pvalues.get('OH_distance', np.nan)})

        print("\n"); print("="*80); print("MODEL 3"); print("EE ~ OH_DISTANCE + OH_COUNT"); print("="*80)
        m3 = smf.ols("EE ~ OH_distance + OH_count", data=oh_df).fit()
        print(m3.summary())

        print("\n"); print("="*80); print("MODEL 4"); print("EE ~ OH_DISTANCE + OH_COUNT + MolWt + LogP"); print("="*80)
        m4 = smf.ols("EE ~ OH_distance + OH_count + MolWt + LogP", data=oh_df).fit()
        print(m4.summary())

        # --------------------------- VIF analysis (FIX: add intercept before computing VIF) ---------------------------
        # variance_inflation_factor() assumes the matrix passed to it ALREADY contains an
        # intercept/constant column; without one it computes VIF-through-the-origin, which
        # inflates every value and makes the usual ">5" / ">10" collinearity thresholds
        # meaningless. sm.add_constant() fixes this -- the constant's own VIF row is dropped
        # from the printed table since it isn't a real predictor.
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        X = oh_df[["OH_distance", "OH_count", "MolWt", "LogP"]].dropna()
        X_vif = sm.add_constant(X)
        vif_df = pd.DataFrame()
        vif_df["Feature"] = X_vif.columns
        vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        vif_df = vif_df[vif_df["Feature"] != "const"].reset_index(drop=True)
        print(vif_df)

        summary_table = pd.DataFrame({
            "Model":["OH_count","OH_distance","OH_distance + OH_count","OH_distance + OH_count + MolWt + LogP"],
            "R2":[m1.rsquared, m2.rsquared, m3.rsquared, m4.rsquared]
        })
        summary_table.to_csv("table_OH_regression_models.csv", index=False)
        print("\nSaved: table_OH_regression_models.csv")

        # --------------------------- OH DEEP-DIVE ANALYSIS ---------------------------
        OH_PATTERN = Chem.MolFromSmarts("[OX2H]")

        def get_oh_metrics(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return pd.Series({"OH_count": np.nan, "Headgroup_OH_count": np.nan, "Linker_OH_count": np.nan,
                                   "Tail_OH_count": np.nan, "OH_min_distance": np.nan, "OH_mean_distance": np.nan,
                                   "OH_burden": np.nan})
            OH_matches = mol.GetSubstructMatches(OH_PATTERN)
            N_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
            if len(N_atoms) == 0:
                return pd.Series({"OH_count": 0, "Headgroup_OH_count": 0, "Linker_OH_count": 0, "Tail_OH_count": 0,
                                   "OH_min_distance": np.nan, "OH_mean_distance": np.nan, "OH_burden": np.nan})
            distances = []; head = 0; linker = 0; tail = 0
            for match in OH_matches:
                O_idx = match[0]
                nearest = 999
                for n in N_atoms:
                    try:
                        path = Chem.rdmolops.GetShortestPath(mol, O_idx, n)
                        dist = len(path) - 1
                        nearest = min(nearest, dist)
                    except:
                        pass
                distances.append(nearest)
                if nearest <= 4: head += 1
                elif nearest <= 8: linker += 1
                else: tail += 1

            if len(distances) == 0:
                min_dist = np.nan; mean_dist = np.nan; burden = 0
            else:
                min_dist = np.min(distances); mean_dist = np.mean(distances)
                burden = np.sum([1/d for d in distances])
            return pd.Series({"OH_count": len(distances), "Headgroup_OH_count": head, "Linker_OH_count": linker,
                               "Tail_OH_count": tail, "OH_min_distance": min_dist, "OH_mean_distance": mean_dist,
                               "OH_burden": burden})

        oh_df = pd.DataFrame()
        oh_df["EE"] = clean_df["EE"]
        oh_df["MolWt"] = (clean_df["ionizable_lipid_smiles"].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x))))
        oh_df["LogP"] = (clean_df["ionizable_lipid_smiles"].apply(lambda x: Descriptors.MolLogP(Chem.MolFromSmiles(x))))
        metrics = (clean_df["ionizable_lipid_smiles"].apply(get_oh_metrics))
        oh_df = pd.concat([oh_df, metrics], axis=1)
        oh_df = oh_df[oh_df["OH_count"] > 0].copy()

        print("\n"); print("="*80); print("OH DESCRIPTOR CORRELATIONS"); print("="*80)
        corr_rows = []
        for col in ["OH_count","Headgroup_OH_count","Linker_OH_count","Tail_OH_count",
                    "OH_min_distance","OH_mean_distance","OH_burden"]:
            rho,p = spearmanr(oh_df[col], oh_df["EE"])
            corr_rows.append({"Descriptor": col, "rho": rho, "p": p})
        corr_df = pd.DataFrame(corr_rows)
        print(corr_df.round(4).to_string(index=False))
        global_pvals += [{'analysis': 'OH_deep_dive_correlation', 'feature': r['Descriptor'], 'p': r['p']}
                          for _, r in corr_df.iterrows()]
        corr_df.to_csv("table_OH_correlations.csv", index=False)
        # Keep a stable reference to THIS specific correlation table -- 'corr_df' as a
        # variable name gets reassigned by later sub-analyses in this method, so Figure2
        # (below) needs its own copy to plot real numbers instead of typed-in ones.
        oh_deepdive_corr_df = corr_df.copy()

        print("\n"); print("="*80); print("REGRESSION COMPARISON"); print("="*80)
        models_oh = {
            "OH_count": "EE ~ OH_count",
            "Headgroup_OH": "EE ~ Headgroup_OH_count",
            "OH_burden": "EE ~ OH_burden",
            "Burden+Count": "EE ~ OH_burden + OH_count",
            "Burden+Headgroup": "EE ~ OH_burden + Headgroup_OH_count",
            "Full": "EE ~ OH_burden + OH_count + Headgroup_OH_count + MolWt + LogP"
        }
        reg_results = []
        for name, formula in models_oh.items():
            model = smf.ols(formula, data=oh_df).fit()
            reg_results.append({"Model": name, "R2": model.rsquared, "Adj_R2": model.rsquared_adj, "AIC": model.aic})
            print("\n"); print("-"*80); print(name); print("-"*80)
            print(model.summary())

        reg_df = pd.DataFrame(reg_results)
        reg_df.to_csv("table_OH_model_comparison.csv", index=False)

        # --------------------------- VIF ANALYSIS (FIX: add intercept) ---------------------------
        print("\n"); print("="*80); print("VIF ANALYSIS"); print("="*80)
        X = oh_df[["OH_burden","OH_count","Headgroup_OH_count","MolWt","LogP"]].dropna()
        X_vif = sm.add_constant(X)
        vif_df = pd.DataFrame()
        vif_df["Feature"] = X_vif.columns
        vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
        vif_df = vif_df[vif_df["Feature"] != "const"].reset_index(drop=True)
        print(vif_df.round(2).to_string(index=False))
        vif_df.to_csv("table_OH_VIF.csv", index=False)

        try:
            from pingouin import partial_corr
            print("\n"); print("="*80); print("PARTIAL CORRELATIONS"); print("="*80)
            pc1 = partial_corr(data=oh_df, x="OH_burden", y="EE", covar=["MolWt","LogP"])
            print("\nOH burden vs EE controlling MolWt+LogP")
            print(pc1)
            pc2 = partial_corr(data=oh_df, x="Headgroup_OH_count", y="EE", covar=["MolWt","LogP"])
            print("\nHeadgroup OH count vs EE controlling MolWt+LogP")
            print(pc2)
        except Exception as e:
            print("\nPartial correlation skipped:", e)

        print("\nSaved:")
        print(" table_OH_correlations.csv")
        print(" table_OH_model_comparison.csv")
        print(" table_OH_VIF.csv")

        # --------------------------- SCAFFOLD-STRATIFIED OH ANALYSIS ---------------------------
        OH_PATTERN = Chem.MolFromSmarts("[OX2H]")
        def count_oh(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None: return np.nan
            return len(mol.GetSubstructMatches(OH_PATTERN))

        tmp = pd.DataFrame()
        tmp["EE"] = clean_df["EE"]
        tmp["lipid"] = clean_df["ionizable_lipid"]
        tmp["smiles"] = clean_df["ionizable_lipid_smiles"]
        tmp["OH_count"] = (tmp["smiles"].apply(count_oh))

        print("\n"); print("="*90); print("OH COUNT DISTRIBUTION"); print("="*90)
        print(tmp.groupby("OH_count")["EE"].agg(["count","mean","median","std"]).round(2))

        print("\n"); print("="*90); print("SCAFFOLD SUMMARY"); print("="*90)
        scaffold_summary = (tmp.groupby("lipid").agg(n=("EE","size"), mean_EE=("EE","mean"), sd_EE=("EE","std"), OH_count=("OH_count","mean")).sort_values("n", ascending=False))
        print(scaffold_summary.round(2).to_string())
        scaffold_summary.to_csv("table_scaffold_oh_summary.csv")

        print("\n"); print("="*90); print("WITHIN-SCAFFOLD OH EFFECTS"); print("="*90)
        rows = []
        for lipid, sub in tmp.groupby("lipid"):
            if len(sub) < 5: continue
            if sub["OH_count"].nunique() < 2: continue
            rho,p = spearmanr(sub["OH_count"], sub["EE"])
            rows.append({"lipid": lipid, "n": len(sub), "rho": rho, "p": p, "mean_EE": sub["EE"].mean()})

        within_df = pd.DataFrame(rows)
        if len(within_df) > 0:
            print(within_df.sort_values("rho").round(3).to_string(index=False))
            within_df.to_csv("table_within_scaffold_OH_effects.csv", index=False)
        else:
            print("\nNo scaffold contains sufficient OH-count variation.")

        print("\n"); print("="*90); print("TOP HIGH-OH SCAFFOLDS"); print("="*90)
        high_oh = (scaffold_summary.sort_values("OH_count", ascending=False).head(20))
        print(high_oh.round(2).to_string())

        print("\n"); print("="*90); print("BETWEEN-SCAFFOLD ANALYSIS"); print("="*90)
        rho,p = spearmanr(scaffold_summary["OH_count"],scaffold_summary["mean_EE"])
        print(f"\nScaffold mean OH count vs scaffold mean EE\nrho = {rho:.3f}\np   = {p:.5g}")

        overall_var = np.var(tmp["EE"])
        between_var = np.var(scaffold_summary["mean_EE"])
        print("\n"); print("="*90); print("VARIANCE DECOMPOSITION"); print("="*90)
        print(f"Overall EE variance     : {overall_var:.2f}")
        print(f"Between-scaffold variance: {between_var:.2f}")
        print(f"Fraction explained by scaffold: {between_var/overall_var:.3f}")
        print("\nSaved:")
        print(" table_scaffold_oh_summary.csv")
        print(" table_within_scaffold_OH_effects.csv")

        # --------------------------- ETHER + ESTER FINAL ARBITRATION ANALYSIS ---------------------------
        try:
            from pingouin import partial_corr
            HAS_PINGOUIN = True
        except:
            HAS_PINGOUIN = False

        PATTERNS = {"Ether": "[#6]-O-[#6]", "Ester": ESTER_SMARTS}

        def get_group_metrics(smiles, smarts):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return pd.Series({"count": np.nan, "min_dist": np.nan, "mean_dist": np.nan, "burden": np.nan,
                                   "head_count": np.nan, "linker_count": np.nan, "tail_count": np.nan})
            patt = Chem.MolFromSmarts(smarts)
            matches = mol.GetSubstructMatches(patt)
            N_atoms = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 7]
            if len(matches) == 0 or len(N_atoms) == 0:
                return pd.Series({"count": 0, "min_dist": np.nan, "mean_dist": np.nan, "burden": 0,
                                   "head_count": 0, "linker_count": 0, "tail_count": 0})
            distances = []; head = 0; linker = 0; tail = 0
            for match in matches:
                anchor = match[0]
                nearest = 999
                for n in N_atoms:
                    try:
                        path = Chem.rdmolops.GetShortestPath(mol, anchor, n)
                        dist = len(path)-1
                        nearest = min(nearest, dist)
                    except:
                        pass
                distances.append(nearest)
                if nearest <= 4: head += 1
                elif nearest <= 8: linker += 1
                else: tail += 1
            burden = sum(1/d for d in distances if d > 0)
            return pd.Series({"count": len(distances), "min_dist": np.min(distances), "mean_dist": np.mean(distances),
                               "burden": burden, "head_count": head, "linker_count": linker, "tail_count": tail})

        chem_df = pd.DataFrame()
        chem_df["EE"] = clean_df["EE"]
        chem_df["MolWt"] = (clean_df["ionizable_lipid_smiles"].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x))))
        chem_df["LogP"] = (clean_df["ionizable_lipid_smiles"].apply(lambda x: Descriptors.MolLogP(Chem.MolFromSmiles(x))))
        for group, smarts in PATTERNS.items():
            tmp = (clean_df["ionizable_lipid_smiles"].apply(lambda x: get_group_metrics(x, smarts)))
            tmp.columns = [f"{group}_{c}" for c in tmp.columns]
            chem_df = pd.concat([chem_df, tmp], axis=1)

        # Store the two adjusted-regression models fit below so Figure1_Topology (further
        # down) can plot their ACTUAL fitted beta coefficients instead of typed-in numbers.
        ether_ester_adjusted_models = {}

        for GROUP in ["Ether","Ester"]:
            print("\n"); print("="*90); print(GROUP.upper()); print("="*90)
            descriptors = [f"{GROUP}_count", f"{GROUP}_min_dist", f"{GROUP}_mean_dist", f"{GROUP}_burden",
                           f"{GROUP}_head_count", f"{GROUP}_linker_count", f"{GROUP}_tail_count"]

            print("\nCORRELATIONS\n")
            rows = []
            for col in descriptors:
                sub = chem_df[["EE", col]].dropna()
                if len(sub) < 20: continue
                rho,p = spearmanr(sub[col], sub["EE"])
                rows.append({"Descriptor": col, "rho": rho, "p": p})

            corr_df = pd.DataFrame(rows)
            print(corr_df.round(4).to_string(index=False))
            global_pvals += [{'analysis': f'{GROUP}_topology_correlation', 'feature': r['Descriptor'], 'p': r['p']}
                              for _, r in corr_df.iterrows()]
            corr_df.to_csv(f"{GROUP}_correlations.csv",index=False)

            print("\nREGRESSIONS\n")
            for col in descriptors:
                sub = chem_df[["EE", col]].dropna()
                if len(sub) < 20: continue
                model = smf.ols(f"EE ~ {col}", data=sub).fit()
                print("\n"); print(col)
                coef = model.params.get(col, np.nan)
                pval = model.pvalues.get(col, np.nan)
                print(f"R²={model.rsquared:.3f} beta={coef:.3f} p={pval:.5g}")

            print("\n"); print("ADJUSTED MODEL"); print("-"*60)
            dcol = f"{GROUP}_mean_dist"
            reg = chem_df[["EE", dcol, "MolWt", "LogP"]].dropna()
            model = smf.ols(f"EE ~ {dcol} + MolWt + LogP", data=reg).fit()
            print(model.summary())
            ether_ester_adjusted_models[GROUP] = model
            global_pvals.append({'analysis': f'{GROUP}_adjusted_regression', 'feature': dcol,
                                  'p': model.pvalues.get(dcol, np.nan)})

            # --------------------------- VIF (FIX: add intercept) ---------------------------
            print("\nVIF\n")
            X = reg[[dcol, "MolWt", "LogP"]]
            X_vif = sm.add_constant(X)
            vif_df = pd.DataFrame()
            vif_df["Feature"] = X_vif.columns
            vif_df["VIF"] = [variance_inflation_factor(X_vif.values, i) for i in range(X_vif.shape[1])]
            vif_df = vif_df[vif_df["Feature"] != "const"].reset_index(drop=True)
            print(vif_df.round(2).to_string(index=False))

            if HAS_PINGOUIN:
                print("\nPARTIAL CORRELATION\n")
                pc = partial_corr(data=reg, x=dcol, y="EE", covar=["MolWt","LogP"])
                print(pc)

        chem_df.to_csv("table_ether_ester_final_analysis.csv",index=False)
        print("\nSaved: table_ether_ester_final_analysis.csv")

        # --------------------------- NITROGEN ARCHITECTURE ANALYSIS --------------------------
        try:
            from pingouin import partial_corr
            HAS_PINGOUIN = True
        except:
            HAS_PINGOUIN = False

        PRIMARY = Chem.MolFromSmarts("[NX3;H2]")
        SECONDARY = Chem.MolFromSmarts("[NX3;H1]")
        TERTIARY = Chem.MolFromSmarts("[NX3;H0]")
        QUATERNARY = Chem.MolFromSmarts("[N+]")

        def nitrogen_metrics(smiles):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return pd.Series({"Total_N": np.nan, "Primary_N": np.nan, "Secondary_N": np.nan,
                                   "Tertiary_N": np.nan, "Quaternary_N": np.nan, "Min_NN_Distance": np.nan,
                                   "Mean_NN_Distance": np.nan, "N_Cluster_Score": np.nan, "Polyamine": np.nan})
            N_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7]
            total_n = len(N_atoms)
            if total_n == 0:
                return pd.Series({"Total_N": 0, "Primary_N": 0, "Secondary_N": 0, "Tertiary_N": 0,
                                   "Quaternary_N": 0, "Min_NN_Distance": np.nan, "Mean_NN_Distance": np.nan,
                                   "N_Cluster_Score": 0, "Polyamine": 0})
            primary = len(mol.GetSubstructMatches(PRIMARY))
            secondary = len(mol.GetSubstructMatches(SECONDARY))
            tertiary = len(mol.GetSubstructMatches(TERTIARY))
            quaternary = len(mol.GetSubstructMatches(QUATERNARY))
            pair_distances = []
            for i in range(len(N_atoms)):
                for j in range(i+1,len(N_atoms)):
                    try:
                        path = Chem.rdmolops.GetShortestPath(mol, N_atoms[i], N_atoms[j])
                        dist = len(path)-1
                        pair_distances.append(dist)
                    except:
                        pass
            if len(pair_distances)==0:
                min_nn = np.nan; mean_nn = np.nan; cluster_score = 0
            else:
                min_nn = np.min(pair_distances); mean_nn = np.mean(pair_distances)
                cluster_score = np.sum([1/d for d in pair_distances])
            polyamine = int(total_n >= 2)
            return pd.Series({"Total_N": total_n, "Primary_N": primary, "Secondary_N": secondary,
                               "Tertiary_N": tertiary, "Quaternary_N": quaternary, "Min_NN_Distance": min_nn,
                               "Mean_NN_Distance": mean_nn, "N_Cluster_Score": cluster_score, "Polyamine": polyamine})

        n_df = pd.DataFrame()
        n_df["EE"] = clean_df["EE"]
        n_df["MolWt"] = (clean_df["ionizable_lipid_smiles"].apply(lambda x: Descriptors.MolWt(Chem.MolFromSmiles(x))))
        n_df["LogP"] = (clean_df["ionizable_lipid_smiles"].apply(lambda x: Descriptors.MolLogP(Chem.MolFromSmiles(x))))
        metrics = (clean_df["ionizable_lipid_smiles"].apply(nitrogen_metrics))
        n_df = pd.concat([n_df, metrics],axis=1)

        print("\n"); print("="*90); print("NITROGEN CORRELATIONS"); print("="*90)
        descriptors = ["Total_N","Primary_N","Secondary_N","Tertiary_N","Quaternary_N",
                       "Min_NN_Distance","Mean_NN_Distance","N_Cluster_Score","Polyamine"]
        corr_rows = []
        for col in descriptors:
            sub = n_df[["EE", col]].dropna()
            if len(sub) < 20: continue
            rho,p = spearmanr(sub[col], sub["EE"])
            corr_rows.append({"Descriptor": col, "rho":rho, "p": p})

        corr_df = pd.DataFrame(corr_rows)
        print(corr_df.round(4).to_string(index=False))
        global_pvals += [{'analysis': 'nitrogen_correlation', 'feature': r['Descriptor'], 'p': r['p']}
                          for _, r in corr_df.iterrows()]
        corr_df.to_csv("table_nitrogen_correlations.csv",index=False)
        # Stable reference for Figure3 (see the oh_deepdive_corr_df comment above).
        nitrogen_corr_df = corr_df.copy()

        print("\n"); print("="*90); print("REGRESSIONS"); print("="*90)
        # Store fitted single-variable models keyed by descriptor for Figure3 to reuse.
        nitrogen_single_models = {}
        for col in descriptors:
            sub = n_df[["EE",col]].dropna()
            if len(sub) < 20: continue
            model = smf.ols(f"EE ~ {col}", data=sub).fit()
            nitrogen_single_models[col] = model
            coef = model.params.get(col, np.nan)
            pval = model.pvalues.get(col, np.nan)
            print("\n"); print(col)
            print(f"$R^2$={model.rsquared:.3f}  beta={coef:.3f}  p={pval:.5g}")

        print("\n"); print("="*90); print("ADJUSTED MODELS"); print("="*90)
        adj_rows = []
        nitrogen_adjusted_models = {}
        for col in descriptors:
            reg = n_df[["EE", col, "MolWt", "LogP"]].dropna()
            if len(reg) < 30: continue
            model = smf.ols(f"EE ~ {col} + MolWt + LogP", data=reg).fit()
            nitrogen_adjusted_models[col] = model
            adj_rows.append({"Descriptor": col, "Beta": model.params[col], "P": model.pvalues[col], "R2": model.rsquared})
            global_pvals.append({'analysis': 'nitrogen_adjusted_regression', 'feature': col, 'p': model.pvalues[col]})

        adj_df = pd.DataFrame(adj_rows)
        print(adj_df.round(4).to_string(index=False))
        adj_df.to_csv("table_nitrogen_adjusted_models.csv", index=False)

        # --------------------------- VIF (FIX: add intercept) ---------------------------
        print("\n"); print("="*90); print("VIF ANALYSIS"); print("="*90)
        for col in ["Total_N","Primary_N","Secondary_N","Tertiary_N","N_Cluster_Score"]:
            reg = n_df[[col, "MolWt", "LogP"]].dropna()
            if len(reg) < 30: continue
            reg_vif = sm.add_constant(reg)
            vif_df = pd.DataFrame()
            vif_df["Feature"] = reg_vif.columns
            vif_df["VIF"] = [variance_inflation_factor(reg_vif.values, i) for i in range(reg_vif.shape[1])]
            vif_df = vif_df[vif_df["Feature"] != "const"].reset_index(drop=True)
            print("\n"); print(col)
            print(vif_df.round(2).to_string(index=False))

        if HAS_PINGOUIN:
            print("\n"); print("="*90); print("PARTIAL CORRELATIONS"); print("="*90)
            for col in ["Total_N","Primary_N","Secondary_N","Tertiary_N","N_Cluster_Score"]:
                sub = n_df[["EE",col,"MolWt","LogP"]].dropna()
                if len(sub) < 30: continue
                print("\n"); print(col)
                pc = partial_corr(data=sub, x=col, y="EE", covar=["MolWt", "LogP"])
                print(pc)

        print("\n"); print("="*90); print("TOTAL N SUMMARY"); print("="*90)
        print(n_df.groupby("Total_N")["EE"].agg(["count", "mean", "median", "std"]).round(2))

        print("\n"); print("="*90); print("PRIMARY AMINE SUMMARY"); print("="*90)
        print(n_df.groupby("Primary_N")["EE"].agg(["count", "mean", "median"]).round(2))

        print("\n"); print("="*90); print("POLYAMINE SUMMARY"); print("="*90)
        print(n_df.groupby("Polyamine")["EE"].agg(["count", "mean", "median"]).round(2))

        print("\nSaved:")
        print(" table_nitrogen_correlations.csv")
        print(" table_nitrogen_adjusted_models.csv")

        # --------------------------- FIGURE : ETHER / ESTER TOPOLOGY ---------------------------
        colors = PUB_PALETTE[:16]
        plot_df = bin_df[bin_df["Feature"].isin(["Ether","Ester"])].copy()
        order = ["0-2", "3-5", "6-8", ">8"]
        fig, axes = plt.subplots(1, 3, figsize=(8, 3))

        ether = plot_df[plot_df.Feature=="Ether"]
        sns.barplot(data=ether, x="Distance_bin", y="Delta_EE", order=order, color=colors[1], ax=axes[0])
        axes[0].axhline(0, ls="--", c="black")
        axes[0].set_title("Ether Topology")
        axes[0].set_ylabel(f"$\\Delta$ EE (%)")

        ester = plot_df[plot_df.Feature=="Ester"]
        sns.barplot(data=ester, x="Distance_bin", y="Delta_EE", order=order, color=colors[2], ax=axes[1])
        axes[1].axhline(0, ls="--", c="black")
        axes[1].set_title("Ester Topology")
        axes[1].set_ylabel("$\\Delta$ EE (%)")

        # FIX: previously coef_df was hand-typed as {"Ether": 3.3412, "Ester": 3.1863} --
        # disconnected from the adjusted OLS models fit above. Now pulls the actual fitted
        # beta_{group}_mean_dist coefficient out of ether_ester_adjusted_models, so the bar
        # heights always match what the printed regression summaries above say.
        coef_df = pd.DataFrame({
            "Feature": ["Ether", "Ester"],
            "Beta": [
                ether_ester_adjusted_models["Ether"].params[f"Ether_mean_dist"],
                ether_ester_adjusted_models["Ester"].params[f"Ester_mean_dist"],
            ]
        })
        sns.barplot(data=coef_df, x="Feature", y="Beta", palette=[colors[0], colors[3]], ax=axes[2])
        axes[2].set_title("Adjusted Distance Effects")
        axes[2].set_ylabel(f"Regression $\\beta$")

        plt.tight_layout()
        save_pub_figure(fig, 'Figure1_Topology', width='double')
        plt.show()

        # --------------------------- FIGURE : HYDROXYL BURDEN RATHER THAN DISTANCE DRIVES EE ---------------------------
        def _lookup(corr_table, descriptor):
            row = corr_table.loc[corr_table['Descriptor'] == descriptor]
            if len(row) == 0:
                return np.nan, np.nan
            return float(row['rho'].iloc[0]), float(row['p'].iloc[0])

        fig, axes = plt.subplots(1, 3, figsize=(8, 3))

        tmp = (oh_df.groupby("OH_count")["EE"].agg(["mean","sem","count"]).reset_index())
        axes[0].errorbar(tmp["OH_count"], tmp["mean"], yerr=tmp["sem"], marker="o", lw=2, capsize=4, color=colors[11])
        axes[0].set_xlabel("Number of OH Groups")
        axes[0].set_ylabel("Mean EE (%)")
        # FIX: rho/p now read from the actual computed oh_deepdive_corr_df instead of typed literals.
        rho_a, p_a = _lookup(oh_deepdive_corr_df, 'OH_count')
        axes[0].text(0.05, 0.95, f"$\\rho$ = {rho_a:.3f}\np {'< 0.001' if p_a < 0.001 else f'= {p_a:.3g}'}",
                     transform=axes[0].transAxes, va="top", bbox=dict(fc="white", alpha=0.6))

        tmp = (oh_df.groupby("Headgroup_OH_count")["EE"].agg(["mean","sem","count"]).reset_index())
        axes[1].errorbar(tmp["Headgroup_OH_count"], tmp["mean"], yerr=tmp["sem"], marker="o", lw=2, capsize=4, color=colors[12])
        axes[1].set_xlabel("Headgroup OHs")
        axes[1].set_ylabel("Mean EE (%)")
        rho_b, p_b = _lookup(oh_deepdive_corr_df, 'Headgroup_OH_count')
        axes[1].text(0.05, 0.95, f"$\\rho$ = {rho_b:.3f}\np = {p_b:.3g}",
                     transform=axes[1].transAxes, va="top", bbox=dict(fc="white", alpha=0.6))

        sns.regplot(data=oh_df, x="OH_mean_distance", y="EE", scatter_kws={"alpha":0.5,"s":30}, line_kws={"color":"black"}, ax=axes[2])
        axes[2].set_xlabel("Mean OH Distance")
        axes[2].set_ylabel("EE (%)")
        rho_c, p_c = _lookup(oh_deepdive_corr_df, 'OH_mean_distance')
        ns_flag = "\nn.s." if p_c >= 0.05 else ""
        axes[2].text(0.05, 0.95, f"$\\rho$ = {rho_c:.3f}\np = {p_c:.3g}{ns_flag}",
                     transform=axes[2].transAxes, va="top", bbox=dict(fc="white", alpha=0.6))

        plt.tight_layout()
        save_pub_figure(fig, "Figure2_HydroxylBurden", width="double")
        plt.show()

        # --------------------------- FIGURE : NITROGEN ARCHITECTURE ---------------------------
        fig, axes = plt.subplots(1, 3, figsize=(8, 3))

        tmp = (n_df.groupby("Primary_N")["EE"].agg(["mean","sem","count"]).reset_index())
        axes[0].errorbar(tmp["Primary_N"], tmp["mean"], yerr=tmp["sem"], marker="^", lw=2, capsize=4, color=colors[9])
        axes[0].set_xlabel("Primary Amine Count")
        axes[0].set_ylabel("Mean EE (%)")
        rho_a, p_a = _lookup(nitrogen_corr_df, 'Primary_N')
        beta_a = nitrogen_single_models['Primary_N'].params.get('Primary_N', np.nan) if 'Primary_N' in nitrogen_single_models else np.nan
        axes[0].text(0.05, 0.95, f"$\\rho$ = {rho_a:.3f}\np {'< 0.001' if p_a < 0.001 else f'= {p_a:.3g}'}\n"
                     f"$\\beta$ = {beta_a:.2f}", transform=axes[0].transAxes, va="top", bbox=dict(fc="white", alpha=0.6))

        tmp = (n_df.groupby("Total_N")["EE"].agg(["mean","sem","count"]).reset_index())
        axes[1].errorbar(tmp["Total_N"], tmp["mean"], yerr=tmp["sem"], marker="d", lw=2, capsize=4, color=colors[10])
        axes[1].set_xlabel("Total N Count")
        axes[1].set_ylabel("Mean EE (%)")
        rho_b, p_b = _lookup(nitrogen_corr_df, 'Total_N')
        beta_b = nitrogen_single_models['Total_N'].params.get('Total_N', np.nan) if 'Total_N' in nitrogen_single_models else np.nan
        axes[1].text(0.05, 0.95, f"$\\rho$ = {rho_b:.3f}\np {'< 0.001' if p_b < 0.001 else f'= {p_b:.3g}'}\n"
                     f"$\\beta$ = {beta_b:.2f}", transform=axes[1].transAxes, va="top", bbox=dict(fc="white", alpha=0.6))

        sns.regplot(data=n_df, x="N_Cluster_Score", y="EE", scatter_kws={"alpha":0.45,"s":30}, line_kws={"color":"black"}, ax=axes[2])
        axes[2].set_xlabel("Cluster Score")
        axes[2].set_ylabel("EE (%)")
        rho_c, p_c = _lookup(nitrogen_corr_df, 'N_Cluster_Score')
        beta_c = nitrogen_single_models['N_Cluster_Score'].params.get('N_Cluster_Score', np.nan) if 'N_Cluster_Score' in nitrogen_single_models else np.nan
        axes[2].text(0.05, 0.95, f"$\\rho$ = {rho_c:.3f}\np {'< 0.001' if p_c < 0.001 else f'= {p_c:.3g}'}\n"
                     f"$\\beta$ = {beta_c:.2f}", transform=axes[2].transAxes, va="top", bbox=dict(fc="white", alpha=0.6))

        plt.tight_layout()
        save_pub_figure(fig, "Figure3_NitrogenArchitecture", width="double")
        plt.show()

        # --------------------------- FIGURE 4 : MECHANISTIC DESIGN RULES (DELTA EE) ---------------------------
        # FIX: summary_df below was previously hand-typed (Delta_EE and P values pasted in
        # as literals, disconnected from rules_df / region_df / n_df / oh_df computed above).
        # It is now built entirely from an indicator-vs-EE Mann-Whitney comparison computed
        # in-place from the same dataframes used throughout this section, so every bar in
        # Figure4 traces back to an actual computed value.
        def _delta_and_mwu(indicator, ee):
            indicator = np.asarray(indicator).astype(bool)
            ee = np.asarray(ee, dtype=float)
            mask = ~pd.isna(ee)
            indicator, ee = indicator[mask], ee[mask]
            present, absent = ee[indicator], ee[~indicator]
            if len(present) < 3 or len(absent) < 3:
                return np.nan, np.nan
            delta = present.mean() - absent.mean()
            _, p = mannwhitneyu(present, absent, alternative='two-sided')
            return delta, p

        design_rule_rows = []

        # OH-rich headgroup: loc_df['OH_Headgroup'] is the binary "has a headgroup OH"
        # indicator already computed in the headgroup/linker/tail analysis above.
        d, p = _delta_and_mwu(loc_df['OH_Headgroup'], loc_df['EE'])
        design_rule_rows.append({'Feature': 'OH-rich headgroup', 'Delta_EE': d, 'P': p})

        # Primary / secondary amine: reuse the already-fit MWU design rules (rules_df).
        for label, frag_name in [('Primary amine', 'Primary amine'), ('Secondary amine', 'Secondary amine')]:
            row = rules_df.loc[rules_df['Feature'] == frag_name]
            if len(row):
                design_rule_rows.append({'Feature': label, 'Delta_EE': float(row['delta'].iloc[0]), 'P': float(row['p_mwu'].iloc[0])})

        # Nitrogen clustering: median split of N_Cluster_Score.
        nz = n_df['N_Cluster_Score'].dropna()
        if len(nz) > 0:
            median_cluster = nz.median()
            d, p = _delta_and_mwu(n_df['N_Cluster_Score'] > median_cluster, n_df['EE'])
            design_rule_rows.append({'Feature': 'Nitrogen clustering', 'Delta_EE': d, 'P': p})

        # Polyamine architecture: n_df['Polyamine'] is already a 0/1 indicator.
        d, p = _delta_and_mwu(n_df['Polyamine'], n_df['EE'])
        design_rule_rows.append({'Feature': 'Polyamine architecture', 'Delta_EE': d, 'P': p})

        # High nitrogen count: at/above the median Total_N vs below.
        tn = n_df['Total_N'].dropna()
        if len(tn) > 0:
            median_n = tn.median()
            d, p = _delta_and_mwu(n_df['Total_N'] >= median_n, n_df['EE'])
            design_rule_rows.append({'Feature': 'High nitrogen count', 'Delta_EE': d, 'P': p})

        # Distal ester / ether: topo_df['{group}_region'] == 'Tail' indicator (computed earlier).
        for label, group in [('Distal ester', 'Ester'), ('Distal ether', 'Ether')]:
            indicator = (topo_df[f'{group}_region'] == 'Tail')
            d, p = _delta_and_mwu(indicator, topo_df['EE'])
            design_rule_rows.append({'Feature': label, 'Delta_EE': d, 'P': p})

        summary_df = pd.DataFrame(design_rule_rows).dropna(subset=['Delta_EE'])
        global_pvals += [{'analysis': 'mechanistic_design_rule_summary', 'feature': r['Feature'], 'p': r['P']}
                          for _, r in summary_df.iterrows()]

        def p_to_stars(p):
            if p < 0.001: return "***"
            elif p < 0.01: return "**"
            elif p < 0.05: return "*"
            return ""

        summary_df["Stars"] = (summary_df["P"].apply(p_to_stars))
        summary_df = summary_df.sort_values("Delta_EE")

        fig, ax = plt.subplots(figsize=(7, 3))
        bar_colors = [colors[12] if x < 0 else colors[10] for x in summary_df["Delta_EE"]]
        bars = ax.barh(summary_df["Feature"], summary_df["Delta_EE"], color=bar_colors, edgecolor="white", linewidth=2)
        ax.axvline(0, color="black", lw=2, ls=":")

        for bar, (_, row) in zip(bars, summary_df.iterrows()):
            effect = row["Delta_EE"]
            x_loc = (effect + 0.8 if effect > 0 else effect - 0.8)
            ax.text(x_loc, bar.get_y() + bar.get_height()/2, row["Stars"], fontweight="bold", va="center", ha="left" if effect > 0 else "right")

        ax.set_xlabel(r"$\Delta$EE (%)")
        ax.set_ylabel("Chemical Design Rule")
        span = max(abs(summary_df["Delta_EE"].min()), abs(summary_df["Delta_EE"].max())) * 1.3 + 2
        ax.set_xlim(-span, span)
        ax.text(0.98, -0.12, "* p<0.05   ** p<0.01   *** p<0.001", transform=ax.transAxes, ha="center", style="italic")
        plt.legend()
        plt.tight_layout()
        save_pub_figure(fig, "Figure4_MechanisticDesignRules_DeltaEE", width="single")
        plt.show()

        # --- SHAP-Guided Design Case Study ----------------------------------------
        # FIX: uses the shared ESTER_SMARTS constant (was a separate, slightly different
        # bare 'C(=O)O' pattern before) and now prints the actual edited SMILES plus a
        # basic sanity check (atom-count / validity) rather than only sanitizing silently,
        # so a chemically odd substructure-replacement result is visible before anyone
        # treats it as evidence.
        ester_smarts_mol = Chem.MolFromSmarts(ESTER_SMARTS)
        ether_template = Chem.MolFromSmiles('CO')

        def ester_to_ether_edit(smiles):
            """Replace the first ester linkage with an ether as a minimal structural probe
            of the SHAP ether/ester design rule. Returns None if no ester is present or the
            edited molecule fails to sanitize. NOTE: this is a crude atom-substitution edit,
            not a validated retrosynthetic transform -- inspect the printed SMILES/atom counts
            before treating the result as more than a rough in-silico probe (see caveat below)."""
            mol = Chem.MolFromSmiles(str(smiles))
            if mol is None or not mol.HasSubstructMatch(ester_smarts_mol):
                return None
            edited_mols = AllChem.ReplaceSubstructs(mol, ester_smarts_mol, ether_template)
            edited = edited_mols[0]
            try:
                Chem.SanitizeMol(edited)
                edited_smiles = Chem.MolToSmiles(edited)
                # Sanity check: report the heavy-atom-count change so an obviously wrong
                # edit (e.g. losing/gaining far more atoms than the O-for-C(=O)O swap
                # should) is visible in the printed output, not silently accepted.
                n_before = mol.GetNumHeavyAtoms()
                n_after = edited.GetNumHeavyAtoms()
                print(f'    [ester_to_ether_edit sanity check] heavy atoms before={n_before}, '
                      f'after={n_after} (delta={n_after - n_before}; inspect if this looks larger '
                      f'than expected for a single ester->ether substitution)')
                return edited_smiles
            except Exception:
                return None

        candidates = clean_df[clean_df['EE'] < 60].copy()
        candidates['edited_smiles'] = candidates['ionizable_lipid_smiles'].apply(ester_to_ether_edit)
        candidates = candidates[candidates['edited_smiles'].notna()]

        if len(candidates) == 0:
            print('No Low-EE ester-containing ionizable lipid found in this dataset slice -- '
                  'widen the EE threshold above, or supply a formulation manually.')
        else:
            row = candidates.iloc[0]
            print(f'Original ionizable lipid: {row["ionizable_lipid_smiles"]}')
            print(f'Edited (ester->ether):    {row["edited_smiles"]}')
            print(f'Measured EE% (original formulation): {row["EE"]:.1f}%')

            with open('lnp_classifier.pkl', 'rb') as f:
                bundle_case = pickle.load(f)

            ratio_str = f'{row["ion_ratio"]:.2f}:{row["peg_ratio"]:.2f}:{row["sterol_ratio"]:.2f}:{row["helper_ratio"]:.2f}'

            pred_original = predict_lnp(row['ionizable_lipid_smiles'], row['helper_lipid_smiles'],
                                         row['sterol_lipid_smiles'], row['peg_lipid_smiles'],
                                         molar_ratio=ratio_str, model_bundle=bundle_case)
            pred_edited   = predict_lnp(row['edited_smiles'], row['helper_lipid_smiles'],
                                         row['sterol_lipid_smiles'], row['peg_lipid_smiles'],
                                         molar_ratio=ratio_str, model_bundle=bundle_case)

            print(f'\nPredicted P(High EE) -- original (ester): {pred_original["prob_high_EE"]:.3f}')
            print(f'Predicted P(High EE) -- edited (ether):    {pred_edited["prob_high_EE"]:.3f}')
            print('Shift consistent with SHAP rule (ether favored): ' +
                  ('YES' if pred_edited['prob_high_EE'] > pred_original['prob_high_EE'] else 'NO'))
            print('\nNote: in-silico structural probe only -- not experimental validation. '
                  'Synthetic accessibility of the edited structure was not assessed, and the '
                  'atom-count sanity check above should be reviewed before citing this result.')

        # --------------------------- GLOBAL MULTIPLE-TESTING CORRECTION ---------------------------
        # FIX: every sub-analysis above (functional groups, MWU design rules, SMARTS
        # enrichment, headgroup/linker/tail, topology distance/bin/region, OH descriptors,
        # ether/ester topology, nitrogen architecture, design-rule summary) corrected its
        # own p-values IN ISOLATION. Section12 as a whole runs several dozen overlapping,
        # often correlated hypothesis tests -- the true family-wise false-discovery rate
        # across the section is understated by any single table's q-values. This block
        # pools every p-value collected into `global_pvals` during this method and applies
        # ONE FDR (and, more conservatively, one Bonferroni) correction across all of them,
        # so a reviewer (or you) can see how many "significant" findings survive a fair,
        # section-wide accounting rather than being read off degrees table-by-table.
        global_test_df = pd.DataFrame(global_pvals).dropna(subset=['p'])
        if len(global_test_df) > 0:
            global_test_df['q_fdr_global']    = multipletests(global_test_df['p'], method='fdr_bh')[1]
            global_test_df['p_bonf_global']   = multipletests(global_test_df['p'], method='bonferroni')[1]
            global_test_df['sig_fdr_global']  = global_test_df['q_fdr_global'] < 0.05
            global_test_df['sig_bonf_global'] = global_test_df['p_bonf_global'] < 0.05
            global_test_df = global_test_df.sort_values('p').reset_index(drop=True)
            global_test_df.to_csv('table_section12_global_multiple_testing_correction.csv', index=False)

            n_tests = len(global_test_df)
            n_locally_would_be_sig_at_p05 = int((global_test_df['p'] < 0.05).sum())
            n_sig_fdr_global = int(global_test_df['sig_fdr_global'].sum())
            n_sig_bonf_global = int(global_test_df['sig_bonf_global'].sum())
            print('\n' + '='*90)
            print('GLOBAL MULTIPLE-TESTING CORRECTION ACROSS SECTION 12')
            print('='*90)
            print(f'Total hypothesis tests pooled across this section: {n_tests}')
            print(f'Raw p<0.05 (uncorrected, per-test):                {n_locally_would_be_sig_at_p05} / {n_tests}')
            print(f'Surviving a GLOBAL FDR (Benjamini-Hochberg) correction:  {n_sig_fdr_global} / {n_tests}')
            print(f'Surviving a GLOBAL Bonferroni correction:                {n_sig_bonf_global} / {n_tests}')
            print('\nUse this table -- not any single sub-analysis\'s local q-value -- when deciding which '
                  'mechanistic claims in this section are safe to state as confirmatory rather than '
                  'hypothesis-generating. Saved: table_section12_global_multiple_testing_correction.csv')
            print('\nMost significant tests globally:')
            print(global_test_df.head(15)[['analysis','feature','p','q_fdr_global','sig_fdr_global']].to_string(index=False))
        else:
            print('\nNo p-values collected for the global multiple-testing correction table '
                  '(unexpected -- check that global_pvals is being populated above).')

        return self.state


# ============================================================================
# Section13 Chemistry 
# ============================================================================
class Chemistry:
    """
    Chemical-space analysis: UMAP, ratio optimiser, applicability domain, robust lipids, physicochemical descriptors, cargo-type effects, data-driven lipid classes, pKa calibration.
    Reads from state:  PUB_PALETTE, save_pub_figure, clean_df, count_fp_array, fp_matrix, X_D, y, groups, all_y_true, all_y_prob, MC3, DSPC, CHOLESTEROL, DMG_PEG, bundle, predict_lnp, X_ion, ion_fp_cols, knn, ad_threshold, gkf, df
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        PUB_PALETTE = self.state.PUB_PALETTE
        save_pub_figure = self.state.save_pub_figure
        clean_df = self.state.clean_df
        count_fp_array = self.state.count_fp_array
        fp_matrix = self.state.fp_matrix
        X_D = self.state.X_D
        y = self.state.y
        groups = self.state.groups
        all_y_true = self.state.all_y_true
        all_y_prob = self.state.all_y_prob
        MC3 = self.state.MC3
        DSPC = self.state.DSPC
        CHOLESTEROL = self.state.CHOLESTEROL
        DMG_PEG = self.state.DMG_PEG
        bundle = self.state.bundle
        predict_lnp = self.state.predict_lnp
        X_ion = self.state.X_ion
        ion_fp_cols = self.state.ion_fp_cols
        knn = self.state.knn
        ad_threshold = self.state.ad_threshold
        gkf = self.state.gkf
        df = self.state.df

        try:
            import umap
        except ImportError:
            import subprocess, sys
            subprocess.run([sys.executable,'-m','pip','install','umap-learn','-q'], capture_output=True)
            import umap

        ion_data = (
            clean_df.groupby('ionizable_lipid_smiles')
            .agg(mean_EE=('EE','mean'), n_obs=('EE','count'), label=('EE', lambda x: int(x.mean()>=80)))
            .reset_index()
        )

        fps_unique = np.vstack(ion_data['ionizable_lipid_smiles'].apply(count_fp_array).values)
        reducer    = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        embedding  = reducer.fit_transform(fps_unique)

        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        colors = PUB_PALETTE[:8]
        for cls, label, color in [(0,'Low EE (<80%)',colors[0]),(1,'High EE (>=80%)',colors[1])]:
            mask = ion_data['label'] == cls
            axes[0].scatter(embedding[mask,0], embedding[mask,1], c=color,
                            s=ion_data.loc[mask,'n_obs']*10+20, alpha=0.7, label=label, edgecolors='white', lw=0.3)
        axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')
        axes[0].legend()

        sc = axes[1].scatter(embedding[:,0], embedding[:,1], c=ion_data['mean_EE'],
                              cmap='Purples', s=ion_data['n_obs']*10+20, alpha=0.8,
                              edgecolors='white', lw=0.3, vmin=0, vmax=100)
        plt.colorbar(sc, ax=axes[1], label='Mean EE%')
        axes[1].set_xlabel('UMAP 1'); axes[1].set_ylabel('UMAP 2')

        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec13A_umap', width='double')
        plt.show()

        def optimise_ratio(ionizable_smiles, helper_smiles, sterol_smiles, peg_smiles,
                           bundle, il_range=(25, 50, 2.5), peg_range=(1,3,0.5)):
            rows = []
            for il in np.arange(*il_range):
                for peg in np.arange(*peg_range):
                    rem = 100 - il - peg
                    if rem < 10: continue
                    sterol = int(rem*0.78); helper = rem - sterol
                    ratio = f'{il}:{peg}:{sterol}:{helper}'
                    r = predict_lnp(ionizable_smiles, helper_smiles, sterol_smiles, peg_smiles,
                                    molar_ratio=ratio, model_bundle=bundle)
                    rows.append({'molar_ratio':ratio,'IL%':il,'PEG%':peg,'sterol%':sterol,'helper%':helper,
                                 'prob_high_EE':r['prob_high_EE'],'FQS':r.get('FQS','N/A')})
            return pd.DataFrame(rows).sort_values('prob_high_EE', ascending=False)

        print('Optimising ratio for MC3 + DSPC + Cholesterol + DMG-PEG2000...')
        opt = optimise_ratio(MC3, DSPC, CHOLESTEROL, DMG_PEG, bundle)
        print('Top 10 predicted ratios:')
        print(opt.head(10).to_string(index=False))

        ion_fp_cols = [c for c in fp_matrix.columns if c.startswith('ionizable_')]
        X_ion = fp_matrix[ion_fp_cols].values

        k = 5
        knn = NearestNeighbors(n_neighbors=k+1, metric='jaccard', n_jobs=-1)
        knn.fit(X_ion > 0)

        dists, _ = knn.kneighbors(X_ion > 0)
        knn_dists = dists[:, 1:].mean(axis=1)

        ad_threshold = np.percentile(knn_dists, 95)
        print(f'AD threshold (95th percentile of kNN Jaccard distances): {ad_threshold:.4f}')
        print(f'Formulations INSIDE AD:  {(knn_dists <= ad_threshold).sum()} / {len(knn_dists)}')
        print(f'Formulations OUTSIDE AD: {(knn_dists > ad_threshold).sum()} / {len(knn_dists)}')

        clean_df['knn_dist']   = knn_dists
        clean_df['in_AD']      = knn_dists <= ad_threshold

        inside_mask  = clean_df['in_AD'].values
        outside_mask = ~inside_mask

        if outside_mask.sum() > 10:
            auc_inside  = roc_auc_score(all_y_true[inside_mask],  all_y_prob[inside_mask])
            auc_outside = roc_auc_score(all_y_true[outside_mask], all_y_prob[outside_mask])
            print(f'\nAUC inside AD:  {auc_inside:.3f}  (n={inside_mask.sum()})')
            print(f'AUC outside AD: {auc_outside:.3f}  (n={outside_mask.sum()})')
            print(f'Performance drop outside AD: \\Delta ={auc_outside-auc_inside:+.3f}')

        try:
            import umap as umap_lib
            fig, axes = plt.subplots(1, 2, figsize=(7, 3))
            colors = PUB_PALETTE[:8]

            axes[0].scatter(embedding[inside_mask[:len(embedding)], 0],
                            embedding[inside_mask[:len(embedding)], 1],
                            c=colors[0], s=15, alpha=0.6, label='Inside AD')
            axes[0].scatter(embedding[outside_mask[:len(embedding)], 0],
                            embedding[outside_mask[:len(embedding)], 1],
                            c=colors[1], s=40, alpha=0.9, marker='^', label='Outside AD')
            axes[0].legend()
            axes[0].set_xlabel('UMAP 1'); axes[0].set_ylabel('UMAP 2')

            axes[1].hist(knn_dists, bins=30, color=colors[2], edgecolor='white', alpha=0.6)
            axes[1].axvline(ad_threshold, color=colors[1], lw=1.5, ls='--',
                            label=f'AD threshold={ad_threshold:.3f} (95th pct)')
            axes[1].set_xlabel(f'Mean Jaccard distance to {k} nearest neighbours')
            axes[1].set_ylabel('Count')
            axes[1].legend()

            plt.tight_layout()
            save_pub_figure(fig, 'fig_sec13C_applicability_domain', width='double')
            plt.show()
        except Exception as e:
            print(f'UMAP plot skipped (run after Section 9 UMAP): {e}')

        lip_stats = (
            clean_df.groupby('ionizable_lipid_smiles')
            .agg(
                n_obs=('EE','count'), mean_EE=('EE','mean'), std_EE=('EE','std'),
                pct_high_EE=('true_label','mean'),
                n_helper=('helper_lipid_smiles','nunique'), n_peg=('peg_lipid_smiles','nunique'),
            )
            .reset_index()
        )
        lip_stats['std_EE'] = lip_stats['std_EE'].fillna(0)

        lip_multi = lip_stats[lip_stats['n_obs'] >= 3].copy()
        print(f'Ionizable lipids with ≥3 observations: {len(lip_multi)}')

        lip_multi['robust'] = ((lip_multi['mean_EE'] >= 75) & (lip_multi['std_EE'] <= 15))
        lip_multi['universal'] = ((lip_multi['pct_high_EE'] >= 0.75) & (lip_multi['std_EE'] <= 15))
        print(f'\nUniversally robust lipids (≥75% formulations high EE, std≤15%): {lip_multi["universal"].sum()}')
        print(f'Context-dependent lipids (std>15%): {(lip_multi["std_EE"]>15).sum()}')
        print('\nTop 10 most universally robust ionizable lipids:')
        print(lip_multi.sort_values('pct_high_EE', ascending=False).head(10)[
            ['n_obs','mean_EE','std_EE','pct_high_EE','n_helper','n_peg','robust']
        ].round(2).to_string())

        fig, axes = plt.subplots(1, 2, figsize=(7, 3))
        color = PUB_PALETTE[:8]

        sc = axes[0].scatter(
            lip_multi['mean_EE'], lip_multi['std_EE'],
            c=lip_multi['pct_high_EE'], cmap='RdYlGn',
            s=lip_multi['n_obs']*15+20, alpha=0.75, edgecolors='white', lw=0.4,
            vmin=0, vmax=1
        )
        plt.colorbar(sc, ax=axes[0], label='Fraction High EE')
        axes[0].axhline(15, ls='--', color='grey', lw=1, label='std=15% boundary')
        axes[0].axvline(75, ls='--', color='grey', lw=1, label='mean=75% boundary')
        axes[0].set_xlabel('Mean EE% across all formulations')
        axes[0].set_ylabel('EE% Std (across formulations)')
        axes[0].set_xlim(45, 100); axes[0].set_ylim(0, 40)
        axes[0].legend()

        cat = lip_multi['universal'].map({True:'Universal', False:'Context-dependent'})
        lip_multi['category'] = cat
        axes[1].boxplot(
            [lip_multi.loc[lip_multi['category']=='Universal','std_EE'].dropna(),
             lip_multi.loc[lip_multi['category']=='Context-dependent','std_EE'].dropna()],
            labels=['Universal\n(works broadly)','Context-dependent\n(needs specific partner)'],
            patch_artist=True,
        )
        axes[1].set_ylabel('Within-lipid EE% std')

        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec13D_universality', width='double')
        plt.show()


        df['ionizable_lipid_smiles_canon'] = df['ionizable_lipid_smiles'].apply(
            lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(str(s))) if pd.notna(s) and
            Chem.MolFromSmiles(str(s)) is not None else np.nan)

        def best_name(rows):
            primary = rows['ionizable_lipid'].dropna()
            primary = primary[primary != 'Custom lipid']
            if len(primary): return primary.iloc[0]
            original = rows['ionizable_lipid_original'].dropna()
            if len(original): return original.iloc[0]
            return None

        name_lookup = {}
        if 'ionizable_lipid' in df.columns or 'ionizable_lipid_original' in df.columns:
            for smi, rows in df.groupby('ionizable_lipid_smiles_canon'):
                n = best_name(rows)
                if n: name_lookup[smi] = n
            print(f"Resolved names for {len(name_lookup)} canonical ionizable-lipid structures "
                  f"from 'ionizable_lipid' / 'ionizable_lipid_original'.")
        smiles_to_name = name_lookup

        def mol_formula(smiles):
            mol = Chem.MolFromSmiles(str(smiles))
            return rdMolDescriptors.CalcMolFormula(mol) if mol else 'invalid'

        lip_stats_full = (
            clean_df.groupby('ionizable_lipid_smiles')
            .agg(n_obs=('EE', 'count'), mean_EE=('EE', 'mean'), std_EE=('EE', 'std'),
                 pct_high_EE=('true_label', 'mean'),
                 n_helper=('helper_lipid_smiles', 'nunique'), n_peg=('peg_lipid_smiles', 'nunique'))
            .reset_index()
        )
        lip_stats_full['std_EE'] = lip_stats_full['std_EE'].fillna(0)
        lip_multi_full = lip_stats_full[lip_stats_full['n_obs'] >= 3].copy()
        lip_multi_full['universal'] = ((lip_multi_full['pct_high_EE'] >= 0.75) & (lip_multi_full['std_EE'] <= 15))
        lip_multi_full['label'] = lip_multi_full['ionizable_lipid_smiles'].map(
            lambda s: smiles_to_name.get(s, mol_formula(s)))

        robust_lipids = lip_multi_full[lip_multi_full['universal']].sort_values('pct_high_EE', ascending=False)
        print(f'\n{len(robust_lipids)} universally robust ionizable lipids '
              f'(>=70% High-EE across partners, std<=15%, n_obs>=3)\n')

        report_rows = []
        for _, r in robust_lipids.iterrows():
            subset = clean_df[clean_df['ionizable_lipid_smiles'] == r['ionizable_lipid_smiles']]
            best = subset.loc[subset['EE'].idxmax()]
            report_rows.append({
                'Ionizable lipid label':                  r['label'],
                'Ionizable lipid SMILES':                 r['ionizable_lipid_smiles'],
                'n_formulations':                         int(r['n_obs']),
                'Mean EE%':                               round(r['mean_EE'], 1),
                'Std EE%':                                round(r['std_EE'], 1),
                'Frac. High EE (>=80%)':                  round(r['pct_high_EE'], 2),
                'Best formulation EE%':                   round(best['EE'], 1),
                'Best molar ratio (Ion:PEG:Ster:Helper)':  f"{best['ion_ratio']:.2f}:{best['peg_ratio']:.2f}:"
                                                            f"{best['sterol_ratio']:.2f}:{best['helper_ratio']:.2f}",
                'Helper lipid SMILES':                    best['helper_lipid_smiles'],
                'Sterol lipid SMILES':                    best['sterol_lipid_smiles'],
                'PEG lipid SMILES':                       best['peg_lipid_smiles'],
                'Cargo':                                  best.get('target_type', np.nan),
                'Source (DOI/title)':                     best.get('paper_doi', best.get('paper_title', np.nan)),
            })

        report_df = pd.DataFrame(report_rows)
        report_df.to_csv('table_universally_robust_lipid_compositions.csv', index=False)
        print('Saved table_universally_robust_lipid_compositions.csv\n')

        pd.set_option('display.max_colwidth', 60)
        print(report_df[['Ionizable lipid label', 'n_formulations', 'Mean EE%', 'Std EE%',
                          'Frac. High EE (>=80%)', 'Best formulation EE%',
                          'Best molar ratio (Ion:PEG:Ster:Helper)']].to_string(index=False))

        print('\nFull table (incl. all 4-component SMILES, cargo, and source paper) is in '
              'table_universally_robust_lipid_compositions.csv -- pull that directly into the report.')

        def longest_aliphatic_run(mol):
            if mol is None: return np.nan
            patt = Chem.MolFromSmarts('[CX4;!R]')
            matches = [m[0] for m in mol.GetSubstructMatches(patt)]
            if not matches: return 0
            sub = set(matches)
            adj = {a: [] for a in sub}
            for bond in mol.GetBonds():
                a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                if a in sub and b in sub:
                    adj[a].append(b); adj[b].append(a)
            def farthest(start):
                seen = {start: 0}; stack = [start]
                while stack:
                    cur = stack.pop()
                    for nb in adj[cur]:
                        if nb not in seen:
                            seen[nb] = seen[cur] + 1; stack.append(nb)
                f = max(seen, key=seen.get)
                return f, seen[f]
            a0 = next(iter(sub))
            b, _ = farthest(a0)
            _, d = farthest(b)
            return d + 1

        def peg_repeat_units(smi):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None: return np.nan
            patt = Chem.MolFromSmarts('OCC')
            return len(mol.GetSubstructMatches(patt))

        print('Computing physicochemical descriptors for all formulations...')
        clean_df['ion_LogP']         = clean_df['ionizable_lipid_smiles'].apply(lambda s: Descriptors.MolLogP(Chem.MolFromSmiles(str(s))) if Chem.MolFromSmiles(str(s)) else np.nan)
        clean_df['ion_tail_len']     = clean_df['ionizable_lipid_smiles'].apply(lambda s: longest_aliphatic_run(Chem.MolFromSmiles(str(s))))
        clean_df['peg_repeat_units'] = clean_df['peg_lipid_smiles'].apply(peg_repeat_units)

        desc_cols = {
            'Ionizable LogP':          'ion_LogP',
            'Longest aliphatic run':   'ion_tail_len',
            'PEG repeat units (est.)': 'peg_repeat_units',
            'Sterol fraction':         'sterol_frac',
            'Helper fraction':         'helper_frac',
        }

        corr_rows = []
        for label, col in desc_cols.items():
            m = clean_df[col].notna()
            r, p = stats.spearmanr(clean_df.loc[m, col], clean_df.loc[m, 'EE'])
            corr_rows.append({'Descriptor': label, 'spearman_r': r, 'p': p, 'n': int(m.sum())})
        desc_corr_df = pd.DataFrame(corr_rows).sort_values('spearman_r', key=abs, ascending=False)
        print('\n=== Physicochemical Descriptors vs EE% (Spearman) ===')
        print(desc_corr_df.to_string(index=False))

        fig, axes = plt.subplots(1, 4, figsize=(18, 4))
        panels = [('ion_LogP', 'Ionizable lipid LogP'),
                  ('ion_tail_len', 'Longest aliphatic run (atoms)'), ('peg_repeat_units', 'PEG repeat units (est.)')]
        for ax, (col, title) in zip(axes, panels):
            m = clean_df[col].notna()
            ax.scatter(clean_df.loc[m, col], clean_df.loc[m, 'EE'], s=14, alpha=0.4, color=color[0], edgecolor='white', lw=0.3)
            r, p = stats.spearmanr(clean_df.loc[m, col], clean_df.loc[m, 'EE'])
            z = np.polyfit(clean_df.loc[m, col], clean_df.loc[m, 'EE'], 1)
            xs = np.linspace(clean_df.loc[m, col].min(), clean_df.loc[m, col].max(), 50)
            ax.plot(xs, np.polyval(z, xs), color=color[1], lw=2)
            ax.set_xlabel(title); ax.set_ylabel('EE%')
            ax.set_title(f'ρ={r:.2f}, p={p:.1e}', fontsize=10)
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec13F_descriptors', width='double')
        plt.show()

        # --- Does Cargo Type (mRNA / siRNA / DNA) Matter? ----------------------------------------
        print('=== EE% by cargo (target_type) ===')
        print(clean_df.groupby('target_type')['EE'].agg(['count', 'mean', 'median', 'std']).round(1))
        # FIX: fixed cargo order used everywhere below (plot + annotation loop), instead of
        # relying on whatever order pandas/seaborn happen to infer. Previously the boxplot
        # had no explicit `order=`, so the hardcoded 'n=' label loop below it could silently
        # attach sample sizes to the wrong box if seaborn's inferred category order ever
        # differed from ['DNA','mRNA','siRNA'].
        CARGO_ORDER = [c for c in ['DNA', 'mRNA', 'siRNA'] if c in clean_df['target_type'].unique()]
        cargo_groups = [clean_df.loc[clean_df['target_type'] == t, 'EE'].values for t in CARGO_ORDER]
        h, p_cargo = stats.kruskal(*cargo_groups)
        print(f'Kruskal-Wallis across cargo types: H={h:.2f}, p={p_cargo:.3f}')

        fig, ax = plt.subplots(figsize=(5.5, 4))
        color = PUB_PALETTE[:8]
        sns.boxplot(
            data=clean_df, x='target_type', y='EE', order=CARGO_ORDER,
            showfliers=False, width=0.5, color='white', ax=ax
        )
        sns.stripplot(
            data=clean_df, x='target_type', y='EE', order=CARGO_ORDER,
            color=color[4], alpha=0.25, jitter=0.25, size=5, ax=ax
        )
        ax.set_xlabel('Cargo type')
        ax.set_ylabel('EE (%)')
        # FIX: p-value now taken from the Kruskal-Wallis test computed two lines above
        # (p_cargo) instead of a hardcoded 'p = 0.096' string.
        ax.text(
            0.75, 0.05,
            f'Kruskal-Wallis p = {p_cargo:.3g}',
            transform=ax.transAxes,
            ha='left',
            va='bottom',
        )

        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec13G_cargo', width='single')
        plt.show()
        counts = clean_df['target_type'].value_counts()
        # FIX: iterate CARGO_ORDER (the same order the plot above actually used) instead of
        # a hardcoded ['DNA','mRNA','siRNA'] list that could silently mismatch the plotted
        # box order; also guard with .get() in case a cargo type has zero rows.
        for i, grp in enumerate(CARGO_ORDER):
            ax.text(
                i, -0.13,
                f'n={counts.get(grp, 0)}',
                transform=ax.get_xaxis_transform(),
                ha='center'
            )

        cargo_oh = pd.get_dummies(clean_df['target_type'].fillna('Unknown'), prefix='cargo').reset_index(drop=True)
        X_cargo = pd.concat([X_D.reset_index(drop=True), cargo_oh], axis=1)

        def cv_eval(X, y, groups, model):
            aucs, aps = [], []
            for tr, te in gkf.split(X, y, groups):
                model.fit(X.iloc[tr], y[tr])
                prob = model.predict_proba(X.iloc[te])[:, 1]
                aucs.append(roc_auc_score(y[te], prob)); aps.append(average_precision_score(y[te], prob))
            return np.mean(aucs), np.std(aucs), np.mean(aps), np.std(aps)

        rfc_k = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
        auc_m, auc_s, ap_m, ap_s = cv_eval(X_D, y, groups, rfc_k)
        auc_m2, auc_s2, ap_m2, ap_s2 = cv_eval(X_cargo, y, groups, rfc_k)
        print(f'\nX_D (chemistry only):  AUC={auc_m:.3f}±{auc_s:.3f}  AP={ap_m:.3f}±{ap_s:.3f}')
        print(f'X_D + cargo one-hot:   AUC={auc_m2:.3f}±{auc_s2:.3f}  AP={ap_m2:.3f}±{ap_s2:.3f}')
        print(f'Delta AUC from adding cargo: {auc_m2 - auc_m:+.3f}')
        print('Interpretation: if Delta AUC is small/negative, this supports the framing that intrinsic')
        print('lipid chemistry — not payload identity — dominates EE% prediction in this dataset.')

        def desc_row(smi):
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None: return None
            return {'LogP': Descriptors.MolLogP(mol), 'MolWt': Descriptors.MolWt(mol),
                    'RingCount': Lipinski.RingCount(mol), 'RotBonds': Lipinski.NumRotatableBonds(mol),
                    'n_tert_amine': Fragments.fr_NH0(mol), 'n_sec_amine': Fragments.fr_NH1(mol),
                    'n_prim_amine': Fragments.fr_NH2(mol), 'n_ester': Fragments.fr_ester(mol)}

        uniq_lipids = clean_df['ionizable_lipid_smiles'].unique()
        ldesc = pd.DataFrame([desc_row(s) for s in uniq_lipids], index=uniq_lipids).dropna()
        Xs = StandardScaler().fit_transform(ldesc.values)
        print('\n=== KMeans Cluster Validation ===')
        k_results = []
        for k in range(2, 9):
            km_tmp = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km_tmp.fit_predict(Xs)
            k_results.append({'k': k, 'inertia': km_tmp.inertia_, 'silhouette': silhouette_score(Xs, labels)})
        k_df = pd.DataFrame(k_results)
        print(k_df.to_string(index=False))
        km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(Xs)
        ldesc['cluster'] = km.labels_
        clean_df['lipid_class'] = clean_df['ionizable_lipid_smiles'].map(ldesc['cluster'].to_dict())

        class_ee = clean_df.groupby('lipid_class')['EE'].agg(['count', 'mean', 'median', 'std']).round(1)
        class_profile = ldesc.groupby('cluster')[['LogP', 'MolWt', 'RingCount', 'n_tert_amine', 'n_sec_amine', 'n_ester']].mean().round(2)
        print('=== EE% by data-driven ionizable-lipid class (k=4 KMeans on descriptors) ===')
        print(class_ee)
        print('\nCluster chemistry profile (mean descriptor values):')
        print(class_profile)

        class_groups = [clean_df.loc[clean_df['lipid_class'] == c, 'EE'].values for c in sorted(clean_df['lipid_class'].dropna().unique())]
        h2, p_class = stats.kruskal(*class_groups)
        print(f'\nKruskal-Wallis across lipid classes: H={h2:.2f}, p={p_class:.2e}')
        fig, ax = plt.subplots(figsize=(7, 3))
        sns.boxplot(
            data=clean_df, x='lipid_class', y='EE',
            showfliers=False, width=0.5, palette=PUB_PALETTE[8:12], ax=ax
        )
        sns.stripplot(
            data=clean_df, x='lipid_class', y='EE',
            color='black', alpha=0.25, jitter=0.25, size=5, ax=ax
        )
        ax.set_xlabel('Data-driven ionizable lipid class')
        ax.set_ylabel('EE (%)')
        ax.text(
            0.95, 0.05,
            f'Kruskal-Wallis p = {p_class:.2e}',
            transform=ax.transAxes,
            ha='right',
            va='bottom'
        )
        counts = clean_df['lipid_class'].value_counts().sort_index()
        ax.set_xticklabels([
            f'Class 0\n(n={counts.get(0, 0)})',
            f'Class 1\n(n={counts.get(1, 0)})',
            f'Class 2\n(n={counts.get(2, 0)})',
            f'Class 3\n(n={counts.get(3, 0)})'
        ])
        plt.legend()
        plt.tight_layout()
        save_pub_figure(fig, 'fig_sec13H_lipid_class', width='double')
        plt.show()

        return self.state


# ============================================================================
# Section14 FeedbackLoop
# ============================================================================
class FeedbackLoop:
    """
    Human-in-the-loop feedback pipeline for future model updating (not fully automatic online learning).

    Also provides suggest_next_candidates() -- an active-learning acquisition
    function: given a pool of hypothetical formulations, rank them by how
    informative labeling them would be, and return a diverse shortlist.

    Reads from state:  count_fp_array, X_D, y, groups, results_df, gkf
    """
    def __init__(self, state: PipelineState):
        self.state = state

    def run(self):
        """Execute this section and update shared state."""
        count_fp_array = self.state.count_fp_array
        X_D = self.state.X_D
        y = self.state.y
        groups = self.state.groups
        results_df = self.state.results_df
        gkf = self.state.gkf

        FEEDBACK_LOG = 'lnp_feedback_log.jsonl'
        MIN_NEW_SAMPLES_FOR_RETRAIN = 20
        AUC_REGRESSION_TOLERANCE = 0.01

        def log_prediction_feedback(ionizable_smiles, helper_smiles, sterol_smiles, peg_smiles,
                                     molar_ratio, prediction_result, true_EE=None, source='user_submitted'):
            record = {
                'timestamp': datetime.utcnow().isoformat(),
                'ionizable_smiles': ionizable_smiles, 'helper_smiles': helper_smiles,
                'sterol_smiles': sterol_smiles, 'peg_smiles': peg_smiles,
                'molar_ratio': molar_ratio,
                'predicted_prob_high_EE': prediction_result.get('prob_high_EE'),
                'conformal_set': prediction_result.get('conformal_set'),
                'true_EE': true_EE,
                'source': source,
            }
            with open(FEEDBACK_LOG, 'a') as f:
                f.write(json.dumps(record) + '\n')
            return record

        def load_verified_feedback():
            if not os.path.exists(FEEDBACK_LOG):
                return pd.DataFrame()
            rows = [json.loads(l) for l in open(FEEDBACK_LOG)]
            df_fb = pd.DataFrame(rows)
            return df_fb[df_fb['true_EE'].notna()] if len(df_fb) else df_fb

        def featurize_feedback_row(row):
            fp_row = {}
            for prefix, smi in [('ionizable', row['ionizable_smiles']), ('helper', row['helper_smiles']),
                                 ('sterol', row['sterol_smiles']), ('peg', row['peg_smiles'])]:
                arr = count_fp_array(smi)
                fp_row.update({f'{prefix}_fp{i}': arr[i] for i in range(len(arr))})
            parts = [float(x) for x in str(row['molar_ratio']).split(':')]
            total = sum(parts)
            ion_f, peg_f, ste_f, hel_f = [p / total for p in parts]
            fp_row.update({'ion_frac': ion_f, 'peg_frac': peg_f, 'sterol_frac': ste_f, 'helper_frac': hel_f})
            return fp_row

        def evaluate_candidate_vs_current(X_current, y_current, groups_current, new_rows, y_new,
                                           new_groups, current_auc_ref):
            X_combined = pd.concat([X_current, pd.DataFrame(new_rows)[X_current.columns]], ignore_index=True)
            y_combined = np.concatenate([y_current, y_new])
            groups_combined = np.concatenate([groups_current, new_groups])

            gkf_c = GroupKFold(n_splits=5)
            aucs = []
            rfc_c = RandomForestClassifier(n_estimators=500, class_weight='balanced', random_state=42, n_jobs=-1)
            for tr, te in gkf_c.split(X_combined, y_combined, groups_combined):
                rfc_c.fit(X_combined.iloc[tr], y_combined[tr])
                prob = rfc_c.predict_proba(X_combined.iloc[te])[:, 1]
                aucs.append(roc_auc_score(y_combined[te], prob))
            candidate_auc = np.mean(aucs)
            delta = candidate_auc - current_auc_ref
            promote = delta >= -AUC_REGRESSION_TOLERANCE
            print(f'Candidate model AUC (original + verified feedback): {candidate_auc:.3f} '
                  f'(current reference: {current_auc_ref:.3f}, delta: {delta:+.3f})')
            print('Promote candidate model: ' + ('YES' if promote else 'NO -- regression exceeds tolerance'))
            return (rfc_c if promote else None), candidate_auc

        verified = load_verified_feedback()
        print(f'Verified feedback rows available: {len(verified)} '
              f'(retrain triggers at {MIN_NEW_SAMPLES_FOR_RETRAIN})')

        if len(verified) >= MIN_NEW_SAMPLES_FOR_RETRAIN:
            new_rows    = [featurize_feedback_row(r) for _, r in verified.iterrows()]
            y_new       = (verified['true_EE'].values >= 80).astype(int)
            new_groups  = verified['ionizable_smiles'].values
            current_ref_auc = results_df['AUC'].mean()
            candidate_model, candidate_auc = evaluate_candidate_vs_current(
                X_D, y, groups, new_rows, y_new, new_groups, current_ref_auc)
        else:
            print('Not enough verified feedback yet -- logging only. Call log_prediction_feedback(...) '
                  'as outcomes come in (prioritize formulations with low conformal confidence for '
                  'verification, active-learning style), and re-run this cell periodically.')

        print("\n=== Conformal Diagnostics ===")
        print(f"q_hat = {self.state.q_hat}")

        print('\n=== Active Learning Demonstration ===')
        clean_df = self.state.clean_df

        lipid_classes = sorted(clean_df['lipid_class'].dropna().unique())
        candidate_lipids = []
        for cls in lipid_classes:
            top_in_class = (
                clean_df[clean_df['lipid_class'] == cls]
                .groupby('ionizable_lipid_smiles')['EE']
                .mean()
                .sort_values(ascending=False)
                .head(3)
                .index
                .tolist()
            )
            candidate_lipids.extend(top_in_class)
        candidate_pools = []

        for ion_lipid in candidate_lipids:
            pool = self.generate_candidate_pool(
                ion_lipid, self.state.DSPC, self.state.CHOLESTEROL, self.state.DMG_PEG
            )
            candidate_pools.append(pool)
        candidate_pool = pd.concat(candidate_pools, ignore_index=True)
        print(f"Candidate pool size: {len(candidate_pool)}")
        shortlist = self.suggest_next_candidates(candidate_pool, n_suggestions=10)
        print(
            shortlist[['ionizable_smiles', 'molar_ratio', 'prob_high_EE', 'knn_distance',
                       'acquisition_score', 'acquisition_type']]
            .to_string(index=False)
        )
        print(shortlist.head(10).to_string(index=False))

        winners, losers, uncertain = (
            self.rank_candidates_for_validation(candidate_pool, top_n=10)
        )

        print("\n=== TOP PREDICTED WINNERS ===")
        print(winners[['prob_high_EE', 'molar_ratio', 'knn_distance']].to_string(index=False))

        print("\n=== TOP PREDICTED LOSERS ===")
        print(losers[['prob_high_EE', 'molar_ratio', 'knn_distance']].to_string(index=False))

        print("\n=== MOST UNCERTAIN CANDIDATES ===")
        print(uncertain[['prob_high_EE', 'molar_ratio', 'knn_distance']].to_string(index=False))
        return self.state

    def generate_candidate_pool(self, ionizable_smiles, helper_smiles, sterol_smiles, peg_smiles,
                                 il_range=(25, 50, 2.5), peg_range=(1, 3, 0.5)):
        rows = []
        for il in np.arange(*il_range):
            for peg in np.arange(*peg_range):
                rem = 100 - il - peg
                if rem < 10:
                    continue
                sterol = int(rem * 0.78)
                helper = rem - sterol
                rows.append({
                    "ionizable_smiles": ionizable_smiles, "helper_smiles": helper_smiles,
                    "sterol_smiles": sterol_smiles, "peg_smiles": peg_smiles,
                    "molar_ratio": f"{il}:{helper}:{sterol}:{peg}"
                })
        return pd.DataFrame(rows)

    def suggest_next_candidates(self, candidate_pool, n_suggestions=10, ad_multiplier_cutoff=2.0, min_novelty=1e-6):
        """
        Active-learning acquisition function -- see original docstring for full details:
        ranks a pool of hypothetical formulations by uncertainty (conformal ambiguity),
        novelty vs. the applicability domain, and promise, then greedily diversifies
        the shortlist across ionizable-lipid chemotypes.
        """
        predict_lnp = self.state.predict_lnp
        conformal_predict_set = self.state.conformal_predict_set
        q_hat = self.state.q_hat
        knn = self.state.knn
        ion_fp_cols = self.state.ion_fp_cols
        ad_threshold = self.state.ad_threshold
        count_fp_array = self.state.count_fp_array
        bundle = self.state.bundle

        missing = [name for name, val in [
            ('predict_lnp', predict_lnp), ('conformal_predict_set', conformal_predict_set),
            ('q_hat', q_hat), ('knn', knn), ('ion_fp_cols', ion_fp_cols),
            ('ad_threshold', ad_threshold), ('count_fp_array', count_fp_array), ('bundle', bundle),
        ] if val is None]
        if missing:
            raise RuntimeError(
                f"suggest_next_candidates() needs {missing} in state -- make sure "
                f"SaveAndPredictAPI, ConformalPrediction, and ErrorAnalysis have all "
                f"run() before calling this."
            )

        active_bit_indices = [int(c.replace('ionizable_fp', '')) for c in ion_fp_cols]

        rows = []
        for _, cand in candidate_pool.iterrows():
            result = predict_lnp(cand['ionizable_smiles'], cand['helper_smiles'],
                                  cand['sterol_smiles'], cand['peg_smiles'],
                                  molar_ratio=cand['molar_ratio'], model_bundle=bundle)
            if not result.get('valid', True):
                continue

            pset = conformal_predict_set(result['prob_high_EE'], q_hat)
            is_ambiguous = len(pset) == 2

            ion_arr = count_fp_array(cand['ionizable_smiles'])
            ion_fp_vec = ion_arr[active_bit_indices]
            dist, _ = knn.kneighbors((ion_fp_vec > 0).reshape(1, -1), n_neighbors=5)
            knn_dist = float(dist[0].mean())
            in_ad = knn_dist <= ad_threshold

            rows.append({
                **cand.to_dict(),
                'prob_high_EE': result['prob_high_EE'],
                'conformal_set': pset,
                'is_ambiguous': is_ambiguous,
                'knn_distance': knn_dist,
                'in_AD': in_ad,
                '_ion_fp': ion_fp_vec,
            })

        if not rows:
            print('No valid candidates in the pool (all failed SMILES/ratio parsing).')
            return pd.DataFrame()

        cand_df = pd.DataFrame(rows)

        n_before_novelty_filter = len(cand_df)
        cand_df = cand_df[cand_df['knn_distance'] > min_novelty].copy()
        n_excluded_known = n_before_novelty_filter - len(cand_df)
        if n_excluded_known > 0:
            print(f'{n_excluded_known} candidate(s) excluded as already-known lipids '
                f'(knn_distance <= {min_novelty}) -- these are untested ratios of '
                f'existing training-set lipids, not new chemistry.')

        n_before_ad_filter = len(cand_df)
        cand_df = cand_df[cand_df['knn_distance'] <= ad_multiplier_cutoff * ad_threshold].copy()
        if len(cand_df) == 0:
            print(f'All {n_before_ad_filter} candidates fell outside the applicability-domain '
                  f'cutoff ({ad_multiplier_cutoff}x threshold) -- widen ad_multiplier_cutoff, or '
                  f'reconsider whether this candidate pool is too far from the training chemistry.')
            return cand_df.drop(columns=['_ion_fp'], errors='ignore')

        cand_df['uncertainty_score'] = 1.0 - np.abs(cand_df['prob_high_EE'] - 0.5) * 2

        cand_df['novelty_score'] = (
            cand_df['knn_distance'] / cand_df['knn_distance'].max()
        )

        cand_df['ad_penalty'] = np.where(cand_df['in_AD'], 1.0, 0.5)

        cand_df['promise_score'] = cand_df['prob_high_EE']

        cand_df['acquisition_score'] = (
            0.5 * cand_df['uncertainty_score']
            + 0.3 * cand_df['novelty_score']
            + 0.2 * cand_df['promise_score']
        )
        cand_df['acquisition_score'] *= cand_df['ad_penalty']
        cand_df['acquisition_type'] = np.select(
                [cand_df['is_ambiguous'], cand_df['knn_distance'] > ad_threshold, cand_df['prob_high_EE'] > 0.80],
                ['Uncertainty-driven', 'Novelty-driven', 'High-promise'],
                default='Balanced'
            )
        cand_df = cand_df.sort_values('acquisition_score', ascending=False).reset_index(drop=True)

        selected_idx = [0]
        selected_lipids = {cand_df.loc[0, 'ionizable_smiles']}
        remaining_idx = list(range(1, len(cand_df)))
        fps = np.vstack(cand_df['_ion_fp'].values)
        max_score = max(cand_df['acquisition_score'].max(), 1e-9)

        while len(selected_idx) < min(n_suggestions, len(cand_df)) and remaining_idx:
            unpicked_lipid_idx = [i for i in remaining_idx if cand_df.loc[i, 'ionizable_smiles'] not in selected_lipids]
            pool_for_this_slot = unpicked_lipid_idx if unpicked_lipid_idx else remaining_idx
            if not unpicked_lipid_idx:
                print('Ran out of distinct ionizable lipids -- allowing a repeat lipid '
                    'at a different ratio to fill the remaining shortlist slot(s).')

            best_idx, best_combined = None, -1.0
            for i in pool_for_this_slot:
                jaccard_dists_to_selected = [
                    1 - (np.minimum(fps[i], fps[j]).sum() / max(np.maximum(fps[i], fps[j]).sum(), 1))
                    for j in selected_idx
                ]
                diversity = min(jaccard_dists_to_selected)
                combined = 0.5 * diversity + 0.5 * (cand_df.loc[i, 'acquisition_score'] / max_score)
                if combined > best_combined:
                    best_combined, best_idx = combined, i
            selected_idx.append(best_idx)
            selected_lipids.add(cand_df.loc[best_idx, 'ionizable_smiles'])
            remaining_idx.remove(best_idx)

        shortlist = cand_df.iloc[selected_idx].drop(columns=['_ion_fp']).reset_index(drop=True)
        shortlist.to_csv('active_learning_shortlist.csv', index=False)
        print("Saved active_learning_shortlist.csv")
        print("\n=== Active Learning Shortlist ===")
        cols_to_show = ['prob_high_EE', 'conformal_set', 'knn_distance', 'uncertainty_score',
                        'novelty_score', 'promise_score', 'acquisition_score', 'acquisition_type']

        print(shortlist[cols_to_show].round(3).to_string(index=False))
        n_ambiguous = int(cand_df['is_ambiguous'].sum())
        n_excluded_ad = n_before_ad_filter - len(cand_df)
        print(f'suggest_next_candidates(): {len(shortlist)} suggestions from a pool of '
              f'{n_before_ad_filter} ({n_ambiguous} were conformally ambiguous within the AD, '
              f'{n_excluded_ad} excluded as outside the applicability domain).')
        return shortlist

    def rank_candidates_for_validation(self, candidate_pool, top_n=10):
        """
        Generate prospective validation candidates.
        Returns: top_winners, top_losers, uncertain_candidates
        """
        predict_lnp = self.state.predict_lnp
        conformal_predict_set = self.state.conformal_predict_set
        q_hat = self.state.q_hat
        knn = self.state.knn
        ion_fp_cols = self.state.ion_fp_cols
        ad_threshold = self.state.ad_threshold
        count_fp_array = self.state.count_fp_array
        bundle = self.state.bundle

        active_bit_indices = [int(c.replace('ionizable_fp', '')) for c in ion_fp_cols]

        rows = []
        for _, cand in candidate_pool.iterrows():
            result = predict_lnp(cand['ionizable_smiles'], cand['helper_smiles'],
                                  cand['sterol_smiles'], cand['peg_smiles'],
                                  molar_ratio=cand['molar_ratio'], model_bundle=bundle)
            if not result.get("valid", True):
                continue

            ion_arr = count_fp_array(cand['ionizable_smiles'])
            ion_fp_vec = ion_arr[active_bit_indices]
            dist, _ = knn.kneighbors((ion_fp_vec > 0).reshape(1, -1), n_neighbors=5)
            knn_dist = float(dist[0].mean())

            pset = conformal_predict_set(result["prob_high_EE"], q_hat)

            rows.append({
                **cand.to_dict(),
                "prob_high_EE": result["prob_high_EE"],
                "conformal_set": pset,
                "in_AD": knn_dist <= ad_threshold,
                "knn_distance": knn_dist,
                "uncertainty": abs(result["prob_high_EE"] - 0.5)
            })

        df = pd.DataFrame(rows)
        df = df[df["in_AD"]].copy()
        print("\n=== Probability Distribution ===")
        print(df["prob_high_EE"].describe())
        print(f"\nCandidates inside AD = {len(df)}")

        top_winners = (df.sort_values("prob_high_EE", ascending=False).head(top_n))
        top_losers = (df.sort_values("prob_high_EE", ascending=True).head(top_n))
        uncertain_candidates = (df.sort_values("uncertainty", ascending=True).head(top_n))

        top_winners.to_csv("prospective_winners.csv", index=False)
        top_losers.to_csv("prospective_losers.csv", index=False)
        uncertain_candidates.to_csv("prospective_uncertain.csv", index=False)

        print("\nSaved:")
        print("  prospective_winners.csv")
        print("  prospective_losers.csv")
        print("  prospective_uncertain.csv")

        print("\n=== Probability Distribution ===")
        print(df["prob_high_EE"].describe())

        print("\nTop 20 probabilities")
        print(df["prob_high_EE"].sort_values(ascending=False).head(20).to_string(index=False))

        print("\nBottom 20 probabilities")
        print(df["prob_high_EE"].sort_values(ascending=True).head(20).to_string(index=False))

        return (top_winners, top_losers, uncertain_candidates)

# ============================================================================
# Orchestration
# ============================================================================
def main():
    """Run every section top-to-bottom, exactly matching the notebook's order."""
    state = PipelineState()
    pipeline = [
        Setup(state),
        DataCleaning(state),
        EDA(state),
        NoiseFloor(state),
        RigorousNoiseFloor(state),
        FeatureEngineering(state),
        FeatureAblation(state),
        FingerprintBenchmark(state),
        BitWidthComparison(state),
        PrimaryModel(state),
        FQS(state),
        ModelComparison(state),
        SaveAndPredictAPI(state),
        ConformalPrediction(state),
        ErrorAnalysis(state),
        LearningCurve(state),
        SHAP(state),
        Chemistry(state),
        FeedbackLoop(state),
    ]
    for section in pipeline:
        print(f'\n=== Running {type(section).__name__} ===')
        section.run()
    print('\nPipeline complete.')
    return state

if __name__ == "__main__":
    main()