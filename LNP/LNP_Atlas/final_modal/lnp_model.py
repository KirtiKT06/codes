"""
lnp_model.py
============
Knowledge-guided Unified Model for Understanding Design and encapsulation efficiency of Lipid Nanoparticles (KUMUD)
A small, standalone module for USING the already-trained LNP encapsulation-
efficiency classifier -- as opposed to `lnp_pipeline.py`, which is the big
script that BUILDS it (data cleaning, feature engineering, ablations, SHAP,
the whole paper).

Why this file exists
---------------------
If you just trained a Random Forest inside a 4,000-line research script,
"using the model" usually means scrolling through that whole script to find
the one function you need. This file is the opposite of that: it's the
thing you'd actually want to `import` in a Jupyter cell, a Flask app, or a
lab notebook six months from now when you've forgotten how the training
pipeline works but still want a prediction.

It depends on nothing except `rdkit`, `numpy`, `pandas`, and the pickle file
`lnp_pipeline.py` saves (`lnp_classifier.pkl`) -- no matplotlib, no shap, no
umap, no GroupKFold. That's on purpose: loading a model to use it shouldn't
require the heavy machinery you needed to build it.

Quick start
-----------
    from lnp_model import LNPModel

    model = LNPModel.load("lnp_classifier.pkl")

    # The raw trained sklearn classifier is right there if you want it --
    # nothing hidden behind private attributes:
    print(model.model)                    # RandomForestClassifier(...)
    print(model.feature_names[:5])        # ['ionizable_fp0', 'ionizable_fp1', ...]
    print(model.training_auc)             # 0.78-ish

    # The thing you actually came here for:
    result = model.predict(
        ionizable_smiles="O=C(OCCC(OC(=O)CCCCCCC/C=C\\CCCCCCCC)COCCN(CC)CC)CCCCCCC/C=C\\CCCCCCCC",      # MC3
        helper_smiles="CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[NH3+])OC(=O)CCCCCCCCCCCCCCCC",        # DSPC
        sterol_smiles="OC1CCC2(C)C(CCC3C2CC=C2C3(C)CCC(C(C)CCCC(C)C)C2)C1",                             # Cholesterol
        peg_smiles="CCCCCCCCCCCCCCCCCC(=O)OCC(OC(=O)CCCCCCCCCCCCCCCCC)COC(=O)OCC(O)COCCOCCOCCOCC"
                    "OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCC",                                          # DMG-PEG2000
        molar_ratio="50:10:38.5:1.5",
        size_nm=85.0,
        pdi=0.12,
    )
    print(result)

Run this file directly (`python lnp_model.py`) for a live demo using the
same worked example.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


# =============================================================================
# Desirability functions for the Formulation Quality Score (FQS)
# =============================================================================
# These three functions score how "good" a formulation looks on one axis
# each -- predicted encapsulation efficiency, particle size, and
# polydispersity index (PDI, a measure of how uniform the particle sizes
# are). Each returns a number from 0 (bad) to 1 (ideal). FQS then combines
# all three into one number a formulation scientist can skim at a glance.
#
# The size/PDI thresholds below aren't arbitrary -- they're taken from the
# same literature anchors used in the original notebook (Section 10):
#   - 50-120 nm optimal window, <150 nm hard upper limit:
#       US Patent 11,938,227; Lokugamage et al., RSC Pharmaceutics 2024
#       (DOI:10.1039/D4PM00128A)
#   - PDI <= 0.2 target, <= 0.3 FDA outer limit:
#       Danaei et al., Pharmaceutics 2018, 10(2):57 (DOI:10.3390/pharmaceutics10020057)


def _desirability_ee(prob_high_ee: float, r: float = 3) -> float:
    """
    How desirable is this predicted P(EE >= 80%)?

    This is a Derringer-Suich (1980) "larger-the-better" desirability
    function. With r=1 it would just be the probability itself; r=3 (the
    value used throughout this project) bends the curve so that
    low-confidence predictions are penalised more than a straight line
    would, and only genuinely high-confidence "High EE" calls get close
    to a perfect score of 1.0.
    """
    p = np.clip(prob_high_ee, 0, 1)
    return p ** r


def _desirability_size(size_nm: Optional[float]) -> Optional[float]:
    """
    How desirable is this particle size?

    Trapezoidal shape: ramps up from 0 at 30 nm to 1.0 across the ideal
    50-120 nm plateau, then ramps back down to 0 at 150 nm (the accepted
    upper limit for a therapeutic LNP). Returns None if size wasn't provided.
    """
    if size_nm is None or (isinstance(size_nm, float) and np.isnan(size_nm)):
        return None
    s = float(size_nm)
    if s <= 30:
        return 0.0
    if s <= 50:
        return (s - 30) / 20.0
    if s <= 120:
        return 1.0
    if s <= 150:
        return (150 - s) / 30.0
    return 0.0


def _desirability_pdi(pdi: Optional[float]) -> Optional[float]:
    """
    How desirable is this PDI (polydispersity index)?

    PDI <= 0.2 is excellent (score 1.0); it fades linearly to 0 by the FDA's
    outer acceptable limit of 0.3. Returns None if PDI wasn't provided.
    """
    if pdi is None or (isinstance(pdi, float) and np.isnan(pdi)):
        return None
    p = float(pdi)
    if p <= 0.20:
        return 1.0
    if p <= 0.30:
        return (0.30 - p) / 0.10
    return 0.0


def compute_fqs(prob_high_ee: float, size_nm: Optional[float] = None,
                 pdi: Optional[float] = None) -> dict:
    """
    Formulation Quality Score (FQS): a single 0-100 number combining
    predicted encapsulation efficiency with physicochemical QC criteria,
    via a geometric mean (so a formulation that fails badly on ANY one
    axis gets pulled down, rather than being averaged away by good scores
    on the other axes).

    Returns a dict with the overall FQS plus each component score, so you
    can see exactly why a formulation scored the way it did.
    """
    d_ee = _desirability_ee(prob_high_ee)
    components, labels = [d_ee], ["EE"]

    d_size = _desirability_size(size_nm)
    if d_size is not None:
        components.append(d_size)
        labels.append("size")

    d_pdi = _desirability_pdi(pdi)
    if d_pdi is not None:
        components.append(d_pdi)
        labels.append("PDI")

    geometric_mean = float(np.prod(components) ** (1 / len(components)))
    return {
        "FQS": round(geometric_mean * 100, 2),
        "components": "+".join(labels),
        "d_EE": round(d_ee, 4),
        "d_size": round(d_size, 4) if d_size is not None else None,
        "d_PDI": round(d_pdi, 4) if d_pdi is not None else None,
    }


def _fqs_grade(fqs: float) -> str:
    """Turn a numeric FQS into a plain-English grade for quick reading."""
    if fqs >= 75:
        return "Excellent"
    if fqs >= 55:
        return "Good"
    if fqs >= 35:
        return "Moderate"
    return "Poor"


# =============================================================================
# The result object returned by LNPModel.predict()
# =============================================================================
@dataclass
class PredictionResult:
    """
    Everything you get back from a single prediction, in one readable
    object instead of a loose dict. Every field is exactly what it sounds
    like -- print(result) gives you a human-readable summary.
    """
    valid: bool
    error: Optional[str] = None

    prob_high_EE: Optional[float] = None          # model's probability the formulation is High EE
    prediction: Optional[str] = None               # "High EE (>=80%)" or "Low EE", using the model's own threshold
    confidence: Optional[str] = None                # "High" / "Moderate" / "Low", based on distance from 0.5

    FQS: Optional[float] = None                     # 0-100 Formulation Quality Score
    FQS_grade: Optional[str] = None                 # "Excellent" / "Good" / "Moderate" / "Poor"
    FQS_components: Optional[str] = None             # which of EE/size/PDI went into the score
    d_EE: Optional[float] = None
    d_size: Optional[float] = None
    d_PDI: Optional[float] = None

    def __repr__(self) -> str:
        if not self.valid:
            return f"PredictionResult(INVALID: {self.error})"
        return (
            f"PredictionResult(\n"
            f"  prediction     = {self.prediction!r}  (confidence: {self.confidence})\n"
            f"  prob_high_EE   = {self.prob_high_EE}\n"
            f"  FQS            = {self.FQS} / 100  ({self.FQS_grade}, from {self.FQS_components})\n"
            f")"
        )


# =============================================================================
# LNPModel -- the main thing you actually import
# =============================================================================
class LNPModel:
    """
    A loaded, ready-to-use LNP encapsulation-efficiency classifier.

    Everything about the model is a plain, public attribute -- there's
    nothing private to dig through. In particular:

        model.model            the raw trained sklearn RandomForestClassifier.
                                Use this directly if you want `.predict_proba()`,
                                `.feature_importances_`, tree internals, or
                                anything else scikit-learn exposes that this
                                wrapper doesn't.
        model.feature_names    the exact ordered list of columns the model
                                expects (fingerprint bits + molar fractions).
        model.decision_threshold
                                the probability cutoff used to turn a P(High EE)
                                into a High/Low call. Not necessarily 0.5 --
                                see the comment on `decision_threshold` below.
        model.training_auc / model.training_ap
                                the cross-validated performance metrics that
                                were logged when this model was trained
                                (see lnp_pipeline.py, Section 9).

    Typical usage
    -------------
        model = LNPModel.load("lnp_classifier.pkl")
        result = model.predict(ionizable_smiles=..., helper_smiles=..., ...)
    """

    def __init__(self, bundle: dict):
        # `bundle` is the dict that lnp_pipeline.py's SaveAndPredictAPI section
        # pickled to lnp_classifier.pkl. We just unpack it into friendly,
        # documented, public attributes rather than making people dig through
        # a raw dict with string keys.
        self._bundle = bundle

        self.model = bundle["model"]                          # the raw sklearn RandomForestClassifier -- yours to use directly
        self.feature_names = bundle["feature_names"]            # exact column order the model expects
        self.active_fp_columns = bundle["active_fp_cols"]       # which fingerprint bit columns are actually used
        self.fp_radius = bundle["fp_radius"]                    # Morgan fingerprint radius (3 = ECFP6)
        self.fp_nbits = bundle["fp_nbits"]                      # fingerprint bits per lipid component
        self.ee_threshold = bundle["ee_threshold"]              # the EE% cutoff used to define "High EE" (80%)
        self.decision_threshold = bundle["decision_threshold"]  # probability cutoff for High/Low calls (see note below)
        self.training_n = bundle["training_n"]                  # how many formulations the model was trained on
        self.training_auc = bundle["training_auc"]              # cross-validated ROC-AUC at training time
        self.training_ap = bundle["training_ap"]                # cross-validated Average Precision at training time

        # Set up the fingerprint generator once, matching how the model was
        # trained (see lnp_pipeline.py, FeatureEngineering section -- keep
        # fp_radius/fp_nbits here in sync with whatever that section used).
        self._fpgen = rdFingerprintGenerator.GetMorganGenerator(
            radius=self.fp_radius, fpSize=self.fp_nbits
        )

    # -------------------------------------------------------------------
    # Loading
    # -------------------------------------------------------------------
    @classmethod
    def load(cls, path: str = "lnp_classifier.pkl") -> "LNPModel":
        """Load a trained model from the pickle file lnp_pipeline.py saved."""
        with open(path, "rb") as f:
            bundle = pickle.load(f)
        return cls(bundle)

    def __repr__(self) -> str:
        return (
            f"LNPModel(trained on n={self.training_n} formulations, "
            f"AUC={self.training_auc}, AP={self.training_ap}, "
            f"decision_threshold={self.decision_threshold})"
        )

    # -------------------------------------------------------------------
    # Internal helper: turn one lipid's SMILES into its fingerprint
    # -------------------------------------------------------------------
    def _fingerprint(self, mol, prefix: str) -> dict:
        """
        Compute a count-based Morgan fingerprint for one lipid component
        and return it as {'<prefix>_fp0': count, '<prefix>_fp1': count, ...}
        -- the same column-naming convention used when the model was trained,
        so the resulting dict can be dropped straight into a feature row.
        """
        fp = self._fpgen.GetCountFingerprint(mol)
        arr = np.zeros(self.fp_nbits, dtype=np.float32)
        for idx, count in fp.GetNonzeroElements().items():
            arr[idx % self.fp_nbits] += count  # fold hash collisions into nbits, same as training
        return {f"{prefix}_fp{i}": arr[i] for i in range(self.fp_nbits)}

    # -------------------------------------------------------------------
    # The main entry point
    # -------------------------------------------------------------------
    def predict(
        self,
        ionizable_smiles: str,
        helper_smiles: str,
        sterol_smiles: str,
        peg_smiles: str,
        molar_ratio: str,
        size_nm: Optional[float] = None,
        pdi: Optional[float] = None,
    ) -> PredictionResult:
        """
        Predict the encapsulation-efficiency class (and Formulation Quality
        Score) of a new lipid nanoparticle formulation.

        Parameters
        ----------
        ionizable_smiles, helper_smiles, sterol_smiles, peg_smiles : str
            SMILES strings for each of the four lipid components.
        molar_ratio : str
            The four components' molar ratio, formatted
            "ionizable:helper:sterol:peg", e.g. "50:10:38.5:1.5".
            (Note: this is IL:helper:sterol:PEG order to match how the
            training data's `lipid_molar_ratio` column was parsed --
            double check against lnp_pipeline.py's DataCleaning section
            if your ratios look reversed.)
        size_nm, pdi : float, optional
            Measured or target particle size (nm) and PDI, if you have them.
            These only affect the FQS score, not the EE prediction itself --
            leave them out and FQS will be computed from EE alone.

        Returns
        -------
        PredictionResult
            `result.valid` is False (with `result.error` explaining why) if
            any SMILES failed to parse or the molar ratio couldn't be read --
            check that before trusting the rest of the fields.
        """
        # --- Step 1: make sure every SMILES string is actually a valid molecule ---
        components = {
            "ionizable": ionizable_smiles,
            "helper": helper_smiles,
            "sterol": sterol_smiles,
            "peg": peg_smiles,
        }
        molecules = {}
        for name, smi in components.items():
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                return PredictionResult(valid=False, error=f"Invalid SMILES for {name}: {smi!r}")
            molecules[name] = mol

        # --- Step 2: parse the molar ratio into fractions that sum to 1 ---
        try:
            parts = [float(x) for x in str(molar_ratio).strip().split(":")]
            if len(parts) != 4:
                raise ValueError(f"expected 4 colon-separated numbers, got {len(parts)}")
            total = sum(parts)
            ion_frac, helper_frac, sterol_frac, peg_frac = [p / total for p in parts]
        except Exception as exc:
            return PredictionResult(valid=False, error=f"Invalid molar_ratio {molar_ratio!r}: {exc}")

        # --- Step 3: build the fingerprint feature row ---
        fp_row = {}
        for name, mol in molecules.items():
            fp_row.update(self._fingerprint(mol, name))

        feature_row = {col: fp_row.get(col, 0.0) for col in self.active_fp_columns}
        feature_row.update({
            "ion_frac": ion_frac,
            "helper_frac": helper_frac,
            "sterol_frac": sterol_frac,
            "peg_frac": peg_frac,
        })
        X_new = pd.DataFrame([feature_row])[self.feature_names]

        # --- Step 4: ask the model ---
        prob_high_ee = float(self.model.predict_proba(X_new)[0, 1])
        is_high = prob_high_ee >= self.decision_threshold

        # How far is the prediction from a coin-flip? Used only as a rough,
        # human-readable confidence label -- not a calibrated uncertainty
        # estimate. (lnp_pipeline.py's ConformalPrediction section has a
        # statistically rigorous version of this if you need one.)
        distance_from_uncertain = abs(prob_high_ee - 0.5)
        if distance_from_uncertain > 0.25:
            confidence = "High"
        elif distance_from_uncertain > 0.10:
            confidence = "Moderate"
        else:
            confidence = "Low"

        # --- Step 5: Formulation Quality Score ---
        fqs_result = compute_fqs(prob_high_ee, size_nm=size_nm, pdi=pdi)

        return PredictionResult(
            valid=True,
            error=None,
            prob_high_EE=round(prob_high_ee, 4),
            prediction=f"High EE (>={self.ee_threshold}%)" if is_high else "Low EE",
            confidence=confidence,
            FQS=fqs_result["FQS"],
            FQS_grade=_fqs_grade(fqs_result["FQS"]),
            FQS_components=fqs_result["components"],
            d_EE=fqs_result["d_EE"],
            d_size=fqs_result["d_size"],
            d_PDI=fqs_result["d_PDI"],
        )

    # -------------------------------------------------------------------
    # Convenience: predict for many formulations at once
    # -------------------------------------------------------------------
    def predict_batch(self, formulations: pd.DataFrame) -> pd.DataFrame:
        """
        Run .predict() over every row of a DataFrame and return the results
        as new columns appended to a copy of that DataFrame.

        `formulations` must have these columns: ionizable_smiles,
        helper_smiles, sterol_smiles, peg_smiles, molar_ratio, and
        optionally size_nm, pdi.

        This is just a convenience loop around `.predict()` -- for a handful
        of formulations it's plenty fast, but it re-runs the fingerprint
        computation row by row rather than vectorising, so for very large
        batches (thousands of rows) you may want to profile it.
        """
        results = []
        for _, row in formulations.iterrows():
            r = self.predict(
                ionizable_smiles=row["ionizable_smiles"],
                helper_smiles=row["helper_smiles"],
                sterol_smiles=row["sterol_smiles"],
                peg_smiles=row["peg_smiles"],
                molar_ratio=row["molar_ratio"],
                size_nm=row.get("size_nm"),
                pdi=row.get("pdi"),
            )
            results.append({
                "valid": r.valid, "error": r.error,
                "prob_high_EE": r.prob_high_EE, "prediction": r.prediction,
                "confidence": r.confidence, "FQS": r.FQS, "FQS_grade": r.FQS_grade,
            })
        return pd.concat([formulations.reset_index(drop=True), pd.DataFrame(results)], axis=1)

    # -------------------------------------------------------------------
    # Convenience: sweep molar ratios for a fixed set of 4 lipids
    # -------------------------------------------------------------------
    def optimize_ratio(
        self,
        ionizable_smiles: str,
        helper_smiles: str,
        sterol_smiles: str,
        peg_smiles: str,
        ionizable_pct_range: tuple = (30, 66, 5),
        peg_pct_range: tuple = (1, 6, 1),
    ) -> pd.DataFrame:
        """
        Given a fixed choice of the 4 lipid components, sweep the molar
        ratio and return every combination tried, sorted by predicted
        P(High EE) -- highest first. Useful for "I know which lipids I want,
        what ratio should I use?".

        The sterol/helper split for each candidate ratio is fixed at
        roughly 78%/22% of the remaining mass (matching the sterol:helper
        ratio commonly seen in the training data) -- if you want to sweep
        that too, call `.predict()` directly in your own loop instead.

        Parameters
        ----------
        ionizable_pct_range, peg_pct_range : tuple(start, stop, step)
            Passed straight to Python's range() for the ionizable-lipid and
            PEG-lipid mol% to try.
        """
        rows = []
        for ionizable_pct in range(*ionizable_pct_range):
            for peg_pct in range(*peg_pct_range):
                remainder = 100 - ionizable_pct - peg_pct
                if remainder < 10:
                    continue  # not enough room left for sterol + helper to make sense
                sterol_pct = int(remainder * 0.78)
                helper_pct = remainder - sterol_pct
                ratio_str = f"{ionizable_pct}:{peg_pct}:{sterol_pct}:{helper_pct}"

                result = self.predict(
                    ionizable_smiles, helper_smiles, sterol_smiles, peg_smiles,
                    molar_ratio=ratio_str,
                )
                if not result.valid:
                    continue
                rows.append({
                    "molar_ratio": ratio_str,
                    "ionizable_pct": ionizable_pct, "peg_pct": peg_pct,
                    "sterol_pct": sterol_pct, "helper_pct": helper_pct,
                    "prob_high_EE": result.prob_high_EE, "FQS": result.FQS,
                })
        return pd.DataFrame(rows).sort_values("prob_high_EE", ascending=False).reset_index(drop=True)


# =============================================================================
# Demo -- run this file directly to see it in action
# =============================================================================
if __name__ == "__main__":
    # Same four reference lipids used throughout lnp_pipeline.py, so you can
    # sanity-check this file's output against that script's Section G demo.
    MC3 = "O=C(OCCC(OC(=O)CCCCCCC/C=C\\CCCCCCCC)COCCN(CC)CC)CCCCCCC/C=C\\CCCCCCCC"
    DSPC = "CCCCCCCCCCCCCCCCCC(=O)OCC(COP(=O)([O-])OCC[NH3+])OC(=O)CCCCCCCCCCCCCCCC"
    CHOLESTEROL = "OC1CCC2(C)C(CCC3C2CC=C2C3(C)CCC(C(C)CCCC(C)C)C2)C1"
    DMG_PEG = ("CCCCCCCCCCCCCCCCCC(=O)OCC(OC(=O)CCCCCCCCCCCCCCCCC)COC(=O)OCC(O)COCCOCCOCCOCC"
               "OCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCCOCC")

    print("Loading model from lnp_classifier.pkl ...")
    model = LNPModel.load("lnp_classifier.pkl")
    print(model)
    print()

    print("=== A well-formed MC3 mRNA-LNP formulation ===")
    good = model.predict(MC3, DSPC, CHOLESTEROL, DMG_PEG, "50:10:38.5:1.5", size_nm=85.0, pdi=0.12)
    print(good)
    print()

    print("=== Same lipids, but a poor formulation (oversized, high PDI) ===")
    bad = model.predict(MC3, DSPC, CHOLESTEROL, DMG_PEG, "50:10:38.5:1.5", size_nm=220.0, pdi=0.45)
    print(bad)
    print()

    print("=== Sweeping molar ratios to find the best one for these 4 lipids ===")
    sweep = model.optimize_ratio(MC3, DSPC, CHOLESTEROL, DMG_PEG)
    print(sweep.head(5).to_string(index=False))
