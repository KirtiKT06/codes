"""
dna_tis_params.py
==================
Parameter tables for the Three-Interaction-Site (TIS) sequence-dependent
DNA model of Chakraborty, Hori & Thirumalai, J. Chem. Theory Comput.
2018, 14, 3763-3779 ("the TIS-DNA paper").

Every number in this file is taken directly from that paper (its Tables
1-4 and the calibration values quoted in the text). Where the paper
gives a value "before/after including the entropic correction s" (the
numbers in parentheses in Table 3), we use the AFTER value (with s),
since that is the final parametrized model the paper recommends.

Units used THROUGHOUT this file (converted at the point of use in the
OpenMM script):
    length   : Angstrom (A)
    angle    : degrees
    energy   : kcal/mol
    k_bond   : kcal/mol/A^2
    k_angle  : kcal/mol/rad^2
"""

import math

# ─────────────────────────────────────────────────────────────────────
# Table 4: bonded harmonic parameters (Boltzmann-inverted from PDB
# database mining; angle-specific k_alpha further tuned to reproduce
# ssDNA persistence lengths).
# ─────────────────────────────────────────────────────────────────────

# Backbone + sugar-base "bonds". Real chemical bonds are S-P and P-S
# (the sugar-phosphate backbone, alternating S(i)-P(i+1)-S(i+1)...).
# The sugar-base "bond" (S-Base) is NOT a real chemical bond but is
# treated as one in the TIS representation, since each base hangs off
# its own sugar with only mild orientational freedom.
#
# k_r : kcal/mol/A^2      r0 : Angstrom
BOND_PARAMS = {
    "SP": {"k_r": 62.59, "r0": 3.75},   # sugar(i)   -> phosphate(i+1)
    "PS": {"k_r": 17.63, "r0": 3.74},   # phosphate(i)-> sugar(i)
    "SA": {"k_r": 44.31, "r0": 4.85},   # sugar(i)   -> adenine base(i)
    "SG": {"k_r": 48.98, "r0": 4.96},   # sugar(i)   -> guanine base(i)
    "SC": {"k_r": 43.25, "r0": 4.30},   # sugar(i)   -> cytosine base(i)
    "ST": {"k_r": 46.56, "r0": 4.40},   # sugar(i)   -> thymine base(i)
}

# Angle parameters.  k_alpha : kcal/mol/rad^2   alpha0 : degrees
ANGLE_PARAMS = {
    "PSP": {"k_alpha": 25.67, "alpha0": 123.30},
    "SPS": {"k_alpha": 67.50, "alpha0":  94.60},
    "PSA": {"k_alpha": 29.53, "alpha0": 107.38},
    "PST": {"k_alpha": 39.56, "alpha0":  97.18},
    "PSG": {"k_alpha": 26.28, "alpha0": 111.01},
    "PSC": {"k_alpha": 35.25, "alpha0": 101.49},
    "ASP": {"k_alpha": 67.32, "alpha0": 118.94},
    "TSP": {"k_alpha": 93.99, "alpha0": 123.59},
    "GSP": {"k_alpha": 62.94, "alpha0": 116.90},
    "CSP": {"k_alpha": 77.78, "alpha0": 121.43},
}
DNA_SOFTENING = 1.0  # optional factor to soften the bonded terms (for testing only)
# Stacking-dihedral / hydrogen-bond geometric force constants
# (isotropic across all dimers/base-pairs; bottom part of Table 4).
K_L      = 1.45 * DNA_SOFTENING   # A^-2   (stacking distance l)
K_PHI    = 3.00   # rad^-2 (stacking dihedrals phi1, phi2)
K_D      = 4.00 * DNA_SOFTENING   # A^-2   (hydrogen-bond distance d)
K_THETA  = 1.50   # rad^-2 (hydrogen-bond angles theta1, theta2)
K_PSI    = 0.15   # rad^-2 (hydrogen-bond dihedrals psi1, psi2, psi3)

# ─────────────────────────────────────────────────────────────────────
# Excluded volume (WCA), Eq. 8.  Same D0, eps0 for every interaction
# site (P, S, B of any type).
# ─────────────────────────────────────────────────────────────────────
EV_D0   = 3.2   # Angstrom
EV_EPS0 = 1.0   # kcal/mol

# ─────────────────────────────────────────────────────────────────────
# Stacking interaction, Eq. 9:
#   U_S = U0_S * [1 + kl(l-l0)^2 + kphi(phi1-phi1_0)^2
#                   + kphi(phi2-phi2_0)^2]^-1
#   U0_S = -h + kB(T - Tm)*s               (Tm in the formula below is
#                                            fixed at 331.9 K per dimer
#                                            fit in the paper; here we
#                                            store h, s, dG0 directly
#                                            and reconstruct U0_S(T) at
#                                            run time.)
#
# Table 3 gives h (kcal/mol) and s (dimensionless, multiplies kB) AFTER
# the entropy correction (values NOT in parentheses), plus dG0
# (kcal/mol), for the "x/w" step meaning: the top strand step is x
# stacked on w in the 3'->5' direction; because of near-degeneracy the
# paper reports one entry for several equivalent dimers (e.g. A/A & T/T
# style relations). We expand Table 3 into all 16 explicit dinucleotide
# steps (5'->3' reading, top strand only) using the equivalences stated
# in the text (Eqs. 10-11 symmetry + the "similar propensity" grouping
# used to fill in the rest of the table).
#
# A dinucleotide step here is specified as STEP[i] = (base_i, base_{i+1})
# reading 5'->3' along one strand, e.g. ("A","T") means 5'-A-T-3'.
# ─────────────────────────────────────────────────────────────────────

# h (kcal/mol), s (dimensionless), dG0 (kcal/mol) -- Table 3, using the
# values WITH the entropy correction s (second number where two are
# given for a row).
# h (kcal/mol, PARENTHETICAL / pre-ΔG0-correction column -- see note
# below), s (dimensionless), Tm (K, Table 2), dG0 (kcal/mol, kept only
# for diagnostics -- NOT used in eq 9 itself).
#
# *** WHY PARENTHETICAL, NOT MAIN ***
# Table 3 gives two h values per row, e.g. "5.13 (4.73)" for the G/C
# step. The paper's OWN worked numerical example (Results text, G/C
# dimer) explicitly computes the production U0_S using the
# PARENTHETICAL value:
#     "U0_S = -4.73 + kB(T-331.9)s ,  s = 2.41"
# i.e. h=4.73 (parenthetical), Tm=331.9 (Table 2, G/C row), s=2.41.
# A previous version of this table stored the MAIN column (5.13, 5.69,
# 5.03, etc.) instead -- that does not reproduce the paper's own stated
# formula. Fixed below: parenthetical h everywhere it exists; where the
# paper lists only one h (C/T, T/C, C/C, T/T -- no parenthetical given),
# that single value is unchanged.
_STACK_TABLE_RAW = {
    ("A", "A"): (4.67, 0.94, 322.0, 0.60),
    ("A", "T"): (4.28, 0.65, 293.0, 0.49),   # (A/T over T/A) -- T/A value, 3'->5' convention
    ("T", "A"): (4.18, 0.87, 293.0, 0.49),   # A/T value
    ("A", "C"): (4.24, 0.79, 293.0, 0.49),   # (A/C over C/A) -- C/A value
    ("C", "A"): (4.21, 0.92, 293.0, 0.49),   # A/C value
    ("A", "G"): (4.82, 0.98, 333.6, 0.39),   # (A/G over G/A) -- G/A value
    ("G", "A"): (4.83, 1.06, 333.6, 0.39),   # A/G value
    ("C", "T"): (4.18, 0.94, 288.3, 0.00),   # (C/T over T/C) -- no parenthetical given
    ("T", "C"): (4.13, 0.71, 288.3, 0.00),
    ("C", "C"): (4.15, 0.98, 288.3, 0.00),   # no parenthetical given
    ("C", "G"): (4.73, 2.41, 331.9, 0.26),   # (C/G over G/C) -- G/C value; matches worked example exactly
    ("G", "C"): (4.81, 2.33, 331.9, 0.26),   # C/G value
    ("G", "T"): (4.86, 1.73, 332.6, 0.38),   # (G/T over T/G) -- T/G value
    ("T", "G"): (4.70, 1.69, 332.6, 0.38),   # G/T value
    ("G", "G"): (5.13, -0.29, 353.9, 0.38),  # parenthetical (main was 5.66)
    ("T", "T"): (4.17, 0.89, 288.3, 0.00),   # no parenthetical given
}


def get_stacking_params(base5, base3):
    """
    Return (h, s, Tm, dG0) for the dinucleotide step 5'-base5-base3-3'.
    base5, base3 in {"A","T","G","C"}. Falls back to the reverse-order
    entry (rare, only if a step was not explicitly tabulated -- should
    not happen given the 16 entries above, but kept as a safety net).

    Production formula (paper eq, "Using melting temperature of dimers
    to learn the U0_S value"):
        U0_S(T) = -h + kB*(T - Tm)*s
    dG0 is NOT part of this formula -- it's only used in the paper's
    eq 12 to fit h/s against osmometry free energies, not at simulation
    run time. Kept here for reference/diagnostics only.
    """
    key = (base5.upper(), base3.upper())
    if key in _STACK_TABLE_RAW:
        return _STACK_TABLE_RAW[key]
    rev = (key[1], key[0])
    if rev in _STACK_TABLE_RAW:
        return _STACK_TABLE_RAW[rev]
    raise KeyError(f"No stacking parameters for step {key}")


# (STACKING_TREF removed -- Tm is now stored per-dimer in
# _STACK_TABLE_RAW above, Table 2, and used directly in
# U0_S(T) = -h + kB(T-Tm)*s. No single global reference temperature
# is needed or correct here, since Tm genuinely varies by dimer step
# (288.3 K to 353.9 K across the 16 steps).

# kB in kcal/mol/K
KB_KCAL = 1.98720425e-3

# ─────────────────────────────────────────────────────────────────────
# Hydrogen bonding, Eq. 13.  Single free parameter UHB0, calibrated by
# the paper to reproduce a DNA hairpin melting curve.
# ─────────────────────────────────────────────────────────────────────
UHB0 = -1.92  # kcal/mol  (multiplied by 2 for A-T, 3 for G-C, below)
HB_MULTIPLICITY = {"AT": 2, "TA": 2, "GC": 3, "CG": 3}

# Hydrogen-bond "formed" definition (used only for analysis / optional
# restraint switching, not needed for the raw potential itself):
#   A-T pair considered bonded if U_HB < -2 kB T
#   G-C pair considered bonded if U_HB < -3 kB T
HB_NKBT = {"AT": 2, "TA": 2, "GC": 3, "CG": 3}

# ─────────────────────────────────────────────────────────────────────
# Electrostatics: Debye-Hueckel with Oosawa-Manning renormalized
# phosphate charge (Eq. 14-17).
# ─────────────────────────────────────────────────────────────────────
PHOSPHATE_BARE_CHARGE = -1.0   # e, before Manning renormalization
LENGTH_PER_UNIT_CHARGE_B = 4.4  # Angstrom (Olson & Manning, used by DT/paper)


def bjerrum_length_A(T_kelvin, eps_water):
    """
    Bjerrum length l_B(T) = e^2 / (eps * kB * T) in Angstrom, with e in
    esu-equivalent kcal-based units already absorbed via the standard
    332.0637 kcal*A/mol/e^2 Coulomb constant.
    """
    COULOMB_CONST = 332.0637  # kcal*Angstrom/mol/e^2
    return COULOMB_CONST / (eps_water * KB_KCAL * T_kelvin)


def water_dielectric(T_kelvin):
    """eps_water(T), Eq. 17, T in Kelvin (converted to Celsius inside)."""
    Tc = T_kelvin - 273.15
    return 87.740 - 0.4008 * Tc + 9.398e-4 * Tc**2 - 1.410e-6 * Tc**3


def renormalized_phosphate_charge(T_kelvin, eps_water=None):
    """
    Oosawa-Manning renormalized charge magnitude on the phosphate bead,
    Eq. 16: Q = b / l_B(T).  Returns a NEGATIVE charge (electron units).
    At 298 K this reproduces Q ~ -0.6 e as quoted in the paper.
    """
    if eps_water is None:
        eps_water = water_dielectric(T_kelvin)
    lB = bjerrum_length_A(T_kelvin, eps_water)
    Q_mag = LENGTH_PER_UNIT_CHARGE_B / lB
    # Manning condensation only renormalizes charge down to a maximum
    # magnitude of 1 (bare charge); if Q_mag > 1 (very high T / low
    # eps), cap it at the bare charge.
    Q_mag = min(Q_mag, 1.0)
    return -Q_mag


def debye_length_A(T_kelvin, ionic_strength_M, eps_water=None):
    """
    Debye length lambda_D (Angstrom) for a 1:1 salt of given ionic
    strength (mol/L), Eq. 15.
    """
    if eps_water is None:
        eps_water = water_dielectric(T_kelvin)
    COULOMB_CONST = 332.0637  # kcal*Angstrom/mol/e^2
    N_A = 6.02214076e23
    # convert ionic strength (mol/L) -> number density (1/A^3)
    rho = ionic_strength_M * N_A * 1e-27  # ions / A^3 per mol/L (both ion types, q=+-1)
    # sum_n q_n^2 rho_n = 2 * rho (for 1:1 salt, cation + anion, q^2=1 each)
    kappa2 = 4 * math.pi * COULOMB_CONST / (eps_water * KB_KCAL * T_kelvin) * (2 * rho)
    return 1.0 / math.sqrt(kappa2)