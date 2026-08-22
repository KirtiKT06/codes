"""
set_stack_input_dna.py
-------------------------
DNA analogue of set_stack_input.py.

Implements eq 9's well-depth term:
    U_S^0 = -h + kB*(T - Tm)*s
using Table 2 (Tm, per dimer step) and Table 3 (h, s, per dimer step) of
Chakraborty, Hori & Thirumalai, JCTC 2018.

*** IMPORTANT CAVEAT ***
Table 3 lists TWO h values per row, e.g. "5.13 (4.73)" for the G/C step:
the parenthetical is described as "the values of h before the inclusion
of the correction term, dG0". The paper's own worked numerical example
(Results text, G/C dimer) explicitly uses the PARENTHETICAL value in the
final production formula:
    "U_S^0 = -4.73 + kB(T-331.9)s ,  s = 2.41"
So this script uses the PARENTHETICAL h (falling back to the single
listed value where no parenthetical is given, e.g. C/C, T/T, C/T, T/C).
If your own re-derivation of Table 3 disagrees, it's a one-line dict edit
below.

*** SECOND CAVEAT: equilibrium geometry (l0, phi10, phi20) ***
The paper states these come from Boltzmann-inverting an ideal B-DNA helix
(same procedure as bonds/angles in Table 4), but does NOT print per-dimer
numeric values in the main text (likely SI Table S1, not available here).
The placeholders below use a SINGLE representative B-DNA stacking geometry
for all 16 steps -- this is almost certainly an approximation relative to
the real per-dimer values used in the paper. Replace PLACEHOLDER_L0 /
PLACEHOLDER_PHI10 / PLACEHOLDER_PHI20 with numbers from your own PDB
mining (same procedure as extract_CG_param_new.tcl) or the paper's SI
before running production simulations -- getting these wrong will bias
your stacked-state geometry even though the well DEPTH is still correct.
"""

import numpy as np

# --- placeholders: REPLACE with real per-dimer Boltzmann-inversion values ---
PLACEHOLDER_L0 = 4.60          # Angstrom, approx base-base stacking distance in B-DNA
PLACEHOLDER_PHI10 = -2.5       # radians, backbone dihedral phi1 (Fig. 3)
PLACEHOLDER_PHI20 = 3.0        # radians, backbone dihedral phi2 (Fig. 3)

kl_stack = 1.45     # A^-2   (Table 4, universal)
kphi_stack = 3.00   # rad^-2 (Table 4, universal)

# Table 2 Tm (K) + Table 3 h (kcal/mol, PARENTHETICAL/pre-correction value), s (cal/mol/K... actually
# s carries units consistent with kB(T-Tm)*s in kcal/mol when kB=0.001987 kcal/mol/K, so s is dimensionless-ish
# scaling used exactly as printed in Table 3)
STACK_PARAMS = {
    # step "XY" = base X (5') stacked on base Y (3'), i.e. residue i base X, residue i+1 base Y
    'AA': dict(Tm=322.0, h=4.67, s=0.94),
    'AT': dict(Tm=293.0, h=4.18, s=0.87),
    'TA': dict(Tm=293.0, h=4.28, s=0.65),
    'AG': dict(Tm=333.6, h=4.83, s=1.06),
    'GA': dict(Tm=333.6, h=4.82, s=0.98),
    'AC': dict(Tm=293.0, h=4.21, s=0.92),
    'CA': dict(Tm=293.0, h=4.24, s=0.79),
    'CG': dict(Tm=331.9, h=4.81, s=2.33),
    'GC': dict(Tm=331.9, h=4.73, s=2.41),
    'GT': dict(Tm=332.6, h=4.70, s=1.69),
    'TG': dict(Tm=332.6, h=4.86, s=1.73),
    'CC': dict(Tm=288.3, h=4.15, s=0.98),
    'CT': dict(Tm=288.3, h=4.13, s=0.71),
    'TC': dict(Tm=288.3, h=4.18, s=0.94),
    'TT': dict(Tm=288.3, h=4.17, s=0.89),
    'GG': dict(Tm=353.9, h=5.13, s=-0.29),
}

kB = 0.001987  # kcal/mol/K


def consecutive_stack_dna(Temperature, n_beads, seq_file="4_sequence_dna.inp"):
    """
    Returns U0, R0, Phi10, Phi20 lists for every consecutive base-stacking
    interaction along the DNA chain, indexed exactly like the RNA
    version's consecutive_stack(): iterates bead index i over base
    positions (i.e. i corresponds to bead 3*residue+2 in the P,S,Base
    scheme), skipping the last stackable step at the 3' end.
    """
    seq = np.loadtxt(seq_file, dtype="U1")
    Temp = float(Temperature)

    U0, R0, Phi10, Phi20 = [], [], [], []

    for i in range(len(seq) - 1):
        step = seq[i] + seq[i + 1]
        if step not in STACK_PARAMS:
            raise ValueError("Unrecognized dinucleotide step %s at residue %d" % (step, i))
        p = STACK_PARAMS[step]
        u0 = -p['h'] + kB * (Temp - p['Tm']) * p['s']

        U0.append(u0)
        R0.append(PLACEHOLDER_L0)
        Phi10.append(PLACEHOLDER_PHI10)
        Phi20.append(PLACEHOLDER_PHI20)

    return U0, R0, Phi10, Phi20


if __name__ == "__main__":
    U0, R0, Phi10, Phi20 = consecutive_stack_dna(298.0, None)
    outfile = open("stack_params_dna.inp", "w")
    for i in range(len(U0)):
        outfile.write("%d,%d,%.5lf,%.5lf,%.5lf,%.5lf\n" % (i, i + 1, U0[i], R0[i], Phi10[i], Phi20[i]))
    outfile.close()
    print("Wrote stack_params_dna.inp (%d stacking interactions)" % len(U0))
