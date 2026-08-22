"""
set_harmonic_bonds_dna.py
--------------------------
DNA analogue of set_harmonic_bonds.py.

Bead indexing convention identical to the RNA pipeline:
    residue i -> beads [3i (P), 3i+1 (S), 3i+2 (Base)]

Bond force constants (kr, kcal/mol/A^2) and equilibrium lengths (r0, A)
are taken VERBATIM from Table 4 of Chakraborty, Hori & Thirumalai,
J. Chem. Theory Comput. 2018, 14, 3763-3779 (eq 2, U_B = kr(r-r0)^2 --
note this paper's k already absorbs the factor that would otherwise sit
outside, i.e. eq 5 form, same convention as your RNA code's k_PS etc).

  bond   kr        r0
  SP     62.59      3.75
  PS     17.63      3.74
  SA     44.31      4.85
  SG     48.98      4.96
  SC     43.25      4.30
  ST     46.56      4.40

Mapping onto the RNA code's bond topology:
  - "PS" (Table 4) = the intra-residue P(i) -> S(i) bond -> RNA's k_PS/d_PS role
  - "SP" (Table 4) = the inter-residue S(i) -> P(i+1) bond -> RNA's k_SP/d_SP role
  (Careful: Table 4's naming order does not necessarily match the RNA
   code's P1S1/S1P2 naming -- double check against your own bead order
   convention before production runs. Assumption used below: SP = S->P
   forward bond (inter-residue), PS = P->S bond (intra-residue), matching
   the paper's likely 5'->3' reading order.)
  - "SA","SG","SC","ST" = intra-residue Sugar -> Base bond, base-dependent
"""

import numpy as np

seq = np.loadtxt("4_sequence_dna.inp", dtype='U2')
no_of_residues = len(seq)
no_of_beads = 3 * no_of_residues
no_of_bonds = no_of_beads - 1

print("Number of residues = %d" % no_of_residues)
print("Number of beads    = %d" % no_of_beads)
print("Number of bonds    = %d" % no_of_bonds)

outfile = open("1_harmonic_bonds_dna.inp", "w")

# --- Table 4: bond force constants (kcal/mol/A^2) and r0 (A) ---
k_PS, d_PS = 17.63, 3.74     # intra-residue P -> S
k_SP, d_SP = 62.59, 3.75     # inter-residue S -> P(next)

k_SA, d_SA = 44.31, 4.85
k_SG, d_SG = 48.98, 4.96
k_SC, d_SC = 43.25, 4.30
k_ST, d_ST = 46.56, 4.40

for i_bond in range(0, no_of_bonds + 1, 3):

    base = seq[i_bond // 3]
    if base == 'A':
        k_SB, d_SB = 44.31, d_SA
    elif base == 'G':
        k_SB, d_SB = 48.98, d_SG
    elif base == 'C':
        k_SB, d_SB = 43.25, d_SC
    elif base == 'T':
        k_SB, d_SB = 46.56, d_ST
    else:
        raise ValueError("Unrecognized base %s at residue %d" % (base, i_bond // 3))

    if i_bond == no_of_bonds - 2:
        # terminal residue: only P-S and S-Base bonds remain
        outfile.write("%d,%d,%.2lf,%.5lf\n" % (i_bond, i_bond + 1, k_PS, d_PS))
        outfile.write("%d,%d,%.2lf,%.5lf\n" % (i_bond + 1, i_bond + 2, k_SB, d_SB))
    else:
        outfile.write("%d,%d,%.2lf,%.5lf\n" % (i_bond, i_bond + 1, k_PS, d_PS))
        outfile.write("%d,%d,%.2lf,%.5lf\n" % (i_bond + 1, i_bond + 2, k_SB, d_SB))
        outfile.write("%d,%d,%.2lf,%.5lf\n" % (i_bond + 1, i_bond + 3, k_SP, d_SP))

outfile.close()
