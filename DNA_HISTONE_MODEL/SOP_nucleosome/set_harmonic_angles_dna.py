"""
set_harmonic_angles_dna.py
----------------------------
DNA analogue of set_harmonic_angles.py.

Angle force constants (k_alpha, kcal/mol/rad^2) and equilibrium angles
(alpha0, degrees -> converted to radians) taken VERBATIM from Table 4:

  angle   k_alpha   alpha0(deg)
  PSP     25.67     123.30      backbone: P(i-1) - S(i) - P(i+1)
  SPS     67.50      94.60      backbone: S(i)   - P(i+1) - S(i+1)
  PSA     29.53     107.38      P(i-1) - S(i) - Base(i), when Base=A
  PST     39.56      97.18      P(i-1) - S(i) - Base(i), when Base=T
  PSG     26.28     111.01      P(i-1) - S(i) - Base(i), when Base=G
  PSC     35.25     101.49      P(i-1) - S(i) - Base(i), when Base=C
  ASP     67.32     118.94      Base(i) - S(i) - P(i+1), when Base=A
  TSP     93.99     123.59      Base(i) - S(i) - P(i+1), when Base=T
  GSP     62.94     116.90      Base(i) - S(i) - P(i+1), when Base=G
  CSP     77.78     121.43      Base(i) - S(i) - P(i+1), when Base=C

Convention (matches RNA code's P1/P2 naming): P1 = phosphate 5' of the
sugar (preceding), P2 = phosphate 3' of the sugar (following). "PSx" is
read as P1-S-Base; "xSP" is read as Base-S-P2. Verify this assignment
against the paper's SI if angle labelling turns out reversed -- it does
not change energetics much (both P-S-Base angles are physically similar)
but does matter for reproducing the exact reported persistence lengths.
"""

import numpy as np

seq = np.loadtxt("4_sequence_dna.inp", dtype='U2')
no_of_residues = len(seq)
no_of_beads = 3 * no_of_residues
no_of_angles = 4 * no_of_residues - 3

print("Number of residues = %d" % no_of_residues)
print("Number of beads    = %d" % no_of_beads)
print("Number of angles   = %d" % no_of_angles)

outfile = open("2_harmonic_angles_dna.inp", "w")

k_PSP, a_PSP = 25.67, np.deg2rad(123.30)
k_SPS, a_SPS = 67.50, np.deg2rad(94.60)

k_PSA, a_PSA = 29.53, np.deg2rad(107.38)
k_PST, a_PST = 39.56, np.deg2rad(97.18)
k_PSG, a_PSG = 26.28, np.deg2rad(111.01)
k_PSC, a_PSC = 35.25, np.deg2rad(101.49)

k_ASP, a_ASP = 67.32, np.deg2rad(118.94)
k_TSP, a_TSP = 93.99, np.deg2rad(123.59)
k_GSP, a_GSP = 62.94, np.deg2rad(116.90)
k_CSP, a_CSP = 77.78, np.deg2rad(121.43)

PS_TABLE = {'A': (k_PSA, a_PSA), 'T': (k_PST, a_PST),
            'G': (k_PSG, a_PSG), 'C': (k_PSC, a_PSC)}
SP_TABLE = {'A': (k_ASP, a_ASP), 'T': (k_TSP, a_TSP),
            'G': (k_GSP, a_GSP), 'C': (k_CSP, a_CSP)}

for i_ang in range(1, no_of_beads + 1, 3):

    base = seq[i_ang // 3]
    k_P1SB, a_P1SB = PS_TABLE[base]

    if i_ang == no_of_beads - 2:
        # terminal residue: only P1-S-Base angle survives
        outfile.write("%d,%d,%d,%.2lf,%.4lf\n" % (i_ang - 1, i_ang, i_ang + 1, k_P1SB, a_P1SB))
    else:
        k_BSP2, a_BSP2 = SP_TABLE[base]

        outfile.write("%d,%d,%d,%.2lf,%.4lf\n" % (i_ang - 1, i_ang, i_ang + 1, k_P1SB, a_P1SB))
        outfile.write("%d,%d,%d,%.2lf,%.4lf\n" % (i_ang + 1, i_ang, i_ang + 2, k_BSP2, a_BSP2))
        outfile.write("%d,%d,%d,%.2lf,%.4lf\n" % (i_ang - 1, i_ang, i_ang + 2, k_PSP, a_PSP))
        outfile.write("%d,%d,%d,%.2lf,%.4lf\n" % (i_ang, i_ang + 2, i_ang + 3, k_SPS, a_SPS))

outfile.close()
