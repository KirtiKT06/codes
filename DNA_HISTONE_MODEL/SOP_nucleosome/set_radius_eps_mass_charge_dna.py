"""
set_radius_eps_mass_charge_dna.py
----------------------------------
DNA analogue of set_radius_eps_mass_charge.py (the RNA version).

Key difference from the RNA case (per the paper, Methodology - Excluded
Volume section, eq 8): the TIS-DNA model does NOT use bead-type-specific
WCA radii/epsilons the way the RNA pipeline does. Instead:

    "All the interaction sites are assigned the same D0 and eps0 to
     keep the parametrization as simple as possible."
    D0 = 3.2 A, eps0 = 1 kcal/mol   (Chakraborty, Hori & Thirumalai 2018)

So the radius/eps columns below are uniform across P, S, and base beads.
Mass and charge still vary by bead identity, exactly as in the RNA file.

Masses:
  - Phosphate (P) and base (A, G, C) masses are carried over unchanged
    from the RNA parametrization, since the chemical groups are identical
    for those beads in both nucleic acids.
  - Sugar (S) mass is reduced relative to RNA's ribose bead to account for
    loss of the 2'-OH (deoxyribose instead of ribose): approx -16 amu (one
    oxygen). This is an engineering approximation -- verify against your
    own atom-grouping mass sums.
  - Thymine (T) replaces uracil (U): T = 5-methyluracil, so
    mass(T) = mass(U) + mass(CH2) = 111.07882 + 14.0266 = 125.10542

Charges: P = -1 (bare charge; the Oosawa-Manning renormalization to -0.6
is applied later, in the electrostatics force, NOT here -- keeping bare
charge in this file mirrors how the RNA pipeline stores bare ion charges
and lets the force-field code decide screening/renormalization).
S = 0, bases = 0 (no partial charges on bases in the TIS model).
"""

import numpy as np

D0_DNA = 3.2       # Angstrom, eq 8
EPS0_DNA = 1.0      # kcal/mol, eq 8

# Same PDB-derived, mass-weighted CG structure file convention as the RNA
# pipeline: column 3 (index 2) gives bead identity (P/S/C where C = base
# placeholder position, to be resolved against the sequence file).
atom_name, res_name, x, y, z = [], [], [], [], []
with open("dna_tis.pdb") as f:
    for line in f:
        if line.startswith(("ATOM", "HETATM")):
            atom_name.append(line[12:16].strip())
            res_name.append(line[17:20].strip())
            x.append(float(line[30:38]))
            y.append(float(line[38:46]))
            z.append(float(line[46:54]))

outfile = open("./3_sigma_eps_mass_charge_dna.inp", 'w')
outfile.write("Type, radius(A), eps(kcal/mol), mass(amu), charge(unit)\n")

for bead, base in zip(atom_name, res_name):
    if bead == 'P':
        outfile.write("%s,%lf,%lf,%lf,%lf\n" % ('P', D0_DNA/2.0, EPS0_DNA, 62.9714, -1))
    elif bead == 'S':
        outfile.write("%s,%lf,%lf,%lf,%lf\n" % ('S', D0_DNA/2.0, EPS0_DNA, 115.1083, 0))
    else:
        if base in ('A', 'DA'):
            outfile.write("%s,%lf,%lf,%lf,%lf\n" % ('A', D0_DNA/2.0, EPS0_DNA, 134.11876, 0))
        elif base in ('G', 'DG'):
            outfile.write("%s,%lf,%lf,%lf,%lf\n" % ('G', D0_DNA/2.0, EPS0_DNA, 150.11816, 0))
        elif base in ('C', 'DC'):
            outfile.write("%s,%lf,%lf,%lf,%lf\n" % ('C', D0_DNA/2.0, EPS0_DNA, 110.09406, 0))
        elif base in ('T', 'DT'):
            outfile.write("%s,%lf,%lf,%lf,%lf\n" %
                         ('T', D0_DNA/2.0, EPS0_DNA, 125.10542, 0))
        else:
            print("UNKNOWN BASE:", repr(base), "BEAD:", bead)

outfile.close()
print("Wrote 3_sigma_eps_mass_charge_dna.inp with %d beads" % len(atom_name))

outfile.close()
print("Wrote 3_sigma_eps_mass_charge_dna.inp with %d beads" % len(atom_name))