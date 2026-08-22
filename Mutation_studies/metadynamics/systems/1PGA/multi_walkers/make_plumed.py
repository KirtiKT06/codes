"""
make_plumed.py  –  generates plumed_template.dat for multi-walker metaD.

Usage:
    python make_plumed.py --walkers 4

Each walker's run_meta.py reads plumed_template.dat, replaces {WALKER_ID}
with its own ID at runtime, and writes a local plumed.dat before launching.

HILLS files land in the shared base directory:
    base/HILLS.0, base/HILLS.1, ...  (WALKERS_DIR=../ from each walker subdir)
COLVAR, traj, logs stay inside each walker's own subdirectory.
"""

import argparse
from openmm.app import PDBFile

# ==========================================
# RESIDUE NAMES TO EXCLUDE FROM Rg
# ==========================================

NON_PROTEIN = {"HOH", "NA", "CL", "K", "MG", "CA", "ZN", "NA+", "CL-", "SOD", "CLA"}

# ==========================================
# CLI
# ==========================================

parser = argparse.ArgumentParser()
parser.add_argument("--walkers", type=int, default=4,
                    help="Total number of walkers (default: 4)")
args = parser.parse_args()

N_WALKERS = args.walkers

# ==========================================
# LOAD CONTACTS
# ==========================================

contacts = []

with open("ca_contacts_with_dist.dat") as f:
    for line in f:
        i, j, r0 = line.split()
        i  = int(i)   + 1      # PLUMED is 1-based
        j  = int(j)   + 1
        r0 = float(r0)
        contacts.append((i, j, r0))

# ==========================================
# FIND PROTEIN ATOMS ONLY
# ==========================================

pdb = PDBFile("wt_solvated.pdb")

protein_atoms = []
for atom in pdb.topology.atoms():
    if atom.residue.name not in NON_PROTEIN:
        protein_atoms.append(atom.index + 1)

protein_string = ",".join(map(str, protein_atoms))
print(f"Protein atoms for Rg: {len(protein_atoms)}")

# ==========================================
# WRITE TEMPLATE  (placeholder = {WALKER_ID})
# ==========================================

with open("plumed_template.dat", "w") as out:

    out.write("MOLINFO STRUCTURE=wt_solvated.pdb\n\n")

    # ------------------------------------------
    # Individual contacts (COORDINATION)
    # ------------------------------------------

    contact_labels = []

    for n, (i, j, r0) in enumerate(contacts, start=1):
        label = f"c{n}"
        contact_labels.append(label)
        out.write(
            f"{label}: COORDINATION "
            f"GROUPA={i} GROUPB={j} "
            f"SWITCH={{RATIONAL R_0={1.2 * r0:.3f}}}\n"
        )

    out.write("\n")

    # ------------------------------------------
    # Fraction native contacts  (q)
    # ------------------------------------------

    out.write(
        "qsum: COMBINE ARG="
        + ",".join(contact_labels)
        + " PERIODIC=NO\n"
    )
    out.write(
        f"q: MATHEVAL ARG=qsum FUNC=x/{len(contacts)} PERIODIC=NO\n\n"
    )

    # ------------------------------------------
    # Radius of gyration  (protein only)
    # ------------------------------------------

    out.write(
        f"rg: GYRATION TYPE=RADIUS ATOMS={protein_string}\n\n"
    )

    # ------------------------------------------
    # Well-tempered metaD  –  multi-walker block
    #
    # {WALKER_ID} is filled in at runtime by run_meta.py.
    # FILE=../HILLS  →  each walker writes  ../HILLS.<id>
    # WALKERS_DIR=../  →  each walker reads  ../HILLS.*
    # ------------------------------------------

    out.write(
        "METAD LABEL=metad "
        "ARG=q,rg "
        "PACE=500 HEIGHT=1.2 "
        "SIGMA=0.02,0.05 "
        "BIASFACTOR=10 TEMP=300 "
        "FILE=../HILLS "                    # lands in base dir as HILLS.<id>
        f"WALKERS_N={N_WALKERS} "
        "WALKERS_ID={WALKER_ID} "          # filled at runtime
        "WALKERS_DIR=../ "                 # shared HILLS directory
        "WALKERS_RSTRIDE=500\n\n"          # re-read other walkers every 500 steps
    )

    # ------------------------------------------
    # Output  (local to each walker subdir)
    # ------------------------------------------

    out.write(
        "PRINT STRIDE=500 "
        "ARG=q,rg,metad.bias "
        "FILE=COLVAR\n"
    )

print(f"Wrote {len(contacts)} contacts")
print(f"Created plumed_template.dat  (N_WALKERS={N_WALKERS})")
print("Run:  python run_meta.py --walker 0  (and --walker 1, 2, 3 …)")