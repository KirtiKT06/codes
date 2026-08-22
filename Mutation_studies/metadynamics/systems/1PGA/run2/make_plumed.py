from openmm.app import PDBFile

# ==========================================
# RESIDUE NAMES TO EXCLUDE FROM Rg
# HOH = water; the rest are common ions
# ==========================================

NON_PROTEIN = {"HOH", "NA", "CL", "K", "MG", "CA", "ZN", "NA+", "CL-", "SOD", "CLA"}

# ==========================================
# LOAD CONTACTS
# ==========================================

contacts = []

with open("ca_contacts_with_dist.dat") as f:
    for line in f:

        i, j, r0 = line.split()

        # PLUMED uses 1-based indexing
        i = int(i) + 1
        j = int(j) + 1
        r0 = float(r0)

        contacts.append((i, j, r0))

# ==========================================
# FIND PROTEIN ATOMS ONLY
# ==========================================

pdb = PDBFile("wt_solvated.pdb")

protein_atoms = []

for atom in pdb.topology.atoms():
    if atom.residue.name not in NON_PROTEIN:   # FIXED: was only checking != "HOH"
        protein_atoms.append(atom.index + 1)

protein_string = ",".join(map(str, protein_atoms))

print(f"Protein atoms for Rg: {len(protein_atoms)}")

# ==========================================
# WRITE PLUMED INPUT
# ==========================================

with open("plumed.dat", "w") as out:

    out.write("MOLINFO STRUCTURE=wt_solvated.pdb\n\n")

    # --------------------------------------
    # Individual contacts
    # FIXED: use COORDINATION instead of CONTACT
    # --------------------------------------

    contact_labels = []

    for n, (i, j, r0) in enumerate(contacts, start=1):

        label = f"c{n}"
        contact_labels.append(label)

        # COORDINATION with single-atom groups = pairwise switched contact
        out.write(
            f"{label}: COORDINATION "
            f"GROUPA={i} GROUPB={j} "
            f"SWITCH={{RATIONAL R_0={1.2*r0:.3f}}}\n"
        )

    out.write("\n")

    # --------------------------------------
    # Fraction native contacts
    # --------------------------------------

    out.write(
        "qsum: COMBINE ARG="
        + ",".join(contact_labels)
        + " PERIODIC=NO\n"
    )

    out.write(
        f"q: MATHEVAL ARG=qsum FUNC=x/{len(contacts)} PERIODIC=NO\n\n"
    )

    # --------------------------------------
    # Radius of gyration (protein only)
    # --------------------------------------

    out.write(
        f"rg: GYRATION TYPE=RADIUS ATOMS={protein_string}\n\n"
    )

    # --------------------------------------
    # Well-tempered metadynamics
    # --------------------------------------

    out.write(
        "METAD LABEL=metad ARG=q,rg PACE=500 HEIGHT=1.2 "
        "SIGMA=0.02,0.05 BIASFACTOR=10 TEMP=300 FILE=HILLS\n\n"
    )

    # --------------------------------------
    # Output
    # --------------------------------------

    out.write(
        "PRINT STRIDE=500 "
        "ARG=q,rg,metad.bias "
        "FILE=COLVAR\n"
    )

print(f"Wrote {len(contacts)} contacts")
print("Created plumed.dat")