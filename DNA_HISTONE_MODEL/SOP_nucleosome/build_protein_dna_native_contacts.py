"""
build_protein_dna_contacts.py
==============================
Structure-based (Go-like) protein-DNA NATIVE contacts, built the same
way build_contacts.py builds protein-protein native contacts, but now
crossing the protein CG beads against the DNA TIS beads.

Parameters are taken from Table S1 of Reddy & Thirumalai (SOP model for
the nucleosome, supplement to the NAR 2021 paper you attached):

    HPC-DNA:  R_c = 11 A   (native-contact distance cutoff)
              eps_h = 1.18 kcal/mol  (native contact well depth)
              sigma  = 5.4 A          (used for the NON-native / generic
                                       excluded-volume repulsion between
                                       protein and DNA beads, applied in
                                       the main OpenMM script rather than
                                       here)

IMPORTANT: this only makes physical sense if the protein CG PDB (with
tails) and the DNA TIS PDB were both built from the SAME original
all-atom structure (e.g. 2CV5), in the same coordinate frame, since
"native" here means "in contact in the crystal/cryo-EM structure".  If
you built the DNA TIS structure from a different / idealized helix
(e.g. via fd_helix.py, to get proper TIS-DNA equilibrium geometry),
skip this script -- there is no meaningful native protein-DNA structure
to encode, and the system should rely purely on the generic
excluded-volume + electrostatic protein-DNA coupling instead (which is
built directly in sop_protein_dna_openmm.py).
"""

import json
import numpy as np


def load_beads(path, tag):
    beads = []
    with open(path) as f:
        for line in f:
            if line[:6].strip() != "ATOM":
                continue
            beads.append({
                "chain": line[21],
                "resid": int(line[22:26]),
                "name": line[12:16].strip(),
                "resname": line[17:20].strip(),
                "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
                "kind": tag,
            })
    return beads


def main():
    config = json.load(open("input.json"))
    protein_pdb = config["cg_pdb_with_tails"]
    dna_pdb = config.get("cg_pdb_dna", "dna_tis.pdb")
    Rc = config.get("protein_dna_native_cutoff", 11.0)     # Angstrom, Table S1
    eps_h = config.get("protein_dna_eps_h", 1.18)          # kcal/mol, Table S1

    protein_beads = load_beads(protein_pdb, "protein")
    dna_beads = load_beads(dna_pdb, "dna")

    print(f"Protein beads: {len(protein_beads)}   DNA beads: {len(dna_beads)}")
    print(f"Using Rc = {Rc} A, eps_h = {eps_h} kcal/mol (Reddy & Thirumalai, Table S1)")

    p_coords = np.array([[b["x"], b["y"], b["z"]] for b in protein_beads])
    d_coords = np.array([[b["x"], b["y"], b["z"]] for b in dna_beads])

    contacts = []
    for i in range(len(protein_beads)):
        diffs = d_coords - p_coords[i]
        dists = np.linalg.norm(diffs, axis=1)
        hits = np.where(dists < Rc)[0]
        for j in hits:
            contacts.append((i, j, float(dists[j])))

    print(f"Found {len(contacts)} protein-DNA native contact pairs.")

    with open("protein_dna_native_contacts.dat", "w") as f:
        f.write(f"# protein_bead_idx  dna_bead_idx  r0(A)  eps_h(kcal/mol)={eps_h}\n")
        for i, j, r0 in contacts:
            f.write(f"{i} {j} {r0:.4f}\n")

    print("Saved: protein_dna_native_contacts.dat")


if __name__ == "__main__":
    main()