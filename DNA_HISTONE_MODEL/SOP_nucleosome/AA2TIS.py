"""
AA2TIS_dna.py
=============
Converts an all-atom DNA structure (PDB) into the Three-Interaction-Site
(TIS) coarse-grained representation used by Chakraborty, Hori &
Thirumalai (JCTC 2018): each nucleotide -> 3 beads (Phosphate P, Sugar
S, Base B), beads placed at the CENTER OF MASS of the corresponding
heavy-atom group (Figure 1 of the paper).

This mirrors the style of your existing AA2CG.py for the histones:
same input.json-driven config, same fixed-width PDB writer.

ATOM GROUPING (heavy atoms only; matches the standard TIS/CG mapping
used in the Hyeon-Thirumalai / Denesyuk-Thirumalai family of models,
which the DT-DNA paper explicitly says it inherits its functional form
from):

    Phosphate (P bead):  P, OP1, OP2, O5'
    Sugar     (S bead):  C1', C2', C3', C4', C5', O4', O3'
    Base      (B bead):  everything else attached to the nucleotide
                          (the purine/pyrimidine ring + exocyclic atoms:
                          N1,C2,N3,C4,C5,C6,N6/O6/O4,N7,C8,N9,N2,O2,C7...)

The 5'-terminal nucleotide of a chain has no phosphate group in most
crystal/fiber-diffraction structures; in that case no P bead is
written for residue 1 of the chain (consistent with how real DNA/RNA
in the PDB is deposited, and with how the TIS paper counts "P,S,B per
nucleotide" for internal residues only -- the model simply starts the
chain at the first sugar).

Output:
    - a CG PDB with one ATOM record per bead (name "P", "S", or "B",
      resname = one-letter base packed into a 3-letter DNA code DA/DT/
      DG/DC so it round-trips through the same fixed-width parsers used
      elsewhere in this project)
    - a plain-text sequence file per chain (5'->3'), used later by
      build_dna_topology.py to look up stacking parameters etc.
"""

import json
from collections import defaultdict

# ── Atomic masses (heavy atoms only, amu) ──────────────────────────────
MASS = {"C": 12.011, "N": 14.007, "O": 15.999, "P": 30.974}


def element_of(atom_name):
    """Guess element from a PDB atom name (heavy atoms only)."""
    name = atom_name.strip()
    if name.startswith("P"):
        return "P"
    # strip leading digits (e.g. "5'" numbering doesn't have digits first
    # for atom *names*, but some structures prefix with an atom serial
    # convention like "1H5'"; we only handle heavy atoms here anyway)
    for ch in name:
        if ch.isalpha():
            return ch
    return "C"


PHOSPHATE_ATOMS = {"P", "OP1", "OP2", "O1P", "O2P", "O5'"}
SUGAR_ATOMS = {"C1'", "C2'", "C3'", "C4'", "C5'", "O4'", "O3'", "O2'"}
# Everything else observed on a nucleotide residue is treated as base.

BASE_3LETTER = {"A": "DA", "T": "DT", "G": "DG", "C": "DC"}
RESNAME_TO_LETTER = {
    "DA": "A", "DT": "T", "DG": "G", "DC": "C",
    "A": "A", "T": "T", "G": "G", "C": "C",
    "ADE": "A", "THY": "T", "GUA": "G", "CYT": "C",
}


def parse_aa_dna(aa_pdb_file, dna_chains):
    """
    Parse ATOM records for the requested DNA chains.
    Returns: dict chain -> list of residues, each residue a dict with
             resid, resname(one-letter base), and atom-> (x,y,z) coords.
    """
    residues = defaultdict(dict)  # chain -> resid -> {"resname":.., "atoms": {}}
    with open(aa_pdb_file) as f:
        for line in f:
            record = line[:6].strip()
            if record not in ("ATOM", "HETATM"):
                continue
            chain = line[21]
            if chain not in dna_chains:
                continue
            resname = line[17:20].strip()
            if resname not in RESNAME_TO_LETTER:
                continue
            resid = int(line[22:26])
            atom_name = line[12:16].strip()
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])

            if resid not in residues[chain]:
                residues[chain][resid] = {
                    "resname": RESNAME_TO_LETTER[resname],
                    "atoms": {}
                }
            residues[chain][resid]["atoms"][atom_name] = (x, y, z)
    return residues


def center_of_mass(atom_coords):
    """atom_coords: dict atom_name -> (x,y,z). Returns mass-weighted COM."""
    sx = sy = sz = 0.0
    mtot = 0.0
    for name, (x, y, z) in atom_coords.items():
        m = MASS.get(element_of(name), 12.0)
        sx += m * x; sy += m * y; sz += m * z
        mtot += m
    if mtot == 0.0:
        raise ValueError("Empty atom group; cannot compute center of mass.")
    return (sx / mtot, sy / mtot, sz / mtot)


def build_tis_beads(residues, chain_order):
    """
    residues: output of parse_aa_dna.
    Returns: list of bead dicts (in final PDB order) and a dict
             chain -> sequence string (5'->3').
    """
    all_beads = []
    sequences = {}

    for ch in chain_order:
        if ch not in residues:
            continue
        resids_sorted = sorted(residues[ch].keys())
        seq_letters = []
        for k, resid in enumerate(resids_sorted):
            res = residues[ch][resid]
            base_letter = res["resname"]
            seq_letters.append(base_letter)
            atoms = res["atoms"]

            phosphate_atoms = {n: c for n, c in atoms.items() if n in PHOSPHATE_ATOMS}
            sugar_atoms = {n: c for n, c in atoms.items() if n in SUGAR_ATOMS}
            base_atoms = {n: c for n, c in atoms.items()
                          if n not in PHOSPHATE_ATOMS and n not in SUGAR_ATOMS}

            # 5'-terminal residue of the chain typically has no P/OP1/OP2
            # (no phosphate at the very 5' end) -- skip P bead if absent.
            if len(phosphate_atoms) >= 2:  # need at least P + one O to be meaningful
                p_com = center_of_mass(phosphate_atoms)
                all_beads.append({
                    "chain": ch, "resid": resid, "bead": "P",
                    "resname": BASE_3LETTER[base_letter],
                    "base": base_letter,
                    "x": p_com[0], "y": p_com[1], "z": p_com[2],
                })

            if len(sugar_atoms) < 3:
                raise ValueError(
                    f"Chain {ch} resid {resid}: too few sugar atoms found "
                    f"({list(sugar_atoms.keys())}); check atom naming."
                )
            s_com = center_of_mass(sugar_atoms)
            all_beads.append({
                "chain": ch, "resid": resid, "bead": "S",
                "resname": BASE_3LETTER[base_letter],
                "base": base_letter,
                "x": s_com[0], "y": s_com[1], "z": s_com[2],
            })

            if len(base_atoms) < 3:
                raise ValueError(
                    f"Chain {ch} resid {resid}: too few base atoms found "
                    f"({list(base_atoms.keys())}); check atom naming."
                )
            b_com = center_of_mass(base_atoms)
            all_beads.append({
                "chain": ch, "resid": resid, "bead": "B",
                "resname": BASE_3LETTER[base_letter],
                "base": base_letter,
                "x": b_com[0], "y": b_com[1], "z": b_com[2],
            })

        sequences[ch] = "".join(seq_letters)

    return all_beads, sequences


def write_tis_pdb(beads, outfile):
    with open(outfile, "w") as f:
        for i, b in enumerate(beads, start=1):
            f.write(
                "{:<6s}{:5d} {:^4s} {:>3s} {:1s}{:4d}    "
                "{:8.3f}{:8.3f}{:8.3f}  1.00  0.00\n".format(
                    "ATOM", i, b["bead"], b["resname"], b["chain"],
                    b["resid"], b["x"], b["y"], b["z"]
                )
            )
        f.write("END\n")


def write_sequences(sequences, outfile):
    with open(outfile, "w") as f:
        for ch, seq in sequences.items():
            f.write(f"> chain {ch} (5'->3')\n{seq}\n")


def main():
    config = json.load(open("input.json"))
    aa_pdb = config["dna_pdb"]
    dna_chains = config["dna_chains"]
    cg_pdb_dna_out = config.get("cg_pdb_dna", "dna_tis.pdb")
    dna_seq_out = config.get("dna_sequence_file", "dna_sequences.txt")

    print(f"Reading all-atom DNA from {aa_pdb}, chains {dna_chains} ...")
    residues = parse_aa_dna(aa_pdb, dna_chains)
    for ch in dna_chains:
        n = len(residues.get(ch, {}))
        print(f"  Chain {ch}: {n} nucleotides found")

    beads, sequences = build_tis_beads(residues, dna_chains)
    write_tis_pdb(beads, cg_pdb_dna_out)
    write_sequences(sequences, dna_seq_out)

    n_p = sum(1 for b in beads if b["bead"] == "P")
    n_s = sum(1 for b in beads if b["bead"] == "S")
    n_b = sum(1 for b in beads if b["bead"] == "B")
    print(f"\nSaved: {cg_pdb_dna_out}")
    print(f"  {n_p} P beads, {n_s} S beads, {n_b} B beads "
          f"({n_p + n_s + n_b} total TIS beads)")
    print(f"Saved sequences: {dna_seq_out}")
    for ch, seq in sequences.items():
        print(f"  Chain {ch} (5'->3', {len(seq)} nt): {seq}")


if __name__ == "__main__":
    main()