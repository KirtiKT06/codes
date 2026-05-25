from __future__ import print_function

import json
import numpy as np


config = json.load(open("input.json"))

aa_pdb_file = config["aa_pdb_prefix"] + ".pdb"
cg_pdb_file = config["cg_pdb_prefix"] + ".pdb"

sequence_input = config["sequence"].replace(" ", "").replace("\n", "").upper()
fallback_sc_offset_A = float(config.get("fallback_sc_offset", 1.0))
histidine_charge = float(config.get("histidine_charge", 0.0))


THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "CYX": "C", "GLN": "Q", "GLU": "E",
    "GLY": "G", "HIS": "H", "HID": "H", "HIE": "H", "HIP": "H",
    "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M",
    "PHE": "F", "PRO": "P", "SER": "S", "THR": "T",
    "TRP": "W", "TYR": "Y", "VAL": "V"
}


AA_PARAMS = {
    "BB": {"radius_A": 1.90, "charge": 0.0},

    # Gly is one bead only, located at CA, using Gly radius 2.25 A.
    # Ala has explicit side-chain bead.
    "G": {"radius_A": 2.25, "charge": 0.0},
    "A": {"radius_A": 2.52, "charge": 0.0},

    "R": {"radius_A": 3.28, "charge": 1.0},
    "K": {"radius_A": 3.18, "charge": 1.0},
    "H": {"radius_A": 3.04, "charge": histidine_charge},
    "D": {"radius_A": 2.79, "charge": -1.0},
    "E": {"radius_A": 2.96, "charge": -1.0},
    "S": {"radius_A": 2.59, "charge": 0.0},
    "T": {"radius_A": 2.81, "charge": 0.0},
    "N": {"radius_A": 2.84, "charge": 0.0},
    "Q": {"radius_A": 3.01, "charge": 0.0},
    "C": {"radius_A": 2.74, "charge": 0.0},
    "P": {"radius_A": 2.78, "charge": 0.0},
    "I": {"radius_A": 3.09, "charge": 0.0},
    "L": {"radius_A": 3.09, "charge": 0.0},
    "M": {"radius_A": 3.09, "charge": 0.0},
    "F": {"radius_A": 3.18, "charge": 0.0},
    "W": {"radius_A": 3.39, "charge": 0.0},
    "Y": {"radius_A": 3.23, "charge": 0.0},
    "V": {"radius_A": 2.93, "charge": 0.0}
}


BACKBONE_ATOMS = set([
    "N", "CA", "C", "O", "OXT",
    "H", "HN", "H1", "H2", "H3",
    "HA", "HA2", "HA3"
])


ELEMENT_MASS = {
    "H": 1.008,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06
}


def infer_element(atom_name, element_field):
    element_field = element_field.strip()

    if element_field:
        return element_field[0].upper()

    name = atom_name.strip().upper()

    if name.startswith("H"):
        return "H"
    if name.startswith("C"):
        return "C"
    if name.startswith("N"):
        return "N"
    if name.startswith("O"):
        return "O"
    if name.startswith("S"):
        return "S"

    return "C"


def atom_mass(atom_name, element_field):
    e = infer_element(atom_name, element_field)
    return ELEMENT_MASS.get(e, 12.011)


def parse_pdb_atoms(pdb_file):
    """
    Manual parser so non-contiguous Molefacture residue atom order does not break parsing.
    Groups atoms by chain, resid, insertion code, residue name.
    """
    residues = {}

    with open(pdb_file) as f:
        for line in f:
            rec = line[:6].strip()

            if rec not in ["ATOM", "HETATM"]:
                continue

            atom_name = line[12:16].strip()
            resname = line[17:20].strip().upper()
            chain = line[21].strip()
            resid_str = line[22:26].strip()
            icode = line[26].strip()

            if resname not in THREE_TO_ONE:
                continue

            try:
                resid = int(resid_str)
            except ValueError:
                continue

            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])

            element_field = line[76:78] if len(line) >= 78 else ""

            key = (chain, resid, icode, resname)

            residues.setdefault(key, [])
            residues[key].append({
                "atom_name": atom_name,
                "resname": resname,
                "chain": chain,
                "resid": resid,
                "icode": icode,
                "xyz": np.array([x, y, z], dtype=float),
                "mass": atom_mass(atom_name, element_field),
                "element": infer_element(atom_name, element_field)
            })

    keys = sorted(residues.keys(), key=lambda x: (x[0], x[1], x[2], x[3]))
    return keys, residues


def weighted_com(xyz_list, mass_list):
    xyz = np.asarray(xyz_list, dtype=float)
    mass = np.asarray(mass_list, dtype=float)
    return np.sum(xyz * mass[:, None], axis=0) / np.sum(mass)


def fallback_sc_position(ca_pos, index):
    angle = index * 2.399963229728653
    direction = np.array([np.cos(angle), np.sin(angle), 0.3])
    direction /= np.linalg.norm(direction)
    return ca_pos + fallback_sc_offset_A * direction


def expected_bead_count(sequence):
    """
    Gly has one bead.
    All non-Gly residues have BB + SC.
    """
    return 2 * len(sequence) - sequence.count("G")


def write_cg_pdb(cg_records, out_pdb):
    with open(out_pdb, "w") as f:
        serial = 1

        for rec in cg_records:
            atom_name = rec["atom_name"]
            resname = rec["resname"]
            chain = rec["chain"]
            resid = rec["resid"]
            x, y, z = rec["xyz"]

            f.write(
                "{:<6s}{:5d} {:^4s} {:>3s} {:1s}{:4d}    "
                "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}\n".format(
                    "ATOM",
                    serial,
                    atom_name,
                    resname,
                    chain,
                    resid,
                    x, y, z,
                    1.00,
                    0.00,
                    "C"
                )
            )

            serial += 1

        f.write("TER\nEND\n")


def main():
    keys, residues = parse_pdb_atoms(aa_pdb_file)

    sequence = ""
    cg_records = []

    n_gly = 0
    n_sc = 0
    n_fallback_sc = 0

    for idx, key in enumerate(keys):
        chain, resid, icode, resname = key
        aa = THREE_TO_ONE[resname]
        sequence += aa

        atoms = residues[key]

        ca_pos = None
        side_heavy_xyz = []
        side_heavy_mass = []
        side_any_xyz = []
        side_any_mass = []

        for atom in atoms:
            atom_name = atom["atom_name"].strip().upper()

            if atom_name == "CA":
                ca_pos = atom["xyz"]

            is_h = atom["element"].upper() == "H"

            if atom_name not in BACKBONE_ATOMS:
                side_any_xyz.append(atom["xyz"])
                side_any_mass.append(atom["mass"])

                if not is_h:
                    side_heavy_xyz.append(atom["xyz"])
                    side_heavy_mass.append(atom["mass"])

        if ca_pos is None:
            print("Missing CA in residue:", resname, chain, resid)
            print("Atoms found:", [a["atom_name"] for a in atoms])
            raise RuntimeError("Residue has no CA atom.")

        # Always write BB bead at CA position.
        # For Gly, this is the only bead and uses Gly radius in runSOP.
        cg_records.append({
            "atom_name": "BB",
            "resname": aa,
            "chain": "A",
            "resid": idx + 1,
            "xyz": ca_pos
        })

        # Glycine has no explicit SC bead.
        if aa == "G":
            n_gly += 1
            continue

        # All non-Gly residues, including alanine, get an SC bead.
        if len(side_heavy_xyz) > 0:
            sc_pos = weighted_com(side_heavy_xyz, side_heavy_mass)
        elif len(side_any_xyz) > 0:
            sc_pos = weighted_com(side_any_xyz, side_any_mass)
        else:
            sc_pos = fallback_sc_position(ca_pos, idx)
            n_fallback_sc += 1

        cg_records.append({
            "atom_name": "SC",
            "resname": aa,
            "chain": "A",
            "resid": idx + 1,
            "xyz": sc_pos
        })

        n_sc += 1

    if sequence != sequence_input:
        print("PDB sequence length:", len(sequence))
        print("Input sequence length:", len(sequence_input))
        print("PDB sequence:", sequence)
        print("Input sequence:", sequence_input)
        raise RuntimeError("Sequence mismatch between all-atom PDB and input.json.")

    expected_beads = expected_bead_count(sequence)

    if len(cg_records) != expected_beads:
        print("Expected CG beads:", expected_beads)
        print("Actual CG beads:", len(cg_records))
        raise RuntimeError("CG bead count mismatch.")

    write_cg_pdb(cg_records, cg_pdb_file)

    q = 0.0
    for aa in sequence:
        q += AA_PARAMS[aa]["charge"]

    print("Wrote protein-only CG PDB:", cg_pdb_file)
    print("Sequence length:", len(sequence))
    print("Gly residues:", n_gly)
    print("SC beads written:", n_sc)
    print("Fallback SC beads used:", n_fallback_sc)
    print("Expected protein beads:", expected_beads)
    print("Protein beads:", len(cg_records))
    print("Protein charge:", q)


if __name__ == "__main__":
    main()