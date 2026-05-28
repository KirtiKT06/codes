from __future__ import print_function

import json
import random
from tkinter.font import names
import numpy as np


config = json.load(open("input.json"))

random_seed = int(config.get("random_seed", 100))
random.seed(random_seed)
np.random.seed(random_seed)

cg_pdb_file = config["cg_pdb_prefix"] + ".pdb"
out_pdb_file = config["pdb_prefix"] + ".pdb"

sequence = config["sequence"].replace(" ", "").replace("\n", "").upper()

box_length_A = float(config["box_length"])
ion_min_dist_A = float(config.get("ion_min_dist", 4.5))

auto_ions = bool(config.get("auto_ions", True))
neutralize = bool(config.get("neutralize", True))

KCl_mM = float(config.get("KCl_mM", 0.0))
MgCl2_mM = float(config.get("MgCl2_mM", 0.0))


AA_CHARGE = {
    "R": 1.0,
    "K": 1.0,
    "D": -1.0,
    "E": -1.0
}


def protein_charge(sequence):
    return sum(AA_CHARGE.get(aa, 0.0) for aa in sequence)

def ion_formula_units_from_concentration(salt_mM, box_length_A):
    NA = 6.02214076e23
    concentration_M = salt_mM / 1000.0
    box_nm = box_length_A * 0.1
    volume_L = (box_nm ** 3) * 1e-24
    return int(round(concentration_M * NA * volume_L))


def build_ion_names():
    """
    Build ions in the requested order:

        Mg first
        K second
        Cl last

    Logic:
      1. Add Mg according to MgCl2_mM.
      2. Add K according to KCl_mM.
      3. If protein is net negative, add extra K to neutralize protein.
         If protein is net positive, do not add K; Cl will neutralize it.
      4. Add enough Cl to make the total system charge zero.

    This means Cl count is determined by charge neutrality, not by separately
    adding n_KCl + 2*n_MgCl2 directly.
    """
    ions = []

    q_prot = protein_charge(sequence)

    qK = float(config["qK"])
    qMg = float(config["qMg"])
    qCl = float(config["qCl"])

    if qCl >= 0:
        raise RuntimeError("qCl must be negative for chloride.")

    if auto_ions:
        n_KCl = ion_formula_units_from_concentration(KCl_mM, box_length_A)
        n_MgCl2 = ion_formula_units_from_concentration(MgCl2_mM, box_length_A)

        # ------------------------------------------------------------
        # 1. Mg first: concentration of MgCl2 formula units
        # ------------------------------------------------------------
        nMg = n_MgCl2

        # ------------------------------------------------------------
        # 2. K second: KCl concentration + optional neutralization
        # ------------------------------------------------------------
        nK = n_KCl

        nK_neutral = 0
        if neutralize:
            if q_prot < 0:
            # For net-negative protein, neutralize with K+
                nK_neutral = int(round(abs(q_prot) / qK))
                nK += nK_neutral

        # ------------------------------------------------------------
        # 3. Cl last: choose Cl count to neutralize the whole system
        # ------------------------------------------------------------
        total_positive_charge = qK * nK + qMg * nMg
        total_charge_before_cl = q_prot + total_positive_charge

        if total_charge_before_cl < 0:
            raise RuntimeError(
                "System is still net negative before adding Cl. "
                "This means K/Mg are insufficient or charges are inconsistent."
            )

        nCl = int(round(total_charge_before_cl / abs(qCl)))

        # Optional sanity check
        final_charge = q_prot + qK * nK + qMg * nMg + qCl * nCl

        if abs(final_charge) > 1e-6:
            raise RuntimeError(
                "Could not make system neutral. Final charge = %.6f" % final_charge
            )

        # ------------------------------------------------------------
        # Final order: Mg, K, Cl
        # ------------------------------------------------------------
        ions += ["Mg"] * nMg
        ions += ["K"] * nK
        ions += ["Cl"] * nCl

        print("Auto ion mode: Mg first, then K, then Cl")
        print("KCl_mM:", KCl_mM, "formula units:", n_KCl)
        print("MgCl2_mM:", MgCl2_mM, "formula units:", n_MgCl2)
        print("Protein charge:", q_prot)
        print("Neutralizing K added:", nK_neutral)
        print("Mg ions:", nMg)
        print("K ions:", nK)
        print("Cl ions:", nCl)
        print("Final charge:", final_charge)

    else:
        # Manual mode, same requested order: Mg, K, Cl
        nMg = int(config.get("nMg", 0))
        nK = int(config.get("nK", 0))
        nCl = int(config.get("nCl", 0))

        ions += ["Mg"] * nMg
        ions += ["K"] * nK
        ions += ["Cl"] * nCl

        final_charge = q_prot + qMg * nMg + qK * nK + qCl * nCl

        print("Manual ion mode: Mg first, then K, then Cl")
        print("Protein charge:", q_prot)
        print("Mg ions:", nMg)
        print("K ions:", nK)
        print("Cl ions:", nCl)
        print("Final charge:", final_charge)

        if abs(final_charge) > 1e-6:
            print("WARNING: manual ion counts do not make the system neutral.")

    return ions


def ion_charge(name):
    if name == "K":
        return float(config["qK"])
    if name == "Mg":
        return float(config["qMg"])
    if name == "Cl":
        return float(config["qCl"])
    return 0.0


def read_cg_pdb(pdb_file):
    names = []
    resnames = []
    chains = []
    resids = []
    xyz = []

    with open(pdb_file) as f:
        for line in f:
            rec = line[:6].strip()
            if rec not in ["ATOM", "HETATM"]:
                continue

            names.append(line[12:16].strip())
            resnames.append(line[17:20].strip())
            chains.append(line[21].strip())
            resids.append(int(line[22:26]))
            xyz.append([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54])
            ])

    return names, resnames, chains, resids, np.asarray(xyz, dtype=float)


def minimum_image_distance(a, b):
    delta = a - b
    delta -= box_length_A * np.round(delta / box_length_A)
    return np.linalg.norm(delta)


def generate_ion_positions(ion_names, protein_xyz):
    ion_xyz = []
    max_attempts = 3000000
    attempts = 0

    while len(ion_xyz) < len(ion_names):
        attempts += 1

        if attempts > max_attempts:
            raise RuntimeError("Could not place all ions. Increase box or reduce ion_min_dist.")

        pos = box_length_A * (np.random.rand(3) - 0.5)

        ok = True

        for p in protein_xyz:
            if minimum_image_distance(pos, p) < ion_min_dist_A:
                ok = False
                break

        if not ok:
            continue

        for p in ion_xyz:
            if minimum_image_distance(pos, p) < ion_min_dist_A:
                ok = False
                break

        if ok:
            ion_xyz.append(pos)

    return np.asarray(ion_xyz, dtype=float)


def write_pdb(names, resnames, chains, resids, xyz, ion_names, ion_xyz, out_pdb):
    with open(out_pdb, "w") as f:
        f.write(
            "CRYST1{:9.3f}{:9.3f}{:9.3f}{:7.2f}{:7.2f}{:7.2f} P 1           1\n".format(
                box_length_A, box_length_A, box_length_A, 90.0, 90.0, 90.0
            )
        )

        serial = 1

        # shifted_xyz = xyz + box_length_A / 2.0
        # Fix (center COM at box center):
        com = xyz.mean(axis=0)
        shifted_xyz = xyz - com + np.array([box_length_A/2.0, 
                                            box_length_A/2.0, 
                                            box_length_A/2.0])

        for i in range(len(names)):
            f.write(
                "{:<6s}{:5d} {:^4s} {:>3s} {:1s}{:4d}    "
                "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}\n".format(
                    "ATOM",
                    serial,
                    names[i],
                    resnames[i],
                    "A",
                    resids[i],
                    shifted_xyz[i, 0],
                    shifted_xyz[i, 1],
                    shifted_xyz[i, 2],
                    1.00,
                    0.00,
                    "C"
                )
            )
            serial += 1

        f.write("TER\n")

        shifted_ion_xyz = ion_xyz + box_length_A / 2.0

        for i, ion in enumerate(ion_names):
            elem = ion
            f.write(
                "{:<6s}{:5d} {:^4s} {:>3s} {:1s}{:4d}    "
                "{:8.3f}{:8.3f}{:8.3f}{:6.2f}{:6.2f}          {:>2s}\n".format(
                    "HETATM",
                    serial,
                    ion,
                    ion,
                    "I",
                    i + 1,
                    shifted_ion_xyz[i, 0],
                    shifted_ion_xyz[i, 1],
                    shifted_ion_xyz[i, 2],
                    1.00,
                    0.00,
                    elem
                )
            )
            serial += 1

        f.write("END\n")


def expected_bead_count(sequence):
    """
      Gly = one bead
      all non-Gly residues = BB + SC
    """
    return 2 * len(sequence) - sequence.count("G")


def main():
    names, resnames, chains, resids, xyz = read_cg_pdb(cg_pdb_file)

    expected_beads = expected_bead_count(sequence)

    if len(names) != expected_beads:
        raise RuntimeError(
            "Protein bead count mismatch: CG PDB has %d beads but sequence expects %d. "
            "For 2025-style mapping, expected beads = 2*N - N_Gly."
            % (len(names), expected_beads)
        )

    # center protein around zero before ion placement
    xyz_centered = xyz - xyz.mean(axis=0)

    ions = build_ion_names()
    ion_xyz = generate_ion_positions(ions, xyz_centered)

    write_pdb(names, resnames, chains, resids, xyz_centered, ions, ion_xyz, out_pdb_file)

    nK = ions.count("K")
    nMg = ions.count("Mg")
    nCl = ions.count("Cl")

    q_prot = protein_charge(sequence)
    q_ion = sum(ion_charge(x) for x in ions)

    print("Wrote boxed ionized PDB:", out_pdb_file)
    print("Sequence length:", len(sequence))
    print("Gly residues:", sequence.count("G"))
    print("Expected protein beads:", expected_beads)
    print("Protein beads:", len(names))
    print("Protein charge:", q_prot)
    print("nK:", nK)
    print("nMg:", nMg)
    print("nCl:", nCl)
    print("Ion charge:", q_ion)
    print("Total charge:", q_prot + q_ion)
    print("Box length:", box_length_A, "A")

if __name__ == "__main__":
    main()