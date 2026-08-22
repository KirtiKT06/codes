"""
Check the REAL geometric angle (not force field parameter — actual 3D
angle between atoms) around the flagged ARG guanidinium groups, straight
from the PDBFixer output (fixed_pdb), before compact_tails.py or any
dynamics ever touched it. If these angles are already far from the
correct ~120 degrees (2.094 rad) sp2 target here, the problem is
PDBFixer's addMissingHydrogens() placement, not anything downstream.

Usage:
    python3 check_guanidinium_geometry.py
"""
import numpy as np
from openmm import app

from config import load_config, outpath


def angle_deg(p1, p2, p3):
    """Angle at p2, formed by p1-p2-p3, in degrees."""
    v1 = np.array([p1.x, p1.y, p1.z]) - np.array([p2.x, p2.y, p2.z])
    v2 = np.array([p3.x, p3.y, p3.z]) - np.array([p2.x, p2.y, p2.z])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    return np.degrees(np.arccos(cos_theta))


def main():
    cfg = load_config()
    fixed_pdb = outpath(cfg, "fixed_pdb")

    pdb = app.PDBFile(fixed_pdb)
    positions = pdb.positions

    # the flagged residues from your last run
    flagged = [
        ("E", "72"), ("G", "32"), ("G", "88"), ("F", "39"), ("F", "67"),
    ]

    for chain_id, resid in flagged:
        for residue in pdb.topology.residues():
            if residue.chain.id != chain_id or residue.id != resid or residue.name != "ARG":
                continue

            atom_by_name = {a.name: a for a in residue.atoms()}
            required = ["CD", "NE", "HE", "CZ"]
            if not all(n in atom_by_name for n in required):
                print(f"{chain_id} ARG{resid}: missing one of {required}, skipping")
                continue

            cd = positions[atom_by_name["CD"].index]
            ne = positions[atom_by_name["NE"].index]
            he = positions[atom_by_name["HE"].index]
            cz = positions[atom_by_name["CZ"].index]

            angle_cd_ne_cz = angle_deg(cd, ne, cz)
            angle_cd_ne_he = angle_deg(cd, ne, he)
            angle_cz_ne_he = angle_deg(cz, ne, he)
            angle_sum = angle_cd_ne_he + angle_cz_ne_he + angle_cd_ne_cz

            print(f"{chain_id} ARG{resid} NE geometry:")
            print(f"  CD-NE-CZ = {angle_cd_ne_cz:.1f} deg  (expect ~120)")
            print(f"  CD-NE-HE = {angle_cd_ne_he:.1f} deg  (expect ~120)")
            print(f"  CZ-NE-HE = {angle_cz_ne_he:.1f} deg  (expect ~120)")
            print(f"  sum of all three angles around NE = {angle_sum:.1f} deg  (expect ~360 if planar)")
            print()


if __name__ == "__main__":
    main()