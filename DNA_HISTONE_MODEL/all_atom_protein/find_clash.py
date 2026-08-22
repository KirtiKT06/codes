"""
Run this against your actual solvated_pdb to find out exactly what is
sitting on top of (or very close to) the atoms run_md.py flagged as
having catastrophic residual force after minimization.

Usage:
    python3 find_clash.py
"""
import numpy as np
from openmm import app, unit

from config import load_config, outpath


def main():
    cfg = load_config()
    solvated_pdb = outpath(cfg, "solvated_pdb")

    pdb = app.PDBFile(solvated_pdb)
    positions = pdb.positions.value_in_unit(unit.nanometer)
    positions = np.array([[p.x, p.y, p.z] for p in positions])

    atoms = list(pdb.topology.atoms())

    # the exact indices run_md.py flagged in your run
    flagged_indices = [8956, 8958, 8953, 1405, 11120, 634, 1406, 11121, 635, 10956]

    for idx in flagged_indices:
        atom = atoms[idx]
        pos = positions[idx]

        # distance to every other atom (brute force — fine for a one-off check)
        deltas = positions - pos
        dists = np.linalg.norm(deltas, axis=1)
        dists[idx] = np.inf  # exclude self

        nearest_5 = np.argsort(dists)[:5]

        print(
            f"atom {idx}: {atom.residue.chain.id} {atom.residue.name}"
            f"{atom.residue.id} {atom.name}  at {pos}"
        )
        for n_idx in nearest_5:
            n_atom = atoms[n_idx]
            print(
                f"    nearest: atom {n_idx} "
                f"{n_atom.residue.chain.id} {n_atom.residue.name}"
                f"{n_atom.residue.id} {n_atom.name}  "
                f"dist = {dists[n_idx]*10:.3f} A"
            )
        print()


if __name__ == "__main__":
    main()