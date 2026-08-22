"""
make_pqr.py
-----------
Writes a .pqr file using EXACTLY the same charge model the PINN trains
against (unit -1e on each phosphorus atom, everything else neutral) --
instead of relying on pdb2pqr's AMBER/CHARMM force-field assignment,
which (a) drops modified nucleotides it doesn't recognize and (b) uses
partial charges spread over many atoms rather than our simplified
unit-backbone-charge model.

This makes the APBS run a genuine numerical check of the SAME physics
problem the PINN is solving -- not a comparison against a differently
parameterized system.

Also writes a matching APBS input file (.in) using the same bounding
box and ionic strength as the PINN's training run, so both solves cover
the same domain.

Usage:
    python make_pqr.py 1EHZ.pdb --chain A --ionic-strength 0.15
    # -> writes 1EHZ_simple.pqr and 1EHZ_simple.in
    apbs 1EHZ_simple.in
"""

import argparse
import numpy as np

from structure import load_rna_structure, get_bounding_box

# Generic heavy-atom radius, Angstrom -- matches the PINN's ATOM_RADIUS
# assumption (see pb_pinn.py) for consistency between the two solves.
GENERIC_RADIUS = 1.7
EPS_RNA = 4.0
EPS_WATER = 78.0


def write_pqr(pdb_path, pqr_path, chain_id=None):
    """
    Rebuilds atom records directly from the PDB (not via pdb2pqr) so every
    atom -- including modified nucleotides -- is retained. Charge is -1.0
    on phosphorus atoms, 0.0 everywhere else; radius is GENERIC_RADIUS for
    every atom (matches the PINN's single-radius dielectric surface).
    """
    from Bio.PDB import PDBParser
    from Bio.PDB.Polypeptide import is_aa
    from structure import NON_SOLUTE_HETNAMES

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rna", pdb_path)
    model = next(structure.get_models())

    lines = []
    serial = 1
    n_charged = 0

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            resname = res.get_resname().strip()
            if is_aa(res) or resname in NON_SOLUTE_HETNAMES:
                continue
            for atom in res:
                name = atom.get_name().strip()
                x, y, z = atom.get_coord()
                charge = -1.0 if name == "P" else 0.0
                if charge != 0.0:
                    n_charged += 1
                # PQR column format (whitespace-delimited, APBS-compatible):
                # ATOM serial name resname chain resseq x y z charge radius
                lines.append(
                    f"ATOM  {serial:5d} {name:<4s} {resname:<3s} "
                    f"{chain.id:1s}{res.id[1]:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}{charge:8.4f}{GENERIC_RADIUS:7.4f}"
                )
                serial += 1

    with open(pqr_path, "w") as f:
        f.write("REMARK   Manually generated PQR: unit -1e on P atoms only, "
                "matches pb_pinn.py charge model\n")
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {pqr_path}: {serial - 1} atoms, {n_charged} charged (should match "
          f"n_phosphates from structure.py, e.g. 76 for 1EHZ)")
    return serial - 1, n_charged


def write_apbs_input(pqr_path, in_path, lo, hi, ionic_strength_M=0.15,
                      eps_in=EPS_RNA, eps_out=EPS_WATER, grid_spacing=0.5):
    """
    APBS input file (.in) using the SAME bounding box and ionic strength as
    the PINN's training run, and nonlinear PBE (npbe) to match pb_pinn.py's
    sinh(phi) term.

    NOTE: APBS requires grid dimensions of the form c*2^l + 1 for multigrid
    -- we round up to the nearest valid "good" dimension here via a short
    lookup, since arbitrary dims are rejected.
    """
    extent = hi - lo
    center = (hi + lo) / 2.0

    def good_dim(n):
        candidates = [33, 65, 97, 129, 161, 193, 225, 257, 321, 385, 449, 513]
        for c in candidates:
            if c >= n:
                return c
        return candidates[-1]

    dims = [good_dim(int(np.ceil(e / grid_spacing)) + 1) for e in extent]

    in_text = f"""read
    mol pqr {pqr_path}
end

elec
    mg-auto
    dime {dims[0]} {dims[1]} {dims[2]}
    cglen {extent[0]:.2f} {extent[1]:.2f} {extent[2]:.2f}
    fglen {extent[0]:.2f} {extent[1]:.2f} {extent[2]:.2f}
    cgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
    fgcent {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}
    mol 1
    npbe
    bcfl sdh
    pdie {eps_in}
    sdie {eps_out}
    srfm smol
    chgm spl2
    sdens 10.0
    srad 1.4
    swin 0.3
    temp 298.15
    ion charge 1 conc {ionic_strength_M} radius 2.0
    ion charge -1 conc {ionic_strength_M} radius 2.0
    calcenergy no
    calcforce no
    write pot dx {pqr_path.replace('.pqr', '_potential')}
end

quit
"""
    with open(in_path, "w") as f:
        f.write(in_text)
    print(f"Wrote {in_path}  (grid dims: {dims}, box extent: {extent.round(1)} Å)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_path")
    parser.add_argument("--chain", default=None)
    parser.add_argument("--ionic-strength", type=float, default=0.15)
    args = parser.parse_args()

    base = args.pdb_path.rsplit(".", 1)[0]
    pqr_path = f"{base}_simple.pqr"
    in_path = f"{base}_simple.in"

    _, _, allatom_xyz = load_rna_structure(args.pdb_path, chain_id=args.chain)
    lo, hi = get_bounding_box(allatom_xyz, padding=20.0)

    write_pqr(args.pdb_path, pqr_path, chain_id=args.chain)
    write_apbs_input(pqr_path, in_path, lo, hi, ionic_strength_M=args.ionic_strength)

    print(f"\nNext: run   apbs {in_path}")
    print(f"This writes a .dx file named like "
          f"'{base}_simple_potential*.dx' -- feed that into compare_apbs.py")


if __name__ == "__main__":
    main()