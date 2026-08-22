"""
qm_cluster_builder.py
Build QM cluster models (Gaussian .com) for metal-binding sites in a
metalloprotein, with automatic Cβ-truncation, capping-H placement,
charge calculation, and metal-identity swapping.

Usage:
    python qm_cluster_builder.py --pdb 2eli.pdb --metal ZN \
        --swap ZN CD PB --outdir clusters/
"""

import argparse
import os
import numpy as np
import MDAnalysis as mda

# ---------------------------------------------------------------------
# Chemistry tables — edit here if your site has unusual ligands
# ---------------------------------------------------------------------

# Which sidechain atom(s) count as a "donor" for metal coordination search
DONOR_ATOMS = {
    "CYS": ["SG"],
    "HIS": ["ND1", "NE2"],
    "ASP": ["OD1", "OD2"],
    "GLU": ["OE1", "OE2"],
    "MET": ["SD"],
}

# Formal charge contributed by each residue TYPE when coordinating
# (assumes the standard coordinating tautomer/protonation state —
#  VERIFY this against your specific system before trusting it)
RESIDUE_CHARGE = {
    "CYS": -1,   # deprotonated thiolate
    "HIS": 0,    # neutral imidazole, one N protonated
    "ASP": -1,   # deprotonated carboxylate
    "GLU": -1,   # deprotonated carboxylate
    "MET": 0,    # neutral thioether
}

METAL_CHARGE = {"ZN": 2, "CD": 2, "PB": 2}

# Atom names considered "backbone" — everything else on a residue is
# kept as sidechain. This is what makes the truncation general-purpose
# instead of hardcoded per residue type.
BACKBONE_NAMES = {"N", "CA", "C", "O", "H", "H1", "H2", "H3",
                   "HA", "HA2", "HA3", "OXT"}

DEF2_LIGAND_BASIS = "def2SVP"
DEF2_METAL_BASIS = "def2TZVP"


# ---------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------

def find_metal_sites(u, metal_resname, cutoff=3.6):
    """Find each metal atom and its coordinating residues within cutoff (Å)."""
    metals = u.select_atoms(f"resname {metal_resname}")
    if len(metals) == 0:
        raise ValueError(f"No atoms with resname {metal_resname} found.")

    donor_sel = " or ".join(
        f"(resname {res} and name {' '.join(atoms)})"
        for res, atoms in DONOR_ATOMS.items()
    )

    sites = []
    for metal_atom in metals:
        nearby = u.select_atoms(
            f"({donor_sel}) and around {cutoff} (index {metal_atom.index})"
        )
        residues = sorted(set(nearby.residues), key=lambda r: r.resid)
        sites.append({"metal_atom": metal_atom, "residues": residues})
    return sites


def get_capping_hydrogen(residue):
    """Place a capping H along the real CA->CB bond vector at 1.09 Å from CB."""
    try:
        ca = residue.atoms.select_atoms("name CA").positions[0]
        cb = residue.atoms.select_atoms("name CB").positions[0]
    except IndexError:
        raise ValueError(
            f"{residue.resname}{residue.resid} missing CA or CB — "
            "cannot cap (e.g. Gly can't coordinate via a normal sidechain cut)."
        )
    vec = (cb - ca)
    vec /= np.linalg.norm(vec)
    return cb + 1.09 * vec


def truncate_residue(residue):
    """Return (elements, coords) for sidechain-only atoms + one capping H."""
    sidechain = residue.atoms.select_atoms(
        "not name " + " ".join(BACKBONE_NAMES)
    )
    elements = list(sidechain.elements) if hasattr(sidechain, "elements") \
        else [a.name[0] for a in sidechain]  # fallback if elements not guessed
    coords = list(sidechain.positions)

    h_cap = get_capping_hydrogen(residue)
    elements.append("H")
    coords.append(h_cap)
    return elements, coords


def compute_total_charge(residues, metal_symbol):
    charge = METAL_CHARGE.get(metal_symbol.upper(), 0)
    for res in residues:
        charge += RESIDUE_CHARGE.get(res.resname, 0)
    return charge


def build_cluster(u, site, metal_symbol):
    """Assemble full cluster: capped residues + (possibly swapped) metal."""
    all_elements, all_coords = [], []
    for res in site["residues"]:
        elems, coords = truncate_residue(res)
        all_elements.extend(elems)
        all_coords.extend(coords)

    all_elements.append(metal_symbol.capitalize())
    all_coords.append(site["metal_atom"].position)

    charge = compute_total_charge(site["residues"], metal_symbol)
    return all_elements, all_coords, charge


# ---------------------------------------------------------------------
# Gaussian .com writer — correct blank-line spacing, mixed basis
# ---------------------------------------------------------------------

def write_gaussian_com(filename, elements, coords, charge, mult=1,
                        chk=None, nproc=8, mem="8GB",
                        functional="b3lyp", dispersion="gd3bj"):
    chk = chk or os.path.basename(filename).replace(".com", ".chk")
    unique_elements = sorted(set(e for e in elements if e.upper() not in METAL_CHARGE))
    metal_elements = sorted(set(e for e in elements if e.upper() in METAL_CHARGE))

    with open(filename, "w") as f:
        f.write(f"%chk={chk}\n")
        f.write(f"%nprocshared={nproc}\n")
        f.write(f"%mem={mem}\n")
        f.write(f"# opt freq {functional}/gen empiricaldispersion={dispersion} "
                f"scf=(xqc,maxcycle=512) int=ultrafine nosymm\n")
        f.write("\n")
        f.write(f"{os.path.basename(filename)} QM cluster\n")
        f.write("\n")
        f.write(f"{charge} {mult}\n")
        for el, xyz in zip(elements, coords):
            f.write(f" {el:<2s}  {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}\n")
        f.write("\n")
        f.write(" ".join(unique_elements) + " 0\n")
        f.write(f"{DEF2_LIGAND_BASIS}\n")
        f.write("****\n")
        for el in metal_elements:
            f.write(f"{el} 0\n")
            f.write(f"{DEF2_METAL_BASIS}\n")
            f.write("****\n")
        f.write("\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb", required=True)
    ap.add_argument("--metal", required=True, help="Native metal resname in PDB, e.g. ZN")
    ap.add_argument("--swap", nargs="+", default=None,
                     help="Element symbols to generate, e.g. ZN CD PB")
    ap.add_argument("--cutoff", type=float, default=3.6)
    ap.add_argument("--outdir", default="clusters")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    u = mda.Universe(args.pdb)

    sites = find_metal_sites(u, args.metal, cutoff=args.cutoff)
    swap_list = args.swap or [args.metal]

    for i, site in enumerate(sites, start=1):
        residues_str = ", ".join(f"{r.resname}{r.resid}" for r in site["residues"])
        print(f"Site {i}: metal idx {site['metal_atom'].index}, "
              f"coordinating residues: {residues_str}")
        if len(site["residues"]) < 3:
            print(f"  WARNING: only {len(site['residues'])} donor residues found "
                  f"within {args.cutoff} Å — check cutoff or structure completeness.")

        for metal_symbol in swap_list:
            elements, coords, charge = build_cluster(u, site, metal_symbol)
            fname = os.path.join(args.outdir, f"site{i}_{metal_symbol.lower()}.com")
            write_gaussian_com(fname, elements, coords, charge)
            print(f"  -> wrote {fname}  (charge={charge}, mult=1, "
                  f"{len(elements)} atoms)")


if __name__ == "__main__":
    main()