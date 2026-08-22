"""
structure.py
------------
Turn an RNA PDB file into the two things the PB-PINN needs:

1. A fixed-charge distribution (Gaussian-smeared phosphate charges, -1e each).
   Delta functions are not differentiable-friendly for a PINN residual, so
   every point charge is replaced by a normalized Gaussian of width `sigma_q`
   (typically 0.5-1.0 Angstrom -- small compared to the box, large enough that
   autograd doesn't choke on a near-singular source term).

2. An all-atom point cloud used to build a smoothed dielectric field,
   eps(r): ~2-20 inside the RNA, 78 in bulk water, with a sigmoid transition
   across the (approximate) molecular surface. This is the "easy" interface
   treatment mentioned in the plan -- no explicit SES/SASA meshing yet.

NOTE ON CHARGE MODEL: this uses unit backbone charge (-1e per phosphate),
the standard simplification for coarse PB treatments of nucleic acids. Swap
in partial charges (e.g. from AMBER RNA force field) later if you want
atomic-resolution accuracy -- the PINN code doesn't care, it just consumes
a list of (position, charge).
"""

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

RNA_RESNAMES = {"A", "U", "G", "C", "DA", "DU", "DG", "DC", "RA", "RU", "RG", "RC"}

# Crystallographic solvent / bound ions that show up as HETATM and should NOT
# be treated as part of the "solute" when building the dielectric surface --
# they're either explicit water (already represented implicitly by eps_water)
# or ions the continuum PB treatment itself is trying to model implicitly.
# Extend this set if a given PDB has other crystallization additives (e.g.
# SO4, GOL, ACT, PO4 buffer ions, spermine/SPM, spermidine/SPD, other metals).
NON_SOLUTE_HETNAMES = {"HOH", "MG", "MN", "NA", "K", "CA", "ZN", "CO", "NI", "SO4", "PO4"}


def load_rna_structure(pdb_path: str, chain_id: str | None = None):
    """
    Returns:
        phosphate_xyz : (Np, 3) float64 array, phosphorus atom positions
        phosphate_q   : (Np,)   float64 array, charge in units of e (all -1.0)
        allatom_xyz   : (Na, 3) float64 array, every atom (for the eps(r) surface)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("rna", pdb_path)
    model = next(structure.get_models())

    phosphate_xyz, allatom_xyz = [], []
    skipped_hetnames = set()

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue
        for res in chain:
            resname = res.get_resname().strip()
            if is_aa(res):
                continue  # skip protein residues if this is a complex (e.g. nucleosome)
            if resname in NON_SOLUTE_HETNAMES:
                skipped_hetnames.add(resname)
                continue  # crystallographic water / bound ion -- not part of the RNA solute

            for atom in res:
                allatom_xyz.append(atom.get_coord())
                # Identify backbone phosphates by ATOM NAME, not residue name.
                # Modified nucleotides (1MA, PSU, OMG, H2U, etc. -- routine in
                # tRNA) still carry a normal phosphate; gating on RNA_RESNAMES
                # alone silently drops their charge. The 5'-terminal residue
                # legitimately has no P atom -- that's fine, nothing to add.
                if atom.get_name().strip() == "P":
                    phosphate_xyz.append(atom.get_coord())

    if skipped_hetnames:
        print(f"[structure.py] Excluded non-solute HETATM residues from the "
              f"solute cloud: {sorted(skipped_hetnames)}")

    if len(phosphate_xyz) == 0:
        raise ValueError(
            f"No phosphorus atoms found in {pdb_path} (chain={chain_id}). "
            "Check chain_id, and that this is RNA not a bare nucleoside/5' terminus-only fragment."
        )

    phosphate_xyz = np.asarray(phosphate_xyz, dtype=np.float64)
    allatom_xyz = np.asarray(allatom_xyz, dtype=np.float64)
    phosphate_q = -1.0 * np.ones(len(phosphate_xyz), dtype=np.float64)

    return phosphate_xyz, phosphate_q, allatom_xyz


def get_bounding_box(allatom_xyz: np.ndarray, padding: float = 15.0):
    """
    Padding in Angstrom. PB potentials from a screened (Debye-Hückel-decaying)
    source die off over a few Debye lengths, so ~15-25 A padding at physiological
    ionic strength (kappa^-1 ~ 8-10 A at 150 mM) is a reasonable box margin --
    tighten/loosen once you check kappa for your actual ionic strength.
    """
    lo = allatom_xyz.min(axis=0) - padding
    hi = allatom_xyz.max(axis=0) + padding
    return lo, hi