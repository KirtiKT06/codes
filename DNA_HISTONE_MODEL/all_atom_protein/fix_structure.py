"""
Stage 2 — run PDBFixer on the protein-only PDB:
  - find missing residues (the disordered tails) and missing atoms
  - build them in with idealized local geometry
  - add hydrogens at the configured pH

Also writes out a JSON list of exactly which (chain, resnum) residues were
newly built, so 04_run_md.py can restrain the ordered core and leave the
built tails free during equilibration.

Reads:  paths.protein_only_pdb
Writes: paths.fixed_pdb, paths.added_residues_json
"""
import json

from openmm.app import PDBFile
from pdbfixer import PDBFixer

from config import load_config, outpath


def main():
    cfg = load_config()

    in_pdb = outpath(cfg, "protein_only_pdb")
    out_pdb = outpath(cfg, "fixed_pdb")
    out_json = outpath(cfg, "added_residues_json")
    ph = cfg["structure"]["ph"]

    fixer = PDBFixer(filename=in_pdb)

    fixer.findMissingResidues()

    # findMissingResidues() also flags real chain-terminal gaps between
    # separate chains as "missing" if they're not actually part of the
    # sequence span you want built. For a protein-only octamer this isn't
    # an issue, but if you ever feed this script a multi-domain construct,
    # inspect fixer.missingResidues before proceeding.
    missing_before = {
        f"{k[0]}_{k[1]}": v for k, v in fixer.missingResidues.items()
    }

    fixer.findMissingAtoms()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)

    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    with open(out_pdb, "w") as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f, keepIds=True)

    # Build a flat list of every residue (chain id, resSeq, resname) that
    # PDBFixer inserted, for use as the "flexible" set downstream.
    added = []
    chains = list(fixer.topology.chains())
    for key, res_names in missing_before.items():
        if isinstance(key, tuple) and len(key) == 2:
            chain_index, insert_after_res_index = key
        elif isinstance(key, str) and "_" in key:
            chain_index_str, insert_after_res_index_str = key.split("_")
            chain_index = int(chain_index_str)
            insert_after_res_index = int(insert_after_res_index_str)
        else:
            raise ValueError(f"Unexpected key shape in missingResidues: {key!r} ({type(key)})")

        chain_id = chains[chain_index].id
        for offset, resname in enumerate(res_names):
            added.append(
                {
                    "chain": chain_id,
                    "chain_index": chain_index,
                    "insert_after_residue_index": insert_after_res_index,
                    "offset": offset,
                    "resname": resname,
                }
            )

    with open(out_json, "w") as f:
        json.dump(added, f, indent=2)

    print(f"Missing-residue stretches found: {len(missing_before)}")
    print(f"Total residues built in: {len(added)}")
    print(f"Fixed PDB written to: {out_pdb}")
    print(f"Added-residue manifest written to: {out_json}")


if __name__ == "__main__":
    main()