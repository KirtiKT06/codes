"""
Stage 1 — remove DNA chains (and waters/HETATM if requested) from the raw
nucleosome PDB, keeping only the histone protein chains.

This is a text-level PDB filter rather than an MDAnalysis rewrite,
specifically to KEEP the SEQRES records for the protein chains. PDBFixer's
missing-residue detection (stage 2) reads SEQRES to know the true full
sequence — if SEQRES is dropped, PDBFixer only sees the residues already
present and will not detect the missing tails at all.

Reads:  paths.raw_pdb
Writes: paths.protein_only_pdb
"""
from config import load_config, outpath


def main():
    cfg = load_config()
    struct_cfg = cfg["structure"]

    raw_pdb = cfg["paths"]["raw_pdb"]
    out_pdb = outpath(cfg, "protein_only_pdb")

    protein_chains = set(struct_cfg["protein_chain_ids"])
    keep_hetatm = not struct_cfg.get("remove_waters_and_hetatm", True)

    kept_seqres_chains = set()
    kept_atom_chains = set()
    n_atom_lines = 0

    with open(raw_pdb) as fin, open(out_pdb, "w") as fout:
        for line in fin:
            record = line[0:6]

            if record == "SEQRES":
                chain_id = line[11]
                if chain_id in protein_chains:
                    fout.write(line)
                    kept_seqres_chains.add(chain_id)

            elif record in ("ATOM  ", "HETATM"):
                chain_id = line[21]
                if chain_id not in protein_chains:
                    continue
                if record == "HETATM" and not keep_hetatm:
                    continue
                fout.write(line)
                kept_atom_chains.add(chain_id)
                n_atom_lines += 1

            elif record == "TER   ":
                chain_id = line[21] if len(line) > 21 else ""
                if chain_id == "" or chain_id in protein_chains:
                    fout.write(line)

            elif record in ("CRYST1", "REMARK", "HEADER", "TITLE "):
                fout.write(line)

        fout.write("END\n")

    missing = protein_chains - kept_atom_chains
    if missing:
        print(f"WARNING: requested chains not found in ATOM records: {missing}")

    print(f"Chains with SEQRES kept: {sorted(kept_seqres_chains)}")
    print(f"Chains with coordinates kept: {sorted(kept_atom_chains)}")
    print(f"Atom/hetatm lines written: {n_atom_lines}")
    print(f"Protein-only PDB written to: {out_pdb}")


if __name__ == "__main__":
    main()
