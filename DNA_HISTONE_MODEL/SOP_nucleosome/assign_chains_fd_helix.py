"""
assign_chains_fd_helix.py
==========================
fd_helix.py (used to build an idealized reference B-DNA helix for
deriving correct TIS-DNA equilibrium bond/angle/stacking/H-bond
geometry -- see build_dna_topology.py) writes PDB records with a BLANK
chain-ID column and uses a single "TER" line to separate the two
strands. AA2TIS_dna.py (like AA2CG.py elsewhere in this project)
expects real chain IDs in column 22, matching the "dna_chains" entry
of input.json.

This tiny utility rewrites an fd_helix.py PDB, assigning chain_id_1 to
every record before the first TER and chain_id_2 to everything after.

Usage:
    python3 assign_chains_fd_helix.py raw_fd_helix.pdb fixed.pdb I J
"""

import sys


def assign_chains(infile, outfile, chain1="I", chain2="J"):
    current_chain = chain1
    with open(infile) as fin, open(outfile, "w") as fout:
        for line in fin:
            if line.startswith("TER"):
                fout.write(line)
                current_chain = chain2
                continue
            if line[:6].strip() in ("ATOM", "HETATM"):
                line = line[:21] + current_chain + line[22:]
            fout.write(line)


if __name__ == "__main__":
    infile, outfile = sys.argv[1], sys.argv[2]
    chain1 = sys.argv[3] if len(sys.argv) > 3 else "I"
    chain2 = sys.argv[4] if len(sys.argv) > 4 else "J"
    assign_chains(infile, outfile, chain1, chain2)
    print(f"Wrote {outfile} with chains '{chain1}' / '{chain2}'.")