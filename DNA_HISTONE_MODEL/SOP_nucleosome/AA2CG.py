"""
The purpose of this script is to read in the all-atom PDB structure of the nucleosome, and generate a coarse-grained PDB file containing only the CA atoms of the histone proteins. 
This will be used as input for the SOP nucleosome simulations.
"""

# Standard imports
from __future__ import print_function
import json

# Load config
config = json.load(open("input.json"))

aa_pdb_file = config["aa_pdb"]
cg_pdb_file = config["cg_pdb"]

histone_chains = config["histone_chains"]
# dna_chains = config["dna_chains"]
# native_cutoff = config["native_cutoff"]

# Parse AA PDB and extract CA atoms of histone chains

chain_counts = {}
beads = []
with open(aa_pdb_file) as f:

    for line in f:

        # Step 1: Only consider ATOM lines (ignore HETATM, etc.)
        record = line[:6].strip()
        if record != "ATOM":
            continue
        
        # Step 2: Extract relevant fields
        atom_name = line[12:16].strip()
        chain = line[21]

        # Step 3: Only consider specified histone chains
        if chain not in histone_chains:
            continue

        # Step 4: Only consider CA atoms
        if atom_name != "CA":
            continue

        # Step 5: Extract residue info and coordinates
        resname = line[17:20].strip()
        resid = int(line[22:26])
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])

        # Step 6: Store bead info and update chain counts
        chain_counts[chain] = chain_counts.get(chain, 0) + 1
        beads.append(
        {
            "chain": chain,
            "resid": resid,
            "resname": resname,
            "x": x,
            "y": y,
            "z": z
        }
        )

# Step 7: Write out CG PDB file
with open(cg_pdb_file, "w") as f:
    for i, bead in enumerate(beads, start=1):
        f.write(
    "{:<6s}{:5d} {:^4s} {:>3s} {:1s}{:4d}    "
    "{:8.3f}{:8.3f}{:8.3f}\n".format(
        "ATOM",
        i,
        "BB",
        bead["resname"],
        bead["chain"],
        bead["resid"],
        bead["x"],
        bead["y"],
        bead["z"]
    )
)
    f.write("END\n")
print(f"Saved CG PDB with {len(beads)} beads to: {cg_pdb_file}")
print("Chain counts:", chain_counts)