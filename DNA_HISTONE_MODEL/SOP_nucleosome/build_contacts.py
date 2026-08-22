"""
This script reads in the CG PDB file, identifies native contacts based on a distance cutoff, and saves the contact list to a file. 
The contact list will be used in the SOP model to define native interactions.
"""
import numpy as np
import json

# Load config
config = json.load(open("input.json"))
cg_pdb_file = config["cg_pdb"]
native_cutoff = config["native_cutoff"]

# Load CG beads
beads = []
with open(cg_pdb_file) as f:
    for line in f:
        if line[:6].strip() != "ATOM":
            continue
        chain = line[21]
        resid = int(line[22:26])
        resname = line[17:20].strip()
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
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
        

native_contacts = []
# Build contacts with index-based separation, not resid-based
for i in range(len(beads)):
    for j in range(i+1, len(beads)):
        bead_i = beads[i]
        bead_j = beads[j]

        if bead_i["chain"] == bead_j["chain"]:
            if (j - i) < 3:   # sequential bead index gap, not resid gap
                continue

        # ... distance check
        x1, y1, z1 = bead_i["x"], bead_i["y"], bead_i["z"]
        x2, y2, z2 = bead_j["x"], bead_j["y"], bead_j["z"]

        r_i = np.array([x1, y1, z1])
        r_j = np.array([x2, y2, z2])

        r_ij = np.linalg.norm(r_j - r_i)

        if r_ij < native_cutoff:
            native_contacts.append((i, j, r_ij))
print(f"Found {len(native_contacts)} native contacts:")

# Create contact file
with open("native_contacts.dat", "w") as f:
    for i, j, r0 in native_contacts:
        f.write(f"{i} {j} {r0:.4f}\n")

print(f"Saved native contacts to: native_contacts.dat")