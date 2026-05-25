import json
import numpy as np

config = json.load(open("input.json"))

cg_pdb_file = config["cg_pdb_prefix"] + ".pdb"

native_cutoff = float(config.get("native_cutoff", 8.0))

min_seq_sep = int(config.get("min_seq_sep", 3))

def read_cg_pdb(pdb_file):
    names = []
    resnames = []
    chains = []
    resids = []
    xyz = []

    with open(pdb_file) as f:
        for line in f:
            rec = line[:6].strip()
            if rec != "ATOM":
                continue

            names.append(line[12:16].strip())
            resnames.append(line[17:20].strip())
            chains.append(line[21].strip())
            resids.append(int(line[22:26]))
            xyz.append([
                float(line[30:38]),
                float(line[38:46]),
                float(line[46:54])
            ])

    return names, resnames, chains, resids, np.asarray(xyz, dtype=float)

native_contacts = []

names, resnames, chains, resids, xyz = read_cg_pdb(cg_pdb_file)

for i in range(len(xyz)):
    for j in range(i+1, len(xyz)):
        if abs(resids[i] - resids[j]) <= min_seq_sep:
            continue
        
        dist = np.linalg.norm(xyz[i] - xyz[j])
        if dist < native_cutoff:
            native_contacts.append((i, j, dist))

print("Total beads:", len(xyz))
print("Native contacts found:", len(native_contacts))

if len(native_contacts) == 0:
    raise RuntimeError("No native contacts found. Increase cutoff.")

distances = [x[2] for x in native_contacts]
print("Min native distance:", min(distances))
print("Max native distance:", max(distances))
print("Average native distance:", np.mean(distances))

with open("native_contacts.dat", "w") as f:
    f.write("# i j r0_A\n")

    for i, j, dist in native_contacts:
        f.write(f"{i} {j} {dist:.3f}\n")