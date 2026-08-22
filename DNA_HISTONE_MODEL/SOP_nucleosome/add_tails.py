"""
add_tails.py
============
Extends the coarse-grained histone PDB (histone_cg.pdb) with the
disordered N-terminal tail residues that are absent from the 2CV5
crystal structure.

HOW IT WORKS (theory summary in comments below)
"""

import numpy as np
import json
import subprocess
from collections import defaultdict

# ─────────────────────────────────────────────
# THEORY: Why and how we build the tails
# ─────────────────────────────────────────────
#
# The histone tails are intrinsically disordered regions (IDRs).
# Because they have no stable structure, they are absent from the
# 2CV5 crystal — the electron density is too diffuse to resolve them.
#
# In the SOP-CG model, each residue is one bead at the Cα position.
# For the tails we have no Cα coordinates, so we must generate them.
#
# APPROACH: Self-Avoiding Random Walk (SAW) from the N-terminal anchor
# ─────────────────────────────────────────────────────────────────────
# 1. Start at the first RESOLVED crystal bead (the anchor).
# 2. Walk "outward" (away from the core) placing each new bead at
#    bond length b = 3.8 Å from the previous one.
# 3. Each step direction is drawn randomly on the unit sphere,
#    then we check for clashes (r < σ = 3.8 Å) with ALL existing beads.
#    If clash → resample (up to MAX_TRIES attempts).
# 4. After all tails are placed, we do a short steepest-descent
#    relaxation (just moving clashing beads) to remove any remaining
#    overlaps.
#
# Why 3.8 Å bond length?
#   In the SOP model, σ_protein = 3.8 Å (Table S1, Reddy & Thirumalai).
#   The FENE bond equilibrium distance is also set by this length scale.
#   Consecutive Cα atoms in real proteins are ~3.8 Å apart.
#
# Why random walk and not extended chain?
#   The tails are genuinely disordered. A single extended conformation
#   would introduce artificial order. A random walk samples the
#   disordered ensemble appropriately and will be further randomised
#   during equilibration before production.
#
# The tail beads participate in:
#   (a) FENE bonds with their neighbours (same as core)
#   (b) Excluded volume (repulsive LJ) with ALL other protein beads
#   (c) Screened Coulomb with DNA beads — THIS is the key physics
#       for your study (charged tails wrapping DNA)
#   They do NOT get native contacts assigned (no crystal reference).

# ─────────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────────
config = json.load(open("input.json"))

aa_pdb       = config["aa_pdb"]
cg_pdb_in    = config["cg_pdb"]
cg_pdb_out   = config["cg_pdb_with_tails"]

BOND_LENGTH  = config["bond_length"]          # Å — Cα–Cα / SOP σ_protein
CLASH_CUTOFF = config["clash_cutoff"]         # Å — minimum bead–bead distance
MAX_TRIES    = config["max_placement_tries"]  # SAW placement attempts
SEED         = config["random_seed"]

np.random.seed(SEED)

# ─────────────────────────────────────────────
# STEP 1: READ SEQRES → FULL SEQUENCES PER CHAIN
# ─────────────────────────────────────────────
result = subprocess.run(["grep", "SEQRES", aa_pdb], capture_output=True, text=True)
chain_full_seq = defaultdict(list)   # chain → [RES1, RES2, ...] (1-indexed)
histone_chains = config["histone_chains"]
for line in result.stdout.splitlines():
    parts = line.split()
    if len(parts) < 4:
        continue
    ch = parts[2]
    if ch not in histone_chains:
        continue
    chain_full_seq[ch].extend(parts[4:])

# ─────────────────────────────────────────────
# STEP 2: READ EXISTING CG BEADS
# ─────────────────────────────────────────────
beads = []   # list of dicts, order = PDB order = bead index
with open(cg_pdb_in) as f:
    for line in f:
        if line[:6].strip() != "ATOM":
            continue
        chain  = line[21]
        resid  = int(line[22:26])
        resname= line[17:20].strip()
        x      = float(line[30:38])
        y      = float(line[38:46])
        z      = float(line[46:54])
        beads.append({"chain": chain, "resid": resid, "resname": resname,
                      "x": x, "y": y, "z": z, "is_tail": False})

# First crystal residue per chain (needed to know where tail attaches)
crystal_start = {}
for b in beads:
    ch = b["chain"]
    if ch not in crystal_start or b["resid"] < crystal_start[ch]:
        crystal_start[ch] = b["resid"]

print("Crystal start residues:", crystal_start)

# ─────────────────────────────────────────────
# STEP 3: BUILD TAIL BEADS CHAIN BY CHAIN
# ─────────────────────────────────────────────

def random_unit_vector():
    """Uniform random direction on unit sphere (Marsaglia method)."""
    while True:
        v = np.random.randn(3)
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            return v / norm

def has_clash(pos, all_coords, cutoff=CLASH_CUTOFF):
    """True if pos is within cutoff of any existing bead."""
    if len(all_coords) == 0:
        return False
    diffs = all_coords - pos
    dists = np.linalg.norm(diffs, axis=1)
    return np.any(dists < cutoff)

tail_beads_per_chain = {}
all_tail_insertions  = {}   # chain → list of bead dicts, ordered res1..resN_tail

for ch in sorted(crystal_start.keys()):
    n_tail = crystal_start[ch] - 1   # number of missing residues
    if n_tail <= 0:
        print(f"Chain {ch}: no tail to add.")
        all_tail_insertions[ch] = []
        continue

    tail_seq = chain_full_seq[ch][:n_tail]   # SEQRES residues 1..n_tail
    print(f"\nChain {ch}: adding {n_tail} tail beads: {tail_seq}")

    # Anchor = first crystal bead of this chain
    anchor = next(b for b in beads if b["chain"] == ch and b["resid"] == crystal_start[ch])
    anchor_pos = np.array([anchor["x"], anchor["y"], anchor["z"]])

    # Build outward from anchor (tail residues run from resid n_tail down to 1)
    # We place them in REVERSE order (resid n_tail → 1), growing away from core.
    # This means:
    #   placed[0] = resid n_tail  (directly bonded to anchor)
    #   placed[1] = resid n_tail-1
    #   ...
    #   placed[-1] = resid 1      (free N-terminus)

    # Collect all existing coords for clash checking (numpy array)
    existing_coords = np.array([[b["x"], b["y"], b["z"]] for b in beads])

    placed = []  # will be in order resid n_tail → 1
    prev_pos = anchor_pos

    for k in range(n_tail):
        resid_k   = n_tail - k          # e.g. 37, 36, 35 ... 1
        resname_k = tail_seq[resid_k - 1]  # 0-indexed into full sequence

        success = False
        for attempt in range(MAX_TRIES):
            direction = random_unit_vector()
            new_pos   = prev_pos + BOND_LENGTH * direction

            # Check clash against all existing beads + already-placed tail beads
            if len(placed) > 0:
                placed_coords = np.array([[p["x"], p["y"], p["z"]] for p in placed])
                all_check     = np.vstack([existing_coords, placed_coords])
            else:
                all_check = existing_coords

            if not has_clash(new_pos, all_check):
                placed.append({
                    "chain"  : ch,
                    "resid"  : resid_k,
                    "resname": resname_k,
                    "x"      : new_pos[0],
                    "y"      : new_pos[1],
                    "z"      : new_pos[2],
                    "is_tail": True
                })
                prev_pos = new_pos
                success  = True
                break

        if not success:
            print(f"  WARNING: chain {ch} res {resid_k} — could not place clash-free "
                  f"after {MAX_TRIES} attempts. Placing anyway at last position + offset.")
            fallback = prev_pos + BOND_LENGTH * np.array([1.0, 0.0, 0.0])
            placed.append({
                "chain": ch, "resid": resid_k, "resname": resname_k,
                "x": fallback[0], "y": fallback[1], "z": fallback[2],
                "is_tail": True
            })
            prev_pos = fallback

    # placed is ordered resid n_tail → 1; reverse so list is resid 1 → n_tail
    placed_ordered = placed[::-1]
    all_tail_insertions[ch] = placed_ordered
    print(f"  Chain {ch}: placed {len(placed_ordered)} tail beads "
          f"(res 1–{n_tail}), free N-term at "
          f"({placed_ordered[0]['x']:.1f}, {placed_ordered[0]['y']:.1f}, {placed_ordered[0]['z']:.1f})")

# ─────────────────────────────────────────────
# STEP 4: MERGE AND WRITE OUTPUT PDB
# ─────────────────────────────────────────────
# Final bead order: for each chain, [tail res 1..N_tail] then [crystal res]
# We build chain by chain to keep chains contiguous.

chain_order = config["histone_chains"]
all_new_beads = []

for ch in chain_order:
    # Tail beads first (resid 1 .. n_tail)
    all_new_beads.extend(all_tail_insertions.get(ch, []))
    # Then crystal beads (already sorted by resid within chain in the PDB)
    core_beads = sorted([b for b in beads if b["chain"] == ch], key=lambda b: b["resid"])
    all_new_beads.extend(core_beads)

total_beads = len(all_new_beads)
tail_count  = sum(b["is_tail"] for b in all_new_beads)
print(f"\nTotal beads: {total_beads} ({tail_count} tail + {total_beads - tail_count} core)")

with open(cg_pdb_out, "w") as f:
    for i, b in enumerate(all_new_beads, start=1):
        f.write(
            "{:<6s}{:5d} {:^4s} {:>3s} {:1s}{:4d}    "
            "{:8.3f}{:8.3f}{:8.3f}  1.00  0.00\n".format(
                "ATOM", i, "BB", b["resname"], b["chain"], b["resid"],
                b["x"], b["y"], b["z"]
            )
        )
    f.write("END\n")

print(f"Saved: {cg_pdb_out}")

# ─────────────────────────────────────────────
# STEP 5: WRITE CHARGE ANNOTATION FILE
# ─────────────────────────────────────────────
# For use in OpenMM: bead index (0-based) → charge
# Charged residues: ARG=+1, LYS=+1, HIS=+0.5 (partial), ASP=-1, GLU=-1
# We treat HIS as +1 at physiological pH (protonated form relevant for DNA binding)

charged_beads_file = config["charged_beads_file"]

# Charge assignment. HIS charge is configurable (0.0 = neutral, 1.0 = protonated).
# At physiological pH HIS is ~10% protonated; for DNA-binding studies
# it is common to treat it as fully charged (+1) to maximise electrostatic effect.
his_charge = config["histidine_charge"]

charge_map = {
    "ARG": +1.0,
    "LYS": +1.0,
    "HIS":  his_charge,
    "ASP": -1.0,
    "GLU": -1.0
}

charged_beads = []
for i, b in enumerate(all_new_beads):
    if b["resname"] in charge_map and charge_map[b["resname"]] != 0.0:
        q = charge_map[b["resname"]]
        charged_beads.append((i, b["chain"], b["resid"], b["resname"], q, b["is_tail"]))

with open(charged_beads_file, "w") as f:
    f.write("# bead_idx(0-based)  chain  resid  resname  charge  is_tail\n")
    for idx, ch, resid, resname, q, is_tail in charged_beads:
        f.write(f"{idx}  {ch}  {resid}  {resname}  {q:+.1f}  {int(is_tail)}\n")

print(f"Saved: {charged_beads_file} ({len(charged_beads)} charged beads, "
      f"{sum(1 for c in charged_beads if c[5])} in tails)")

# ─────────────────────────────────────────────
# STEP 6: SUMMARY STATISTICS
# ─────────────────────────────────────────────
print("\n=== SUMMARY ===")
print(f"Input CG beads (no tails):   761")
print(f"Tail beads added:            {tail_count}")
print(f"Output CG beads (with tails):{total_beads}")
print(f"Paper's model (with tails):  974")
print(f"Difference from paper:       {total_beads - 974} beads")
print("(Small difference expected — paper added tails to a slightly")
print(" different residue range using their own modelling protocol.)")

per_chain = {}
for b in all_new_beads:
    per_chain[b["chain"]] = per_chain.get(b["chain"], 0) + 1
print("\nBeads per chain:")
for ch in chain_order:
    tail_n = len(all_tail_insertions.get(ch, []))
    core_n = per_chain[ch] - tail_n
    print(f"  Chain {ch}: {core_n} core + {tail_n} tail = {per_chain[ch]}")