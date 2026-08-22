"""
build_dna_topology.py
======================
Takes the TIS-DNA coarse-grained PDB produced by AA2TIS_dna.py (P,S,B
beads) plus the per-chain sequence, and builds every interaction list
needed to run the Chakraborty-Hori-Thirumalai TIS-DNA force field in
OpenMM:

    dna_bonds.dat     -- harmonic S-P / P-S / S-Base bonds      (Eq. 2)
    dna_angles.dat    -- harmonic P-S-P / S-P-S / P-S-B / B-S-P  (Eq. 3)
    dna_stacking.dat  -- Eq. 9 stacking interactions (consecutive nt)
    dna_hbonds.dat     -- Eq. 13 Watson-Crick hydrogen-bond interactions
    dna_charges.dat   -- phosphate bead charges (Oosawa-Manning, Eq. 16)

Equilibrium geometric parameters that are NOT tabulated per-type in the
paper (the stacking distance/dihedral references l0, phi1_0, phi2_0,
and the hydrogen-bond geometry d0, theta1_0, theta2_0, psi1_0, psi2_0,
psi3_0) are, exactly as described in the Methodology section of the
paper ("equilibrium values ... obtained by coarse-graining an ideal
B-form DNA helix"), MEASURED DIRECTLY from the input TIS structure.
In practice you should run AA2TIS_dna.py on an idealized B-DNA
structure (e.g. generated with fd_helix.py, helix type "abdna" or
"lbdna") for a sequence that contains every dimer/base-pair type you
need, so that these reference values reflect ideal B-form geometry
rather than a distorted crystal structure. If instead you run this on
the actual nucleosomal DNA (bent, distorted), these "equilibrium"
values will reflect that bent geometry -- which is fine if your goal is
a structure-based (Go-like) elastic network for that particular
substrate, but is NOT what the original TIS-DNA paper parametrized.
See the printed warnings at run time.
"""

import json
import numpy as np
import dna_tis_params as P


# ─────────────────────────────────────────────────────────────────────
# Geometry helpers (all in Angstrom / radians)
# ─────────────────────────────────────────────────────────────────────
def _dist(a, b):
    return float(np.linalg.norm(np.array(a) - np.array(b)))


def _angle(a, b, c):
    """Angle a-b-c in radians."""
    v1 = np.array(a) - np.array(b)
    v2 = np.array(c) - np.array(b)
    cos_t = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    cos_t = np.clip(cos_t, -1.0, 1.0)
    return float(np.arccos(cos_t))


def _dihedral(a, b, c, d):
    """Dihedral a-b-c-d in radians (standard formula)."""
    b1 = np.array(b) - np.array(a)
    b2 = np.array(c) - np.array(b)
    b3 = np.array(d) - np.array(c)
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m1, n2)
    return float(np.arctan2(y, x))


# ─────────────────────────────────────────────────────────────────────
# Load the TIS-DNA PDB
# ─────────────────────────────────────────────────────────────────────
def load_tis_pdb(path):
    """Returns beads: list of dicts, and lookup[(chain,resid,beadtype)] -> index."""
    beads = []
    lookup = {}
    with open(path) as f:
        for line in f:
            if line[:6].strip() != "ATOM":
                continue
            bead_type = line[12:16].strip()
            resname = line[17:20].strip()
            chain = line[21]
            resid = int(line[22:26])
            x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
            idx = len(beads)
            beads.append({
                "chain": chain, "resid": resid, "bead": bead_type,
                "resname": resname, "x": x, "y": y, "z": z,
            })
            lookup[(chain, resid, bead_type)] = idx
    return beads, lookup


def load_sequences(path):
    seqs = {}
    chain = None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                chain = line.split()[2]
            else:
                seqs[chain] = line
    return seqs


def xyz(beads, idx):
    b = beads[idx]
    return (b["x"], b["y"], b["z"])


# ─────────────────────────────────────────────────────────────────────
# BONDS  (Eq. 2)
# ─────────────────────────────────────────────────────────────────────
def build_bonds(beads, lookup, sequences, chain_order):
    bonds = []  # (i, j, r0, k_r, type_label)
    for ch in chain_order:
        if ch not in sequences:
            continue
        seq = sequences[ch]
        resids = sorted({b["resid"] for b in beads if b["chain"] == ch})
        for n, resid in enumerate(resids):
            base = seq[n]
            s_idx = lookup.get((ch, resid, "S"))
            b_idx = lookup.get((ch, resid, "B"))
            p_idx = lookup.get((ch, resid, "P"))

            # S(i) - Base(i)
            btype = "S" + base
            r0 = P.BOND_PARAMS[btype]["r0"]
            k_r = P.BOND_PARAMS[btype]["k_r"]
            bonds.append((s_idx, b_idx, r0, k_r, btype))

            # P(i) - S(i)  ("PS" bond)
            if p_idx is not None:
                r0 = P.BOND_PARAMS["PS"]["r0"]
                k_r = P.BOND_PARAMS["PS"]["k_r"]
                bonds.append((p_idx, s_idx, r0, k_r, "PS"))

            # S(i) - P(i+1)  ("SP" bond)
            if n + 1 < len(resids):
                p_next = lookup.get((ch, resids[n + 1], "P"))
                if p_next is not None:
                    r0 = P.BOND_PARAMS["SP"]["r0"]
                    k_r = P.BOND_PARAMS["SP"]["k_r"]
                    bonds.append((s_idx, p_next, r0, k_r, "SP"))
    return bonds


# ─────────────────────────────────────────────────────────────────────
# ANGLES  (Eq. 3)
# ─────────────────────────────────────────────────────────────────────
def build_angles(beads, lookup, sequences, chain_order):
    angles = []  # (i, j, k, alpha0_rad, k_alpha, type_label)
    for ch in chain_order:
        if ch not in sequences:
            continue
        seq = sequences[ch]
        resids = sorted({b["resid"] for b in beads if b["chain"] == ch})
        for n, resid in enumerate(resids):
            base = seq[n]
            s_idx = lookup.get((ch, resid, "S"))
            b_idx = lookup.get((ch, resid, "B"))
            p_idx = lookup.get((ch, resid, "P"))
            p_next = lookup.get((ch, resids[n + 1], "P")) if n + 1 < len(resids) else None
            s_next = lookup.get((ch, resids[n + 1], "S")) if n + 1 < len(resids) else None

            # P(i)-S(i)-Base(i)   -> "PS" + base
            if p_idx is not None:
                key = "PS" + base
                a0 = np.deg2rad(P.ANGLE_PARAMS[key]["alpha0"])
                ka = P.ANGLE_PARAMS[key]["k_alpha"]
                angles.append((p_idx, s_idx, b_idx, a0, ka, key))

            # Base(i)-S(i)-P(i+1) -> base + "SP"
            if p_next is not None:
                key = base + "SP"
                a0 = np.deg2rad(P.ANGLE_PARAMS[key]["alpha0"])
                ka = P.ANGLE_PARAMS[key]["k_alpha"]
                angles.append((b_idx, s_idx, p_next, a0, ka, key))

            # P(i)-S(i)-P(i+1)   -> "PSP"
            if p_idx is not None and p_next is not None:
                a0 = np.deg2rad(P.ANGLE_PARAMS["PSP"]["alpha0"])
                ka = P.ANGLE_PARAMS["PSP"]["k_alpha"]
                angles.append((p_idx, s_idx, p_next, a0, ka, "PSP"))

            # S(i)-P(i+1)-S(i+1) -> "SPS"
            if p_next is not None and s_next is not None:
                a0 = np.deg2rad(P.ANGLE_PARAMS["SPS"]["alpha0"])
                ka = P.ANGLE_PARAMS["SPS"]["k_alpha"]
                angles.append((s_idx, p_next, s_next, a0, ka, "SPS"))
    return angles


# ─────────────────────────────────────────────────────────────────────
# STACKING (Eq. 9) -- consecutive nucleotides i, i+1 on the same chain.
#
# *** IMPORTANT: these quadruplets must match whatever the MD driver's
# StackForce CustomCompoundBondForce expression actually computes, or
# you'll silently enforce the wrong equilibrium value on the wrong
# dihedral. This function measures the BACKBONE-ONLY dihedrals used by
# dnamd.py's StackForce (the same functional form/atom pattern as the
# validated RNA pipeline, which the DNA paper states it shares):
#
#   l0     = dist(B_i, B_j)                         [B_i,B_j stacked bases]
#   phi1_0 = dihedral(P_i, S_i, P_j, S_j)            [backbone P-S-P-S]
#   phi2_0 = dihedral(S_i, P_j, S_j, P_k)            [backbone S-P-S-P,
#                                                      P_k = phosphate of
#                                                      the residue AFTER j]
#
# (An earlier version of this function measured base-centric dihedrals,
#  dihedral(S_i,B_i,B_j,S_j) and dihedral(P_i,S_i,B_i,B_j) -- those do
#  NOT correspond to what dnamd.py's StackForce evaluates, and mixing
#  the two silently biases the stacked-state geometry. Fixed here.)
#
# At the 3' end, where P_j or P_k don't exist, this step's stacking
# interaction is simply not defined (there aren't enough backbone atoms
# for phi1/phi2) -- matches dnamd.py's StackForce, which needs beads
# out to P_(i+2) for every stacked step and therefore only adds a
# stacking bond for steps that have that atom available.
# ─────────────────────────────────────────────────────────────────────
def build_stacking(beads, lookup, sequences, chain_order):
    stacks = []
    for ch in chain_order:
        if ch not in sequences:
            continue
        seq = sequences[ch]
        resids = sorted({b["resid"] for b in beads if b["chain"] == ch})
        for n in range(len(resids) - 1):
            resid_i, resid_j = resids[n], resids[n + 1]
            base_i, base_j = seq[n], seq[n + 1]
            s_i = lookup[(ch, resid_i, "S")]
            b_i = lookup[(ch, resid_i, "B")]
            s_j = lookup[(ch, resid_j, "S")]
            b_j = lookup[(ch, resid_j, "B")]

            p_i = lookup.get((ch, resid_i, "P"))
            p_j = lookup.get((ch, resid_j, "P"))
            p_k = lookup.get((ch, resids[n + 2], "P")) if n + 2 < len(resids) else None

            if p_i is None or p_j is None or p_k is None:
                # not enough backbone atoms to define phi1/phi2 for this
                # step (5' terminus has no P_i; 3'-most steps have no P_k)
                continue

            l0 = _dist(xyz(beads, b_i), xyz(beads, b_j))
            phi1_0 = _dihedral(xyz(beads, p_i), xyz(beads, s_i),
                                xyz(beads, p_j), xyz(beads, s_j))
            phi2_0 = _dihedral(xyz(beads, s_i), xyz(beads, p_j),
                                xyz(beads, s_j), xyz(beads, p_k))

            h, s, Tm, dG0 = P.get_stacking_params(base_i, base_j)

            stacks.append({
                "s_i": s_i, "b_i": b_i, "b_j": b_j, "s_j": s_j,
                "p_i": p_i, "p_j": p_j, "p_k": p_k,
                "l0": l0, "phi1_0": phi1_0, "phi2_0": phi2_0,
                "h": h, "s": s, "Tm": Tm, "dG0": dG0,
                "step": f"{base_i}{base_j}",
            })
    return stacks


# ─────────────────────────────────────────────────────────────────────
# WATSON-CRICK PAIR DETECTION + HYDROGEN BONDING (Eq. 13)
# ─────────────────────────────────────────────────────────────────────
COMPLEMENT = {"A": "T", "T": "A", "G": "C", "C": "G"}


def find_wc_pairs(beads, lookup, sequences, chain_order, cutoff=6.5):
    """
    Auto-detect Watson-Crick pairs: for each Base bead, find the
    nearest Base bead on a DIFFERENT chain with complementary identity
    and distance < cutoff (Angstrom). Returns list of
    (chain1, resid1, chain2, resid2).
    """
    base_beads = [(i, b) for i, b in enumerate(beads) if b["bead"] == "B"]
    pairs = []
    used = set()
    for i, bi in base_beads:
        if (bi["chain"], bi["resid"]) in used:
            continue
        target_base = COMPLEMENT.get(
            sequences[bi["chain"]][
                sorted({b["resid"] for b in beads if b["chain"] == bi["chain"]}).index(bi["resid"])
            ]
        )
        best = None
        best_d = cutoff
        for j, bj in base_beads:
            if bj["chain"] == bi["chain"]:
                continue
            if (bj["chain"], bj["resid"]) in used:
                continue
            seq_j = sequences[bj["chain"]]
            resid_list_j = sorted({b["resid"] for b in beads if b["chain"] == bj["chain"]})
            base_j = seq_j[resid_list_j.index(bj["resid"])]
            if base_j != target_base:
                continue
            d = _dist((bi["x"], bi["y"], bi["z"]), (bj["x"], bj["y"], bj["z"]))
            if d < best_d:
                best_d = d
                best = bj
        if best is not None:
            pairs.append((bi["chain"], bi["resid"], best["chain"], best["resid"]))
            used.add((bi["chain"], bi["resid"]))
            used.add((best["chain"], best["resid"]))
    return pairs


def build_hbonds(beads, lookup, sequences, chain_order, wc_pairs):
    """
    Builds the geometric parameters needed for Eq. 13 for each WC pair.
    Reference geometry (d0, theta1_0, theta2_0, psi1_0, psi2_0, psi3_0)
    is measured from the input structure, per Fig. 3 definitions:
        d      = dist(B1, B5)                       [B1,B5 = the two paired bases]
        theta1 = angle(S1, B1, B5)
        theta2 = angle(S5, B5, B1)
        psi1   = dihedral(S1, B1, B5, S5)
        psi2   = dihedral(P6, S5, B5, B1)   (P of the residue AFTER base5)
        psi3   = dihedral(P2, S1, B1, B5)   (P of the residue AFTER base1)
    """
    hbonds = []
    for (ch1, resid1, ch5, resid5) in wc_pairs:
        s1 = lookup[(ch1, resid1, "S")]
        b1 = lookup[(ch1, resid1, "B")]
        s5 = lookup[(ch5, resid5, "S")]
        b5 = lookup[(ch5, resid5, "B")]

        resids1 = sorted({b["resid"] for b in beads if b["chain"] == ch1})
        resids5 = sorted({b["resid"] for b in beads if b["chain"] == ch5})
        seq1 = sequences[ch1][resids1.index(resid1)]
        seq5 = sequences[ch5][resids5.index(resid5)]
        pair_key = seq1 + seq5
        if pair_key not in P.HB_MULTIPLICITY:
            continue  # not a canonical WC pair; skip

        d0 = _dist(xyz(beads, b1), xyz(beads, b5))
        theta1_0 = _angle(xyz(beads, s1), xyz(beads, b1), xyz(beads, b5))
        theta2_0 = _angle(xyz(beads, s5), xyz(beads, b5), xyz(beads, b1))
        psi1_0 = _dihedral(xyz(beads, s1), xyz(beads, b1), xyz(beads, b5), xyz(beads, s5))

        # P after residue5 / residue1 (may not exist at 3' chain ends)
        idx5 = resids5.index(resid5)
        idx1 = resids1.index(resid1)
        p6 = lookup.get((ch5, resids5[idx5 + 1], "P")) if idx5 + 1 < len(resids5) else None
        p2 = lookup.get((ch1, resids1[idx1 + 1], "P")) if idx1 + 1 < len(resids1) else None

        psi2_0 = (_dihedral(xyz(beads, p6), xyz(beads, s5), xyz(beads, b5), xyz(beads, b1))
                  if p6 is not None else psi1_0)
        psi3_0 = (_dihedral(xyz(beads, p2), xyz(beads, s1), xyz(beads, b1), xyz(beads, b5))
                  if p2 is not None else psi1_0)

        mult = P.HB_MULTIPLICITY[pair_key]
        hbonds.append({
            "s1": s1, "b1": b1, "b5": b5, "s5": s5, "p6": p6, "p2": p2,
            "d0": d0, "theta1_0": theta1_0, "theta2_0": theta2_0,
            "psi1_0": psi1_0, "psi2_0": psi2_0, "psi3_0": psi3_0,
            "UHB0": P.UHB0 * mult, "pair": pair_key,
        })
    return hbonds


# ─────────────────────────────────────────────────────────────────────
# PHOSPHATE CHARGES
# ─────────────────────────────────────────────────────────────────────
def build_phosphate_charges(beads, T_kelvin):
    q = P.renormalized_phosphate_charge(T_kelvin)
    charges = []
    for i, b in enumerate(beads):
        if b["bead"] == "P":
            charges.append((i, q))
    return charges, q


# ─────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────
def write_bonds(bonds, path):
    with open(path, "w") as f:
        f.write("# i  j  r0(A)  k_r(kcal/mol/A^2)  type\n")
        for i, j, r0, kr, t in bonds:
            f.write(f"{i} {j} {r0:.4f} {kr:.4f} {t}\n")


def write_angles(angles, path):
    with open(path, "w") as f:
        f.write("# i  j  k  alpha0(rad)  k_alpha(kcal/mol/rad^2)  type\n")
        for i, j, k, a0, ka, t in angles:
            f.write(f"{i} {j} {k} {a0:.6f} {ka:.4f} {t}\n")


def write_stacking(stacks, path):
    # Columns give every bead needed to reconstruct dnamd.py's 7-particle
    # StackForce window (p_i, s_i, b_i, p_j, s_j, b_j, p_k) directly --
    # no index arithmetic (3*j+2 etc.) needed on the MD-driver side.
    # Tm (Table 2) is now included so U0_S(T) = -h + kB(T-Tm)*s can be
    # computed correctly at any simulation temperature, not just 331.9K.
    with open(path, "w") as f:
        f.write("# p_i s_i b_i p_j s_j b_j p_k l0(A) phi1_0(rad) phi2_0(rad) h(kcal/mol) s Tm(K) dG0(kcal/mol) step\n")
        for st in stacks:
            f.write(f"{st['p_i']} {st['s_i']} {st['b_i']} {st['p_j']} {st['s_j']} {st['b_j']} {st['p_k']} "
                    f"{st['l0']:.4f} {st['phi1_0']:.6f} {st['phi2_0']:.6f} "
                    f"{st['h']:.4f} {st['s']:.4f} {st['Tm']:.2f} {st['dG0']:.4f} {st['step']}\n")


def write_hbonds(hbonds, path):
    with open(path, "w") as f:
        f.write("# s1 b1 b5 s5 p6 p2 d0(A) theta1_0 theta2_0 psi1_0 psi2_0 psi3_0 UHB0(kcal/mol) pair\n")
        for hb in hbonds:
            p6 = hb["p6"] if hb["p6"] is not None else -1
            p2 = hb["p2"] if hb["p2"] is not None else -1
            f.write(f"{hb['s1']} {hb['b1']} {hb['b5']} {hb['s5']} {p6} {p2} "
                    f"{hb['d0']:.4f} {hb['theta1_0']:.6f} {hb['theta2_0']:.6f} "
                    f"{hb['psi1_0']:.6f} {hb['psi2_0']:.6f} {hb['psi3_0']:.6f} "
                    f"{hb['UHB0']:.4f} {hb['pair']}\n")


def write_charges(charges, path):
    with open(path, "w") as f:
        f.write("# bead_idx  charge(e)\n")
        for i, q in charges:
            f.write(f"{i} {q:.4f}\n")


def main():
    config = json.load(open("input.json"))
    cg_pdb_dna = config.get("cg_pdb_dna", "dna_tis.pdb")
    seq_file = config.get("dna_sequence_file", "dna_sequences.txt")
    dna_chains = config["dna_chains"]
    T_kelvin = config.get("Temp", 300.0)

    print(f"Loading TIS-DNA structure if i <(nbeads-5): from {cg_pdb_dna} ...")
    beads, lookup = load_tis_pdb(cg_pdb_dna)
    sequences = load_sequences(seq_file)
    print(f"  {len(beads)} DNA beads loaded, chains: {list(sequences.keys())}")

    print("NOTE: equilibrium bond/angle/stacking/H-bond geometry is measured "
          "from THIS structure. For a faithful reproduction of the published "
          "TIS-DNA parametrization, run AA2TIS_dna.py on an idealized B-DNA "
          "structure (fd_helix.py, 'abdna'/'lbdna') rather than a bent, "
          "nucleosome-wrapped crystal structure.")

    bonds = build_bonds(beads, lookup, sequences, dna_chains)
    angles = build_angles(beads, lookup, sequences, dna_chains)
    stacks = build_stacking(beads, lookup, sequences, dna_chains)
    wc_pairs = find_wc_pairs(beads, lookup, sequences, dna_chains)
    hbonds = build_hbonds(beads, lookup, sequences, dna_chains, wc_pairs)
    charges, q_phos = build_phosphate_charges(beads, T_kelvin)

    write_bonds(bonds, "dna_bonds.dat")
    write_angles(angles, "dna_angles.dat")
    write_stacking(stacks, "dna_stacking.dat")
    write_hbonds(hbonds, "dna_hbonds.dat")
    write_charges(charges, "dna_charges.dat")

    print(f"\n  {len(bonds)} bonds       -> dna_bonds.dat")
    print(f"  {len(angles)} angles      -> dna_angles.dat")
    print(f"  {len(stacks)} stacks      -> dna_stacking.dat")
    print(f"  {len(wc_pairs)} WC pairs found; {len(hbonds)} canonical H-bond "
          f"interactions -> dna_hbonds.dat")
    print(f"  {len(charges)} phosphate charges (q = {q_phos:.3f} e at "
          f"T={T_kelvin} K) -> dna_charges.dat")


if __name__ == "__main__":
    main()