"""
sop_protein_dna_openmm.py
==========================
Coupled simulation of the SOP-CG histone protein core (+ tails) with a
sequence-dependent TIS-DNA polymer (Chakraborty, Hori & Thirumalai,
JCTC 2018), in explicit monovalent/divalent salt.

This extends your original sop_hpc_openmm.py (protein-only) by adding:

  DNA INTERNAL FORCES (TIS-DNA, all from dna_tis_params.py / the
  topology files produced by build_dna_topology.py):
    - harmonic S-P / P-S / S-Base bonds                         (Eq. 2)
    - harmonic P-S-P / S-P-S / P-S-Base / Base-S-P angles        (Eq. 3)
    - single-stranded stacking (rational functional form)        (Eq. 9)
    - Watson-Crick hydrogen bonding (rational functional form)   (Eq. 13)
    - WCA excluded volume between all non-bonded DNA beads        (Eq. 8)
    - Debye-Hueckel electrostatics on phosphate beads with
      Oosawa-Manning renormalized charge                    (Eq. 14-17)

  PROTEIN-DNA COUPLING (Reddy & Thirumalai, SOP nucleosome model,
  Table S1 of the NAR 2021 supplement you attached):
    - generic excluded volume, sigma_HPC-DNA = 5.4 A, eps_l = 1.0 kcal/mol
    - OPTIONAL native contacts (12-6 LJ, r0 = crystal distance,
      eps_h = 1.18 kcal/mol) IF protein_dna_native_contacts.dat exists
      (built by build_protein_dna_contacts.py; only meaningful if the
      DNA TIS structure and the protein CG structure came from the
      SAME original all-atom crystal, e.g. 2CV5)
    - screened electrostatics between charged protein residues (tails)
      and DNA phosphates, using the SAME low dielectric (eps=10) that
      Reddy & Thirumalai use for protein-protein charge pairs (as
      opposed to the eps=80/Manning-charge convention the TIS-DNA
      paper uses for DNA-DNA electrostatics -- this is a genuine,
      documented difference in convention between the two source
      papers, and we honor each paper's own choice for its own
      "half" of the interactions, using the protein convention for the
      protein-DNA cross term exactly as instructed).

  PROTEIN FORCES: unchanged from sop_hpc_openmm.py (FENE, excluded
  volume, native contacts, Coulomb with explicit ions).

REQUIRED UPSTREAM FILES (see the companion scripts):
    AA2CG.py                     -> histone_cg.pdb           (protein core)
    add_tails.py                 -> histone_cg_with_tails.pdb, charged_beads.dat
    build_contacts.py            -> native_contacts.dat      (protein-protein)
    AA2TIS_dna.py                -> dna_tis.pdb, dna_sequences.txt
    build_dna_topology.py        -> dna_bonds.dat, dna_angles.dat,
                                     dna_stacking.dat, dna_hbonds.dat,
                                     dna_charges.dat
    build_protein_dna_contacts.py (OPTIONAL) -> protein_dna_native_contacts.dat
"""

import json, sys, os
import numpy as np
from openmm import *
from openmm.app import *
import openmm.unit as unit
from openmm.app import PDBxFile

import dna_tis_params as DP

# ─────────────────────────────────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────────────────────────────────
cfg = json.load(open("input.json"))

# Protein files
protein_pdb_core   = cfg["cg_pdb"]
protein_pdb_tails   = cfg["cg_pdb_with_tails"]
native_contacts_file = cfg["native_contacts_file"]
charged_beads_file  = cfg["charged_beads_file"]
chain_order         = cfg["histone_chains"]

# DNA files
dna_pdb          = cfg.get("cg_pdb_dna", "dna_tis.pdb")
dna_seq_file     = cfg.get("dna_sequence_file", "dna_sequences.txt")
dna_bonds_file   = cfg.get("dna_bonds_file", "dna_bonds.dat")
dna_angles_file  = cfg.get("dna_angles_file", "dna_angles.dat")
dna_stack_file   = cfg.get("dna_stacking_file", "dna_stacking.dat")
dna_hbond_file   = cfg.get("dna_hbonds_file", "dna_hbonds.dat")
dna_charge_file  = cfg.get("dna_charges_file", "dna_charges.dat")
protein_dna_native_file = "protein_dna_native_contacts.dat"

# Output
dcd_out   = cfg["dcd_prefix"] + ".dcd"
log_out   = cfg["data_name"]
chk_equil = cfg["checkpoint_name"]
chk_prod  = chk_equil.replace(".chk", "_prod_final.chk")
final_pdb = cfg["final_state_name"]

# Protein SOP forcefield (Reddy & Thirumalai)
K_FENE          = cfg["fene_k"]
R0_FENE         = cfg["fene_R0"]
SIGMA_PROTEIN   = cfg["sigma_protein"]
EPS_H_NATIVE    = cfg["eps_h_native"]
EPS_L_NONNATIVE = cfg["eps_local"]

DIELECTRIC_PROTEIN = cfg["dielectric_protein"]   # eps=10, used for protein-protein
                                                  # AND protein-DNA (Reddy convention)
DIELECTRIC_WATER   = cfg["dielectric_water"]     # eps=80 baseline (ions, DNA-DNA)

# Protein-DNA coupling (Reddy & Thirumalai Table S1)
SIGMA_HPC_DNA = cfg.get("sigma_hpc_dna", 5.4)     # Angstrom
EPS_H_HPC_DNA = cfg.get("protein_dna_eps_h", 1.18)  # kcal/mol (native only)

# Explicit ions
KCl_mM, MgCl2_mM = cfg["KCl_mM"], cfg["MgCl2_mM"]
SIGMA_K, SIGMA_CL, SIGMA_Mg = cfg["rK"], cfg["rCl"], cfg["rMg"]

# Simulation
TEMPERATURE   = cfg["Temp"]
FRICTION      = cfg["friction_coeff"]
TIMESTEP_PS   = cfg["time_step"]
N_STEPS_EQUIL = cfg["numsteps_equil"]
N_STEPS_PROD  = cfg["numsteps_prod"]
REPORT_EVERY  = cfg["data_interval"]
BOX_PADDING   = cfg["box_padding"]
BOX_SCALE     = cfg["box_scale_factor"]
DO_MINIMISE   = cfg["minimization"]
MIN_MAXITER   = cfg["minimization_maxiter"]
MIN_TOL       = cfg["minimization_tol"]
PLATFORM      = cfg["platform_type"]
RESTART       = cfg["restart"]
DO_CMR        = cfg["ComMotionRemover"]
CMR_FREQ      = cfg["CMMR_frequency"]
PME_CUTOFF    = cfg["pme_cutoff"] * 0.1  # Angstrom -> nm


def kcal_to_kJ(x): return x * 4.184
def A_to_nm(x):    return x * 0.1
def deg_to_rad(x): return np.deg2rad(x)


ONE_4PI_EPS0 = 138.935458  # kJ*nm/(mol*e^2)

tcent = TEMPERATURE - 273.15
eps_water = 87.740 - 0.40008*tcent + 9.398e-4*tcent**2 - 1.410e-6*tcent**3
dielectric_sqrt = np.sqrt(DIELECTRIC_WATER)
print(f"T = {TEMPERATURE} K, water dielectric eps(T) = {eps_water:.2f}, using DIELECTRIC_WATER = {DIELECTRIC_WATER}")

# ─────────────────────────────────────────────────────────────────────
# LOAD PROTEIN STRUCTURE (dynamic chain/tail sizing, no hardcoding)
# ─────────────────────────────────────────────────────────────────────
def read_pdb_beads(path):
    beads = []
    with open(path) as f:
        for line in f:
            if line[:6].strip() != "ATOM":
                continue
            beads.append({
                "chain": line[21], "resid": int(line[22:26]),
                "resname": line[17:20].strip(),
                "x": float(line[30:38]), "y": float(line[38:46]), "z": float(line[46:54]),
            })
    return beads

print("Loading protein structures ...")
core_beads  = read_pdb_beads(protein_pdb_core)
beads       = read_pdb_beads(protein_pdb_tails)
N_protein   = len(beads)

def per_chain_count(bead_list):
    counts = {}
    for b in bead_list:
        counts[b["chain"]] = counts.get(b["chain"], 0) + 1
    return counts

old_chain_sizes = per_chain_count(core_beads)
new_chain_sizes = per_chain_count(beads)
tail_sizes = {ch: new_chain_sizes[ch] - old_chain_sizes.get(ch, 0) for ch in chain_order}
print(f"  {N_protein} protein beads ({sum(tail_sizes.values())} tail + "
      f"{sum(old_chain_sizes.values())} core).")

old_offset = 0
old_idx_to_chain_pos = {}
for ch in chain_order:
    for pos in range(old_chain_sizes[ch]):
        old_idx_to_chain_pos[old_offset + pos] = (ch, pos)
    old_offset += old_chain_sizes[ch]

new_chain_start = {}
offset = 0
for ch in chain_order:
    new_chain_start[ch] = offset
    offset += tail_sizes[ch] + old_chain_sizes[ch]

def remap_contact_index(old_idx):
    ch, pos = old_idx_to_chain_pos[old_idx]
    return new_chain_start[ch] + tail_sizes[ch] + pos

native_contacts = []
with open(native_contacts_file) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        i_old, j_old, r0 = line.split()
        native_contacts.append((remap_contact_index(int(i_old)),
                                 remap_contact_index(int(j_old)), float(r0)))
print(f"  {len(native_contacts)} protein-protein native contacts loaded.")

charged_protein = {}
with open(charged_beads_file) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.split()
        charged_protein[int(parts[0])] = float(parts[4])
print(f"  {len(charged_protein)} charged protein beads.")

chain_boundaries = []
off = 0
for ch in chain_order:
    n_ch = tail_sizes[ch] + old_chain_sizes[ch]
    chain_boundaries.append((off, off + n_ch - 1))
    off += n_ch

# ─────────────────────────────────────────────────────────────────────
# LOAD DNA STRUCTURE + TOPOLOGY
# ─────────────────────────────────────────────────────────────────────
print("Loading DNA TIS structure and topology ...")
dna_beads = read_pdb_beads(dna_pdb)
for b, line_bead in zip(dna_beads, dna_beads):
    pass
# bead type (P/S/B) is stored in the atom-name column, re-read separately:
dna_bead_types = []
with open(dna_pdb) as f:
    for line in f:
        if line[:6].strip() != "ATOM":
            continue
        dna_bead_types.append(line[12:16].strip())

N_dna = len(dna_beads)
N_protein_total = N_protein
DNA_OFFSET = N_protein_total   # global index offset for DNA beads
print(f"  {N_dna} DNA TIS beads loaded (global index offset {DNA_OFFSET}).")

def load_table(path, ncols_min):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            rows.append(line.split())
    return rows

dna_bonds_raw   = load_table(dna_bonds_file, 5)
dna_angles_raw  = load_table(dna_angles_file, 6)
dna_stack_raw   = load_table(dna_stack_file, 11)
dna_hbond_raw   = load_table(dna_hbond_file, 13)
dna_charge_raw  = load_table(dna_charge_file, 2)

print(f"  {len(dna_bonds_raw)} DNA bonds, {len(dna_angles_raw)} DNA angles, "
      f"{len(dna_stack_raw)} stacks, {len(dna_hbond_raw)} H-bonds, "
      f"{len(dna_charge_raw)} charged phosphates.")

protein_dna_native = []
if os.path.exists(protein_dna_native_file):
    for row in load_table(protein_dna_native_file, 3):
        protein_dna_native.append((int(row[0]), int(row[1]) + DNA_OFFSET, float(row[2])))
    print(f"  {len(protein_dna_native)} protein-DNA native contacts loaded.")
else:
    print("  No protein-DNA native contacts file found (generic protein "
          "generic-DNA excluded volume + electrostatics only).")

# ─────────────────────────────────────────────────────────────────────
# BUILD TOPOLOGY
# ─────────────────────────────────────────────────────────────────────
print("Building OpenMM topology ...")
topology = Topology()
omm_chains = {ch: topology.addChain(id=ch) for ch in chain_order}
for b in beads:
    res = topology.addResidue(b["resname"], omm_chains[b["chain"]])
    topology.addAtom("BB", Element.getBySymbol("C"), res)

dna_chain_ids = sorted(set(b["chain"] for b in dna_beads))
omm_dna_chains = {ch: topology.addChain(id=ch) for ch in dna_chain_ids}
for b, btype in zip(dna_beads, dna_bead_types):
    res = topology.addResidue(b["resname"], omm_dna_chains[b["chain"]])
    elem = Element.getBySymbol("P") if btype == "P" else Element.getBySymbol("C")
    topology.addAtom(btype, elem, res)

# ─────────────────────────────────────────────────────────────────────
# COMBINED COORDINATES / BOX
# ─────────────────────────────────────────────────────────────────────
prot_coords_nm = np.array([[b["x"], b["y"], b["z"]] for b in beads]) * 0.1
dna_coords_nm  = np.array([[b["x"], b["y"], b["z"]] for b in dna_beads]) * 0.1

# compute protein-only span first, to know how far to push DNA away
prot_min, prot_max = prot_coords_nm.min(0), prot_coords_nm.max(0)
prot_span = prot_max - prot_min

dna_offset_vec = np.array([prot_span[0] + 5.0, 0.0, 0.0])  # clear separation along x
dna_coords_nm = dna_coords_nm + dna_offset_vec

all_solute_coords = np.vstack([prot_coords_nm, dna_coords_nm])

solute_min, solute_max = all_solute_coords.min(0), all_solute_coords.max(0)
solute_span = solute_max - solute_min   # NOW computed on the shifted, combined set

min_box = 2.0 * cfg.get("nonlocal_cutoff", 30.0) * 0.1
box_size = np.maximum(solute_span * BOX_SCALE, min_box * np.ones(3))
centre = 0.5 * box_size
solute_centre = 0.5 * (solute_min + solute_max)
all_solute_coords = all_solute_coords - solute_centre + centre

topology.setPeriodicBoxVectors((Vec3(box_size[0], 0, 0)*unit.nanometer,
                                 Vec3(0, box_size[1], 0)*unit.nanometer,
                                 Vec3(0, 0, box_size[2])*unit.nanometer))

# ─────────────────────────────────────────────────────────────────────
# CHARGES (protein raw + DNA Manning-renormalized) and ION COUNT
# ─────────────────────────────────────────────────────────────────────
dna_charges = {int(r[0]) + DNA_OFFSET: float(r[1]) for r in dna_charge_raw}
net_charge = sum(charged_protein.values()) + sum(dna_charges.values())
print(f"  Net solute charge (protein+DNA): {net_charge:+.1f} e")

N_A = 6.022e23
box_vol_L = float(np.prod(box_size)) * 1e-27 * 1e3
n_kcl = max(0, round(KCl_mM * 1e-3 * N_A * box_vol_L))
n_mgcl2 = max(0, round(MgCl2_mM * 1e-3 * N_A * box_vol_L))
n_k, n_mg = n_kcl, n_mgcl2
n_cl = n_kcl + 2*n_mgcl2
if net_charge > 0:
    n_cl += int(round(net_charge))
else:
    n_k += int(round(-net_charge))
N_ions = n_k + n_cl + n_mg
print(f"  Ions: {n_k} K+, {n_cl} Cl-, {n_mg} Mg2+ = {N_ions} total")

ion_chain = topology.addChain(id="Z")
for _ in range(n_k):
    res = topology.addResidue("K", ion_chain); topology.addAtom("K", Element.getBySymbol("K"), res)
for _ in range(n_cl):
    res = topology.addResidue("CL", ion_chain); topology.addAtom("CL", Element.getBySymbol("Cl"), res)
for _ in range(n_mg):
    res = topology.addResidue("MG", ion_chain); topology.addAtom("MG", Element.getBySymbol("Mg"), res)

print("Placing ions ...")
np.random.seed(cfg.get("random_seed", 42))
ion_positions = []
placed = all_solute_coords.copy()
for _ in range(N_ions):
    for _ in range(50000):
        pos = np.random.rand(3) * box_size
        if np.linalg.norm(placed - pos, axis=1).min() > 0.6:
            ion_positions.append(pos)
            placed = np.vstack([placed, pos])
            break
    else:
        pos = np.random.rand(3) * box_size
        ion_positions.append(pos)
        placed = np.vstack([placed, pos])

all_positions = np.vstack([all_solute_coords, np.array(ion_positions) if N_ions else np.zeros((0, 3))])
positions_openmm = [Vec3(*p) * unit.nanometer for p in all_positions]
N_total = N_protein + N_dna + N_ions
print(f"  Total particles: {N_total}")

ION_K_START  = DNA_OFFSET + N_dna
ION_CL_START = ION_K_START + n_k
ION_MG_START = ION_CL_START + n_cl

# ─────────────────────────────────────────────────────────────────────
# SYSTEM / PARTICLES
# ─────────────────────────────────────────────────────────────────────
print("Building system ...")
system = System()
system.setDefaultPeriodicBoxVectors(Vec3(box_size[0],0,0)*unit.nanometer,
                                     Vec3(0,box_size[1],0)*unit.nanometer,
                                     Vec3(0,0,box_size[2])*unit.nanometer)
for _ in range(N_total):
    system.addParticle(1.0 * unit.amu)

protein_indices = list(range(0, N_protein))
dna_indices = list(range(DNA_OFFSET, DNA_OFFSET + N_dna))

# ═══════════════════════════════════════════════════════════════════
# PROTEIN FORCES  (unchanged physics from sop_hpc_openmm.py)
# ═══════════════════════════════════════════════════════════════════
print("  [protein] FENE bonds ...")
k_fene = kcal_to_kJ(K_FENE) * 100.0
R0_nm = A_to_nm(R0_FENE)
fene_bond = CustomBondForce("-0.5*K*R0*R0*log(1 - ((r-r0)/R0)^2)")
fene_bond.addPerBondParameter("r0"); fene_bond.addPerBondParameter("R0")
fene_bond.addPerBondParameter("K")
fene_bond.setUsesPeriodicBoundaryConditions(True)

bond_pairs = set()
for ch_start, ch_end in chain_boundaries:
    for i in range(ch_start, ch_end):
        j = i + 1
        r0 = float(np.linalg.norm(all_positions[j] - all_positions[i]))
        fene_bond.addBond(i, j, [r0, R0_nm, k_fene])
        bond_pairs.add((i, j))
fene_bond.setForceGroup(0)
system.addForce(fene_bond)
print(f"    {fene_bond.getNumBonds()} FENE bonds.")

print("  [protein] excluded volume ...")
ev_protein = CustomNonbondedForce("eps_l*(sigma_p/max(r,0.15))^6")
ev_protein.addGlobalParameter("eps_l", kcal_to_kJ(EPS_L_NONNATIVE))
ev_protein.addGlobalParameter("sigma_p", A_to_nm(SIGMA_PROTEIN))
ev_protein.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
ev_protein.setCutoffDistance(PME_CUTOFF * unit.nanometer)
for _ in range(N_total):
    ev_protein.addParticle([])

native_contact_set = set((min(i,j), max(i,j)) for i, j, _ in native_contacts)
protein_master_exclusions = set()
for i, j in bond_pairs:
    protein_master_exclusions.add((min(i,j), max(i,j)))
for ch_start, ch_end in chain_boundaries:
    for i in range(ch_start, ch_end + 1):
        for j in range(i+1, min(i+3, ch_end+1)):
            protein_master_exclusions.add((i, j))
for i, j, _ in native_contacts:
    protein_master_exclusions.add((min(i,j), max(i,j)))

ev_protein.addInteractionGroup(protein_indices, protein_indices)
for i, j in protein_master_exclusions:
    ev_protein.addExclusion(i, j)
ev_protein.setForceGroup(1)
system.addForce(ev_protein)

print("  [protein] native contacts ...")
native_force = CustomBondForce("eps_h*((r0/r)^12 - 2*(r0/r)^6)")
native_force.addPerBondParameter("r0"); native_force.addPerBondParameter("eps_h")
native_force.setUsesPeriodicBoundaryConditions(True)
eps_h_kJ = kcal_to_kJ(EPS_H_NATIVE)
for i, j, r0_A in native_contacts:
    native_force.addBond(i, j, [A_to_nm(r0_A), eps_h_kJ])
native_force.setForceGroup(2)
system.addForce(native_force)
print(f"    {native_force.getNumBonds()} native contacts.")

# ═══════════════════════════════════════════════════════════════════
# DNA INTERNAL FORCES  (TIS-DNA, Chakraborty/Hori/Thirumalai)
# ═══════════════════════════════════════════════════════════════════
print("  [DNA] harmonic bonds ...")
dna_bond_force = CustomBondForce("k*(r-r0)^2")
dna_bond_force.addPerBondParameter("r0"); dna_bond_force.addPerBondParameter("k")
dna_bond_force.setUsesPeriodicBoundaryConditions(True)
dna_bond_pairs = set()
for row in dna_bonds_raw:
    i, j = int(row[0]) + DNA_OFFSET, int(row[1]) + DNA_OFFSET
    r0, k = A_to_nm(float(row[2])), kcal_to_kJ(float(row[3])) * 100.0
    dna_bond_force.addBond(i, j, [r0, k])
    dna_bond_pairs.add((min(i,j), max(i,j)))
dna_bond_force.setForceGroup(5)
system.addForce(dna_bond_force)
print(f"    {dna_bond_force.getNumBonds()} DNA bonds.")

print("  [DNA] harmonic angles ...")
dna_angle_force = CustomAngleForce("k*(theta-theta0)^2")
dna_angle_force.addPerAngleParameter("theta0"); dna_angle_force.addPerAngleParameter("k")
for row in dna_angles_raw:
    i, j, k_idx = int(row[0])+DNA_OFFSET, int(row[1])+DNA_OFFSET, int(row[2])+DNA_OFFSET
    theta0, k = float(row[3]), kcal_to_kJ(float(row[4]))
    dna_angle_force.addAngle(i, j, k_idx, [theta0, k])
dna_angle_force.setForceGroup(6)
system.addForce(dna_angle_force)
print(f"    {dna_angle_force.getNumAngles()} DNA angles.")

print("  [DNA] stacking ...")
# 5-particle form: p1=S_i p2=B_i p3=B_j p4=S_j p5=P_i
TWOPI = "6.283185307"
stack5 = CustomCompoundBondForce(
    5,
    "U0/max(1 + kl*(l-l0)^2 + kphi*dphi1w^2 + kphi*dphi2w^2, 0.05);"
    "l = distance(p2,p3);"
    f"dphi1w = dphi1 - {TWOPI}*floor(dphi1/{TWOPI} + 0.5);"
    "dphi1 = dihedral(p1,p2,p3,p4) - phi1_0;"
    f"dphi2w = dphi2 - {TWOPI}*floor(dphi2/{TWOPI} + 0.5);"
    "dphi2 = dihedral(p5,p1,p2,p3) - phi2_0"
)
for p in ("U0", "l0", "phi1_0", "phi2_0"):
    stack5.addPerBondParameter(p)
stack5.addGlobalParameter("kl", DP.K_L / 0.01)      # A^-2 -> nm^-2
stack5.addGlobalParameter("kphi", DP.K_PHI)         # rad^-2, no unit conversion

# 4-particle fallback (no P_i, drop the phi2 term): p1=S_i p2=B_i p3=B_j p4=S_j
stack4 = CustomCompoundBondForce(
    4,
    "U0/max(1 + kl*(l-l0)^2 + kphi*dphi1w^2, 0.05);"
    "l = distance(p2,p3);"
    f"dphi1w = dphi1 - {TWOPI}*floor(dphi1/{TWOPI} + 0.5);"
    "dphi1 = dihedral(p1,p2,p3,p4) - phi1_0"
)
for p in ("U0", "l0", "phi1_0"):
    stack4.addPerBondParameter(p)
stack4.addGlobalParameter("kl", DP.K_L / 0.01)
stack4.addGlobalParameter("kphi", DP.K_PHI)

kB = DP.KB_KCAL
n_stack5 = n_stack4 = 0
for row in dna_stack_raw:
    s_i, b_i, b_j, s_j, p_i = [int(x) for x in row[0:5]]
    l0_nm = A_to_nm(float(row[5]))
    phi1_0, phi2_0 = float(row[6]), float(row[7])
    h, s, dG0 = float(row[8]), float(row[9]), float(row[10])
    # U0_S = -h + kB*(T - Tref)*s   (kcal/mol), converted to kJ/mol
    U0_S = kcal_to_kJ(-h + kB * (TEMPERATURE - DP.STACKING_TREF) * s)
    gi = [x + DNA_OFFSET for x in (s_i, b_i, b_j, s_j)]
    if p_i >= 0:
        stack5.addBond(gi + [p_i + DNA_OFFSET], [U0_S, l0_nm, phi1_0, phi2_0])
        n_stack5 += 1
    else:
        stack4.addBond(gi, [U0_S, l0_nm, phi1_0])
        n_stack4 += 1
stack5.setForceGroup(7); stack4.setForceGroup(7)
system.addForce(stack5); system.addForce(stack4)
print(f"    {n_stack5} 5-particle + {n_stack4} 4-particle stacking terms.")

print("  [DNA] Watson-Crick hydrogen bonds ...")
# 6-particle form: p1=S1 p2=B1 p3=B5 p4=S5 p5=P6 p6=P2
hb6 = CustomCompoundBondForce(
    6,
    "UHB0/max(1 + kd*(d-d0)^2 + kth*dth1w^2 + kth*dth2w^2"
    " + kps*dpsi1w^2 + kps*dpsi2w^2 + kps*dpsi3w^2, 0.05);"
    "d = distance(p2,p3);"
    f"dth1w = dth1 - {TWOPI}*floor(dth1/{TWOPI} + 0.5); dth1 = angle(p1,p2,p3) - th1_0;"
    f"dth2w = dth2 - {TWOPI}*floor(dth2/{TWOPI} + 0.5); dth2 = angle(p4,p3,p2) - th2_0;"
    f"dpsi1w = dpsi1 - {TWOPI}*floor(dpsi1/{TWOPI} + 0.5); dpsi1 = dihedral(p1,p2,p3,p4) - psi1_0;"
    f"dpsi2w = dpsi2 - {TWOPI}*floor(dpsi2/{TWOPI} + 0.5); dpsi2 = dihedral(p5,p4,p3,p2) - psi2_0;"
    f"dpsi3w = dpsi3 - {TWOPI}*floor(dpsi3/{TWOPI} + 0.5); dpsi3 = dihedral(p6,p1,p2,p3) - psi3_0"
)
for p in ("UHB0", "d0", "th1_0", "th2_0", "psi1_0", "psi2_0", "psi3_0"):
    hb6.addPerBondParameter(p)
hb6.addGlobalParameter("kd", DP.K_D / 0.01)
hb6.addGlobalParameter("kth", DP.K_THETA)
hb6.addGlobalParameter("kps", DP.K_PSI)

# 4-particle fallback (chain-end pairs missing P6/P2): p1=S1 p2=B1 p3=B5 p4=S5
hb4 = CustomCompoundBondForce(
    4,
    "UHB0/max(1 + kd*(d-d0)^2 + kth*dth1w^2 + kth*dth2w^2 + kps*dpsi1w^2, 0.05);"
    "d = distance(p2,p3);"
    f"dth1w = dth1 - {TWOPI}*floor(dth1/{TWOPI} + 0.5); dth1 = angle(p1,p2,p3) - th1_0;"
    f"dth2w = dth2 - {TWOPI}*floor(dth2/{TWOPI} + 0.5); dth2 = angle(p4,p3,p2) - th2_0;"
    f"dpsi1w = dpsi1 - {TWOPI}*floor(dpsi1/{TWOPI} + 0.5); dpsi1 = dihedral(p1,p2,p3,p4) - psi1_0"
)
for p in ("UHB0", "d0", "th1_0", "th2_0", "psi1_0"):
    hb4.addPerBondParameter(p)
hb4.addGlobalParameter("kd", DP.K_D / 0.01)
hb4.addGlobalParameter("kth", DP.K_THETA)
hb4.addGlobalParameter("kps", DP.K_PSI)

n_hb6 = n_hb4 = 0
dna_hbond_pairs = set()
for row in dna_hbond_raw:
    s1, b1, b5, s5, p6, p2 = [int(x) for x in row[0:6]]
    d0 = A_to_nm(float(row[6]))
    th1_0, th2_0, psi1_0, psi2_0, psi3_0 = [float(x) for x in row[7:12]]
    UHB0_kJ = kcal_to_kJ(float(row[12]))
    gi = [x + DNA_OFFSET for x in (s1, b1, b5, s5)]
    dna_hbond_pairs.add((min(b1,b5)+DNA_OFFSET, max(b1,b5)+DNA_OFFSET))
    if p6 >= 0 and p2 >= 0:
        hb6.addBond(gi + [p6+DNA_OFFSET, p2+DNA_OFFSET],
                    [UHB0_kJ, d0, th1_0, th2_0, psi1_0, psi2_0, psi3_0])
        n_hb6 += 1
    else:
        hb4.addBond(gi, [UHB0_kJ, d0, th1_0, th2_0, psi1_0])
        n_hb4 += 1
hb6.setForceGroup(8); hb4.setForceGroup(8)
system.addForce(hb6); system.addForce(hb4)
print(f"    {n_hb6} 6-particle + {n_hb4} 4-particle H-bond terms.")

print("  [DNA] excluded volume (WCA) ...")
ev_dna = CustomNonbondedForce(
    "step(D0-r) * eps0 * ((D0/r)^12 - 2*(D0/r)^6 + 1)"
)
ev_dna.addGlobalParameter("D0", A_to_nm(DP.EV_D0))
ev_dna.addGlobalParameter("eps0", kcal_to_kJ(DP.EV_EPS0))
ev_dna.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
ev_dna.setCutoffDistance(A_to_nm(DP.EV_D0) * unit.nanometer)
for _ in range(N_total):
    ev_dna.addParticle([])
ev_dna.addInteractionGroup(dna_indices, dna_indices)

# Build DNA bond-graph exclusions (bonded + 1-3), plus stacking base
# pairs and WC H-bonded base pairs (already have their own explicit
# potential, so exclude from generic WCA repulsion).
from collections import defaultdict
adj = defaultdict(set)
for i, j in dna_bond_pairs:
    adj[i].add(j); adj[j].add(i)
dna_master_exclusions = set(dna_bond_pairs)
for i in adj:
    for j in adj[i]:
        for k in adj[j]:
            if k != i:
                dna_master_exclusions.add((min(i,k), max(i,k)))
for pair in dna_hbond_pairs:
    dna_master_exclusions.add((min(pair), max(pair)))
for i, j in dna_master_exclusions:
    ev_dna.addExclusion(i, j)
ev_dna.setForceGroup(9)
system.addForce(ev_dna)
print(f"    {len(dna_master_exclusions)} DNA EV exclusions applied.")

# ═══════════════════════════════════════════════════════════════════
# PROTEIN-DNA COUPLING  (Reddy & Thirumalai, Table S1)
# ═══════════════════════════════════════════════════════════════════
print("  [protein-DNA] generic excluded volume (sigma=5.4 A) ...")
ev_pd = CustomNonbondedForce("eps_l*(sigma_pd/max(r,0.15))^6")
ev_pd.addGlobalParameter("eps_l", kcal_to_kJ(EPS_L_NONNATIVE))
ev_pd.addGlobalParameter("sigma_pd", A_to_nm(SIGMA_HPC_DNA))
ev_pd.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
ev_pd.setCutoffDistance(PME_CUTOFF * unit.nanometer)
for _ in range(N_total):
    ev_pd.addParticle([])
ev_pd.addInteractionGroup(protein_indices, dna_indices)
pd_native_set = set((min(i,j), max(i,j)) for i, j, _ in protein_dna_native)
for i, j in pd_native_set:
    ev_pd.addExclusion(i, j)
ev_pd.setForceGroup(10)
system.addForce(ev_pd)
print(f"ev_pd interaction group sizes: protein={len(protein_indices)}, dna={len(dna_indices)}")

if protein_dna_native:
    print("  [protein-DNA] native contacts (eps_h=1.18 kcal/mol) ...")
    pd_native_force = CustomBondForce("eps_h*((r0/r)^12 - 2*(r0/r)^6)")
    pd_native_force.addPerBondParameter("r0"); pd_native_force.addPerBondParameter("eps_h")
    pd_native_force.setUsesPeriodicBoundaryConditions(True)
    eps_h_pd_kJ = kcal_to_kJ(EPS_H_HPC_DNA)
    for i, j, r0_A in protein_dna_native:
        pd_native_force.addBond(i, j, [A_to_nm(r0_A), eps_h_pd_kJ])
    pd_native_force.setForceGroup(11)
    system.addForce(pd_native_force)
    print(f"    {pd_native_force.getNumBonds()} protein-DNA native contacts.")

# ═══════════════════════════════════════════════════════════════════
# ELECTROSTATICS: protein charges + DNA phosphate charges + explicit ions
# ═══════════════════════════════════════════════════════════════════
print("  Electrostatics (explicit ions, PME baseline eps=water) ...")
nb_force = NonbondedForce()
nb_force.setNonbondedMethod(NonbondedForce.PME)
nb_force.setCutoffDistance(PME_CUTOFF * unit.nanometer)
nb_force.setEwaldErrorTolerance(1e-4)
nb_force.setReactionFieldDielectric(DIELECTRIC_WATER)

for i in range(N_protein):
    q = charged_protein.get(i, 0.0) / dielectric_sqrt
    nb_force.addParticle(q, A_to_nm(SIGMA_PROTEIN)*unit.nanometer, 0.0)
for i in range(DNA_OFFSET, DNA_OFFSET + N_dna):
    q = dna_charges.get(i, 0.0) / dielectric_sqrt
    nb_force.addParticle(q, A_to_nm(DP.EV_D0)*unit.nanometer, 0.0)
for _ in range(n_k):
    nb_force.addParticle(+1.0/dielectric_sqrt, A_to_nm(SIGMA_K)*unit.nanometer, 0.0)
for _ in range(n_cl):
    nb_force.addParticle(-1.0/dielectric_sqrt, A_to_nm(SIGMA_CL)*unit.nanometer, 0.0)
for _ in range(n_mg):
    nb_force.addParticle(+2.0/dielectric_sqrt, A_to_nm(SIGMA_Mg)*unit.nanometer, 0.0)

# Exceptions: zero out Coulomb for bonded/native/EV-excluded pairs that
# already have explicit potentials (protein, DNA, and cross native).
all_zero_exceptions = set()
all_zero_exceptions |= protein_master_exclusions
all_zero_exceptions |= dna_master_exclusions
all_zero_exceptions |= pd_native_set
for i, j in all_zero_exceptions:
    nb_force.addException(i, j, 0.0, 1.0, 0.0)
nb_force.setForceGroup(3)
system.addForce(nb_force)

# Dielectric correction: protein-protein AND protein-DNA charged pairs
# use eps=10 (Reddy & Thirumalai convention) instead of the eps=80
# baseline above; DNA-DNA phosphate pairs are left at eps=80 (TIS-DNA
# convention, with Manning-reduced charge already applied).
corr_factor = ONE_4PI_EPS0 * (1.0/DIELECTRIC_PROTEIN - 1.0/DIELECTRIC_WATER)
coul_corr = CustomBondForce(f"{corr_factor} * qi * qj / max(r, 0.3)")
coul_corr.addPerBondParameter("qi"); coul_corr.addPerBondParameter("qj")
coul_corr.setUsesPeriodicBoundaryConditions(True)

charged_prot_idx = sorted(charged_protein.keys())
for a in range(len(charged_prot_idx)):
    for b_ in range(a+1, len(charged_prot_idx)):
        i, j = charged_prot_idx[a], charged_prot_idx[b_]
        pair = (min(i,j), max(i,j))
        if pair in bond_pairs or pair in protein_master_exclusions:
            continue
        coul_corr.addBond(i, j, [charged_protein[i], charged_protein[j]])

charged_dna_idx = sorted(dna_charges.keys())
for i in charged_prot_idx:
    for j in charged_dna_idx:
        pair = (min(i,j), max(i,j))
        if pair in pd_native_set:
            continue
        coul_corr.addBond(i, j, [charged_protein[i], dna_charges[j]])

coul_corr.setForceGroup(4)
system.addForce(coul_corr)
print(f"    {coul_corr.getNumBonds()} dielectric-correction (eps=10) pairs "
      f"(protein-protein + protein-DNA).")

if DO_CMR:
    system.addForce(CMMotionRemover(CMR_FREQ))

from scipy.spatial import cKDTree
tree = cKDTree(all_positions, boxsize=box_size)
pairs = tree.query_pairs(r=0.15)  # nm, i.e. < 1.5 A apart
print(f"Found {len(pairs)} pairs closer than 1.5 A:")
for i, j in list(pairs)[:20]:
    print(f"  {i} - {j}: {np.linalg.norm(all_positions[i]-all_positions[j])*10:.3f} A")

# ─────────────────────────────────────────────────────────────────────
# INTEGRATOR / SIMULATION
# ─────────────────────────────────────────────────────────────────────
integrator = LangevinMiddleIntegrator(TEMPERATURE*unit.kelvin,
                                       FRICTION/unit.picosecond,
                                       TIMESTEP_PS*unit.picoseconds)
platform = Platform.getPlatformByName(PLATFORM)
properties = {"Precision": "double"}
simulation = Simulation(topology, system, integrator, platform, properties)

FORCE_GROUPS = [
    (0, "Protein FENE"), (1, "Protein EV"), (2, "Protein native"),
    (3, "PME (all charges, eps=water)"), (4, "Coulomb corr. (eps=10)"),
    (5, "DNA bonds"), (6, "DNA angles"), (7, "DNA stacking"),
    (8, "DNA H-bonds"), (9, "DNA EV (WCA)"), (10, "Protein-DNA EV"),
    (11, "Protein-DNA native"),
]

def print_energy_breakdown(label):
    print(f"\nEnergy breakdown ({label}):")
    for group, name in FORCE_GROUPS:
        try:
            e = simulation.context.getState(getEnergy=True, groups={group}
                    ).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
            print(f"  {name:32s}: {e:12.2f} kcal/mol")
        except Exception:
            pass

print_energy_breakdown("BEFORE minimization")   # add this line right after building system, before minimizeEnergy

if RESTART:
    simulation.loadCheckpoint(chk_equil)
else:
    simulation.context.setPositions(positions_openmm)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE*unit.kelvin)


for group, name in FORCE_GROUPS:
    e = simulation.context.getState(getEnergy=True, groups={group}).getPotentialEnergy()
    print(f"{name}: {e}")

if DO_MINIMISE and not RESTART:
    print("Energy minimisation ...")
    for i in range(50):
        simulation.minimizeEnergy(maxIterations=20, tolerance=MIN_TOL)
        state = simulation.context.getState(getEnergy=True)
        e = state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
        if e != e:  # NaN check
            raise RuntimeError(f"Energy went NaN at minimization stage {i}")
        print(f"  stage {i}: E = {e:.1f} kcal/mol")

print("Scanning FENE bonds for singularities ...")
for idx in range(fene_bond.getNumBonds()):
    i, j, params = fene_bond.getBondParameters(idx)
    r0, R0, K = params
    r = np.linalg.norm(all_positions[j] - all_positions[i])
    x = ((r - r0) / R0)**2
    if x >= 0.999:
        print(f"  BAD BOND idx={idx}: i={i}, j={j}, r={r:.5f} nm, "
              f"r0={r0:.5f} nm, R0={R0:.5f} nm, x={x:.5f}")


print_energy_breakdown("initial")

os.makedirs(os.path.dirname(dcd_out) if os.path.dirname(dcd_out) else ".", exist_ok=True)
simulation.reporters.append(DCDReporter(dcd_out, REPORT_EVERY))
simulation.reporters.append(StateDataReporter(
    log_out, REPORT_EVERY, step=True, time=True, potentialEnergy=True,
    kineticEnergy=True, temperature=True, progress=True, remainingTime=True,
    totalSteps=N_STEPS_EQUIL+N_STEPS_PROD, separator=","))
simulation.reporters.append(StateDataReporter(
    sys.stdout, 5000, step=True, temperature=True, potentialEnergy=True,
    speed=True, progress=True, remainingTime=True,
    totalSteps=N_STEPS_EQUIL+N_STEPS_PROD, separator=" | "))

print(f"\nRunning equilibration ({N_STEPS_EQUIL} steps) ...")
print("Running short diagnostic steps ...")
for i in range(200):
    simulation.step(1)
    e = simulation.context.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)
    if e != e:
        print(f"NaN at step {i}")
        print_energy_breakdown(f"step {i}")
        raise RuntimeError("stop")
simulation.step(N_STEPS_EQUIL)
print_energy_breakdown("after equilibration")
simulation.saveCheckpoint(chk_equil)

print(f"\nRunning production ({N_STEPS_PROD} steps) ...")
simulation.step(N_STEPS_PROD)
simulation.saveCheckpoint(chk_prod)

simulation.topology.setPeriodicBoxVectors(None)
state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
final_cif = final_pdb.replace(".pdb", ".cif")
with open(final_cif, "w") as f:
    PDBxFile.writeFile(simulation.topology, state.getPositions(), f)

print("\nSimulation complete.")
print(f"  Trajectory      : {dcd_out}")
print(f"  Log             : {log_out}")
print(f"  Final structure : {final_cif}")