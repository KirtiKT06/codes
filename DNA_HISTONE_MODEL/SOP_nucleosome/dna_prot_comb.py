"""
dna_prot_comb.py
Uses TIS model for DNA and SOP model for protein
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import json, sys, os
import numpy as np
from openmm import *
from openmm.app import *
import openmm.unit as unit
from openmm import XmlSerializer
from collections import Counter

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────────────────────────────────────────
cfg = json.load(open("input_dna_prot.json"))

pdb_file         = cfg["nucleosome_pdb"]
prot_chains_cfg  = cfg["histone_chains"]
dna_chains_cfg   = cfg["dna_chains"]
native_contacts_file = cfg["native_contacts_file"]
charged_beads_file   = cfg["charged_beads_file"]

dcd_out             = cfg["dcd_prefix"] + ".dcd"
log_out             = cfg["data_name"]
chk_equil           = cfg["checkpoint_save"]
chk_prod            = chk_equil.replace(".chk", "_prod.chk")
chk_start           = cfg["checkpoint_name"]
energy_breakdown    = cfg["energy_breakdown"]
initial_pdb         = cfg["initial_pdb_name"]
final_pdb           = cfg["final_state_name"]

CHK_DIR   = cfg["CHK_DIR"]
STATE_DIR = cfg["STATE_DIR"]

# ── SOP protein forcefield ──────────
K_FENE          = cfg["fene_k"]           # kcal / (mol · Å²)
R0_FENE         = cfg["fene_R0"]          # Å
SIGMA_PROTEIN   = cfg["sigma_protein"]    # Å
EPS_H_NATIVE    = cfg["eps_h_native"]     # kcal/mol
EPS_L_NONNATIVE = cfg["eps_local"]        # kcal/mol

# ── TIS DNA forcefield ───────────────
D0        = cfg["D0"]                     # eq 8, Angstrom
EPS0      = cfg["eps0"]                   # eq 8, kcal/mol
U0_HB     = cfg["U0_HB"]
K_L       = cfg["kl_stack"]
K_PHI     = cfg["kphi_stack"]
K_D       = cfg["kd_HB"]
K_THETA   = cfg["ktheta_HB"]
K_PSI     = cfg["kpsi_HB"]

# ── ELECTROSTATICS ──────────────────────────────────────────────────────
DIELECTRIC_PROTEIN      = cfg["dielectric_protein"]
DIELECTRIC_WATER        = cfg["dielectric_water"]
PME_CUTOFF              = cfg["pme_cutoff"] * 0.1       # Å → nm
HISTIDINE_CHARGE        = cfg["histidine_charge"]
PHOSPHATE_CHARGE        = cfg["phosphate_charge"]

# ── SIMULATION PARAMETERS ───────────────────────────────────────────────
BOX_PADDING = cfg["box_padding"]            # Å
NONLOCAL_CUTOFF = cfg["nonlocal_cutoff"]    # Å

TEMPERATURE   = cfg["Temp"]                 # K
Boltz_Const   = cfg["Boltz_Const"]          # kcal/(mol·K)
N_A           = cfg["N_A"]                  # Avogadro's number
FRICTION      = cfg["friction_coeff"]       # 1/ps
TIMESTEP_PS   = cfg["time_step"]            # ps
N_STEPS_EQUIL = cfg["numsteps_equil"]
N_STEPS_PROD  = cfg["numsteps_prod"]
REPORT_EVERY  = cfg["data_interval"]

DO_MINIMISE  = cfg["minimization"]
MIN_MAXITER  = cfg["minimization_maxiter"]
MIN_TOL      = cfg["minimization_tol"]
PLATFORM     = cfg["platform_type"]
RESTART      = cfg["restart"]
DO_CMR       = cfg["ComMotionRemover"]
CMR_FREQ     = cfg["CMMR_frequency"]

X            = cfg["box_x_nm"] * 10.0           # nm → Å
Y            = cfg["box_y_nm"] * 10.0           # nm → Å   
Z            = cfg["box_z_nm"] * 10.0           # nm → Å 
pi           = np.pi

#── Explicit ions ─────────────────────────────────────────────────────────────
auto_ions       = cfg.get("auto_ions", True)
KCl_mM          = cfg["KCl_mM"]
MgCl2_mM        = cfg["MgCl2_mM"]
neutralize      = cfg.get("neutralize", True)
SIGMA_K         = cfg["rK"]
SIGMA_CL        = cfg["rCl"]
SIGMA_MG        = cfg["rMg"]
EPS_K           = cfg["epsK"]
EPS_CL          = cfg["epsCl"]
EPS_MG          = cfg["epsMg"]
MASS_K          = cfg["mK"]
MASS_CL         = cfg["mCl"]
MASS_MG         = cfg["mMg"]
Q_K             = cfg["qK"]
Q_CL            = cfg["qCl"]
Q_MG            = cfg["qMg"]
ION_MIN_DIST    = cfg["ion_min_dist"] * 0.1   # Å -> nm

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: unit conversions
# ─────────────────────────────────────────────────────────────────────────────
"""
kcal/(mol·K) → kJ/(mol·K)
Angstrom → nm
"""

def kcal_to_kJ(x):
    return x * 4.184

def A_to_nm(x):
    return x * 0.1

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: temperature dependent dielectric
# ─────────────────────────────────────────────────────────────────────────────
tcent = TEMPERATURE - 273.15

eps_water = 87.740 - 0.40008 * tcent + 9.398e-4 * tcent**2 - 1.410e-6 * tcent**3
dielectric_sqrt = np.sqrt(eps_water)
print(f"    Timestep = {TIMESTEP_PS:.4f} ps, friction = {FRICTION:.1f} /ps, temperature = {TEMPERATURE:.1f} K")
print(f"    Temperature: {TEMPERATURE} K, water dielectric: {eps_water:.2f}, sqrt(ε): {dielectric_sqrt:.2f}")

#─────────────────────────────────────────────────────────────────────────────
# Load System
#─────────────────────────────────────────────────────────────────────────────
print(f"    Loading structure ...")

dna_beads = []
protein_beads = []
chain_counts = Counter()
with open(pdb_file) as f:
    for line in f:
        if not line.startswith(("ATOM")):
            continue
        atom_name = line[12:16].strip()
        chain_id = line[21:22].strip()
        chain_counts[chain_id] += 1
        bead = {
            "name"      : atom_name,
            "res_name"  : line[17: 20].strip(),
            "chain_id"  : line[21: 22].strip(),
            "res_num"   : int(line[22: 26].strip()),
            "x"         : float(line[30: 38].strip()),
            "y"         : float(line[38: 46].strip()),
            "z"         : float(line[46: 54].strip())        
        }
        if atom_name == "BB":
            protein_beads.append(bead)
        else:
            dna_beads.append(bead)        

print(f"    Loaded {len(dna_beads)} DNA beads and {len(protein_beads)} protein beads from {pdb_file}")
print(f"    Chain counts: {dict(chain_counts)}")

all_beads = protein_beads + dna_beads

for idx, bead in enumerate(all_beads):
    bead["index"] = idx

N_protein = len(protein_beads)
DNA_OFFSET = N_protein

#============================
# Proteins
#============================
# ── Native contacts ──────────────────────────────────────────────────────────
# native_contacts.dat was built from histone_cg.pdb (761 beads, 0-indexed).
# After adding 191 tail beads prepended PER CHAIN, the core bead indices shift.
# We must remap: for each chain, tail beads come first.

# Build index map: old_index → new_index
# Old order: A(97) B(78) C(108) D(96) E(99) F(85) G(104) H(94)
# New order: A_tail(37) A_core(97) B_tail(24) B_core(78) ...
chain_order = prot_chains_cfg
old_chain_sizes = {'A':97,'B':78,'C':108,'D':96,'E':99,'F':85,'G':104,'H':94}
tail_sizes = {'A':37,'B':24,'C':10,'D':26,'E':36,'F':17,'G':14,'H':27}

# Old index mapping: old idx → (chain, position_in_chain)
old_offset = 0
old_idx_to_chain_pos = {}
for ch in chain_order:
    for pos in range(old_chain_sizes[ch]):
        old_idx_to_chain_pos[old_offset + pos] = (ch, pos)
    old_offset += old_chain_sizes[ch]

# New index mapping: (chain, position_in_core) → new idx
new_chain_start = {}
offset = 0
for ch in chain_order:
    new_chain_start[ch] = offset
    offset += tail_sizes[ch] + old_chain_sizes[ch]

def remap_contact_index(old_idx):
    """Convert a 0-based index from native_contacts.dat to new bead index."""
    ch, pos_in_core = old_idx_to_chain_pos[old_idx]
    return new_chain_start[ch] + tail_sizes[ch] + pos_in_core

native_contacts = []
with open(native_contacts_file) as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        i_old, j_old, r0 = int(parts[0]), int(parts[1]), float(parts[2])
        i_new = remap_contact_index(i_old)
        j_new = remap_contact_index(j_old)
        native_contacts.append((i_new, j_new, r0))

print(f"    {len(native_contacts)} native contacts loaded and remapped.")

mass_proteins = np.loadtxt(cfg["protein_mass_file"], usecols=(3,), unpack=True, skiprows=1, delimiter=',')
# print(f"    N_beads_proteins =", len(mass_proteins))
# print(f"    len(mass_proteins) =", len(mass_proteins))
assert len(mass_proteins) == N_protein, "Mismatch in number of protein beads and mass entries"

# ── Charged beads ─────────────────────────────────────────────────────────────
charged_protein = {}   # bead_idx → charge (elementary units)
with open(charged_beads_file) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.split()
        idx, charge = int(parts[0]), float(parts[4])
        charged_protein[idx] = charge

print(f"    {len(charged_protein)} charged protein beads.")

#============================
# DNA
#============================
def _load_columns(path, skip_last_as_str=False):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            rows.append(parts[:-1] if skip_last_as_str else parts)
    return rows

dna_bond_rows  = _load_columns(cfg["dna_bonds_file"])
dna_angle_rows = _load_columns(cfg["dna_angles_file"])
dna_stack_rows = _load_columns(cfg["dna_stacking_file"], skip_last_as_str=True)
dna_hbond_rows = _load_columns(cfg["dna_hbonds_file"], skip_last_as_str=True)
dna_charge_rows = _load_columns(cfg["dna_charges_file"])
N_dna = len(dna_beads)
mass_dna = np.loadtxt(cfg["dna_mass_file"], usecols=(3,), unpack=True, skiprows=1, delimiter=',')
# print(f"    N_beads_dna =", N_dna)
# print(f"    len(mass_dna) =", len(mass_dna))
assert len(mass_dna) == N_dna, "Mismatch in number of DNA beads and mass entries"
print(f"    Bonds={len(dna_bond_rows)} Angles={len(dna_angle_rows)} Stacks={len(dna_stack_rows)} HBonds={len(dna_hbond_rows)}")

dna_bond_pairs = [(int(r[0]) + DNA_OFFSET, int(r[1]) + DNA_OFFSET) for r in dna_bond_rows]
dna_angle_triples = [(int(r[0]) + DNA_OFFSET, int(r[2]) + DNA_OFFSET) for r in dna_angle_rows]
dna_phosphate_indices = set(int(r[0]) + DNA_OFFSET for r in dna_charge_rows)
net_dna_charge = PHOSPHATE_CHARGE * len(dna_phosphate_indices)   # bare charge, explicit_ions mode
print(f"    Net DNA charge (bare): {net_dna_charge:+f} e  ({len(dna_phosphate_indices)} phosphates)")

print(f"    Protein beads = {N_protein}")
print(f"    DNA beads     = {N_dna}")
print(f"    Total beads   = {len(all_beads)}")

assert len(all_beads) == N_protein + N_dna
assert min(dna_phosphate_indices) >= DNA_OFFSET
assert max(dna_phosphate_indices) < len(all_beads)

#─────────────────────────────────────────────────────────────────────────────
# Build OpenMM Topology
#─────────────────────────────────────────────────────────────────────────────
print(f"    Building topology ...")
topology = Topology()

# Add chains and residues for protein
protein_chains = {}
protein_residues = []
for ch_id in chain_order:
    protein_chains[ch_id] = topology.addChain(id=ch_id)

for b in protein_beads:
    res = topology.addResidue(b["res_name"], protein_chains[b["chain_id"]])
    protein_residues.append(res)

# Add protein atoms (one per bead)
omm_atoms_protein = []
for i, (b, res) in enumerate(zip(protein_beads, protein_residues)):
    atom = topology.addAtom("BB", Element.getBySymbol("C"), res)
    omm_atoms_protein.append(atom)

dna_chains = {}
for ch_id in dna_chains_cfg:
    dna_chains[ch_id] = topology.addChain(id=ch_id)

omm_atoms_dna = []
for b in dna_beads:
    res = topology.addResidue(b["res_name"], dna_chains[b["chain_id"]])
    elem = {"P": "P", "S": "C"}.get(b["name"], "C")
    atom = topology.addAtom(b["name"], Element.getBySymbol(elem), res)
    omm_atoms_dna.append(atom)

print(f"    Protein topology atoms:", len(omm_atoms_protein))
print(f"    DNA topology atoms:", len(omm_atoms_dna))
print(f"    Total topology atoms:", len(omm_atoms_protein) + len(omm_atoms_dna))

assert (len(omm_atoms_protein) + len(omm_atoms_dna)
        == len(all_beads))

#─────────────────────────────────────────────────────────────────────────────
# Create the simulation box
#─────────────────────────────────────────────────────────────────────────────
print(f"    Creating simulation box ...")
coords_nm = np.array([
    [b["x"], b["y"], b["z"]]
    for b in all_beads]) * 0.1

box_size = np.array([70.0, 70.0, 70.0])  # nm

topology.setPeriodicBoxVectors((
    Vec3(box_size[0],0,0)*unit.nanometer,
    Vec3(0,box_size[1],0)*unit.nanometer,
    Vec3(0,0,box_size[2])*unit.nanometer
))

positions_openmm = unit.Quantity(
    [Vec3(*p) for p in coords_nm],
    unit.nanometer
)
print(f"    Box size: {box_size[0]:.2f} x {box_size[1]:.2f} x {box_size[2]:.2f} nm³")

#─────────────────────────────────────────────────────────────────────────────
# Place ions inside the box
#─────────────────────────────────────────────────────────────────────────────
print(f"    Calculating total charge of the system ...")

net_protein_charge = sum(charged_protein.values())
net_charge = net_protein_charge + net_dna_charge
print(f"    Charge distribution for protein:", Counter(charged_protein.values()))
print(f"    Net protein charge: {net_protein_charge:+.1f} e")
print(f"    Net DNA charge: {net_dna_charge:+.1f} e")
print(f"    Net system charge: {net_charge:+.1f} e")
assert net_charge == net_protein_charge + net_dna_charge

print(f"    Adding explicit ions ...")
if auto_ions:
    print(f"    KCl concentration: {KCl_mM} mM, MgCl2 concentration: {MgCl2_mM} mM")

    box_vol_nm3 = float(np.prod(box_size))
    box_vol_L = box_vol_nm3 * 1e-27 * 1e3

    n_kcl_pairs = max(0, round(KCl_mM * 1e-3 * N_A * box_vol_L))
    n_mgcl2_pairs = max(0, round(MgCl2_mM * 1e-3 * N_A * box_vol_L))    

    n_k_salt = n_kcl_pairs
    n_mg_salt = n_mgcl2_pairs
    n_cl_salt = n_kcl_pairs + 2 * n_mgcl2_pairs

    if neutralize:
        if net_charge < 0:
            n_k = n_k_salt + abs(int(round(net_charge)))
            n_cl = n_cl_salt
        elif net_charge > 0:
            n_k = n_k_salt
            n_cl = n_cl_salt + int(round(net_charge))
        else:
            n_k = n_k_salt
            n_cl = n_cl_salt + net_charge
    else:
        n_k, n_cl = n_k_salt, n_cl_salt
    n_mg = n_mg_salt
else:
    n_k, n_cl, n_mg = cfg.get("nK", 0), cfg.get("nCl", 0), cfg.get("nMg", 0)

N_ions = n_k + n_cl + n_mg
total_charge_after_ions = (net_charge + n_k * Q_K + n_cl * Q_CL + n_mg * Q_MG)

print(f"    Box volume: {box_vol_nm3:.2f} nm³, {box_vol_L:.2e} L")
print(f"    Number of KCl pairs: {n_kcl_pairs}, Number of MgCl2 pairs: {n_mgcl2_pairs}")
print(f"    Total ions to add: K+={n_k}, Cl-={n_cl}, Mg2+={n_mg}")
print(f"    Total ions to add: {N_ions}")
print(f"    Charge after ion addition = {total_charge_after_ions}")
assert N_ions == n_k + n_cl + n_mg

print(f"    Ion parameters: SIGMA_K={SIGMA_K:.3f} nm, SIGMA_CL={SIGMA_CL:.3f} nm, SIGMA_MG={SIGMA_MG:.3f} nm")
print(f"    Ion parameters: EPS_K={EPS_K:.3f} kJ/mol, EPS_CL={EPS_CL:.3f} kJ/mol, EPS_MG={EPS_MG:.3f} kJ/mol")
print(f"    Ion parameters: MASS_K={MASS_K:.3f} amu, MASS_CL={MASS_CL:.3f} amu, MASS_MG={MASS_MG:.3f} amu")
print(f"    Ion parameters: Q_K={Q_K:.3f} e, Q_CL={Q_CL:.3f} e, Q_MG={Q_MG:.3f} e")
print(f"    Minimum distance between ions and other beads: {ION_MIN_DIST:.3f} nm")
print(f"    Minimum distance between ions: {ION_MIN_DIST:.3f} nm")

print(f"    Creating ions topology ...")
ion_chain = topology.addChain(id="Z")
k_residues, cl_residues, mg_residues = [], [], []
k_atoms, cl_atoms, mg_atoms = [], [], []
for k in range(n_k):
    res  = topology.addResidue("K", ion_chain)
    atom = topology.addAtom("K", Element.getBySymbol("K"), res)
    k_residues.append(res); k_atoms.append(atom)
for k in range(n_cl):
    res  = topology.addResidue("CL", ion_chain)
    atom = topology.addAtom("CL", Element.getBySymbol("Cl"), res)
    cl_residues.append(res); cl_atoms.append(atom)
for k in range(n_mg):
    res = topology.addResidue("MG", ion_chain)
    atom = topology.addAtom("MG", Element.getBySymbol("Mg"), res)
    mg_residues.append(res); mg_atoms.append(atom)

print(f"    Placing ions in the box ...")
np.random.seed(42)
ion_positions = []

for _ in range(N_ions):
    placed = False
    for attempt in range(50000):
        pos = np.random.rand(3) * box_size
        ok = True
        bead_dists = np.linalg.norm(coords_nm - pos, axis=1)

        if bead_dists.min() <= ION_MIN_DIST:
            ok = False
        if ok and len(ion_positions):
            ion_dists = np.linalg.norm(
                np.array(ion_positions) - pos,
                axis=1
            )
            if ion_dists.min() <= ION_MIN_DIST:
                ok = False
        if ok:
            ion_positions.append(pos)
            placed = True
            break
    if not placed:
        ion_positions.append(np.random.rand(3) * box_size)

all_positions = np.vstack([coords_nm, np.array(ion_positions)]) if N_ions else coords_nm

positions_openmm = unit.Quantity([Vec3(*p) for p in all_positions], unit.nanometer)

print(f"    Placed ions = {len(ion_positions)}")
print(f"    Expected ions = {N_ions}")
print(f"    Total particles: {len(all_positions)}")

# A quick check of the minimum ion-ion distance, to see if any are unreasonably close
if len(ion_positions) > 1:
    ions = np.array(ion_positions)

    mind = np.inf
    for i in range(len(ions)):
        for j in range(i+1, len(ions)):
            d = np.linalg.norm(ions[i]-ions[j])
            mind = min(mind, d)

    print(f"    Minimum ion-ion distance = {mind:.3f} nm")

assert len(all_positions) == (N_protein + N_dna + N_ions)

print(f"    Writing initial structure ...")
topology.setPeriodicBoxVectors(None)
with open(initial_pdb, "w") as f:
    PDBFile.writeFile(
        topology,
        positions_openmm,
        f
    )
print(f"    Initial structure written to: {initial_pdb}")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD FORCE FIELD
# ─────────────────────────────────────────────────────────────────────────────
print(f"    Constructing OpenMM System")

system = System()
system.setDefaultPeriodicBoxVectors(
    Vec3(box_size[0], 0, 0) * unit.nanometer,
    Vec3(0, box_size[1], 0) * unit.nanometer,
    Vec3(0, 0, box_size[2]) * unit.nanometer,
)

for i in range(len(protein_beads)):
    system.addParticle(float(mass_proteins[i]) * unit.amu)
for i in range(len(dna_beads)):
    system.addParticle(float(mass_dna[i]) * unit.amu)
for _ in range(n_k):
    system.addParticle(MASS_K * unit.amu)
for _ in range(n_cl):
    system.addParticle(MASS_CL * unit.amu)
for _ in range(n_mg):
    system.addParticle(MASS_MG * unit.amu)

expected_particles = N_protein + N_dna + N_ions

print(f"    System particles = {system.getNumParticles()}")
print(f"    Expected particles = {expected_particles}")
assert system.getNumParticles() == expected_particles

print(f"    Topology atoms = {topology.getNumAtoms()}")
print(f"    System particles = {system.getNumParticles()}")
assert topology.getNumAtoms() == system.getNumParticles()

#============================
# Protein Forces
#============================
print(f"    Adding only protein forces")
"""
── FORCE (0): FENE bonds ─────────────────────────────────────────────────────

The Finitely Extensible Nonlinear Elastic (FENE) potential:

  U_FENE(r) = - (k/2) R₀² ln[1 - (r - r_cry)² / R₀²]

where:
  r     = current bond length between beads i and i+1
  r_cry = equilibrium (crystal) bond length = distance in CG structure
  R₀    = maximum extension beyond equilibrium (2.0 Å)
  k     = spring constant (20 kcal/mol/Å²)

Why FENE instead of harmonic?
  A harmonic spring can stretch to infinity.  FENE diverges as
  r - r_cry → R₀, which prevents unphysical chain crossing.
  This is critical for CG polymer simulations.

CustomBondForce formula in OpenMM (r is bond length in nm):
  -0.5 * K * R0² * log(1 - ((r - r0)/R0)²)
  All distances in nm, energies in kJ/mol.
"""
print(f"    Adding FENE bonds ...")
k_fene_kJ_nm2 = kcal_to_kJ(K_FENE) * 100.0   # kcal/(mol·Å²) → kJ/(mol·nm²)
R0_nm         = A_to_nm(R0_FENE)

fene_bond = CustomBondForce("-0.5 * K * R0*R0 * log(1 - ((r - r0)/R0)^2)")
fene_bond.addPerBondParameter("r0")   # equilibrium length (nm)
fene_bond.addPerBondParameter("R0")   # max extension (nm)
fene_bond.addPerBondParameter("K")    # spring constant (kJ/mol/nm²)
fene_bond.setUsesPeriodicBoundaryConditions(True)

# Add bonds between consecutive beads in the same chain
chain_boundaries = []
offset = 0
for ch in chain_order:
    n_ch = tail_sizes[ch] + old_chain_sizes[ch]
    chain_boundaries.append((offset, offset + n_ch - 1))
    offset += n_ch

bond_pairs = set()
for ch_start, ch_end in chain_boundaries:
    for i in range(ch_start, ch_end):
        j = i + 1
        pos_i = all_positions[i]
        pos_j = all_positions[j]
        r0_bond = float(np.linalg.norm(pos_j - pos_i))  # nm
        # r0_bond = min(r0_bond, R0_nm * 0.95)  # prevent numerical issues if initial bond is near or above R0
        fene_bond.addBond(i, j, [r0_bond, R0_nm, k_fene_kJ_nm2])
        bond_pairs.add((min(i,j), max(i,j)))

fene_bond.setForceGroup(0)

"""
── FORCE (1): Native contacts (Lennard-Jones 12-6) ──────────────────────────

For pairs in the native contact list:

  U_native(r) = ε_h · [(r_cry/r)¹² - 2(r_cry/r)⁶]

This is a 12-6 LJ potential with:
  minimum at r = r_cry  (the crystal contact distance)
  well depth = ε_h = 2.0 kcal/mol  (for HPC-HPC, Table S1)

Note the factor of 2 in front of the attractive term ensures the
minimum value is exactly -ε_h (not -ε_h/4 as in standard LJ).

r_cry values come from native_contacts.dat (distances in Å).
"""
print(f"    Adding native contacts ...")
native_force = CustomBondForce("eps_h * ((r0/r)^12 - 2*(r0/r)^6)")
native_force.addPerBondParameter("r0")     # crystal distance (nm)
native_force.addPerBondParameter("eps_h")  # well depth (kJ/mol)
native_force.setUsesPeriodicBoundaryConditions(True)

eps_h_kJ = kcal_to_kJ(EPS_H_NATIVE)
for i, j, r0_A in native_contacts:
    native_force.addBond(i, j, [A_to_nm(r0_A), eps_h_kJ])

native_force.setForceGroup(1)

#────────────────────────── Add protein forces to system ─────────────────────────────
system.addForce(fene_bond)
print(f"    {fene_bond.getNumBonds()} FENE bonds added.")
system.addForce(native_force)
print(f"    {native_force.getNumBonds()} native contact pairs added.")


#============================
# DNA Forces
#============================
print(f"    Adding DNA forces now!")
"""
── FORCE (2): Bond Forces ─────────────────────────────────────────────────────
U_B = k_r * (r - r0)²
r0 = equilibrium bond length (from pdb)
k_r = bond spring constant (from dna_bonds.dat)
"""
print(f"    Adding Harmonic Bond Forces...")
dna_bond_force = HarmonicBondForce()
for row in dna_bond_rows:
    i = int(row[0]) + DNA_OFFSET
    j = int(row[1]) + DNA_OFFSET
    r0 = float(row[2])
    kr = float(row[3])
    dna_bond_force.addBond(i, j, r0 * unit.angstrom, 2.0 * kr * unit.kilocalorie_per_mole / (unit.angstrom) ** 2)

dna_bond_force.setForceGroup(2)

"""
── FORCE (3): Angle Forces ─────────────────────────────────────────────────────
U_A = k_a * (θ - θ0)²
θ0 = equilibrium angle (from pdb)
k_a = angle spring constant (from dna_angles.dat)
"""
print(f"    Adding Harmonic Angle Forces...")
dna_angle_force = HarmonicAngleForce()
for row in dna_angle_rows:
    i = int(row[0]) + DNA_OFFSET
    j = int(row[1]) + DNA_OFFSET
    k = int(row[2]) + DNA_OFFSET
    a0 = float(row[3])
    ka = float(row[4])
    dna_angle_force.addAngle(i, j, k, a0 * unit.radians, 2.0 * ka * (unit.kilocalories_per_mole / (unit.radians) ** 2))

dna_angle_force.setForceGroup(3)

"""
── FORCE (4): Stack Forces ─────────────────────────────────────────────────────
U_S = u0ST * (1 + kr*delRsq + kphi1*delphi1sq + kphi2*delphi2sq)⁻¹
"""
print(f"    Adding Stack Forces...")
krST    = K_L / (unit.angstrom) ** 2
kphi1ST = K_PHI / (unit.radian) ** 2
kphi2ST = K_PHI / (unit.radian) ** 2

StackEnergyExpression  = "u0ST /(1.0 + kr*delRsq + kphi1*delphi1sq + kphi2*delphi2sq);"
StackEnergyExpression += "delRsq = delR*delR; delR = (RST-R0ST); RST = distance(p3,p6);"
StackEnergyExpression += "delphi1sq = delphi1*delphi1;"
StackEnergyExpression += "delphi1 = dphi1 - 2.0*pi*step(dphi1-pi) + 2.0*pi*step(-dphi1-pi);"
StackEnergyExpression += "dphi1 =(phi1ST-phi10ST); phi1ST= dihedral(p1,p2,p4,p5);"
StackEnergyExpression += "delphi2sq = delphi2*delphi2;"
StackEnergyExpression += "delphi2 = dphi2 - 2.0*pi*step(dphi2-pi) + 2.0*pi*step(-dphi2-pi);"
StackEnergyExpression += "dphi2 =(phi2ST-phi20ST); phi2ST= dihedral(p2,p4,p5,p7);"

StackForce = CustomCompoundBondForce(7, StackEnergyExpression)
StackForce.addPerBondParameter("u0ST")
StackForce.addPerBondParameter("R0ST")
StackForce.addPerBondParameter("phi10ST")
StackForce.addPerBondParameter("phi20ST")
StackForce.addGlobalParameter("kr", krST)
StackForce.addGlobalParameter("kphi1", kphi1ST)
StackForce.addGlobalParameter("kphi2", kphi2ST)
StackForce.addGlobalParameter("pi", pi)

for row in dna_stack_rows:
    p_i, s_i, b_i, p_j, s_j, b_j, p_k = [int(x) for x in row[0:7]]
    p_i += DNA_OFFSET
    s_i += DNA_OFFSET
    b_i += DNA_OFFSET
    p_j += DNA_OFFSET
    s_j += DNA_OFFSET
    b_j += DNA_OFFSET
    p_k += DNA_OFFSET

    l0, phi1_0, phi2_0 = float(row[7]), float(row[8]), float(row[9])
    h, s, Tm, dG0 = float(row[10]), float(row[11]), float(row[12]), float(row[13])
    u0ST_val = -h + float(Boltz_Const) * (TEMPERATURE - Tm) * s
    StackForce.addBond([p_i, s_i, b_i, p_j, s_j, b_j, p_k],
                        [u0ST_val * unit.kilocalorie_per_mole, l0 * unit.angstroms,
                         phi1_0 * unit.radian, phi2_0 * unit.radian])

StackForce.setForceGroup(4)

"""
── FORCE (5): Hydrogen Bond Forces ─────────────────────────────────────────────────────
U_HB = u0HB_bond * (1 + kd_hb*delDsq + ktheta_hb*deltheta1sq + ktheta_hb*deltheta2sq
                   + kpsi_hb*delpsi1sq + kpsi_hb*delpsi2sq + kpsi_hb*delpsi3sq)⁻¹
"""
print(f"    Adding Hydrogen Bond Forces...")
kdHB     = K_D / (unit.angstrom) ** 2
kthetaHB = K_THETA / (unit.radian) ** 2
kpsiHB   = K_PSI / (unit.radian) ** 2

HBEnergyExpression  = "u0HB_bond / (1.0 + kd_hb*delDsq + ktheta_hb*deltheta1sq + ktheta_hb*deltheta2sq"
HBEnergyExpression += " + kpsi_hb*delpsi1sq + kpsi_hb*delpsi2sq + kpsi_hb*delpsi3sq);"
HBEnergyExpression += "delDsq = delD*delD; delD = (DHB - D0HB); DHB = distance(p3,p4);"
HBEnergyExpression += "deltheta1sq = deltheta1*deltheta1; deltheta1 = (theta1HB - theta10HB); theta1HB = angle(p2,p3,p4);"
HBEnergyExpression += "deltheta2sq = deltheta2*deltheta2; deltheta2 = (theta2HB - theta20HB); theta2HB = angle(p3,p4,p5);"
HBEnergyExpression += "delpsi1sq = delpsi1*delpsi1;"
HBEnergyExpression += "delpsi1 = dpsi1 - 2.0*pi*step(dpsi1-pi) + 2.0*pi*step(-dpsi1-pi);"
HBEnergyExpression += "dpsi1 = (psi1HB - psi10HB); psi1HB = dihedral(p2,p3,p4,p5);"
HBEnergyExpression += "delpsi2sq = delpsi2*delpsi2;"
HBEnergyExpression += "delpsi2 = dpsi2 - 2.0*pi*step(dpsi2-pi) + 2.0*pi*step(-dpsi2-pi);"
HBEnergyExpression += "dpsi2 = (psi2HB - psi20HB); psi2HB = dihedral(p6,p5,p4,p3);"
HBEnergyExpression += "delpsi3sq = delpsi3*delpsi3;"
HBEnergyExpression += "delpsi3 = dpsi3 - 2.0*pi*step(dpsi3-pi) + 2.0*pi*step(-dpsi3-pi);"
HBEnergyExpression += "dpsi3 = (psi3HB - psi30HB); psi3HB = dihedral(p1,p2,p3,p4);"

HBForce_WC = CustomCompoundBondForce(6, HBEnergyExpression)
for pname in ["D0HB", "theta10HB", "theta20HB", "psi10HB", "psi20HB", "psi30HB", "u0HB_bond"]:
    HBForce_WC.addPerBondParameter(pname)
HBForce_WC.addGlobalParameter("kd_hb", kdHB)
HBForce_WC.addGlobalParameter("ktheta_hb", kthetaHB)
HBForce_WC.addGlobalParameter("kpsi_hb", kpsiHB)
HBForce_WC.addGlobalParameter("pi", pi)

n_hbonds_skipped = 0
for row in dna_hbond_rows:

    s1, b1, b5, s5, p6, p2 = [int(x) for x in row[0:6]]
    d0, theta1_0, theta2_0, psi1_0, psi2_0, psi3_0 = [float(x) for x in row[6:12]]
    UHB0_bond = float(row[12])

    if p6 == -1 or p2 == -1:
        n_hbonds_skipped += 1
        continue

    s1 += DNA_OFFSET
    b1 += DNA_OFFSET
    b5 += DNA_OFFSET
    s5 += DNA_OFFSET
    p6 += DNA_OFFSET
    p2 += DNA_OFFSET

    HBForce_WC.addBond([p2, s1, b1, b5, s5, p6],
                        [d0 * unit.angstrom, theta1_0 * unit.radian, theta2_0 * unit.radian,
                         psi1_0 * unit.radian, psi2_0 * unit.radian, psi3_0 * unit.radian,
                         UHB0_bond * unit.kilocalorie_per_mole])
if n_hbonds_skipped:
    print(f"    WARNING: skipped %d WC pair(s) at chain ends lacking a flanking phosphate" % n_hbonds_skipped)

HBForce_WC.setForceGroup(5)

#────────────────────────── Add DNA forces to system ─────────────────────────────
print(f"    DNA offset =", DNA_OFFSET)
print(f"    DNA bonds =", dna_bond_force.getNumBonds())
print(f"    DNA angles =", dna_angle_force.getNumAngles())
print(f"    DNA stacks =", StackForce.getNumBonds())
print(f"    DNA hbonds =", HBForce_WC.getNumBonds())

system.addForce(dna_bond_force)
print(f"    {dna_bond_force.getNumBonds()} DNA bonds added.")
system.addForce(dna_angle_force)
print(f"    {dna_angle_force.getNumAngles()} DNA angles added.")
system.addForce(StackForce)
print(f"    {StackForce.getNumBonds()} DNA stacking interactions added.")
system.addForce(HBForce_WC)
print(f"    {HBForce_WC.getNumBonds()} DNA hydrogen bonds added.")

#============================
# Protein + DNA Forces
#============================
print(f"    Adding Protein-DNA interaction Forces now!")
nbeads_protein = N_protein
nbeads_dna = N_dna
nbeads = N_protein + N_dna + N_ions
dna_indices = list(range(DNA_OFFSET, nbeads_dna+DNA_OFFSET))
protein_indices = list(range(N_protein))
ion_indices = list(range(nbeads_dna+DNA_OFFSET, nbeads))
ion_species_ranges = {}
if N_ions:
    idx = nbeads_protein + nbeads_dna
    for species, count in [('K', n_k), ('Cl', n_cl), ('Mg', n_mg)]:
        ion_species_ranges[species] = (idx, idx + count)
        idx += count

print(f"    Protein indices =", len(protein_indices))
print(f"    DNA indices =", len(dna_indices))
print(f"    Ion indices =", len(ion_indices))

"""
── FORCE (6): Excluded Volume DNA-DNA (non-native repulsion) ────────────────────────
"""
print(f"    Adding Excluded Volume DNA-DNA... ")
EVEnergyExpression = "select(step(D0-r), eps0*(((D0/r)^12) - 2.0*((D0/r)^6) + 1.0), 0);"
EVForce_dna = CustomNonbondedForce(EVEnergyExpression)
EVForce_dna.addGlobalParameter('D0', D0 * unit.angstrom)
EVForce_dna.addGlobalParameter('eps0', EPS0 * unit.kilocalorie_per_mole)
EVForce_dna.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
EVForce_dna.setCutoffDistance(float(D0) * unit.angstrom)
for i in range(nbeads):
    EVForce_dna.addParticle([])

EVForce_dna.addInteractionGroup(set(dna_indices), set(dna_indices))
for (i, j) in dna_bond_pairs:
    EVForce_dna.addExclusion(i, j)
for (i, k) in dna_angle_triples:
    try:
        EVForce_dna.addExclusion(i, k)
    except OpenMMException:
        pass
print(f"    EVForce_dna particles =", EVForce_dna.getNumParticles())
assert EVForce_dna.getNumParticles() == system.getNumParticles()
EVForce_dna.setForceGroup(6)

"""
── FORCE (7, 8): Excluded Volume DNA-ION and ION-ION (non-native repulsion) ────────────────────────
"""
if N_ions:
    ion_radius = {'K': SIGMA_K, 'Cl': SIGMA_CL, 'Mg': SIGMA_MG}
    ion_eps    = {'K': EPS_K, 'Cl': EPS_CL, 'Mg': EPS_MG}
    print(f"    Adding Excluded Volume ION-ION... ")
    MixExpr = ("select(step(sigma_ij-r), 4.0*eps_ij*(((sigma_ij/r)^12)-((sigma_ij/r)^6)) + eps_ij, 0);"
               "sigma_ij = radius1+radius2; eps_ij = sqrt(epsval1*epsval2);")

    LJForce_ion_ion = CustomNonbondedForce(MixExpr)
    LJForce_ion_ion.addPerParticleParameter('radius')
    LJForce_ion_ion.addPerParticleParameter('epsval')
    LJForce_ion_ion.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    LJForce_ion_ion.setCutoffDistance(2.0 * max(SIGMA_K, SIGMA_CL, SIGMA_MG) * unit.angstrom)
    for i in range(nbeads):
        if i in protein_indices:
            LJForce_ion_ion.addParticle([0.0 * unit.angstrom, 0.0 * unit.kilocalorie_per_mole])
        elif i in dna_indices:
            LJForce_ion_ion.addParticle([0.0 * unit.angstrom, 0.0 * unit.kilocalorie_per_mole])
        else:
            for species, (lo, hi) in ion_species_ranges.items():
                if lo <= i < hi:
                    LJForce_ion_ion.addParticle([ion_radius[species] * unit.angstrom, ion_eps[species] * unit.kilocalorie_per_mole])
                    break
    LJForce_ion_ion.addInteractionGroup(set(ion_indices), set(ion_indices))
    
    for (i, j) in dna_bond_pairs:
        LJForce_ion_ion.addExclusion(i, j)
    for (i, k) in dna_angle_triples:
        try:
            LJForce_ion_ion.addExclusion(i, k)
        except OpenMMException:
            pass
    print(f"    LJForce_ion_ion particles =", LJForce_ion_ion.getNumParticles())
    assert LJForce_ion_ion.getNumParticles() == system.getNumParticles()

    print(f"    Adding Excluded Volume DNA-ION... ")
    LJForce_dna_ion = CustomNonbondedForce(MixExpr)
    LJForce_dna_ion.addPerParticleParameter('radius')
    LJForce_dna_ion.addPerParticleParameter('epsval')
    LJForce_dna_ion.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    LJForce_dna_ion.setCutoffDistance((D0 / 2.0 + max(SIGMA_K, SIGMA_CL, SIGMA_MG)) * unit.angstrom)
    for i in range(nbeads):
        if i in protein_indices:
            LJForce_dna_ion.addParticle([0.0 * unit.angstrom, 0.0 * unit.kilocalorie_per_mole])
        elif i in dna_indices:
            LJForce_dna_ion.addParticle([D0 / 2.0 * unit.angstrom, EPS0 * unit.kilocalorie_per_mole])
        else:
            for species, (lo, hi) in ion_species_ranges.items():
                if lo <= i < hi:
                    LJForce_dna_ion.addParticle([ion_radius[species] * unit.angstrom, ion_eps[species] * unit.kilocalorie_per_mole])
                    break
    LJForce_dna_ion.addInteractionGroup(set(dna_indices), set(ion_indices))

    for (i, j) in dna_bond_pairs:
        LJForce_dna_ion.addExclusion(i, j)

    for (i, k) in dna_angle_triples:
        try:
            LJForce_dna_ion.addExclusion(i, k)
        except OpenMMException:
            pass
    print(f"    LJForce_dna_ion particles =", LJForce_dna_ion.getNumParticles())
    assert LJForce_dna_ion.getNumParticles() == system.getNumParticles()
    print(f"    System particles =", system.getNumParticles())
    assert (len(protein_indices) + len(dna_indices) + len(ion_indices) == system.getNumParticles())
    LJForce_ion_ion.setForceGroup(7)
    LJForce_dna_ion.setForceGroup(8)


"""
── FORCE (9, 10, 11): Excluded Volume (non-native repulsion) PROTEIN-PROTEIN,  PROTEIN-DNA, PROTEIN-ION────────────────────────

For all pairs NOT in native contacts AND not bonded (seq_sep ≥ 3):

  U_EV(r) = ε_l · (sigma/r)⁶

This is the purely repulsive r⁻⁶ term (the repulsive half of LJ).
It prevents bead overlap without any attractive well.

For protein-protein:  sigma = sigma_protein = 3.8 Å
For protein-ion:      sigma = (sigma_protein + sigma_ion) / 2  (Lorentz-Berthelot)
For ion-ion:          sigma = sigma_ion (same type) or mixed

We implement this via CustomNonbondedForce with a per-particle sigma.
The formula: eps_l * (sigma/r)^6 where sigma = (sigma_i + sigma_j)/2
"""
print(f"    Adding Excluded Volume PROTEIN-PROTEIN,  PROTEIN-DNA, PROTEIN-ION")

master_exclusions = set()
# (a) bonded pairs (seq_sep = 1)
for i, j in bond_pairs:
    master_exclusions.add((min(i,j), max(i,j)))

# (b) seq_sep = 2 on same chain
for ch_start, ch_end in chain_boundaries:
    for i in range(ch_start, ch_end + 1):
        for j in range(i + 1, min(i + 3, ch_end + 1)):
            master_exclusions.add((i, j))

# (c) native contacts — excluded from EV (they have their own attractive force)
for i, j, _ in native_contacts:
    master_exclusions.add((min(i,j), max(i,j)))

def create_sop_ev():
    force = CustomNonbondedForce("eps_l * (sigma / r)^6 ; sigma = 0.5*(sigma1 + sigma2)")
    force.addGlobalParameter("eps_l", kcal_to_kJ(EPS_L_NONNATIVE))
    force.addPerParticleParameter("sigma")
    force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    force.setCutoffDistance(PME_CUTOFF * unit.nanometer)

    # Protein
    for _ in range(N_protein):
        force.addParticle([A_to_nm(SIGMA_PROTEIN)])
    # DNA
    for _ in range(N_dna):
        force.addParticle([A_to_nm(D0/2.0)])
    # K
    for _ in range(n_k):
        force.addParticle([A_to_nm(SIGMA_K)])
    # Cl
    for _ in range(n_cl):
        force.addParticle([A_to_nm(SIGMA_CL)])
    # Mg
    for _ in range(n_mg):
        force.addParticle([A_to_nm(SIGMA_MG)])

    for i, j in master_exclusions:
        force.addExclusion(i, j)
    return force

EV_PP = create_sop_ev()
EV_PP.addInteractionGroup(set(protein_indices), set(protein_indices))

EV_PD = create_sop_ev()
EV_PD.addInteractionGroup(set(protein_indices), set(dna_indices))

EV_PI = create_sop_ev()
EV_PI.addInteractionGroup(set(protein_indices), set(ion_indices))

# Exclude bonded pairs AND seq_sep < 3 on same chain AND native contacts from EV.
# CRITICAL: OpenMM requires CustomNonbondedForce and NonbondedForce to have
# IDENTICAL exclusion sets. We build one master set and apply to both.
# native_contact_set = set((min(i,j), max(i,j)) for i, j, _ in native_contacts)

print(f"    Protein-Protein EV particles =", EV_PP.getNumParticles())
print(f"    Protein-DNA EV particles =", EV_PD.getNumParticles())
print(f"     Protein-ION EV particles =", EV_PI.getNumParticles())
print(f"     System particles =", system.getNumParticles())

print(f"    Master exclusions =", len(master_exclusions))
print(f"    Protein-Protein EV exclusions =", EV_PP.getNumExclusions())
print(f"    Protein-DNA EV exclusions =", EV_PD.getNumExclusions())
print(f"    Protein-Ion EV exclusions =", EV_PI.getNumExclusions())

assert EV_PP.getNumParticles() == system.getNumParticles()
assert EV_PD.getNumParticles() == system.getNumParticles()
assert EV_PI.getNumParticles() == system.getNumParticles()
assert (len(protein_indices) + len(dna_indices) + len(ion_indices) == system.getNumParticles())

EV_PP.setForceGroup(9)
EV_PD.setForceGroup(10)
EV_PI.setForceGroup(11)

"""
── FORCE (12): Coulombic Forces using PME────────────────────────
"""
print(f"    Adding Coulombic interactions now!")
ESForce = NonbondedForce()
ESForce.setNonbondedMethod(NonbondedForce.PME)
ESForce.setCutoffDistance(PME_CUTOFF * unit.nanometer)
ESForce.setEwaldErrorTolerance(0.0005)

for i in protein_indices:
    q = charged_protein.get(i, 0.0) / dielectric_sqrt
    ESForce.addParticle(q, 1.0 * unit.angstrom, 0.0 * unit.kilocalorie_per_mole)

for i in dna_indices:
    if i in dna_phosphate_indices:
        q = PHOSPHATE_CHARGE / dielectric_sqrt
    else:
        q = 0.0
    ESForce.addParticle(q, 1.0 * unit.angstrom, 0.0 * unit.kilocalorie_per_mole)

for _ in range(n_k):
    ESForce.addParticle(Q_K / dielectric_sqrt, 1.0 * unit.angstrom, 0.0 * unit.kilocalorie_per_mole)
for _ in range(n_cl):
    ESForce.addParticle(Q_CL / dielectric_sqrt, 1.0 * unit.angstrom, 0.0 * unit.kilocalorie_per_mole)
for _ in range(n_mg):
    ESForce.addParticle(Q_MG / dielectric_sqrt, 1.0 * unit.angstrom, 0.0 * unit.kilocalorie_per_mole)

electrostatic_exclusions = set(master_exclusions)

for i, j in dna_bond_pairs:
    electrostatic_exclusions.add((min(i, j), max(i, j)))
for i, k in dna_angle_triples:
    electrostatic_exclusions.add((min(i, k), max(i, k)))
for i, j in electrostatic_exclusions:
    ESForce.addException(i, j, 0.0, 1.0, 0.0 * unit.kilocalorie_per_mole)

print(f"    ESForce particles =", ESForce.getNumParticles())
print(f"    Electrostatic exclusions =", len(electrostatic_exclusions))
print(f"    ESForce exceptions =", ESForce.getNumExceptions())
assert(ESForce.getNumParticles() == system.getNumParticles())
ESForce.setForceGroup(12)

#────────────────────────── Add PROTEIN-DNA forces to system ─────────────────────────────
system.addForce(EVForce_dna)
system.addForce(LJForce_ion_ion)
system.addForce(LJForce_dna_ion)
system.addForce(EV_PP)
system.addForce(EV_PD)
system.addForce(EV_PI)
system.addForce(ESForce)

if DO_CMR:
    system.addForce(CMMotionRemover(CMR_FREQ))

for i in range(system.getNumForces()):
    system.getForce(i).setForceGroup(i)

fe = ["FENE force", "Native_force", "dna_bond_force", "dna_angle_force", "StackForce", "HBForce_WC", "EVForce_dna",
      "LJForce_ion_ion", "LJForce_dna_ion", "EV_PP", "EV_PD", "EV_PI", "ESForce"]

if DO_CMR:
    fe.append("CMMotionRemover")

#─────────────────────────────────────────────────────────────────────
# REPORTER: Energy breakdown by force group
#─────────────────────────────────────────────────────────────────────
class EnergyBreakdownReporter:
    def __init__(self, file, reportInterval, forceNames):
        self._out = open(file, "w")
        self._reportInterval = reportInterval
        self._forceNames = forceNames

        header = "Step"
        for name in forceNames:
            header += f",{name}"
        self._out.write(header + "\n")

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep % self._reportInterval
        return (steps, False, False, False, True, None)

    def report(self, simulation, state):

        step = simulation.currentStep

        values = [str(step)]

        print(f"\n=== Energy Breakdown @ step %d ===" % step)

        for i, name in enumerate(self._forceNames):

            e = simulation.context.getState(
                getEnergy=True,
                groups={i}
            ).getPotentialEnergy()

            ekj = e.value_in_unit(unit.kilojoule_per_mole)

            values.append(f"{ekj:.6f}")

            print(f"{name:30s} : {ekj:15.6f} kJ/mol")

        self._out.write(",".join(values) + "\n")
        self._out.flush()

    def __del__(self):
        try:
            self._out.close()
        except:
            pass

# --------------------------------------------------
# Save unique checkpoints
# --------------------------------------------------

class MultiCheckpointReporter:

    def __init__(self, prefix, interval):
        self.prefix = prefix
        self.interval = interval

    def describeNextReport(self, simulation):
        steps = self.interval - simulation.currentStep % self.interval
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):
        fname = f"{self.prefix}_{simulation.currentStep}.chk"
        simulation.saveCheckpoint(fname)
        print(f"[CHECKPOINT] Saved {fname}")


# --------------------------------------------------
# Save positions + velocities
# --------------------------------------------------

class StateSaverReporter:

    def __init__(self, prefix, interval):
        self.prefix = prefix
        self.interval = interval

    def describeNextReport(self, simulation):
        steps = self.interval - simulation.currentStep % self.interval
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):

        state = simulation.context.getState(
            getPositions=True,
            getVelocities=True,
            getEnergy=True
        )
        fname = f"{self.prefix}_{simulation.currentStep}.xml"

        with open(fname, "w") as f:
            f.write(XmlSerializer.serialize(state))

        print(f"[STATE] Saved {fname}")


# --------------------------------------------------
# Record largest force in system
# --------------------------------------------------

class MaxForceReporter:
    def __init__(self, filename, interval):
        self.interval = interval
        self.out = open(filename, "w")

        self.out.write(
            "Step,Particle,MaxForce\n"
        )

    def describeNextReport(self, simulation):
        steps = self.interval - simulation.currentStep % self.interval
        return (steps, False, False, False, False, False)

    def report(self, simulation, state):

        state = simulation.context.getState(getForces=True)
        forces = state.getForces(asNumpy=True)
        fmag = np.sqrt(np.sum(forces._value**2, axis=1))
        imax = np.argmax(fmag)
        self.out.write(f"{simulation.currentStep},{imax},{fmag[imax]}\n")
        self.out.flush()

# ─────────────────────────────────────────────────────────────────────
# INTEGRATOR + SIMULATION
# ─────────────────────────────────────────────────────────────────────
# integrator = LangevinMiddleIntegrator(TEMPERATURE * unit.kelvin, FRICTION / unit.picosecond, TIMESTEP_PS * unit.picoseconds)
integrator = BrownianIntegrator(TEMPERATURE * unit.kelvin, FRICTION / unit.picosecond, TIMESTEP_PS * unit.picoseconds)
integrator.setRandomNumberSeed(69)
platform = Platform.getPlatformByName(PLATFORM)
simulation = Simulation(topology, system, integrator, platform)

if RESTART:
    print(f"    Restarting from checkpoint: {chk_start}...")
    simulation.loadCheckpoint(chk_start)
else:
    simulation.context.setPositions(positions_openmm)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE * unit.kelvin, 69)

if DO_MINIMISE and not RESTART:
    print(f"    Energy minimisation ...")
    simulation.minimizeEnergy(maxIterations=MIN_MAXITER, tolerance=MIN_TOL)
    print(f"    Energy after minimisation:",
          simulation.context.getState(getEnergy=True).getPotentialEnergy())
    
# A quick check of the forces on each particle, to see if any are unreasonably large
state = simulation.context.getState(getForces=True)
forces = state.getForces(asNumpy=True)
fmag = np.sqrt(np.sum(forces._value**2, axis=1))
imax = np.argmax(fmag)
print(f"    Largest force particle =", imax)
print(f"    Largest force magnitude =", fmag[imax])
print(f"    Mean force =", np.mean(fmag))
print(f"    95 percentile =", np.percentile(fmag,95))

# A quick check of the forces on each particle, to see if any are unreasonably large, broken down by force group
for i,name in enumerate(fe):

    state = simulation.context.getState(
        getForces=True,
        groups={i}
    )

    forces = state.getForces(asNumpy=True)

    fmag = np.sqrt(np.sum(forces._value**2,axis=1))

    imax = np.argmax(fmag)

    print(name)
    print(f"   max particle =",imax)
    print(f"   max force =",fmag[imax])

os.makedirs(os.path.dirname(dcd_out) if os.path.dirname(dcd_out) else ".", exist_ok=True)
os.makedirs(CHK_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)
simulation.reporters.append(DCDReporter(dcd_out, REPORT_EVERY))
simulation.reporters.append(StateDataReporter(
    log_out, REPORT_EVERY, step=True, time=True, potentialEnergy=True, kineticEnergy=True,
    temperature=True, progress=True, remainingTime=True, speed=True,
    totalSteps=N_STEPS_EQUIL + N_STEPS_PROD, separator=","))
simulation.reporters.append(StateDataReporter(
    sys.stdout, 5000, step=True, temperature=True, potentialEnergy=True,
    speed=True, progress=True, remainingTime=True,
    totalSteps=N_STEPS_EQUIL + N_STEPS_PROD, separator=" | "))
# simulation.saveCheckpoint(chk_prod, 5000)
simulation.reporters.append(
    EnergyBreakdownReporter(
        energy_breakdown,
        100000,      # every 100k steps
        fe
    )
)

# checkpoint every 100k Steps
simulation.reporters.append(MultiCheckpointReporter(os.path.join(CHK_DIR, "chk"), 100000))

# full state every 10k steps
simulation.reporters.append(StateSaverReporter(os.path.join(STATE_DIR, "state"), 100000))

# largest force every 10k steps
simulation.reporters.append(MaxForceReporter(os.path.join(STATE_DIR, "maxforce.csv"), 10000))

print("\nEnergy components before dynamics")
for i, name in enumerate(fe):
    e = simulation.context.getState(getEnergy=True, groups={i}).getPotentialEnergy()
    print(f"    {name}: {e}")

print(f"\nRunning equilibration ({N_STEPS_EQUIL} steps) ...")
simulation.step(N_STEPS_EQUIL)
simulation.saveCheckpoint(chk_equil)

print(f"\nRunning production ({N_STEPS_PROD} steps) ...")
simulation.step(N_STEPS_PROD)
simulation.saveCheckpoint(chk_prod)

state = simulation.context.getState(getPositions=True, enforcePeriodicBox=True)
simulation.topology.setPeriodicBoxVectors(None)

with open(final_pdb, "w") as f:
    PDBFile.writeFile(
        simulation.topology,
        state.getPositions(),
        f
    )

print("\nSimulation complete.")
print(f"  Trajectory: {dcd_out}")
print(f"  Log: {log_out}")
print(f"  Final structure: {final_pdb}")