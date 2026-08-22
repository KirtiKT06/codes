"""
sop_hpc_openmm.py
=================
Self-Organised Polymer (SOP) simulation of the Histone Protein Core (HPC)
with histone tails, in explicit monovalent salt (K⁺ / Cl⁻).

SYSTEM
------
  • 952 CG protein beads  (761 core + 191 tail)
  • Explicit K⁺ and Cl⁻  ions at 100 mM (the Debye-Hückel reference
    concentration used by Reddy & Thirumalai, but now treated explicitly)

HAMILTONIAN (forces)
--------------------
The total energy is:

  E_total = E_FENE + E_EV + E_native + E_Coul

  (1) E_FENE  — chain connectivity (bonds between consecutive beads)
  (2) E_EV    — excluded volume (repulsive, between non-bonded beads)
  (3) E_native— native contacts (attractive LJ for crystal contacts)
  (4) E_Coul  — Coulomb interaction (charged beads ↔ charged beads ↔ ions)

Each term is explained in detail below.

CHANGES FROM REDDY-THIRUMALAI (screened Coulomb → explicit ions)
-----------------------------------------------------------------
Reddy & Thirumalai use a SCREENED Coulomb (Debye-Hückel):
    U_DH(r) = q_i q_j / (4π ε ε₀ r) · exp(−κ r)
where κ = 1/λ_D is the inverse Debye length pre-computed for 100 mM
monovalent salt.  This is a MEAN-FIELD treatment — the ions are
averaged out and replaced by an exponential decay factor.

WE USE EXPLICIT IONS instead:
    U_Coul(r) = q_i q_j / (4π ε ε₀ r)
    (bare Coulomb with no screening pre-factor)

The screening then emerges DYNAMICALLY from the ion positions.  This is
more expensive but physically richer:
  • You can watch ions condensing onto charged tail residues.
  • You can directly compute ion-mediated bridging between tail and DNA.
  • Salt concentration is controlled by the NUMBER of ions added, not κ.

We set ε = 80 (water at 300 K) for the Coulomb interactions.
The protein–protein Coulomb uses ε = 10 as in Reddy-Thirumalai for the
intra-HPC part (the lower dielectric models the partially desolvated
protein interior). Ion–protein and ion–ion use ε = 80.

Ion model: WCA-sphere + charge.  No LJ well (purely repulsive + Coulomb).
Ion radius σ_ion = 2.35 Å  (K⁺ effective radius), 1.81 Å  (Cl⁻).
These are consistent with SPC/E water-calibrated parameters.
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import json, sys
import numpy as np
from openmm import *
from openmm.app import *
import openmm.unit as unit
from openmm.app import PDBxFile, PDBFile

# ─────────────────────────────────────────────────────────────────────────────
# LOAD CONFIG
# ─────────────────────────────────────────────────────────────────────────────
cfg = json.load(open("input.json"))

# ── File paths ────────────────────────────────────────────────────────────────
pdb_file             = cfg["cg_pdb_with_tails"]
native_contacts_file = cfg["native_contacts_file"]
charged_beads_file   = cfg["charged_beads_file"]
dcd_out              = cfg["dcd_prefix"] + ".dcd"
log_out              = cfg["data_name"]
chk_equil            = cfg["checkpoint_name"]
chk_prod             = chk_equil.replace(".chk", "_prod_final.chk")
final_pdb            = cfg["final_state_name"]

# ── SOP protein forcefield (from Table S1, Reddy & Thirumalai 2021) ──────────
K_FENE          = cfg["fene_k"]           # kcal / (mol · Å²)
R0_FENE         = cfg["fene_R0"]          # Å
SIGMA_PROTEIN   = cfg["sigma_protein"]    # Å
EPS_H_NATIVE    = cfg["eps_h_native"]     # kcal/mol
EPS_L_NONNATIVE = cfg["eps_local"]        # kcal/mol

# ── Coulomb / dielectric ──────────────────────────────────────────────────────
DIELECTRIC_PROTEIN = cfg["dielectric_protein"]
DIELECTRIC_WATER   = cfg["dielectric_water"]

# ── Explicit ions ─────────────────────────────────────────────────────────────
KCl_mM       = cfg["KCl_mM"]
MgCl2_mM     = cfg["MgCl2_mM"]
SIGMA_K      = cfg["rK"]                  # Å
SIGMA_CL     = cfg["rCl"]                 # Å
SIGMA_Mg     = cfg["rMg"]                 # Å
EPS_ION_K    = cfg["epsK"]                # kcal/mol (same for both ions here)
EPS_ION_Cl   = cfg["epsCl"]
EPS_ION_Mg   = cfg["epsMg"]

# ── Simulation ────────────────────────────────────────────────────────────────
TEMPERATURE   = cfg["Temp"]               # K      # K
FRICTION      = cfg["friction_coeff"]     # ps⁻¹
TIMESTEP_PS   = cfg["time_step"]          # ps
N_STEPS_EQUIL = cfg["numsteps_equil"]
N_STEPS_PROD  = cfg["numsteps_prod"]
REPORT_EVERY  = cfg["data_interval"]
BOX_PADDING   = cfg["box_padding"]        # Å
BOX_SCALE     = cfg["box_scale_factor"]   # scale factor for box size

DO_MINIMISE   = cfg["minimization"]
MIN_MAXITER   = cfg["minimization_maxiter"]
MIN_TOL       = cfg["minimization_tol"]
PLATFORM      = cfg["platform_type"]
RESTART       = cfg["restart"]
DO_CMR        = cfg["ComMotionRemover"]
CMR_FREQ      = cfg["CMMR_frequency"]
CHAIN_ORDER   = cfg["histone_chains"]
PME_CUTOFF    = cfg["pme_cutoff"] * 0.1  # Å → nm

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: unit conversions
# ─────────────────────────────────────────────────────────────────────────────
# OpenMM native units: nm, kJ/mol, ps, elementary charge, K
# We work in Å / kcal throughout config, then convert at force creation.

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
print(f"  Temperature: {TEMPERATURE} K, water dielectric: {eps_water:.2f}, sqrt(ε): {dielectric_sqrt:.2f}")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD INPUT FILES
# ─────────────────────────────────────────────────────────────────────────────
print("Loading input files ...")

# ── Protein beads ─────────────────────────────────────────────────────────────
beads = []
with open(pdb_file) as f:
    for line in f:
        if line[:6].strip() != "ATOM":
            continue
        beads.append({
            "chain"  : line[21],
            "resid"  : int(line[22:26]),
            "resname": line[17:20].strip(),
            "x"      : float(line[30:38]),
            "y"      : float(line[38:46]),
            "z"      : float(line[46:54]),
        })

N_protein = len(beads)
print(f"  {N_protein} protein beads loaded.")

# ── Native contacts ──────────────────────────────────────────────────────────
# native_contacts.dat was built from histone_cg.pdb (761 beads, 0-indexed).
# After adding 191 tail beads prepended PER CHAIN, the core bead indices shift.
# We must remap: for each chain, tail beads come first.

# Build index map: old_index → new_index
# Old order: A(97) B(78) C(108) D(96) E(99) F(85) G(104) H(94)
# New order: A_tail(37) A_core(97) B_tail(24) B_core(78) ...
chain_order = CHAIN_ORDER
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

print(f"  {len(native_contacts)} native contacts loaded and remapped.")

# ── Charged beads ─────────────────────────────────────────────────────────────
charged_protein = {}   # bead_idx → charge (elementary units)
with open(charged_beads_file) as f:
    for line in f:
        if line.startswith("#"):
            continue
        parts = line.split()
        idx, charge = int(parts[0]), float(parts[4])
        charged_protein[idx] = charge

print(f"  {len(charged_protein)} charged protein beads.")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD OpenMM TOPOLOGY
# ─────────────────────────────────────────────────────────────────────────────
print("Building topology ...")
topology = Topology()

# Add chains and residues for protein
omm_chains = {}
omm_residues = []
for ch_id in chain_order:
    omm_chains[ch_id] = topology.addChain(id=ch_id)

for b in beads:
    res = topology.addResidue(b["resname"], omm_chains[b["chain"]])
    omm_residues.append(res)

# Add protein atoms (one per bead)
omm_atoms_protein = []
for i, (b, res) in enumerate(zip(beads, omm_residues)):
    atom = topology.addAtom("BB", Element.getBySymbol("C"), res)
    omm_atoms_protein.append(atom)

# Placeholder for ion atoms (added later after computing count)
# We'll add them to topology after computing box size

# ─────────────────────────────────────────────────────────────────────────────
# COMPUTE BOX SIZE AND ION COUNT
# ─────────────────────────────────────────────────────────────────────────────
# CG BOX PHILOSOPHY:
# In an explicit-solvent AA simulation, the box volume sets the salt
# concentration directly because water fills the space.
# In a CG simulation there is NO explicit solvent — the "box" is just a
# periodic boundary to avoid finite-size artefacts.  Setting the box from
# concentration × volume gives nonsensically large ion counts.
#
# Instead we:
#   (1) Set the box to the protein's own extent + a small padding (minimum
#       image convention: box > 2 × cutoff on each side so no particle
#       interacts with its own image).
#   (2) Set the ion COUNT directly in input.json (nK, nCl), just like your
#       lab's standard input file does with nK, nMg, nCl.
#       Neutralisation ions are added on top to make the system charge-neutral.

coords_nm = np.array([[b["x"], b["y"], b["z"]] for b in beads]) * 0.1  # Å→nm

# Protein geometric extent
prot_min  = coords_nm.min(axis=0)
prot_max  = coords_nm.max(axis=0)
prot_span = prot_max - prot_min   # nm, actual size of protein

# Minimum image convention:
#   box must be > 2 × nonlocal_cutoff so no bead sees its own image.
#   nonlocal_cutoff comes from input.json (in Å → convert to nm).
min_box   = 2.0 * cfg.get("nonlocal_cutoff", 30.0) * 0.1   # Å → nm

# Box = max(protein span + padding, minimum image requirement) per axis
padding_nm = BOX_PADDING * 0.1
box_scale  = BOX_SCALE

box_size = np.maximum(
    prot_span * box_scale,
    min_box * np.ones(3)
)

# Centre protein in box
centre     = 0.5 * box_size
prot_centre= 0.5 * (prot_min + prot_max)
coords_nm  = coords_nm - prot_centre + centre   # protein centred in box

topology.setPeriodicBoxVectors((
    Vec3(box_size[0], 0, 0) * unit.nanometer,
    Vec3(0, box_size[1], 0) * unit.nanometer,
    Vec3(0, 0, box_size[2]) * unit.nanometer
))

# ION COUNT — read directly from input.json, same convention as lab standard.
# Set nK / nCl explicitly.  If auto_ions=true, add neutralisation ions on top.
net_protein_charge = int(round(sum(charged_protein.values())))

if cfg.get("auto_ions", True):
    # Volume-based ion count: N = C × Nₐ × V
    # V is the box volume in litres
    N_A         = 6.022e23
    box_vol_nm3 = float(np.prod(box_size))           # nm³
    box_vol_L   = box_vol_nm3 * 1e-27 * 1e3          # nm³ → L
    n_kcl_pairs = max(0, round(KCl_mM * 1e-3 * N_A * box_vol_L))
    n_mgcl2_pairs = max(0, round(MgCl2_mM * 1e-3 * N_A * box_vol_L))
    n_k = n_kcl_pairs
    n_mg = n_mgcl2_pairs
    n_cl = n_kcl_pairs + 2 * n_mgcl2_pairs

    # Neutralisation: add extra counterions on top of salt ions
    if net_protein_charge > 0:
        n_k = n_kcl_pairs + cfg.get("nK", 0)
        n_cl = (n_kcl_pairs + 2 * n_mgcl2_pairs) + cfg.get("nCl", 0) + net_protein_charge
    else:
        n_k = n_kcl_pairs + cfg.get("nK", 0) + abs(net_protein_charge)
        n_cl = (n_kcl_pairs + 2 * n_mgcl2_pairs) + cfg.get("nCl", 0)
else:
    n_k = cfg.get("nK", 0)
    n_cl = cfg.get("nCl", 0)
    n_mg = cfg.get("nMg", 0)

N_ions = n_k + n_cl + n_mg

print(f"  Protein span:       {prot_span[0]:.2f} × {prot_span[1]:.2f} × {prot_span[2]:.2f} nm")
print(f"  Box size:           {box_size[0]:.2f} × {box_size[1]:.2f} × {box_size[2]:.2f} nm")
print(f"  Net protein charge: {net_protein_charge:+d} e")
print(f"  Ions:               {n_k} K⁺ + {n_cl} Cl⁻ + {n_mg} Mg²⁺ = {N_ions} total")

# Add ion residues to topology
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

# ─────────────────────────────────────────────────────────────────────────────
# INITIAL POSITIONS
# ─────────────────────────────────────────────────────────────────────────────
# Place ions randomly in box, avoiding clashes with protein
print("Placing ions ...")
np.random.seed(42)
ion_positions = []
protein_kdtree_coords = coords_nm.copy()

for _ in range(N_ions):
    for attempt in range(50000):
        pos = np.random.rand(3) * box_size + np.array([0.0, 0.0, 0.0])
        dists = np.linalg.norm(protein_kdtree_coords - pos, axis=1)
        if dists.min() > 0.8:   # 8 Å minimum from protein bead
            ion_positions.append(pos)
            break
    else:
        # fallback: place at random (will be relaxed)
        ion_positions.append(np.random.rand(3) * box_size)

all_positions = np.vstack([coords_nm, np.array(ion_positions)])
positions_openmm = [Vec3(*p) * unit.nanometer for p in all_positions]

N_total = N_protein + N_ions
print(f"  Total particles: {N_total}")

# ─────────────────────────────────────────────────────────────────────────────
# BUILD FORCE FIELD
# ─────────────────────────────────────────────────────────────────────────────
print("Building force field ...")
system = System()
system.setDefaultPeriodicBoxVectors(
    Vec3(box_size[0], 0, 0) * unit.nanometer,
    Vec3(0, box_size[1], 0) * unit.nanometer,
    Vec3(0, 0, box_size[2]) * unit.nanometer
)
for _ in range(N_total):
    system.addParticle(1.0 * unit.amu)   # unit mass (rescaled via friction)

# ── FORCE (1): FENE bonds ─────────────────────────────────────────────────────
#
# The Finitely Extensible Nonlinear Elastic (FENE) potential:
#
#   U_FENE(r) = - (k/2) R₀² ln[1 − (r − r_cry)² / R₀²]
#
# where:
#   r     = current bond length between beads i and i+1
#   r_cry = equilibrium (crystal) bond length = distance in CG structure
#   R₀    = maximum extension beyond equilibrium (2.0 Å)
#   k     = spring constant (20 kcal/mol/Å²)
#
# Why FENE instead of harmonic?
#   A harmonic spring can stretch to infinity.  FENE diverges as
#   r − r_cry → R₀, which prevents unphysical chain crossing.
#   This is critical for CG polymer simulations.
#
# CustomBondForce formula in OpenMM (r is bond length in nm):
#   -0.5 * K * R0² * log(1 - ((r - r0)/R0)²)
#   All distances in nm, energies in kJ/mol.

print("  Adding FENE bonds ...")
k_fene_kJ_nm2 = kcal_to_kJ(K_FENE) * 100.0   # kcal/(mol·Å²) → kJ/(mol·nm²)
R0_nm         = A_to_nm(R0_FENE)

fene_bond = CustomBondForce(
    "-0.5 * K * R0*R0 * log(1 - ((r - r0)/R0)^2)"
)
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
system.addForce(fene_bond)
print(f"    {fene_bond.getNumBonds()} FENE bonds added.")

# ── FORCE (2): Excluded Volume (non-native repulsion) ────────────────────────
#
# For all pairs NOT in native contacts AND not bonded (seq_sep ≥ 3):
#
#   U_EV(r) = ε_l · (σ/r)⁶
#
# This is the purely repulsive r⁻⁶ term (the repulsive half of LJ).
# It prevents bead overlap without any attractive well.
#
# For protein-protein:  σ = σ_protein = 3.8 Å
# For protein-ion:      σ = (σ_protein + σ_ion) / 2  (Lorentz-Berthelot)
# For ion-ion:          σ = σ_ion (same type) or mixed
#
# We implement this via CustomNonbondedForce with a per-particle σ.
# The formula: eps_l * (sigma/r)^6 where sigma = (sigma_i + sigma_j)/2

print("  Adding excluded volume ...")

ev_force = CustomNonbondedForce(
    "eps_l * (sigma / r)^6 ; sigma = 0.5*(sigma1 + sigma2)"
)
ev_force.addGlobalParameter("eps_l", kcal_to_kJ(EPS_L_NONNATIVE))
ev_force.addPerParticleParameter("sigma")
ev_force.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
ev_force.setCutoffDistance(PME_CUTOFF * unit.nanometer)

# Protein beads
for i in range(N_protein):
    ev_force.addParticle([A_to_nm(SIGMA_PROTEIN)])

# Ions
for k in range(n_k):
    ev_force.addParticle([A_to_nm(SIGMA_K)])
for k in range(n_cl):
    ev_force.addParticle([A_to_nm(SIGMA_CL)])
for k in range(n_mg):
    ev_force.addParticle([A_to_nm(SIGMA_Mg)])

# Exclude bonded pairs AND seq_sep < 3 on same chain AND native contacts from EV.
# CRITICAL: OpenMM requires CustomNonbondedForce and NonbondedForce to have
# IDENTICAL exclusion sets. We build one master set and apply to both.
native_contact_set = set((min(i,j), max(i,j)) for i, j, _ in native_contacts)

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

# Apply to EV force
for i, j in master_exclusions:
    ev_force.addExclusion(i, j)

ev_force.setForceGroup(1)
system.addForce(ev_force)

# ── FORCE (3): Native contacts (Lennard-Jones 12-6) ──────────────────────────
#
# For pairs in the native contact list:
#
#   U_native(r) = ε_h · [(r_cry/r)¹² − 2(r_cry/r)⁶]
#
# This is a 12-6 LJ potential with:
#   minimum at r = r_cry  (the crystal contact distance)
#   well depth = ε_h = 2.0 kcal/mol  (for HPC-HPC, Table S1)
#
# Note the factor of 2 in front of the attractive term ensures the
# minimum value is exactly −ε_h (not −ε_h/4 as in standard LJ).
#
# r_cry values come from native_contacts.dat (distances in Å).

print("  Adding native contacts ...")
native_force = CustomBondForce(
    "eps_h * ((r0/r)^12 - 2*(r0/r)^6)"
)
native_force.addPerBondParameter("r0")     # crystal distance (nm)
native_force.addPerBondParameter("eps_h")  # well depth (kJ/mol)
native_force.setUsesPeriodicBoundaryConditions(True)

eps_h_kJ = kcal_to_kJ(EPS_H_NATIVE)
for i, j, r0_A in native_contacts:
    native_force.addBond(i, j, [A_to_nm(r0_A), eps_h_kJ])

native_force.setForceGroup(2)
system.addForce(native_force)
print(f"    {native_force.getNumBonds()} native contact pairs added.")

# ── FORCE (4): Coulomb interactions ──────────────────────────────────────────
#
# EXPLICIT-ION COULOMB (replaces Debye-Hückel)
# ─────────────────────────────────────────────
# U_Coul(r) = (q_i · q_j · e²) / (4π ε₀ ε r)
#
# In Reddy-Thirumalai the ions are implicit and ε = 10 for intra-protein
# charges, yielding an effective screened interaction.  Here we use
# explicit ions, so:
#
#   Protein–protein charged pairs: ε = DIELECTRIC_PROTEIN = 10
#     (models the low-dielectric protein interior where charges are
#      partially shielded from bulk water)
#
#   Protein–ion and ion–ion:       ε = DIELECTRIC_WATER = 80
#     (these interactions occur in the aqueous phase)
#
# The emergent screening of protein charges by the ions naturally
# reproduces Debye-Hückel-like behaviour at long range, but with
# full many-body accuracy at short range.
#
# WHY THIS MATTERS FOR YOUR STUDY:
#   The 60 positively charged tail residues (net +56) are the primary
#   electrostatic handle by which tails grip the negatively charged DNA.
#   With explicit ions you can directly observe:
#     (a) K⁺ condensation on the DNA (Manning condensation)
#     (b) Cl⁻ exclusion from the tail-DNA interface
#     (c) Ion-mediated bridging between tail ARG/LYS and DNA phosphates
#
# Implementation: we use TWO NonbondedForce objects (OpenMM allows this):
#   Force A: protein–protein Coulomb  (ε = 10, PME or reaction-field)
#   Force B: protein–ion + ion–ion    (ε = 80, PME)
# This requires careful exclusion management.

print("  Adding Coulomb forces ...")

ONE_4PI_EPS0 = 138.935458   # kJ·nm / (mol · e²)  — OpenMM's value

# We use a single NonbondedForce with per-particle charges and
# a UNIFORM ε = 80, then correct protein-protein to ε = 10 via
# a CustomBondForce for charged protein pairs.
#
# Strategy:
#   (a) Main NonbondedForce: ALL charges, ε_r = 80, PME
#   (b) Correction CustomBondForce: for each protein–protein charged pair,
#       add a correction term:
#       ΔU = q_i q_j / (4π ε₀ r) · (1/ε_prot − 1/ε_water)
#       = ONE_4PI_EPS0 · q_i q_j · (1/10 − 1/80) / r

# (a) Main NonbondedForce
nb_force = NonbondedForce()
nb_force.setNonbondedMethod(NonbondedForce.PME)
nb_force.setCutoffDistance(PME_CUTOFF * unit.nanometer)
nb_force.setEwaldErrorTolerance(1e-4)
nb_force.setReactionFieldDielectric(DIELECTRIC_WATER)

for i in range(N_protein):
    # q = charged_protein.get(i, 0.0)
    q = charged_protein.get(i, 0.0) / dielectric_sqrt
    nb_force.addParticle(q, A_to_nm(SIGMA_PROTEIN) * unit.nanometer, 0.0)

for k in range(n_k):
    nb_force.addParticle(+1.0/dielectric_sqrt, A_to_nm(SIGMA_K) * unit.nanometer, 0.0)
for k in range(n_cl):
    nb_force.addParticle(-1.0/dielectric_sqrt, A_to_nm(SIGMA_CL) * unit.nanometer, 0.0)
for k in range(n_mg):
    nb_force.addParticle(+2.0/dielectric_sqrt, A_to_nm(SIGMA_Mg) * unit.nanometer, 0.0)

# Apply the SAME master exclusion set to NonbondedForce (must match EV force exactly)
for i, j in master_exclusions:
    # For native contact pairs and seq_sep pairs: zero out Coulomb too
    # (their electrostatics are handled by the correction CustomBondForce)
    nb_force.addException(i, j, 0.0, 1.0, 0.0)

nb_force.setForceGroup(3)
system.addForce(nb_force)

# (b) Protein–protein dielectric correction:
#     ΔU_corr = q_i q_j · C · (1/ε_prot − 1/ε_water) / r
#     C = ONE_4PI_EPS0 = 138.935 kJ·nm/mol/e²

corr_factor = ONE_4PI_EPS0 * (1.0/DIELECTRIC_PROTEIN - 1.0/DIELECTRIC_WATER)

prot_prot_coul = CustomBondForce(f"{corr_factor} * qi * qj / max(r, 0.1)")
prot_prot_coul.addPerBondParameter("qi")
prot_prot_coul.addPerBondParameter("qj")
prot_prot_coul.setUsesPeriodicBoundaryConditions(True)

charged_indices = sorted(charged_protein.keys())
# Add correction for all charged protein pairs within reasonable range.
# We add ALL pairs; the 1/r decay takes care of long-range.
# For very large systems this would need cutoff management, but at 952 beads
# the number of charged pairs is manageable.
for ii in range(len(charged_indices)):
    for jj in range(ii + 1, len(charged_indices)):
        i_idx = charged_indices[ii]
        j_idx = charged_indices[jj]
        qi = charged_protein[i_idx] / dielectric_sqrt
        qj = charged_protein[j_idx] / dielectric_sqrt
        # Skip if bonded
        if (min(i_idx, j_idx), max(i_idx, j_idx)) in bond_pairs:
            continue
        if (min(i_idx, j_idx), max(i_idx, j_idx)) in master_exclusions:  # ADD THIS
            continue
        prot_prot_coul.addBond(i_idx, j_idx, [qi, qj])

prot_prot_coul.setForceGroup(4)
system.addForce(prot_prot_coul)
print(f"    {prot_prot_coul.getNumBonds()} protein-protein Coulomb correction pairs.")

# ─────────────────────────────────────────────────────────────────────────────
# INTEGRATOR: Langevin (Brownian dynamics limit)
# ─────────────────────────────────────────────────────────────────────────────
#
# The Langevin integrator at high friction (ζ >> √(k·m)) gives
# Brownian/Overdamped dynamics, which is what Reddy-Thirumalai use.
# γ = FRICTION = 50 ps⁻¹ for protein beads (ζ_P = 50 τ_L⁻¹).
if DO_CMR:
    system.addForce(CMMotionRemover(CMR_FREQ))


integrator = LangevinMiddleIntegrator(
    TEMPERATURE * unit.kelvin,
    FRICTION / unit.picosecond,
    TIMESTEP_PS * unit.picoseconds
)

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION SETUP
# ─────────────────────────────────────────────────────────────────────────────
print("Setting up simulation ...")
platform   = Platform.getPlatformByName(PLATFORM)
simulation = Simulation(topology, system, integrator, platform)

if RESTART:
    print(f"  Restarting from checkpoint: {chk_equil}")
    simulation.loadCheckpoint(chk_equil)
else:
    simulation.context.setPositions(positions_openmm)
    simulation.context.setVelocitiesToTemperature(TEMPERATURE * unit.kelvin)

# ─────────────────────────────────────────────────────────────────────────────
# ENERGY MINIMISATION
# ─────────────────────────────────────────────────────────────────────────────
if DO_MINIMISE and not RESTART:
    print("Energy minimisation ...")
    simulation.minimizeEnergy(maxIterations=MIN_MAXITER, tolerance=MIN_TOL)
    state = simulation.context.getState(getEnergy=True)

    for group, name in [
    (0, "FENE"),
    (1, "ExcludedVolume"),
    (2, "Native"),
    (3, "PME"),
    (4, "ProtProtCorr")
    ]:
        e = simulation.context.getState(
            getEnergy=True,
            groups={group}
        ).getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)

    print(name, e)
    print(f"  Energy after minimisation: "
          f"{state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole):.1f} kcal/mol")

# ─────────────────────────────────────────────────────────────────────────────
# REPORTERS
# ─────────────────────────────────────────────────────────────────────────────
import os
os.makedirs(os.path.dirname(dcd_out) if os.path.dirname(dcd_out) else ".", exist_ok=True)

simulation.reporters.append(
    DCDReporter(dcd_out, REPORT_EVERY)
)
simulation.reporters.append(
    StateDataReporter(
        log_out, REPORT_EVERY,
        step=True, time=True, potentialEnergy=True, kineticEnergy=True,
        temperature=True, progress=True, remainingTime=True,
        totalSteps=N_STEPS_EQUIL + N_STEPS_PROD, separator=","
    )
)
simulation.reporters.append(
    StateDataReporter(
        sys.stdout,
        5000,
        step=True,
        temperature=True,
        potentialEnergy=True,
        density=True,
        speed=True,
        progress=True,
        remainingTime=True,
        totalSteps=N_STEPS_EQUIL + N_STEPS_PROD,
        separator=" | "
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# RUN: EQUILIBRATION → PRODUCTION
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nRunning equilibration ({N_STEPS_EQUIL} steps = "
      f"{N_STEPS_EQUIL * TIMESTEP_PS * 1e-3:.2f} ns) ...")

simulation.step(N_STEPS_EQUIL)

print("\nEnergy breakdown after equilibration")
for group, name in [
    (0, "FENE"),
    (1, "ExcludedVolume"),
    (2, "Native"),
    (3, "PME"),
    (4, "ProtProtCorr")
]:
    e = simulation.context.getState(
        getEnergy=True,
        groups={group}
    ).getPotentialEnergy()

    print(name, e)

simulation.saveCheckpoint(chk_equil)
print("Equilibration complete. Checkpoint saved.")

print(f"\nRunning production ({N_STEPS_PROD} steps = "
      f"{N_STEPS_PROD * TIMESTEP_PS * 1e-3:.2f} ns) ...")
simulation.step(N_STEPS_PROD)
simulation.saveCheckpoint(chk_prod)

# state = simulation.context.getState(getPositions=True)
# positions_out = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
# with open(final_pdb, "w") as f:
#     PDBFile.writeFile(topology, positions_out * unit.nanometer, f)

simulation.topology.setPeriodicBoxVectors(None)
state = simulation.context.getState(
    getPositions=True,
    enforcePeriodicBox=True
)

final_cif = final_pdb.replace(".pdb", ".cif")

with open(final_cif, "w") as f:
    PDBxFile.writeFile(
        simulation.topology,
        state.getPositions(),
        f
    )

print("\nSimulation complete.")
print(f"  Trajectory : {dcd_out}")
print(f"  Log        : {log_out}")
print(f"  Checkpoint : {chk_prod}")
print(f"  Final structure : {final_cif}")

# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS HELPER (run after simulation)
# ─────────────────────────────────────────────────────────────────────────────
# After simulation, you can analyse tail-ion interactions:
#
#   python3 -c "
#   import mdtraj as md
#   t = md.load('hpc_traj.dcd', top='hpc_final.pdb')
#   # Select tail charged residues and ions
#   tail_charged = t.topology.select('name BB and resSeq < 38')
#   na_ions      = t.topology.select('resname NA')
#   # Compute radial distribution function, contact probabilities etc.
#   "
#
# Key observables for your study:
#   (1) Mean distance between tail K/R beads and their nearest ion
#   (2) Radial distribution g(r) between tail⁺ and Cl⁻
#   (3) Number of ions within 10 Å of each tail residue vs time
#   (4) Native contact fraction Q = Σ(contacts formed) / N_native